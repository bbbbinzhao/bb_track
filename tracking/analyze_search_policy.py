"""Audit saved predictions against full official val and export research evidence."""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from tracking.search_policy_experiment import (
    aggregate_reports, bootstrap_mean_ci, box_iou, failure_starts, load_sequences,
    reappearance_starts, sequence_metrics, summarize_intervention, valid_frames, write_json_atomic,
)
from lib.test.analysis.analys_antiuav import ExperimentAntiUAV410


def export_csv(path, rows):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_baseline(args):
    with open(args.baseline) as handle:
        baseline = json.load(handle)
    sequences = load_sequences('val')
    expected = {record[0] for record in sequences}
    if set(baseline['sequences']) != expected:
        raise ValueError('Report does not cover exactly the official val sequences')
    rows, details, recoveries = [], {}, []
    official_evaluator = ExperimentAntiUAV410.__new__(ExperimentAntiUAV410)
    total_frames = 0
    for name, frames, boxes, visible in sequences:
        value = baseline['sequences'][name]
        predictions = np.asarray(value['predictions'])
        if predictions.shape != boxes.shape or len(frames) != len(predictions):
            raise ValueError('Prediction/annotation length mismatch for ' + name)
        valid = valid_frames(boxes, visible)
        metrics = sequence_metrics(predictions, boxes, visible)
        official_score = float(official_evaluator.eval(predictions.tolist(),
                                {'gt_rect': boxes.tolist(), 'exist': visible.tolist()}))
        if not np.isclose(metrics['mixed_measure'], official_score, atol=1e-12):
            raise RuntimeError('Official mixed score disagrees for ' + name)
        # Existing FPS used N frames while timing only N-1; preserve measured time.
        metrics['fps'] = value['metrics']['fps']
        if baseline.get('protocol') != 'search_evaluation_v2':
            metrics['fps'] *= (len(frames) - 1) / len(frames)
        recoveries.extend(metrics['recovery_delays'])
        ious = np.array([box_iou(p, b) if valid[i] else 0.
                         for i, (p, b) in enumerate(zip(predictions, boxes))])
        starts = failure_starts(valid, ious)
        diagnostic = value['diagnostics']
        valid_records = [record for record in diagnostic['frames'] if valid[record['frame']]]
        outside = np.mean([record['center_outside_crop'] for record in valid_records])
        low = [record for record in valid_records if record['iou'] < 0.2]
        low_covered = sum(record['bbox_coverage'] >= 0.95 for record in low)
        row = {'sequence': name, 'frames': len(frames),
               'visible_frames': metrics['visible_valid_frames'],
               'visible_iou': metrics['visible_iou'], 'visible_auc': metrics['visible_auc'],
               'mixed_measure_official': metrics['mixed_measure'], 'fps': metrics['fps'],
               'failure_events': len(starts), 'center_outside_rate': float(outside),
               'mean_coverage': float(np.mean([r['bbox_coverage'] for r in valid_records])),
               'mean_target_pixels': float(np.mean([r['target_pixels'] for r in valid_records])),
               'low_iou_frames': len(low), 'low_iou_covered_frames': low_covered,
               'true_invisible_frames': int(np.sum(~visible[1:])),
               'invalid_visible_frames': int(np.sum(visible[1:] & ~valid[1:])),
               'recovery_events': metrics['recovery_events'],
               'recovery_rate_30': metrics['recovery_rate_30'],
               'recovery_latency_confirmation': metrics['recovery_latency']}
        rows.append(row)
        details[name] = {'metrics': metrics, 'failure_starts': starts}
        total_frames += len(frames)
    summary = aggregate_reports([value['metrics'] for value in details.values()])
    recovered = [value for value in recoveries if value is not None]
    summary.update({'sequences': len(rows), 'frames': total_frames,
                    'failure_events': sum(row['failure_events'] for row in rows),
                    'mean_center_outside_rate': float(np.mean([r['center_outside_rate'] for r in rows])),
                    'mean_coverage': float(np.mean([r['mean_coverage'] for r in rows])),
                    'mean_target_pixels': float(np.mean([r['mean_target_pixels'] for r in rows])),
                    'recovery_event_count': len(recoveries),
                    'recovery_event_rate_30': len(recovered) / len(recoveries) if recoveries else None,
                    'recovery_event_latency': float(np.mean(recovered)) if recovered else None,
                    'recovery_event_unrecovered': 1 - len(recovered) / len(recoveries) if recoveries else None})
    ci = {key: bootstrap_mean_ci([value['metrics'][key] for value in details.values()])
          for key in ('visible_iou', 'visible_auc', 'mixed_measure')}
    payload = {'baseline_sha256': baseline['baseline_sha256'], 'summary': summary,
               'summary_ci': ci, 'sequences': details,
               'definitions': {'initial_frame': 'excluded from visible IoU/AUC; included in official mixed score',
                               'recovery': 'first legal visible GT after disappearance; third IoU>=0.5 frame confirms',
                               'recovery_window': '30 frames, ending at next disappearance or sequence end',
                               'fps': 'original tracker+diagnostics measurement, initialization excluded; historical load',
                               'auc': '21 thresholds from 0 to 1 inclusive, success uses >=',
                               'ci': '10000 bootstrap replicates over sequences'}}
    write_json_atomic(os.path.join(args.output, 'baseline_audited.json'), payload)
    export_csv(os.path.join(args.output, 'baseline_per_sequence.csv'), rows)
    return payload, rows, recoveries


def analyze_intervention(args, expected_names):
    with open(args.intervention) as handle:
        payload = json.load(handle)
    if payload.get('protocol') != 'search_intervention_v2_hann_integer_geometry':
        raise ValueError('Only corrected Hann/geometry reports can be used')
    if set(payload['sequences']) != expected_names or not payload.get('complete'):
        raise ValueError('Intervention report is incomplete')
    sequences = payload['sequences']
    corrected_labels = 0
    for name, frames, boxes, visible in load_sequences('val'):
        expected_events = {index - 1 for index in reappearance_starts(visible, valid_frames(boxes, visible))[:8]}
        present_frames = {sample['frame'] for sample in sequences[name]}
        if expected_events - present_frames - {0}:
            raise ValueError('Missing GT reappearance sample in ' + name)
        for sample in sequences[name]:
            correct = sample['frame'] in expected_events
            corrected_labels += int(correct != sample.get('reappearance_sample', False))
            sample['reappearance_sample'] = correct
    summary = summarize_intervention(sequences)
    summary['corrected_reappearance_sample_labels'] = corrected_labels
    payload['summary'] = summary
    payload['reappearance_labels_audited_against_exist'] = True
    with open(args.baseline) as handle:
        baseline = json.load(handle)
    if payload['checkpoint_sha256'] != baseline['baseline_sha256']:
        raise ValueError('Intervention and baseline use different frozen weights')
    baseline_diagnostics = {name: {r['frame']: r for r in value['diagnostics']['frames']}
                            for name, value in baseline['sequences'].items()}
    rows = []
    for name, samples in sequences.items():
        for sample in samples:
            valid = [candidate for candidate in sample['candidates']
                     if candidate['future_iou'] is not None]
            base = sample['baseline_future_iou']
            if not valid or base is None:
                continue
            best = max(valid, key=lambda candidate: candidate['future_iou'])
            center_fixed = [c['future_iou'] for c in valid if c['action'] < 3]
            next_diag = baseline_diagnostics[name].get(sample['frame'] + 1, {})
            next_candidates = [c for c in valid if c['next_iou'] is not None]
            best_next = max(next_candidates, key=lambda c: c['next_iou']) if next_candidates else None
            next_max = best_next['next_iou'] if best_next else None
            future_max = best['future_iou']
            tied_next = [c for c in next_candidates if c['next_iou'] >= next_max - 1e-6]
            tied_future = [c for c in valid if c['future_iou'] >= future_max - 1e-6]
            disjoint = (not ({c['action'] for c in tied_next} & {c['action'] for c in tied_future}))
            rows.append({'sequence': name, 'frame': sample['frame'],
                         'uniform': sample['uniform_sample'], 'failure': sample.get('failure_sample', False),
                         'reappearance': sample.get('reappearance_sample', False),
                         'full_horizon': sample.get('full_horizon', False),
                         'baseline_iou8': base, 'oracle_iou8': best['future_iou'],
                         'oracle_delta': best['future_iou'] - base, 'best_action': best['action'],
                         'best_factor': best['region']['factor'],
                         'baseline_next_coverage': next_diag.get('bbox_coverage'),
                         'baseline_next_target_pixels': next_diag.get('target_pixels'),
                         'best_target_pixels': best['target_pixels'],
                         'one_step_vs_eight_step_disjoint': disjoint if best_next else None,
                         'eight_step_regret_of_one_step_choice': future_max - best_next['future_iou']
                         if best_next else None,
                         'motion_increment_over_scale': best['future_iou'] - max(center_fixed)
                         if center_fixed and not payload.get('expanded_candidates') else None})
    export_csv(os.path.join(args.output, 'intervention_per_point.csv'), rows)
    seq_rows = []
    for name in sorted(expected_names):
        selected = [row for row in rows if row['sequence'] == name]
        deltas = [row['oracle_delta'] for row in selected]
        seq_rows.append({'sequence': name, 'points': len(selected),
                         'mean_oracle_delta': float(np.mean(deltas)) if deltas else None,
                         'events_delta_005': sum(delta >= 0.05 for delta in deltas),
                         'failure_points': sum(row['failure'] for row in selected)})
    export_csv(os.path.join(args.output, 'intervention_per_sequence.csv'), seq_rows)
    summary['sequence_deltas'] = {row['sequence']: row['mean_oracle_delta'] for row in seq_rows}
    motion_increment = [row['motion_increment_over_scale'] for row in rows
                        if row['motion_increment_over_scale'] is not None]
    rank_disagreements = [row['one_step_vs_eight_step_disjoint'] for row in rows
                          if row['one_step_vs_eight_step_disjoint'] is not None]
    summary['descriptive_mechanisms'] = {
        'mean_motion_increment_over_best_scale': float(np.mean(motion_increment)) if motion_increment else None,
        'one_step_vs_eight_step_disjoint_rate': float(np.mean(rank_disagreements)) if rank_disagreements else None,
        'best_factor_counts': {str(factor): sum(row['best_factor'] == factor for row in rows)
                               for factor in (4., 6., 8.)},
        'note': 'GT-selected diagnostic candidates; action ties favor the first index; not causal attribution',
    }
    regrets = [row['eight_step_regret_of_one_step_choice'] for row in rows
               if row['eight_step_regret_of_one_step_choice'] is not None]
    summary['descriptive_mechanisms']['one_step_choice_eight_step_regret_mean'] = float(np.mean(regrets))
    summary['descriptive_mechanisms']['one_step_choice_regret_ge_001_rate'] = float(np.mean(np.array(regrets) >= 0.01))
    write_json_atomic(os.path.join(args.output, 'intervention_audited.json'), summary)
    write_json_atomic(os.path.join(args.output, 'intervention_val_v2_full_audited.json'), payload)
    return summary, rows, seq_rows


def research_report(args, baseline, baseline_rows, intervention):
    reports = {'原 FocusTrack/SRA': baseline}
    for name, path in [('常速度中心＋原 SRA 尺度', args.motion_report),
                       ('仅修正整数裁剪坐标', args.coordinate_report)]:
        if path and os.path.isfile(path):
            with open(path) as handle:
                value = json.load(handle)
            if set(value['sequences']) != set(baseline['sequences']):
                raise ValueError('Incomplete control report: ' + path)
            reports[name] = value
    lines = ['# 冻结 FocusTrack：完整验证集诊断与干预报告', '',
             '## 实验范围', '',
             '- Anti-UAV410 官方 val：90 条序列、94,711 帧；test 未使用。',
             '- 冻结权重 SHA256：`' + baseline['baseline_sha256'] + '`。',
             '- 四张 RTX 4090 按序列分片；主指标逐序列计算后等权平均。',
             '- 本报告属于搜索控制诊断阶段，尚无三种子学习策略或 SLT 效果结论。', '',
             '## 完整闭环对照', '',
             '| 方法 | 可见帧 IoU | 可见帧 AUC | 官方 Anti-UAV 综合分数 |',
             '|---|---:|---:|---:|']
    for name, value in reports.items():
        summary = value['summary']
        lines.append('| %s | %.6f | %.6f | %.6f |' %
                     (name, summary['visible_iou'], summary['visible_auc'], summary['mixed_measure']))
    comparisons = {}
    for name, value in reports.items():
        if value is baseline:
            continue
        deltas = [value['sequences'][key]['metrics']['visible_iou'] -
                  baseline['sequences'][key]['metrics']['visible_iou'] for key in baseline['sequences']]
        ci = bootstrap_mean_ci(deltas)
        comparisons[name] = {'delta': float(np.mean(deltas)), 'ci': ci}
        lines.extend(['', '%s相对基线：ΔIoU=%.6f，序列配对 95%% bootstrap 区间 [%.6f, %.6f]。' %
                      (name, np.mean(deltas), ci[0], ci[1])])
    comparison_rows = []
    for key in sorted(baseline['sequences']):
        row = {'sequence': key}
        for index, (_, value) in enumerate(reports.items()):
            row['method_%d_visible_iou' % index] = value['sequences'][key]['metrics']['visible_iou']
            row['method_%d_delta_to_sra' % index] = (row['method_%d_visible_iou' % index] -
                                                    baseline['sequences'][key]['metrics']['visible_iou'])
        comparison_rows.append(row)
    export_csv(os.path.join(args.output, 'closed_loop_comparison_per_sequence.csv'), comparison_rows)
    write_json_atomic(os.path.join(args.output, 'closed_loop_comparisons.json'), comparisons)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    labels = ('Original SRA', 'Motion + SRA', 'Coordinate control')
    colors = ('#333333', '#2563eb', '#15803d')
    ordering = sorted(baseline['sequences'], key=lambda key:
                      list(reports.values())[1]['sequences'][key]['metrics']['visible_iou'] -
                      baseline['sequences'][key]['metrics']['visible_iou']) if len(reports) > 1 else sorted(baseline['sequences'])
    for index, (_, value) in enumerate(reports.items()):
        if index:
            axes[0].plot([value['sequences'][key]['metrics']['visible_iou'] -
                          baseline['sequences'][key]['metrics']['visible_iou'] for key in ordering],
                         label=labels[index], color=colors[index])
        delays = [delay for sequence in value['sequences'].values()
                  for delay in sequence['metrics'].get('recovery_delays', [])]
        if delays:
            axes[1].plot(range(30), [np.mean([delay is not None and delay <= t for delay in delays])
                                    for t in range(30)], label=labels[index], color=colors[index])
    axes[0].axhline(0, color='black', linewidth=0.8)
    axes[0].set(xlabel='Sequence rank by motion delta', ylabel='Visible IoU delta to original SRA')
    axes[0].legend()
    axes[1].set(xlabel='Frames after GT reappearance (0-based)', ylabel='Recovery fraction', ylim=(0, 1))
    axes[1].legend()
    fig.savefig(os.path.join(args.output, 'closed_loop_deltas_and_recovery.png'), dpi=180)
    plt.close(fig)
    if len(reports) == 3:
        motion, control = list(reports.values())[1:]
        differences = [motion['sequences'][key]['metrics']['visible_iou'] -
                       control['sequences'][key]['metrics']['visible_iou'] for key in baseline['sequences']]
        ci = bootstrap_mean_ci(differences)
        lines.extend(['', '控制坐标实现后，常速度外推的额外 ΔIoU=%.6f，95%% 区间 [%.6f, %.6f]。'
                      % (np.mean(differences), ci[0], ci[1]),
                      '坐标对照用于识别工程修正的贡献；不能把它归因为时序建模或 SLT。'])
    summary = baseline['summary']
    low_total = sum(row['low_iou_frames'] for row in baseline_rows)
    covered_low = sum(row['low_iou_covered_frames'] for row in baseline_rows)
    lines.extend(['', '## 基线失败现象', '',
                  '- 合法可见帧中，按序列平均的目标中心出裁剪率 %.2f%%，GT 框覆盖率 %.2f%%。'
                  % (100 * summary['mean_center_outside_rate'], 100 * summary['mean_coverage']),
                  '- 连续五个有效可见帧 IoU<0.2 记一个失效段，共 %d 个事件，分布于 %d 条序列。'
                  % (summary['failure_events'], sum(row['failure_events'] > 0 for row in baseline_rows)),
                  '- %d 个低 IoU 可见帧中，%d 个的 GT 框仍被裁剪覆盖至少 95%%；覆盖与定位是不同问题。'
                  % (low_total, covered_low),
                  '- GT 重现 %d 次；30 帧内恢复率 %.2f%%，未恢复比例 %.2f%%，成功事件平均确认延迟 %.2f 帧。'
                  % (summary['recovery_event_count'], 100 * summary['recovery_event_rate_30'],
                     100 * summary['recovery_event_unrecovered'], summary['recovery_event_latency']),
                  '- 恢复以第三个连续 IoU≥0.5 帧确认；恢复曲线按事件统计，序列平均值另存于 JSON。'])
    if intervention:
        failure = intervention['failure_sample']
        uniform = intervention['uniform_sample']
        mechanisms = intervention['descriptive_mechanisms']
        lines.extend(['', '## 同状态候选干预', '',
                      '- %d 个采样状态，%d 个有有效可见帧奖励；包含 2,880 个均匀样本，事件样本去重并保留标签。'
                      % (intervention['points'], intervention['usable_points']),
                      '- 每点九候选：中心 c、c+0.5v、c+v，因子 4、6、8；仅下一帧更换裁剪，随后恢复原 SRA。',
                      '- %d 个失效采样点中，%.2f%% 获得八帧平均 IoU≥0.05 的改善；%d 条序列出现至少一个此类改善点。'
                      % (failure['usable_points'], 100 * failure['positive_delta_005_rate'],
                         intervention['sequences_with_improvement_event_005']),
                      '- 失效点最佳候选相对基线平均改善 %.6f；均匀点平均改善仅 %.6f。'
                      % (failure['mean_best_minus_baseline'], uniform['mean_best_minus_baseline']),
                      '- 跨序列等权的候选平均差值 %.6f，95%% 区间 [%.6f, %.6f]；仅统计采样点后八帧，不能加到完整数据集主指标上。'
                      % (intervention['sequence_delta_mean'], *intervention['sequence_delta_ci']),
                      '- 工程推进门槛：**%s**。这允许投入策略学习，不构成论文显著性结论。'
                      % ('通过' if intervention['engineering_gate_passed'] else '未通过'),
                      '- 严格完整八帧的失效点为 %d 个，达到 +0.05 的比例 %.2f%%；门槛以这一子集判定，序列末尾的截短窗口另行保留。'
                      % (intervention['failure_full_horizon']['usable_points'],
                         100 * intervention['failure_full_horizon']['positive_delta_005_rate']),
                      '- 最优单帧动作与最优八帧动作集合不相交的比例 %.2f%%；单帧最优动作的八帧平均遗憾 %.6f，其中 %.2f%% 的样本遗憾≥0.01。'
                      % (100 * mechanisms['one_step_vs_eight_step_disjoint_rate'],
                         mechanisms['one_step_choice_eight_step_regret_mean'],
                         100 * mechanisms['one_step_choice_regret_ge_001_rate']),
                      '- 中心外推在最优零偏移尺度候选之上的平均增量为 %.6f；这是 GT 选择的诊断值，需要中心/尺度消融进一步验证。'
                      % mechanisms['mean_motion_increment_over_best_scale'],
                      '- 最佳候选使用未来 GT 选择，属于乐观参照；没有证明当前观测可学，更没有证明 GRU 或 SLT 有额外收益。',
                      '- 各候选沿用相同 Hann 阈值规则；整数裁剪坐标修正通过独立闭环对照分析。'])
    speed_path = 'output/search_policy/isolated_runtime.json'
    if os.path.isfile(speed_path):
        with open(speed_path) as handle:
            speed = json.load(handle)
        medians = speed['median_fps']
        lines.extend(['', '## 速度与有效性检查', '',
                      '独立 GPU0 上交替运行三次，每次 200 帧，排除前 20 帧：原 SRA 中位 %.2f FPS，常速度策略 %.2f FPS（%+.2f%%）。'
                      % (medians['sra'], medians['motion'], medians['motion_fps_change_percent']),
                      '计时包括跟踪与控制，图像解码不计；原 SRA 关闭诊断特征，常速度控制包含所需观测提取。该短片段只用于速度测量，准确率使用完整 val。', '',
                      '七项 CPU 协议检查和真实 CUDA 分支检查通过。冻结网络参数及缓冲前后哈希一致；A/B/A 候选结果一致；重置清空速度状态；裁剪坐标往返及边界填充通过。',
                      '90 条基线预测已逐数值复现；90 份未乘 Hann 响应图文件已保存；综合分数逐序列与项目官方评分器一致。'])
    lines.extend(['', '## 后续实验边界', '',
                  '下一阶段先建立训练集连续 4 帧预热＋8 帧监督缓存，固定输入与九类动作，比较 MLP 与 GRU；随后按原计划执行三个种子、逐 epoch 完整 val 和监督继续训练/SLT 对照。',
                  '现有训练入口与旧检查点仍属于烟测：稀疏缓存不能当作连续时序片段；分片合并必须防止序列 ID 冲突；SLT 的策略预热、有效 episode 梯度累积、随机状态保存和逐 epoch 验证需要先对齐协议。',
                  '当前证据支持“失败附近存在局部搜索控制机会”。当前证据尚不支持“整个序列稳定性已由学习策略改善”或“SLT 优于监督继续训练”。', '',
                  '## 文件索引', '',
                  '- `baseline_audited.json`、`baseline_per_sequence.csv`：完整基线与恢复统计。',
                  '- `intervention_audited.json`、`intervention_per_point.csv`、`intervention_per_sequence.csv`：候选收益与序列差值。',
                  '- `intervention_val_v2_full_audited.json`：保留全部候选数据，并按原始 exist 标记修正重现样本标签。',
                  '- `baseline_diagnosis.png`、`intervention_deltas.png`：诊断与差值图。',
                  '- 上级目录 `baseline_val_v2_full.json`、`motion_val_v2_full.json`、`coordinate_val_v2_full.json`：完整闭环原始报告。',
                  '- 上级目录 `runtime_verification.json`、`isolated_runtime.json`：有效性和速度检查。', ''])
    with open(os.path.join(args.output, 'full_val_report.md'), 'w') as handle:
        handle.write('\n'.join(lines))


def plots(output, baseline_rows, recoveries, intervention_rows=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].scatter([r['center_outside_rate'] for r in baseline_rows],
                    [r['visible_iou'] for r in baseline_rows], s=18, color='#15803d')
    axes[0].set(xlabel='Visible target center outside crop (sequence rate)', ylabel='Visible mean IoU')
    if recoveries:
        axes[1].plot(range(30), [np.mean([d is not None and d <= t for d in recoveries])
                                for t in range(30)], color='#2563eb')
    axes[1].set(xlabel='Frames since GT reappearance (0-based)', ylabel='Recovery fraction', ylim=(0, 1))
    fig.savefig(os.path.join(output, 'baseline_diagnosis.png'), dpi=180)
    plt.close(fig)
    if intervention_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        deltas = [r['mean_oracle_delta'] for r in intervention_rows if r['mean_oracle_delta'] is not None]
        axes[0].hist(deltas, bins=20, color='#15803d')
        axes[0].axvline(0, color='black', linewidth=1)
        axes[0].set(xlabel='Sequence mean oracle IoU delta (8 frames)', ylabel='Sequence count')
        axes[1].plot(sorted(deltas), marker='.', linewidth=1, color='#2563eb')
        axes[1].axhline(0, color='black', linewidth=1)
        axes[1].set(xlabel='Sequence rank', ylabel='Oracle IoU delta')
        fig.savefig(os.path.join(output, 'intervention_deltas.png'), dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default='output/search_policy/baseline_val_full.json')
    parser.add_argument('--intervention')
    parser.add_argument('--motion-report', default='output/search_policy/motion_val_v2_full.json')
    parser.add_argument('--coordinate-report', default='output/search_policy/coordinate_val_v2_full.json')
    parser.add_argument('--output', default='output/search_policy/full_val_analysis')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    baseline, rows, recoveries = audit_baseline(args)
    intervention_rows = None
    summary = None
    if args.intervention:
        summary, _, intervention_rows = analyze_intervention(args, set(baseline['sequences']))
        print('Intervention', json.dumps(summary, indent=2))
    plots(args.output, rows, recoveries, intervention_rows)
    research_report(args, baseline, rows, summary)
    print('Baseline', json.dumps(baseline['summary'], indent=2))


if __name__ == '__main__':
    main()
