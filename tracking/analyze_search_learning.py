"""Audit complete closed-loop results of the seed-zero learning pilot."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tracking.search_policy_experiment import (
    bootstrap_mean_ci, box_iou, failure_starts, sequence_metrics, valid_frames, write_json_atomic,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    sources = {
        'sra': ROOT / 'output/search_policy/baseline_val_v2_full.json',
        'coordinate': ROOT / 'output/search_policy/coordinate_val_v2_full.json',
        'motion': ROOT / 'output/search_policy/motion_val_v2_full.json',
        **{'fixed_%d' % factor: root / ('fixed_%d' % factor) / 'evaluate_val.json'
           for factor in (4, 6, 8)},
        **{'mlp_' + mode: root / ('mlp_' + mode) / 'val_epoch001.json'
           for mode in ('scale', 'joint')},
    }
    for mode in ('scale', 'joint'):
        selected_path = root / ('mlp_' + mode) / 'val_selected.json'
        if selected_path.is_file():
            sources['mlp_' + mode + '_selected'] = selected_path
    selection_path = root / 'convergence_selection.json'
    selection = json.load(selection_path.open()) if selection_path.exists() else None
    reports = {name: json.load(path.open()) for name, path in sources.items()}
    data_root = Path(reports['fixed_4']['data_root'])
    annotations = {}
    for path in sorted((data_root / 'val').glob('*/IR_label.json')):
        data = json.load(path.open())
        annotations[path.parent.name] = (np.asarray(data['gt_rect'], dtype=float),
                                         np.asarray(data['exist'], dtype=bool))
    names = sorted(annotations)
    expected_hash = reports['sra']['baseline_sha256']
    metrics, arrays, frame_groups, summary = {}, {}, {}, {}
    for method, report in reports.items():
        if set(report['sequences']) != set(names) or len(names) != 90:
            raise ValueError('Expected the exact complete 90-sequence official val: ' + method)
        if report['baseline_sha256'] != expected_hash or report['split'] != 'val':
            raise ValueError('Frozen tracker/split mismatch')
        if report.get('complete') is False:
            raise ValueError('Incomplete evaluation: ' + method)
        metrics[method], arrays[method], frame_groups[method] = {}, {}, {}
        for name in names:
            boxes, exist = annotations[name]
            predictions = np.asarray(report['sequences'][name]['predictions'])
            if len(predictions) != len(boxes) or not np.isfinite(predictions).all():
                raise ValueError('Invalid/truncated predictions: ' + method + ' ' + name)
            record = sequence_metrics(predictions, boxes, exist)
            ious = np.array([box_iou(prediction, box) for prediction, box in zip(predictions, boxes)])
            valid = valid_frames(boxes, exist)
            valid[0] = False
            record['low_iou_visible_frames'] = int((valid & (ious < .2)).sum())
            record['low_iou_visible_rate'] = record['low_iou_visible_frames'] / max(int(valid.sum()), 1)
            record['failure_events'] = len(failure_starts(valid, ious))
            metrics[method][name], arrays[method][name] = record, ious
        delays = [delay for name in names for delay in metrics[method][name]['recovery_delays']]
        summary[method] = {
            key: float(np.mean([metrics[method][name][key] for name in names]))
            for key in ('visible_iou', 'visible_auc', 'mixed_measure')}
        summary[method].update(
            recovery_events=len(delays), recovered_30=sum(d is not None for d in delays),
            recovery_rate_30=sum(d is not None for d in delays) / len(delays) if delays else None,
            recovery_unrecovered=sum(d is None for d in delays) / len(delays) if delays else None,
            recovery_latency=float(np.mean([d for d in delays if d is not None]))
                if any(d is not None for d in delays) else None,
            failure_events=sum(metrics[method][name]['failure_events'] for name in names))
        summary[method]['low_iou_visible_frames'] = sum(
            metrics[method][name]['low_iou_visible_frames'] for name in names)
        summary[method]['low_iou_visible_rate'] = float(np.mean([
            metrics[method][name]['low_iou_visible_rate'] for name in names]))
        for output_key, diagnostic_key in (
                ('crop_center_outside_rate', 'visible_crop_center_outside_rate'),
                ('bbox_coverage', 'visible_mean_bbox_coverage')):
            values = [report['sequences'][name]['diagnostics'].get(diagnostic_key) for name in names]
            values = [value for value in values if value is not None]
            summary[method][output_key] = float(np.mean(values)) if values else None
    comparisons = {}
    for method in reports:
        if method == 'sra':
            continue
        for reference in ('sra', 'coordinate', 'fixed_4', 'fixed_6', 'fixed_8'):
            if method == reference:
                continue
            difference = np.array([metrics[method][name]['visible_iou'] -
                                   metrics[reference][name]['visible_iou'] for name in names])
            comparisons[method + '_vs_' + reference] = {
                'mean_delta': float(difference.mean()), 'ci_95': bootstrap_mean_ci(difference),
                'positive_sequences': int((difference > 0).sum()),
                'negative_sequences': int((difference < 0).sum()),
                'per_sequence_deltas': dict(zip(names, difference.tolist()))}
        for group in ('healthy', 'failed'):
            per_sequence = []
            steps = 0
            for name in names:
                boxes, exist = annotations[name]
                valid = valid_frames(boxes, exist)
                valid[0] = False
                selected = valid & ((arrays['sra'][name] >= .5) if group == 'healthy'
                                    else (arrays['sra'][name] < .2))
                if selected.any():
                    per_sequence.append(float((arrays[method][name] - arrays['sra'][name])[selected].mean()))
                    steps += int(selected.sum())
            frame_groups[method][group] = {
                'frames': steps, 'sequences': len(per_sequence),
                'sequence_mean_delta': float(np.mean(per_sequence)) if per_sequence else None,
                'ci_95': bootstrap_mean_ci(per_sequence) if per_sequence else None}
    training = {mode: json.load((root / ('mlp_' + mode) / 'train/training_report.json').open())
                for mode in ('scale', 'joint')}
    actions = {method: {str(action): sum(reports[method]['sequences'][name].get(
                'policy_action_counts', {}).get(str(action), 0) for name in names)
                for action in range(9)} for method in reports if method.startswith('mlp_')}
    scale_counts = {method: {str(factor): sum(count for action, count in counts.items()
                      if int(action) % 3 == index) for index, factor in enumerate((4, 6, 8))}
                    for method, counts in actions.items()}
    expected_actions = sum(len(boxes) - 2 for boxes, _ in annotations.values())
    if any(sum(counts.values()) != expected_actions for counts in actions.values()):
        raise ValueError('Missing per-frame controller action records')
    runtime_path = root / 'isolated_runtime.json'
    runtime = json.load(runtime_path.open()) if runtime_path.exists() else None
    payload = {'scope': 'seed-zero supervision screening and train-development checkpoint selection; not multi-seed validation',
               'backbone_sha256': expected_hash, 'sequences': len(names),
               'frames': sum(len(value[0]) for value in annotations.values()),
               'summary': summary, 'comparisons': comparisons, 'baseline_frame_groups': frame_groups,
               'policy_action_counts': actions, 'policy_scale_counts': scale_counts,
               'three_seed_gate_evaluated': False, 'training': training,
               'isolated_runtime': runtime,
               'supervised_convergence': selection,
               'sources': {name: str(path.resolve()) for name, path in sources.items()}}
    write_json_atomic(str(output / 'learning_pilot_results.json'), payload)
    with (output / 'per_sequence_results.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['method', 'sequence', 'visible_iou', 'visible_auc', 'mixed_measure',
                         'delta_to_sra', 'recovery_events', 'recovered_30', 'recovery_latency',
                         'recovery_unrecovered', 'failure_events', 'low_iou_visible_frames',
                         'low_iou_visible_rate'])
        for method in reports:
            for name in names:
                record = metrics[method][name]
                writer.writerow([method, name, record['visible_iou'], record['visible_auc'],
                    record['mixed_measure'], record['visible_iou'] - metrics['sra'][name]['visible_iou'],
                    record['recovery_events'], sum(d is not None for d in record['recovery_delays']),
                    record['recovery_latency'], record['recovery_unrecovered'],
                    record['failure_events'], record['low_iou_visible_frames'],
                    record['low_iou_visible_rate']])
    lines = ['# 搜索策略学习可行性：种子 0 监督验证', '',
        '- 官方 train 200 条序列生成连续教师；按序列留出 10% 作内部开发集。',
        '- 每片段 4 帧预热、8 帧监督；不可见/非法框不提供定位教师，状态持续推进。',
        '- 首轮两种 MLP 各训练 1 epoch、1,024 个片段、batch=4；AdamW lr=1e-4。',
        '- 官方 val 90 条序列完整闭环评测，初始化帧排除；test 未使用。',
        '- 本轮不包含 GRU、SLT 或三种子有效性结论。', '',
        '| 方法 | 可见 IoU | AUC | Anti-UAV | ΔIoU 对 SRA | 95% 配对区间 | 30 帧恢复 |',
        '|---|---:|---:|---:|---:|---|---:|']
    for method in reports:
        record = summary[method]
        pair = comparisons.get(method + '_vs_sra')
        delta = '%+.6f' % pair['mean_delta'] if pair else '0'
        ci = '[%.6f, %.6f]' % tuple(pair['ci_95']) if pair else ''
        lines.append('| %s | %.6f | %.6f | %.6f | %s | %s | %d/%d |' % (
            method, record['visible_iou'], record['visible_auc'], record['mixed_measure'], delta, ci,
            record['recovered_30'], record['recovery_events']))
    lines.extend(['', '## 固定因子 8 与学习策略', '',
                  '| 方法 | ΔIoU 对坐标控制 | ΔIoU 对固定 8 | 对固定 8 的 95% 配对区间 |',
                  '|---|---:|---:|---|'])
    for method in reports:
        if not method.startswith('mlp_'):
            continue
        pair = comparisons[method + '_vs_fixed_8']
        lines.append('| %s | %+.6f | %+.6f | [%.6f, %.6f] |' % (
            method, comparisons[method + '_vs_coordinate']['mean_delta'], pair['mean_delta'],
            pair['ci_95'][0], pair['ci_95'][1]))
    lines.extend(['', '## 覆盖与失效现象', '',
                  '| 方法 | 可见帧目标中心出裁剪率 | GT 框覆盖率 | 低 IoU 帧比例 | 持续失效事件 |',
                  '|---|---:|---:|---:|---:|'])
    for method, record in summary.items():
        outside = '%.2f%%' % (100 * record['crop_center_outside_rate']) if record['crop_center_outside_rate'] is not None else ''
        coverage = '%.2f%%' % (100 * record['bbox_coverage']) if record['bbox_coverage'] is not None else ''
        lines.append('| %s | %s | %s | %.2f%% | %d |' % (
            method, outside, coverage, 100 * record['low_iou_visible_rate'], record['failure_events']))
    lines.extend(['', '这些指标描述闭环轨迹中的现象；出裁剪也可能是早先定位错误的后果，不能单独确定失效原因。',
                  '失效事件次数需结合低 IoU 帧比例判断：持续丢失可能把多段失效合并成较少事件。'])
    lines.extend(['', '## 教师可预测性与闭环差异', '',
        '内部开发集上的教师收益来自基线状态，不能直接视为部署收益。', ''])
    for mode in ('scale', 'joint'):
        report = training[mode]
        dev = report['history'][-1]['development']
        lines.extend(['- MLP %s：训练 %d 个窗口、开发 %d 个窗口；梯度范数最大 %.6f。' % (
            mode, report['train_windows'], report['development_windows'], report['max_gradient_norm']),
            '  开发教师 selected IoU=%.6f，oracle=%.6f，相对最佳固定动作=%+.6f。' % (
                dev['selected_iou'], dev['oracle_iou'], dev['delta_to_best_fixed_action']),
            '  完整 val 动作计数：`%s`。' % json.dumps(actions['mlp_' + mode]),
            '  完整 val 尺度计数：`%s`。' % json.dumps(scale_counts['mlp_' + mode])])
        for group in ('healthy', 'failed'):
            value = frame_groups['mlp_' + mode][group]
            lines.append('  基线 %s 帧上的逐序列平均变化=%+.6f（%d 帧）。' % (
                group, value['sequence_mean_delta'], value['frames']))
    if selection:
        lines.extend(['', '## 监督训练收敛检查', '',
            '首轮后继续监督训练至 10 个 epoch，保持优化器和随机采样状态；每个 epoch 仍为 1,024 个片段。',
            '仅按训练集内部开发集的教师 selected IoU 选择检查点，同分取最早 epoch；官方 val 不参与这一步选择。',
            '本轮完整 val 覆盖首轮和内部选择的检查点，并未逐 epoch 执行完整 val；这属于筛查协议。', ''])
        for method, record in selection['models'].items():
            dev = record['development']
            lines.append('- %s：内部选择 epoch %d，selected IoU=%.6f，相对最佳固定动作=%+.6f。' % (
                method, record['epoch'], dev['selected_iou'], dev['delta_to_best_fixed_action']))
            distribution = record['teacher_statistics']['development']
            lines.append('  开发集 teacher KL=%.6f；训练集先验分布的交叉熵=%.6f，模型交叉熵=%.6f。' % (
                dev['kl_to_teacher'], distribution.get('train_prior_cross_entropy',
                distribution['constant_distribution_cross_entropy']), dev['loss']))
            selected_name = method + '_selected'
            if selected_name in actions:
                lines.append('  所选模型完整 val 尺度计数：`%s`。' % json.dumps(scale_counts[selected_name]))
                for group in ('healthy', 'failed'):
                    value = frame_groups[selected_name][group]
                    lines.append('  基线 %s 帧上的逐序列平均变化=%+.6f（%d 帧）。' % (
                        group, value['sequence_mean_delta'], value['frames']))
            else:
                lines.append('  所选检查点的完整闭环评测尚未汇总。')
        curve_figure, axis = plt.subplots(figsize=(7, 4))
        for method, record in selection['models'].items():
            axis.plot([entry['epoch'] for entry in record['history']],
                      [entry['development']['delta_to_best_fixed_action'] for entry in record['history']],
                      marker='o', label=method)
        axis.axhline(0, color='grey', linewidth=1)
        axis.set_xlabel('Supervised epoch')
        axis.set_ylabel('Teacher IoU change vs best fixed action')
        axis.set_title('Train-internal development; baseline states')
        axis.legend()
        curve_figure.tight_layout()
        curve_figure.savefig(output / 'supervised_development_curve.png', dpi=160)
        plt.close(curve_figure)
    lines.extend(['', '## 结论边界', '',
        '本轮只回答简单策略训练与完整闭环的初步可行性；单种子筛查不能证明最终有效或无效。',
        '必须同时比较全部固定尺度与坐标控制，防止把恒定动作或几何实现变化归因为学习。',
        '未满足三个种子均为正、平均提升至少 0.01 且配对区间下界大于零前，不宣称达到原计划有效性门槛。',
        '时序与 SLT 贡献仍需各自对照；本轮不支持任何相关主张。', '',
        '逐序列结果见 `per_sequence_results.csv`，完整统计见 `learning_pilot_results.json`。'])
    if runtime:
        lines.extend(['', '## 独立速度', '',
                      '同一 GPU，三次交替运行，每次 200 帧、排除前 20 帧；解码不计时。',
                      'SRA 关闭诊断特征，学习策略计入必需的观测提取和策略前向；检查点以测速 JSON 为准。', '',
                      '| 方法 | 中位 FPS | 相对 SRA |', '|---|---:|---:|'])
        for method in ('sra', 'mlp_scale', 'mlp_joint'):
            lines.append('| %s | %.2f | %+.2f%% |' % (method, runtime['median_fps'][method],
                runtime['relative_fps_change_percent'].get(method, 0.)))
    (output / 'learning_pilot_report.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    methods = [method for method in reports if method != 'sra']
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    values = np.array([comparisons[method + '_vs_sra']['mean_delta'] for method in methods])
    intervals = np.array([comparisons[method + '_vs_sra']['ci_95'] for method in methods])
    axes[0].errorbar(values, np.arange(len(methods)), xerr=np.maximum(0, np.stack([
        values - intervals[:, 0], intervals[:, 1] - values])), fmt='o', capsize=3)
    axes[0].set_yticks(np.arange(len(methods)), methods)
    axes[0].axvline(0, color='grey', linewidth=1)
    axes[0].set_xlabel('Sequence-mean visible IoU change vs SRA (95% paired CI)')
    recovery_methods = ('sra', 'fixed_4', 'fixed_6', 'fixed_8', 'mlp_scale', 'mlp_joint') + tuple(
        method for method in reports if method.endswith('_selected'))
    for method in recovery_methods:
        delays = [delay for name in names for delay in metrics[method][name]['recovery_delays']]
        curve = [sum(delay is not None and delay <= step for delay in delays) / max(len(delays), 1)
                 for step in range(30)]
        axes[1].plot(range(30), curve, label=method)
    axes[1].set_xlabel('Frames since GT reappearance')
    axes[1].set_ylabel('Confirmed recovery fraction')
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / 'learning_deltas_and_recovery.png', dpi=160)
    plt.close(figure)
    print(json.dumps(summary, indent=2), flush=True)
    print('Wrote ' + str(output / 'learning_pilot_report.md'), flush=True)


if __name__ == '__main__':
    main()
