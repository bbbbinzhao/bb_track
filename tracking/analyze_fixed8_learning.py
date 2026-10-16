"""Audit matched one/eight-frame supervision in complete closed-loop validation."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tracking.search_policy_experiment import (
    bootstrap_mean_ci, box_iou, load_sequences, sequence_metrics, valid_frames, write_json_atomic,
)
from tracking.train_fixed8_teacher import load_states


def teacher_statistics(records):
    one = np.stack([record['one'][:3] for record in records])
    eight = np.stack([record['eight'][:3] for record in records])
    gain = eight.max(-1) - eight[:, 2]
    return dict(states=len(records), sequences=len({r['sequence'] for r in records}),
        eight_oracle_gain_mean=float(gain.mean()),
        eight_oracle_gain_ge_005=int((gain >= .05).sum()),
        eight_oracle_gain_ge_005_rate=float((gain >= .05).mean()),
        eight_action_spread_mean=float((eight.max(-1) - eight.min(-1)).mean()),
        one_action_spread_mean=float((one.max(-1) - one.min(-1)).mean()),
        first_reward_low_iou_states=int((one[:, 2] < .2).sum()),
        baseline_eight_low_iou_states=int((eight[:, 2] < .2).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--teachers', required=True)
    parser.add_argument('--entropy-root')
    args = parser.parse_args()
    root = Path(args.root)
    teacher_models = json.loads((root / 'teacher_comparison.json').read_text())
    train, development, identity = load_states(Path(args.teachers))
    sources = dict(sra=ROOT / 'output/search_policy/baseline_val_v2_full.json',
        fixed4=ROOT / 'output/search_policy/learning_pilot_seed0/fixed_4/evaluate_val.json',
        fixed6=ROOT / 'output/search_policy/learning_pilot_seed0/fixed_6/evaluate_val.json',
        fixed8=ROOT / 'output/search_policy/learning_pilot_seed0/fixed_8/evaluate_val.json',
        one=root / 'one/val_selected.json', eight=root / 'eight/val_selected.json')
    if args.entropy_root:
        entropy_root = Path(args.entropy_root)
        entropy_models = json.loads((entropy_root / 'teacher_comparison.json').read_text())
        teacher_models['eight_entropy'] = entropy_models['eight']
        sources['eight_entropy'] = entropy_root / 'eight/val_selected.json'
        if entropy_models['eight']['identity']['cache_identity'] != identity:
            raise ValueError('Entropy control uses different training states')
    reports = {name: json.loads(path.read_text()) for name, path in sources.items()}
    sequences = load_sequences('val', data_root=identity['data_root'])
    names = [record[0] for record in sequences]
    if len(names) != 90:
        raise ValueError('Expected all 90 validation sequences')
    rows, summaries, by_method, action_counts, delays_by_method = [], {}, {}, {}, {}
    for method, report in reports.items():
        if (set(report['sequences']) != set(names) or report['split'] != 'val'
                or report['baseline_sha256'] != identity['checkpoint_sha256']):
            raise ValueError('Incomplete or mismatched evaluation: ' + method)
        values = {}
        for name, frames, boxes, exist in sequences:
            record = report['sequences'][name]
            predictions = np.asarray(record['predictions'])
            if predictions.shape != boxes.shape or not np.isfinite(predictions).all():
                raise ValueError('Truncated or invalid predictions')
            metrics = sequence_metrics(predictions, boxes, exist)
            valid = valid_frames(boxes, exist)
            valid[0] = False
            ious = np.asarray([box_iou(p, b) for p, b in zip(predictions, boxes)])
            metrics['low_iou_rate'] = float((valid & (ious < .2)).sum() / max(valid.sum(), 1))
            metrics['outside_rate'] = record['diagnostics']['visible_crop_center_outside_rate']
            values[name] = metrics
            rows.append(dict(method=method, sequence=name, visible_iou=metrics['visible_iou'],
                visible_auc=metrics['visible_auc'], mixed_measure=metrics['mixed_measure'],
                low_iou_rate=metrics['low_iou_rate'], outside_rate=metrics['outside_rate'],
                recovery_events=metrics['recovery_events'],
                recovered_30=sum(d is not None for d in metrics['recovery_delays'])))
        by_method[method] = values
        delays = [d for record in values.values() for d in record['recovery_delays']]
        delays_by_method[method] = delays
        summaries[method] = {key: float(np.mean([record[key] for record in values.values()]))
            for key in ('visible_iou', 'visible_auc', 'mixed_measure', 'low_iou_rate', 'outside_rate')}
        summaries[method].update(recovery_events=len(delays),
            recovered_30=sum(d is not None for d in delays),
            recovery_rate_30=sum(d is not None for d in delays) / len(delays),
            recovery_latency=float(np.mean([d for d in delays if d is not None]))
                if any(d is not None for d in delays) else None)
        if method in teacher_models:
            counts = {str(a): sum(record.get('policy_action_counts', {}).get(str(a), 0)
                                 for record in report['sequences'].values()) for a in range(9)}
            if sum(counts.values()) != sum(len(record[1]) - 2 for record in sequences):
                raise ValueError('Incomplete controller actions')
            action_counts[method] = counts
    comparisons = {}
    pairs = [('one', 'sra'), ('one', 'fixed8'), ('eight', 'sra'), ('eight', 'fixed8'), ('eight', 'one')]
    pairs.extend([(method, reference) for method in ('one', 'eight') for reference in ('fixed4', 'fixed6')])
    if args.entropy_root:
        pairs.extend([('eight_entropy', reference) for reference in ('sra', 'fixed4', 'fixed6', 'fixed8', 'one', 'eight')])
    for method, reference in pairs:
        deltas = {name: by_method[method][name]['visible_iou'] - by_method[reference][name]['visible_iou']
                  for name in names}
        comparisons[method + '_vs_' + reference] = dict(mean_delta=float(np.mean(list(deltas.values()))),
            ci_95=bootstrap_mean_ci(list(deltas.values())),
            positive_sequences=sum(d > 0 for d in deltas.values()),
            negative_sequences=sum(d < 0 for d in deltas.values()), per_sequence_deltas=deltas)
    runtime_path = root / 'isolated_runtime.json'
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else None
    result = dict(sequences=90, frames=sum(len(record[1]) for record in sequences), seed=0,
        checkpoint_sha256=identity['checkpoint_sha256'], summary=summaries, comparisons=comparisons,
        action_counts=action_counts, training_teacher_statistics=teacher_statistics(train),
        development_teacher_statistics=teacher_statistics(development),
        training_models=teacher_models, runtime=runtime,
        three_seed_gate_evaluated=False, test_used=False,
        scope='Matched scale MLP teachers; no temporal or SLT contribution tested',
        sources={name: str(path.resolve()) for name, path in sources.items()})
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    write_json_atomic(str(output / 'matched_teacher_results.json'), result)
    with (output / 'per_sequence.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 固定 8 状态：单步与八帧教师的匹配对照', '',
        '- 官方 train 全 200 条序列，各 8 个完整八帧窗口；共同排除下一帧无有效 GT 的状态。',
        '- 180/20 序列划分；实际训练状态 %d，内部开发状态 %d。' % (len(train), len(development)),
        '- 同一 seed 0、同一观测/初始化/采样状态、同一两层 64 维 MLP、同一尺度动作 4/6/8。',
        '- 两模型各训练 10 epoch，每 epoch 8,192 个状态、batch=32，共 2,560 次更新。',
        '- 教师 softmax(IoU/0.1)，唯一区别是标签用下一帧 IoU 或八帧可见 IoU 均值。',
        '- 各自仅按内部开发集对应教师效用选检查点，最早同分；完整 val 90 条，test 未使用。',
        '- 这是单状态 MLP 筛查；未进行 GRU、SLT 或三种子最终验证。', '',
        '| 方法 | 可见 IoU | AUC | Anti-UAV | 低 IoU 帧比例 | 30 帧恢复 |',
        '|---|---:|---:|---:|---:|---:|']
    if args.entropy_root:
        lines[8:8] = ['- eight_entropy 为额外对照：仅用训练集将八帧教师熵匹配到单步教师，温度 %.8f；其他条件相同。' %
            teacher_models['eight_entropy']['identity']['teacher_temperature']]
    for method, record in summaries.items():
        lines.append('| %s | %.6f | %.6f | %.6f | %.2f%% | %d/%d |' % (
            method, record['visible_iou'], record['visible_auc'], record['mixed_measure'],
            100 * record['low_iou_rate'], record['recovered_30'], record['recovery_events']))
    lines.extend(['', '| 比较 | ΔIoU | 95% 序列配对区间 | 改善/退化序列 |', '|---|---:|---|---:|'])
    for key, pair in comparisons.items():
        lines.append('| %s | %+.6f | [%.6f, %.6f] | %d/%d |' % (
            key, pair['mean_delta'], *pair['ci_95'], pair['positive_sequences'], pair['negative_sequences']))
    lines.extend(['', '## 教师信号与内部开发', ''])
    for label, records in (('训练', train), ('内部开发', development)):
        statistics = teacher_statistics(records)
        lines.append('- %s：八帧尺度 oracle 相对固定 8 改善 ≥0.05 的状态 %d/%d（%.2f%%），平均动作跨度 %.6f。' % (
            label, statistics['eight_oracle_gain_ge_005'], statistics['states'],
            100 * statistics['eight_oracle_gain_ge_005_rate'], statistics['eight_action_spread_mean']))
    for objective, model in teacher_models.items():
        dev = model['selected_development']
        lines.extend(['- %s：选 epoch %d，开发交叉熵 %.6f，训练先验交叉熵 %.6f，教师熵 %.6f。' % (
            objective, model['selected_epoch'], dev['loss'], dev['train_prior_cross_entropy'], dev['teacher_entropy']),
            '  开发八帧收益：对固定 8=%+.6f；对开发集最佳常量=%+.6f；闭环动作计数 `%s`。' % (
                dev['rewards']['eight']['delta_to_fixed8'], dev['rewards']['eight']['delta_to_best_dev_constant'],
                json.dumps(action_counts[objective]))])
    lines.extend(['', '## 解释边界', '',
        '固定 8 状态上的短片段教师效用不等同于持续部署策略的整序列效用。',
        '开发集最佳常量使用开发标签，只是乐观诊断；实际训练先验和常量动作只用训练集计算。',
        '八帧教师优于单步教师也不足以证明 SLT；必须另与匹配更新次数的监督继续训练比较。',
        '单种子结果不满足原计划最终门槛，不能支持最终论文有效性主张。'])
    if runtime:
        lines.extend(['', '## 独立速度', '', '三次轮换顺序，每次 200 帧、排除 20 帧预热；包含观测/策略、排除解码。', '',
                      '| 方法 | 中位 FPS | 相对 SRA |', '|---|---:|---:|'])
        for method, fps in runtime['median_fps'].items():
            lines.append('| %s | %.2f | %+.2f%% |' % (method, fps, runtime['relative_fps_change_percent'].get(method, 0)))
    (output / 'matched_teacher_report.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    figure, axis = plt.subplots(figsize=(8, max(4, .4 * len(comparisons))))
    keys = list(comparisons)
    for index, key in enumerate(keys):
        pair = comparisons[key]
        delta = pair['mean_delta']
        axis.errorbar(delta, index, xerr=np.array([[delta - pair['ci_95'][0]],
            [pair['ci_95'][1] - delta]]).clip(0), fmt='o', capsize=3)
    axis.set_yticks(range(len(keys)), keys)
    axis.axvline(0, color='grey', linewidth=1)
    axis.set_xlabel('Sequence-mean visible IoU change (95% paired CI)')
    figure.tight_layout()
    figure.savefig(output / 'matched_teacher_deltas.png', dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4))
    for method, delays in delays_by_method.items():
        curve = [sum(delay is not None and delay <= step for delay in delays) / max(len(delays), 1)
                 for step in range(30)]
        axis.plot(range(30), curve, label=method)
    axis.set_xlabel('Frames since GT reappearance')
    axis.set_ylabel('Confirmed recovery fraction')
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / 'matched_teacher_recovery.png', dpi=160)
    plt.close(figure)
    print(json.dumps(dict(summary=summaries, comparisons={k: {f: v for f, v in p.items()
        if f != 'per_sequence_deltas'} for k, p in comparisons.items()}), indent=2), flush=True)
    print('Wrote ' + str(output / 'matched_teacher_report.md'), flush=True)


if __name__ == '__main__':
    main()
