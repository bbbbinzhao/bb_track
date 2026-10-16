"""Audit opportunity beyond fixed eight without treating GT choices as a policy."""
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

from tracking.search_fixed8_intervention import PROTOCOL
from tracking.search_policy_experiment import bootstrap_mean_ci, load_sequences, write_json_atomic


def choose(values):
    best = max(values)
    tied = [action for action, value in enumerate(values) if value >= best - 1e-6]
    return 2 if 2 in tied else tied[0]


def sequence_mean(rows, key):
    groups = {}
    for row in rows:
        if row[key] is not None:
            groups.setdefault(row['sequence'], []).append(row[key])
    return {name: float(np.mean(values)) for name, values in groups.items()}


def group_statistics(rows):
    by_sequence = sequence_mean(rows, 'oracle_delta')
    if not rows:
        return dict(points=0, sequences=0)
    stats = dict(points=len(rows), sequences=len(by_sequence),
        oracle_sequence_mean_delta=float(np.mean(list(by_sequence.values()))),
        oracle_sequence_ci_95=bootstrap_mean_ci(list(by_sequence.values())),
        oracle_point_mean_delta=float(np.mean([row['oracle_delta'] for row in rows])),
        improvement_005_points=sum(row['oracle_delta'] >= .05 for row in rows),
        improvement_005_rate=float(np.mean([row['oracle_delta'] >= .05 for row in rows])),
        sequences_with_improvement_event=sum(any(row['oracle_delta'] > 1e-6 for row in rows
            if row['sequence'] == name) for name in by_sequence),
        sequences_with_improvement_005=sum(any(row['oracle_delta'] >= .05 for row in rows
            if row['sequence'] == name) for name in by_sequence),
        baseline_sequence_mean_iou=float(np.mean(list(sequence_mean(rows, 'baseline_iou8').values()))),
        oracle_sequence_mean_iou=float(np.mean(list(sequence_mean(rows, 'oracle_iou8').values()))),
        per_sequence_oracle_delta=by_sequence)
    for key in ('one_step_chosen_delta8', 'one_step_regret8', 'motion_increment', 'scale_oracle_delta',
                'baseline_coverage', 'oracle_coverage', 'baseline_pixels', 'oracle_pixels'):
        values = list(sequence_mean(rows, key).values())
        stats[key + '_sequence_mean'] = float(np.mean(values)) if values else None
    valid_next = [row for row in rows if row['next_vs_eight_disjoint'] is not None]
    stats['one_step_eight_step_disjoint_rate'] = float(np.mean([
        row['next_vs_eight_disjoint'] for row in valid_next])) if valid_next else None
    stats['best_action_counts'] = {str(action): sum(row['best_action'] == action for row in rows)
                                   for action in range(9)}
    return stats


def analyze(root):
    manifests = sorted(root.glob('shard_*/manifest.json'))
    if not manifests:
        raise ValueError('No intervention manifests')
    identity, sequences, sources = None, {}, []
    for path in manifests:
        payload = json.load(path.open())
        current = {key: value for key, value in payload['identity'].items() if key != 'expected_sequences'}
        if current['protocol'] != PROTOCOL or not payload['complete']:
            raise ValueError('Incomplete or incompatible intervention: ' + str(path))
        if identity is not None and current != identity:
            raise ValueError('Intervention shard identities differ')
        identity = current
        if not payload['frozen_weights_and_buffers_unchanged']:
            raise ValueError('Tracker was not frozen')
        if set(payload['sequences']) != set(payload['identity']['expected_sequences']):
            raise ValueError('Shard sequence coverage mismatch')
        for name, value in payload['sequences'].items():
            if name in sequences:
                raise ValueError('Duplicate sequence ' + name)
            seq = json.load((path.parent / value['file']).open())
            if not seq['baseline_reproduced'] or not seq['branch_order_invariant']:
                raise ValueError('Failed branch verification')
            if seq['baseline_action_max_box_error'] > 1e-4:
                raise ValueError('Baseline action is not matched')
            sequences[name] = seq
        sources.append(str(path.resolve()))
    records = load_sequences(identity['split'], data_root=identity['data_root'])
    if set(sequences) != {record[0] for record in records}:
        raise ValueError('Expected exact full dataset coverage')
    rows, action_rows = [], []
    for name, seq in sequences.items():
        for sample in seq['samples']:
            candidates = sample['candidates']
            if [candidate['action'] for candidate in candidates] != list(range(9)):
                raise ValueError('Incomplete candidate set')
            if sample['baseline_future_iou'] is None:
                continue
            values = [candidate['future_iou'] for candidate in candidates]
            best_action = choose(values)
            best, baseline = candidates[best_action], candidates[2]
            next_values = [candidate['next_iou'] for candidate in candidates]
            next_action = choose(next_values) if next_values[0] is not None else None
            best_value = max(values)
            next_set = ({a for a, v in enumerate(next_values) if v >= max(next_values) - 1e-6}
                        if next_action is not None else set())
            future_set = {a for a, v in enumerate(values) if v >= best_value - 1e-6}
            row = dict(sequence=name, frame=sample['frame'],
                uniform=sample['uniform_sample'], failure=sample['failure_sample'],
                reappearance=sample['reappearance_sample'], full_horizon=sample['full_horizon'],
                valid_reward_frames=sample['valid_reward_frames'], baseline_iou8=baseline['future_iou'],
                oracle_iou8=best_value, oracle_delta=best_value - baseline['future_iou'],
                best_action=best_action, best_factor=best['region']['factor'],
                baseline_next_iou=baseline['next_iou'],
                one_step_chosen_action=next_action,
                one_step_chosen_delta8=values[next_action] - baseline['future_iou'] if next_action is not None else None,
                one_step_regret8=best_value - values[next_action] if next_action is not None else None,
                next_vs_eight_disjoint=not bool(next_set & future_set) if next_action is not None else None,
                scale_oracle_delta=max(values[:3]) - baseline['future_iou'],
                motion_increment=best_value - max(values[:3]),
                baseline_coverage=baseline['frames'][0]['coverage'],
                oracle_coverage=best['frames'][0]['coverage'],
                baseline_pixels=baseline['frames'][0]['target_pixels'],
                oracle_pixels=best['frames'][0]['target_pixels'])
            rows.append(row)
            for candidate in candidates:
                frame_values = [frame for frame in candidate['frames'] if frame['iou'] is not None]
                action_rows.append(dict(sequence=name, frame=sample['frame'], action=candidate['action'],
                    factor=candidate['region']['factor'], uniform=row['uniform'], full_horizon=row['full_horizon'],
                    baseline_next_iou=row['baseline_next_iou'], next_iou=candidate['next_iou'],
                    future_iou=candidate['future_iou'], delta8=candidate['future_iou'] - baseline['future_iou'],
                    low_iou_rate=float(np.mean([frame['iou'] < .2 for frame in frame_values])),
                    mean_coverage=float(np.mean([frame['coverage'] for frame in frame_values])),
                    mean_target_pixels=float(np.mean([frame['target_pixels'] for frame in frame_values]))))
    groups = {'all': rows,
              'full_horizon': [row for row in rows if row['full_horizon']],
              'uniform_full': [row for row in rows if row['uniform'] and row['full_horizon']],
              'failure_full': [row for row in rows if row['failure'] and row['full_horizon']],
              'reappearance_full': [row for row in rows if row['reappearance'] and row['full_horizon']],
              'healthy_uniform_full': [row for row in rows if row['uniform'] and row['full_horizon']
                                       and row['baseline_next_iou'] is not None and row['baseline_next_iou'] >= .5],
              'failed_uniform_full': [row for row in rows if row['uniform'] and row['full_horizon']
                                      and row['baseline_next_iou'] is not None and row['baseline_next_iou'] < .2]}
    stats = {key: group_statistics(selected) for key, selected in groups.items()}
    constant = {}
    for action in range(9):
        selected = [row for row in action_rows if row['action'] == action and row['uniform'] and row['full_horizon']]
        deltas = list(sequence_mean(selected, 'delta8').values())
        healthy = [row for row in selected if row['baseline_next_iou'] is not None and row['baseline_next_iou'] >= .5]
        healthy_deltas = list(sequence_mean(healthy, 'delta8').values())
        constant[str(action)] = dict(sequence_mean_delta=float(np.mean(deltas)),
            ci_95=bootstrap_mean_ci(deltas),
            healthy_sequence_mean_delta=float(np.mean(healthy_deltas)) if healthy_deltas else None)
    failures, all_stats = stats['failure_full'], stats['all']
    gate = bool(all_stats.get('sequences_with_improvement_event', 0) >= 20
                and failures.get('improvement_005_rate', 0) >= .2)
    conservative_gate = bool(stats['full_horizon'].get('sequences_with_improvement_005', 0) >= 20
                             and failures.get('improvement_005_rate', 0) >= .2)
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    result = dict(identity=identity, sequences=len(sequences), frames=sum(len(r[1]) for r in records),
        groups=stats, uniform_constant_action_interventions=constant,
        engineering_gate_passed=gate, conservative_engineering_gate_passed=conservative_gate,
        baseline_reproduced_all_sequences=True, branch_order_invariant_all_sequences=True,
        max_baseline_action_box_error=max(seq['baseline_action_max_box_error'] for seq in sequences.values()),
        sources=sources, notes=['GT oracle is optimistic and cannot be used at inference.',
            'Only one crop changes; remaining frames follow fixed eight, not a learned policy.',
            'Bootstrap uses sequences. Groups overlap; truncated horizons excluded from primary groups.',
            'Tie-breaking prefers unchanged action 2; all reported utility maxima remain exact.'])
    write_json_atomic(str(output / 'fixed8_opportunity.json'), result)
    for filename, records_out in (('per_point.csv', rows), ('per_action.csv', action_rows)):
        with (output / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records_out[0]))
            writer.writeheader()
            writer.writerows(records_out)
    seq_rows = []
    for name in sorted(sequences):
        entry = dict(sequence=name)
        for key, selected in groups.items():
            local = [row for row in selected if row['sequence'] == name]
            entry[key + '_points'] = len(local)
            entry[key + '_oracle_delta'] = float(np.mean([row['oracle_delta'] for row in local])) if local else None
            entry[key + '_improvement_005_points'] = sum(row['oracle_delta'] >= .05 for row in local)
        seq_rows.append(entry)
    with (output / 'per_sequence.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seq_rows[0]))
        writer.writeheader()
        writer.writerows(seq_rows)
    lines = ['# 固定搜索因子 8：同状态八帧干预', '',
        '- 完整 %s：%d 条序列、%d 帧；不使用 test。' % (identity['split'], result['sequences'], result['frames']),
        '- 状态来自固定 8；九候选仅改变下一裁剪，后续七帧恢复固定 8；Hann 沿用原规则。',
        '- 每序列 32 个均匀时刻、最多各 8 个失效/重现事件前帧；重叠去重，事件组保留多重标签。',
        '- 主分组排除不足八帧的片段；IoU 只用有效可见帧。各组先逐序列平均，再跨序列平均。',
        '- 所有序列基线重放、原动作轨迹与分支 A/B/A 一致性检查通过，冻结参数和 buffers 不变。', '',
        '| 分组 | 有效点 | 序列 | GT oracle ΔIoU8 | 95% 序列区间 | 改善≥0.05 点比例 |',
        '|---|---:|---:|---:|---|---:|']
    for name, values in stats.items():
        if not values['points']:
            continue
        lines.append('| %s | %d | %d | %+.6f | [%.6f, %.6f] | %.2f%% |' % (
            name, values['points'], values['sequences'], values['oracle_sequence_mean_delta'],
            *values['oracle_sequence_ci_95'], 100 * values['improvement_005_rate']))
    lines.extend(['', '## 工程门槛', '',
        '- 至少一次改善的序列：%d；至少一次改善 ≥0.05 的序列：%d。' % (
            all_stats['sequences_with_improvement_event'], all_stats['sequences_with_improvement_005']),
        '- 排除短片段后，至少一次完整八帧改善 ≥0.05 的序列：%d。' %
            stats['full_horizon']['sequences_with_improvement_005'],
        '- 失效组完整八帧中改善 ≥0.05 的比例：%.2f%%。' % (100 * failures.get('improvement_005_rate', 0)),
        '- 原工程门槛：%s；按至少 20 条序列获得完整八帧 ≥0.05 事件的保守门槛：%s。' % (gate, conservative_gate),
        '- 门槛只是决定后续投入；GT 选动作不能证明可学习性、GRU 或 SLT 有效。', '',
        '## 单步与多帧目标', '',
        '| 分组 | 单步最优动作的 ΔIoU8 | 对八帧 oracle 的遗憾 | 最优动作集合不相交比例 | 尺度 oracle ΔIoU8 | 运动额外收益 |',
        '|---|---:|---:|---:|---:|---:|'])
    for name in ('uniform_full', 'failure_full', 'healthy_uniform_full'):
        value = stats[name]
        if not value['points']:
            continue
        disjoint = value['one_step_eight_step_disjoint_rate']
        lines.append('| %s | %+.6f | %.6f | %s | %+.6f | %.6f |' % (
            name, value['one_step_chosen_delta8_sequence_mean'], value['one_step_regret8_sequence_mean'],
            '%.2f%%' % (100 * disjoint) if disjoint is not None else '',
            value['scale_oracle_delta_sequence_mean'], value['motion_increment_sequence_mean']))
    lines.extend(['', '## 固定动作干预对照', '',
        '下表是在固定 8 状态上只执行一次动作的收益，并非部署为恒定策略的整序列结果。', '',
        '| 动作 | 中心外推倍数 | 搜索因子 | 均匀组 ΔIoU8 | 95% 区间 | 正常帧组 ΔIoU8 |',
        '|---|---:|---:|---:|---|---:|'])
    for action, value in constant.items():
        a = int(action)
        lines.append('| %s | %.1f | %d | %+.6f | [%.6f, %.6f] | %+.6f |' % (
            action, (0, .5, 1)[a // 3], (4, 6, 8)[a % 3], value['sequence_mean_delta'],
            *value['ci_95'], value['healthy_sequence_mean_delta']))
    lines.extend(['', '动作并列时优先保留原动作；动作频数不能单独解释为偏好。',
        '覆盖和像素大小只描述裁剪现象，不单独提供因果结论。逐候选数据见 `per_action.csv`。',
        '未来 GT 只用于离线 IoU 与事件取样，观测和候选中心不使用未来信息。',
        '完整数据见 `fixed8_opportunity.json`，逐序列差值见 `per_sequence.csv`。'])
    (output / 'fixed8_opportunity_report.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    uniform = groups['uniform_full']
    axes[0].scatter([row['one_step_chosen_delta8'] for row in uniform if row['one_step_chosen_delta8'] is not None],
                    [row['oracle_delta'] for row in uniform if row['one_step_chosen_delta8'] is not None], s=8, alpha=.35)
    axes[0].axvline(0, color='grey', linewidth=1)
    axes[0].set_xlabel('Eight-frame gain of one-step GT choice')
    axes[0].set_ylabel('Eight-frame gain of eight-step GT choice')
    axes[0].set_title('Uniform samples; fixed-eight states')
    for group, label in (('uniform_full', 'Uniform'), ('failure_full', 'Failure'), ('reappearance_full', 'Reappearance')):
        selected = groups[group]
        if selected:
            values = np.sort([row['oracle_delta'] for row in selected])
            axes[1].step(values, np.arange(1, len(values) + 1) / len(values), label=label)
    axes[1].axvline(.05, color='grey', linewidth=1)
    axes[1].set_xlabel('GT oracle eight-frame IoU gain')
    axes[1].set_ylabel('Point-weighted cumulative fraction')
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / 'fixed8_opportunity.png', dpi=160)
    plt.close(figure)
    print(json.dumps({key: value for key, value in result.items() if key not in ('identity', 'sources')}, indent=2), flush=True)
    print('Wrote ' + str(output / 'fixed8_opportunity_report.md'), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    analyze(Path(parser.parse_args().root))
