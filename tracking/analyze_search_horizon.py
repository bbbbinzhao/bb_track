"""Audit nested-horizon counterfactuals; aggregate paired effects by sequence."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tracking.search_horizon_diagnostic import HORIZONS, PROTOCOL, prefix_rewards
from tracking.search_fixed8_intervention import file_hash
from tracking.search_policy_experiment import aggregate_reports, bootstrap_mean_ci, write_json_atomic


def preferred_action(rewards):
    # A tie is not evidence in favor of intervening.
    maximum = np.max(rewards)
    if rewards[2] >= maximum - 1e-12:
        return 2
    return int(np.argmax(rewards))


def point_effects(candidates, policy_action):
    rewards = np.array([[c['rewards'][str(h)]['iou'] for c in candidates] for h in HORIZONS])
    if any(v is None for v in rewards.reshape(-1)):
        return None
    rewards = rewards.astype(float)
    actions = [preferred_action(r) for r in rewards]
    short, middle, long = actions
    d8 = rewards[0, short] - rewards[0, 2]
    d32 = rewards[1, short] - rewards[1, 2]
    d64 = rewards[2, short] - rewards[2, 2]
    tail = []
    for candidate in candidates:
        values = [r['iou'] for r in candidate['frames'][32:64] if r['iou'] is not None]
        tail.append(float(np.mean(values)) if values else None)
    return dict(short_action=short, middle_action=middle, long_action=long,
        short_gain8=float(d8), short_gain32=float(d32), short_gain64=float(d64),
        oracle32_gain=float(np.max(rewards[1]) - rewards[1, 2]),
        oracle64_gain=float(np.max(rewards[2]) - rewards[2, 2]),
        short_regret64=float(np.max(rewards[2]) - rewards[2, short]),
        middle_gain64=float(rewards[2, middle] - rewards[2, 2]),
        middle_regret64=float(np.max(rewards[2]) - rewards[2, middle]),
        longer_selection_gain64=float(rewards[2, middle] - rewards[2, short]),
        policy_gain8=float(rewards[0, policy_action] - rewards[0, 2]),
        policy_gain32=float(rewards[1, policy_action] - rewards[1, 2]),
        policy_gain64=float(rewards[2, policy_action] - rewards[2, 2]),
        policy_regret64=float(np.max(rewards[2]) - rewards[2, policy_action]),
        mismatch=float(short != long), short_intervenes=float(short != 2),
        short_opportunity=float(d8 >= .05), long_opportunity=float(np.max(rewards[2]) - rewards[2, 2] >= .05),
        short_reversal=float(d8 > 1e-12 and d64 < -1e-12),
        reversal_01=float(d8 >= .01 and d64 <= -.01),
        reversal_05=float(d8 >= .05 and d64 <= -.05),
        policy_intervenes=float(policy_action != 2),
        short_tail_gain=None if any(v is None for v in tail) else float(tail[short] - tail[2]))


def long_only_effects(candidates, policy_action):
    values = [candidate['rewards']['64']['iou'] for candidate in candidates]
    if any(value is None for value in values):
        return None
    return dict(policy_gain64=float(values[policy_action] - values[2]),
                oracle64_gain=float(max(values) - values[2]))


def summarize_long_only(rows):
    by_sequence = {}
    for row in rows:
        by_sequence.setdefault(row['sequence'], []).append(row)
    result = dict(sequences=len(by_sequence), points=len(rows), metrics={})
    for key in ('policy_gain64', 'oracle64_gain'):
        values = [float(np.mean([r[key] for r in records])) for records in by_sequence.values()]
        result['metrics'][key] = dict(mean=float(np.mean(values)) if values else None,
            sequence_bootstrap_ci95=bootstrap_mean_ci(values) if values else None)
    return result


METRICS = ('short_gain8', 'short_gain32', 'short_gain64', 'short_regret64',
           'middle_gain64', 'middle_regret64', 'longer_selection_gain64',
           'oracle32_gain', 'oracle64_gain', 'policy_gain8', 'policy_gain32',
           'policy_gain64', 'policy_regret64', 'mismatch', 'short_intervenes',
           'short_opportunity', 'long_opportunity', 'short_reversal',
           'reversal_01', 'reversal_05', 'policy_intervenes', 'short_tail_gain')


def summarize(rows):
    by_sequence = {}
    for row in rows:
        by_sequence.setdefault(row['sequence'], []).append(row)
    result = dict(sequences=len(by_sequence), points=len(rows), metrics={})
    for key in METRICS:
        values = [float(np.mean([r[key] for r in records if r[key] is not None]))
                  for records in by_sequence.values() if any(r[key] is not None for r in records)]
        result['metrics'][key] = dict(mean=float(np.mean(values)) if values else None,
            sequence_bootstrap_ci95=bootstrap_mean_ci(values) if values else None,
            sequences=len(values))
    result['counts'] = {key: int(sum(r[key] for r in rows)) for key in (
        'mismatch', 'short_intervenes', 'short_opportunity', 'long_opportunity',
        'short_reversal', 'reversal_01', 'reversal_05', 'policy_intervenes')}
    selected = [r for r in rows if r['short_gain8'] >= .01]
    result['conditional_short_gain_at_least_01'] = dict(points=len(selected),
        negative64=sum(r['short_gain64'] < 0 for r in selected),
        negative64_at_least_01=sum(r['short_gain64'] <= -.01 for r in selected),
        mean_gain64=float(np.mean([r['short_gain64'] for r in selected])) if selected else None)
    return result


def plot_effects(directory, groups, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, group_name in enumerate(('uniform', 'event')):
        for column, continuation in enumerate(('fixed8', 'aggregate')):
            axis = axes[row, column]
            summary = groups['all/' + group_name + '/' + continuation]
            if not summary['points']:
                continue
            for prefix, label, color in [('short', '8-frame oracle action', '#127b89'),
                                         ('policy', 'Current model action', '#c44e52')]:
                metrics = [summary['metrics'][prefix + '_gain' + str(h)] for h in HORIZONS]
                means = np.array([metric['mean'] for metric in metrics])
                bounds = np.array([metric['sequence_bootstrap_ci95'] for metric in metrics])
                axis.plot(HORIZONS, means, marker='o', color=color, label=label)
                axis.fill_between(HORIZONS, bounds[:, 0], bounds[:, 1], color=color, alpha=.15)
            axis.axhline(0, color='#555555', linewidth=.8)
            axis.set(title=group_name + ' / continuation: ' + continuation,
                     xlabel='Reward horizon (frames)', ylabel='Visible IoU difference vs first factor 8',
                     xticks=list(HORIZONS))
            axis.legend(fontsize=8)
    figure.savefig(directory / 'nested_rewards.png', dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, continuation in zip(axes, ('fixed8', 'aggregate')):
        for group_name, color, marker in [('uniform', '#127b89', 'o'), ('event', '#c44e52', 'x')]:
            subset = [r for r in records if r[group_name] and r['continuation'] == continuation]
            axis.scatter([r['short_gain8'] for r in subset], [r['short_gain64'] for r in subset],
                         label=group_name, color=color, marker=marker, s=15, alpha=.55)
        axis.axhline(0, color='#555555', linewidth=.8)
        axis.axvline(.01, color='#555555', linewidth=.8, linestyle=':')
        axis.set(title='Continuation: ' + continuation, xlabel='8-frame gain of short oracle action',
                 ylabel='64-frame gain of the same action')
        axis.legend(fontsize=8)
    figure.savefig(directory / 'short_long_scatter.png', dpi=180)
    plt.close(figure)


def analyze_development(root):
    root = Path(root)
    registration = json.loads((root / 'protocol.json').read_text())
    if registration['split'] != 'train' or registration['protocol'] != PROTOCOL:
        raise ValueError('Expected registered train-only diagnostic')
    baselines = {}
    for path in root.glob('shard_*/manifest.json'):
        manifest = json.loads(path.read_text())
        if not manifest['complete'] or not manifest['frozen_weights_and_buffers_unchanged']:
            raise ValueError('Incomplete fixed8 reference')
        for key, value in manifest['identity'].items():
            if key != 'expected_sequences' and value != registration.get(key):
                raise ValueError('Fixed8 reference identity differs: ' + key)
        for name, entry in manifest['sequences'].items():
            if name in baselines:
                raise ValueError('Duplicate fixed8 reference sequence')
            payload = json.loads((path.parent / entry['file']).read_text())
            baselines[name] = payload['baseline_metrics']
    if set(baselines) != set(registration['all_sequences']):
        raise ValueError('Fixed8 reference must cover all training sequences')
    policies, hashes = {}, set()
    frames = 0
    for path in (root / 'internal_dev').glob('shard_*/manifest.json'):
        manifest = json.loads(path.read_text())
        if not manifest['complete'] or not manifest['frozen_weights_and_buffers_unchanged']:
            raise ValueError('Incomplete internal development evaluation')
        for key, value in manifest['identity'].items():
            if key not in ('expected_sequences', 'experiment', 'controller', 'trajectory') and value != registration.get(key):
                raise ValueError('Development identity mismatch: ' + key)
        if manifest['identity']['experiment'] != 'unchanged_aggregate_full_internal_development':
            raise ValueError('Unexpected development experiment')
        if (manifest['identity']['controller'] != 'aggregate'
                or manifest['identity']['trajectory'] != 'full_sequence_no_gt_resets'):
            raise ValueError('Unexpected development controller or resets')
        if set(manifest['sequences']) != set(manifest['identity']['expected_sequences']):
            raise ValueError('Incomplete development shard coverage')
        hashes.add(manifest['network_state_sha256'])
        for name, entry in manifest['sequences'].items():
            if name in policies:
                raise ValueError('Duplicate development sequence')
            payload = json.loads((path.parent / entry['file']).read_text())
            if payload['name'] != name or len(payload['predictions']) != payload['frame_count']:
                raise ValueError('Invalid full-trajectory development result')
            policies[name] = payload['metrics']
            frames += payload['frame_count']
    if set(policies) != set(registration['development_sequences']) or len(hashes) != 1:
        raise ValueError('Development split coverage or network identity differs')
    horizon = json.loads((root / 'analysis/results.json').read_text())
    if next(iter(hashes)) != horizon['network_state_sha256']:
        raise ValueError('Horizon/development frozen identities differ')
    rows = [dict(sequence=name, fixed8=baselines[name]['visible_iou'],
                 aggregate=policies[name]['visible_iou'],
                 delta=policies[name]['visible_iou'] - baselines[name]['visible_iou'])
            for name in sorted(policies)]
    delta = [r['delta'] for r in rows]
    result = dict(protocol=PROTOCOL, split='train_internal_dev20', sequences=len(rows), frames=frames,
        selection='unchanged_previously_selected_checkpoint_and_margin',
        fixed8=aggregate_reports([baselines[name] for name in policies]),
        aggregate=aggregate_reports(list(policies.values())), delta=float(np.mean(delta)),
        paired_sequence_ci95=bootstrap_mean_ci(delta),
        improved_sequences=sum(d > 0 for d in delta), worse_sequences=sum(d < 0 for d in delta),
        all_shards_audited=True, network_state_sha256=next(iter(hashes)))
    for label, values in [('fixed8', [baselines[name] for name in policies]), ('aggregate', policies.values())]:
        delays = [delay for value in values for delay in value['recovery_delays']]
        result[label]['recovery_event_count'] = len(delays)
        result[label]['recovered_event_count'] = sum(delay is not None for delay in delays)
    directory = root / 'analysis'
    write_json_atomic(str(directory / 'internal_dev_results.json'), result)
    with (directory / 'internal_dev_per_sequence.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ci = result['paired_sequence_ci95']
    text = ['# 当前汇聚模型的完整内部开发集闭环', '',
        '使用既有 20 条 train 内部开发序列；检查点和 margin 保持上一轮选择，不重新选择。'
        '每条仅首帧用 GT 初始化，此后完整跟踪，不逐帧重置。固定 8 对照来自本轮完整基准轨迹。', '',
        '序列数 %d，原始帧数 %d。' % (len(rows), frames), '',
        '| 方法 | 可见 IoU | AUC | Anti-UAV | 30 帧恢复 |', '|---|---:|---:|---:|---:|']
    for label in ('fixed8', 'aggregate'):
        value = result[label]
        text.append('| %s | %.6f | %.6f | %.6f | %d/%d |' % (
            label, value['visible_iou'], value['visible_auc'], value['mixed_measure'],
            value['recovered_event_count'], value['recovery_event_count']))
    text += ['', '汇聚模型 − 固定 8：%+.6f，序列配对 10,000 次 bootstrap 95%% 区间 [%.6f, %.6f]；'
             '改善 %d 条，退化 %d 条。' % (result['delta'], ci[0], ci[1],
             result['improved_sequences'], result['worse_sequences']), '',
             '该开发集此前已参与模型选择，本轮仅补全闭环诊断，不能称为独立 test 证据。'
             '本轮未训练新策略、未改变阈值、未评测官方 val/test。', '']
    (directory / 'internal_dev_report.md').write_text('\n'.join(text))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def analyze(root):
    root = Path(root)
    registration = json.loads((root / 'protocol.json').read_text())
    if registration['protocol'] != PROTOCOL or registration['split'] != 'train':
        raise ValueError('Only pre-registered train-only horizon diagnostics accepted')
    records, long_only_records, seen, hashes, frames = [], [], set(), set(), 0
    incomplete_points = 0
    all_points = 0
    for path in sorted(root.glob('shard_*/manifest.json')):
        manifest = json.loads(path.read_text())
        if not manifest.get('complete') or not manifest.get('frozen_weights_and_buffers_unchanged'):
            raise ValueError('Incomplete or unaudited shard ' + str(path))
        identity = manifest['identity']
        for key, value in identity.items():
            if key != 'expected_sequences' and registration.get(key) != value:
                raise ValueError('Shard protocol mismatch: ' + key)
        if set(manifest['sequences']) != set(identity['expected_sequences']):
            raise ValueError('Shard expected sequence coverage differs')
        hashes.add(manifest['network_state_sha256'])
        for name, entry in manifest['sequences'].items():
            if name in seen:
                raise ValueError('Duplicate sequence ' + name)
            seen.add(name)
            payload = json.loads((path.parent / entry['file']).read_text())
            if payload['name'] != name or payload['protocol'] != PROTOCOL:
                raise ValueError('Sequence identity differs')
            if not payload['fixed8_replayed'] or not payload['branch_order_invariant']:
                raise ValueError('Missing replay/branch isolation audit')
            frames += payload['frame_count']
            if entry['points'] != len(payload['samples']):
                raise ValueError('Point count differs')
            for sample in payload['samples']:
                all_points += 1
                if not sample['full_horizon'] or sample['frame'] + 64 >= payload['frame_count']:
                    raise ValueError('Incomplete nested horizon')
                if len(sample['observation']) != 32 or not np.isfinite(sample['observation']).all():
                    raise ValueError('Invalid causal observation')
                if set(sample['continuations']) != {'fixed8', 'aggregate'}:
                    raise ValueError('Continuation coverage differs')
                for action in range(3):
                    fixed = sample['continuations']['fixed8'][action]
                    learned = sample['continuations']['aggregate'][action]
                    np.testing.assert_array_equal(fixed['predictions'][0], learned['predictions'][0])
                    if fixed['hann'][0] != learned['hann'][0]:
                        raise ValueError('Future continuation affected first-frame Hann')
                for continuation, candidates in sample['continuations'].items():
                    if [c['action'] for c in candidates] != [0, 1, 2]:
                        raise ValueError('Candidate actions differ')
                    counts = None
                    for candidate in candidates:
                        if candidate['rewards'] != prefix_rewards(candidate['frames']):
                            raise ValueError('Stored rewards do not match nested prefixes')
                        mask = [s['iou'] is not None for s in candidate['frames']]
                        if counts is not None and mask != counts:
                            raise ValueError('Candidate GT visibility masks differ')
                        counts = mask
                    effect = point_effects(candidates, sample['policy_action'])
                    if effect is None:
                        incomplete_points += 1
                        long_effect = long_only_effects(candidates, sample['policy_action'])
                        if long_effect is not None:
                            long_only_records.append(dict(sequence=name, frame=sample['frame'],
                                split='dev20' if name in registration['development_sequences'] else 'train180',
                                continuation=continuation, uniform=sample['uniform'], event=sample['event'],
                                **long_effect))
                        continue
                    records.append(dict(sequence=name, frame=sample['frame'],
                        split='dev20' if name in registration['development_sequences'] else 'train180',
                        continuation=continuation, uniform=sample['uniform'], event=sample['event'],
                        current_iou=sample['current_iou'], policy_action=sample['policy_action'], **effect))
    if seen != set(registration['all_sequences']) or len(hashes) != 1:
        raise ValueError('Global sequence coverage or frozen state identity differs')
    for path_key, hash_key in [('checkpoint', 'checkpoint_sha256'), ('policy', 'policy_sha256')]:
        # Resolve the paths from the immutable launch protocol, not a guessed experiment name.
        key = path_key + '_path'
        if key in registration and file_hash(registration[key]) != registration[hash_key]:
            raise ValueError('Checkpoint file changed during collection')
    groups = {}
    for split in ('all', 'train180', 'dev20'):
        for group in ('uniform', 'event'):
            for continuation in ('fixed8', 'aggregate'):
                subset = [r for r in records if (split == 'all' or r['split'] == split)
                          and r[group] and r['continuation'] == continuation]
                groups['/'.join((split, group, continuation))] = summarize(subset)
    result = dict(protocol=PROTOCOL, split='train', sequences=len(seen), baseline_frames=frames,
        sampled_states=all_points, paired_rows=len(records), missing_early_visible_rows=incomplete_points,
        network_state_sha256=next(iter(hashes)), all_shards_audited=True, groups=groups)
    result['long_only_groups'] = {}
    for split in ('all', 'train180', 'dev20'):
        for group in ('uniform', 'event'):
            for continuation in ('fixed8', 'aggregate'):
                subset = [r for r in long_only_records if (split == 'all' or r['split'] == split)
                          and r[group] and r['continuation'] == continuation]
                result['long_only_groups']['/'.join((split, group, continuation))] = summarize_long_only(subset)
    directory = root / 'analysis'
    directory.mkdir(exist_ok=True)
    result['current_quality_groups'] = {}
    for group_name in ('uniform', 'event'):
        for continuation in ('fixed8', 'aggregate'):
            for condition in ('healthy', 'failed'):
                subset = [r for r in records if r[group_name] and r['continuation'] == continuation
                          and r['current_iou'] is not None
                          and (r['current_iou'] >= .5 if condition == 'healthy' else r['current_iou'] < .2)]
                result['current_quality_groups']['/'.join((group_name, continuation, condition))] = summarize(subset)
    write_json_atomic(str(directory / 'results.json'), result)
    with (directory / 'paired_points.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if long_only_records:
        with (directory / 'long_only_points.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(long_only_records[0]))
            writer.writeheader()
            writer.writerows(long_only_records)
    sequence_rows = []
    for key in groups:
        split, group, continuation = key.split('/')
        if split == 'all':
            continue
        for name in sorted(seen):
            subset = [r for r in records if r['sequence'] == name and r['split'] == split
                      and r[group] and r['continuation'] == continuation]
            if subset:
                sequence_rows.append(dict(sequence=name, split=split, group=group,
                    continuation=continuation, points=len(subset), **{
                        key: float(np.mean([r[key] for r in subset if r[key] is not None]))
                        if any(r[key] is not None for r in subset) else None for key in METRICS}))
    with (directory / 'per_sequence.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sequence_rows[0]))
        writer.writeheader()
        writer.writerows(sequence_rows)
    plot_effects(directory, groups, records)
    lines = ['# 8/32/64 帧回报窗口诊断', '',
        '冻结 FocusTrack；仅使用官方 train，按既有 180/20 划分分别报告。'
        '全部候选只替换下一帧裁剪，之后按指定控制器继续；不存在 GT 重置。', '',
        '完成 %d 条、%d 原始帧、%d 个相同状态；%d 条继续分支因前八帧没有合法可见 GT，'
        '不进入跨窗口配对比较。' % (len(seen), frames, all_points, incomplete_points), '',
        '下表所有均值先在序列内求平均，再跨序列平均；区间为 10,000 次序列 bootstrap。'
        '短期动作是用 GT 选择的八帧最优候选，不能当作可部署策略。', '',
        '| 分组／后续策略 | 序列／状态 | 短期动作 Δ8 | 同动作 Δ32 | 同动作 Δ64 [95% CI] | 64 帧遗憾 | 明显反转数¹ |',
        '|---|---:|---:|---:|---|---:|---:|']
    for key, group in groups.items():
        if not group['points']:
            continue
        metrics = group['metrics']
        mean = lambda name: metrics[name]['mean']
        ci = metrics['short_gain64']['sequence_bootstrap_ci95']
        lines.append('| %s | %d/%d | %+.6f | %+.6f | %+.6f [%.6f, %.6f] | %.6f | %d |' % (
            key, group['sequences'], group['points'], mean('short_gain8'), mean('short_gain32'),
            mean('short_gain64'), ci[0], ci[1], mean('short_regret64'), group['counts']['reversal_01']))
    lines += ['', '¹ 明显反转：相对同后续策略的首帧因子 8，八帧收益≥0.01 且 64 帧收益≤−0.01。'
              '这是一帧动作的局部干预结果，不能解释为整段持续采用该动作的收益。', '',
              '## 延长到 32 帧选动作的作用', '',
              '| 分组／后续策略 | 32 帧动作 Δ64 | 相对八帧动作的 64 帧增益 [95% CI] |',
              '|---|---:|---|']
    for key, group in groups.items():
        if not group['points']:
            continue
        metrics = group['metrics']
        value = metrics['longer_selection_gain64']
        ci = value['sequence_bootstrap_ci95']
        lines.append('| %s | %+.6f | %+.6f [%.6f, %.6f] |' % (
            key, metrics['middle_gain64']['mean'], value['mean'], ci[0], ci[1]))
    lines += ['',
              '## 当前模型动作的配对效果', '',
              '| 分组／后续策略 | 当前动作 Δ8 | 当前动作 Δ32 | 当前动作 Δ64 [95% CI] | 64 帧 oracle 收益 |',
              '|---|---:|---:|---|---:|']
    for key, group in groups.items():
        if not group['points']:
            continue
        metrics = group['metrics']
        ci = metrics['policy_gain64']['sequence_bootstrap_ci95']
        lines.append('| %s | %+.6f | %+.6f | %+.6f [%.6f, %.6f] | %.6f |' % (
            key, metrics['policy_gain8']['mean'], metrics['policy_gain32']['mean'],
            metrics['policy_gain64']['mean'], ci[0], ci[1], metrics['oracle64_gain']['mean']))
    lines += ['', '## 前八帧无定位奖励、64 帧有奖励的状态', '',
              '这些状态没有可定义的八帧最优动作，不进入上面的窗口配对；单列长期动作差值。', '',
              '| 分组／后续策略 | 序列／状态 | 当前模型 Δ64 | 64 帧 oracle 收益 |',
              '|---|---:|---:|---:|']
    for key, group in result['long_only_groups'].items():
        if group['points']:
            lines.append('| %s | %d/%d | %+.6f | %.6f |' % (
                key, group['sequences'], group['points'], group['metrics']['policy_gain64']['mean'],
                group['metrics']['oracle64_gain']['mean']))
    lines += ['', '## 解释边界', '',
              '- 状态来自固定 8 的完整轨迹；当前模型后续分支检验的是同状态下的继续规则，'
              '不覆盖该模型长期部署产生的全部状态。',
              '- 事件样本由 GT 和基准失效记录选取，不能当作自然部署频率；均匀与事件样本分别报告。',
              '- 所有窗口使用相同 64 帧轨迹的前缀；消失帧推进状态但不计算定位奖励。'
              '不同窗口可见帧数可以不同，帧级原始 IoU 已保存。',
              '- 64 帧有限窗口仍不是完整序列目标；oracle 动作差异和遗憾本身不证明可预测或可学习。',
              '- 未重新训练策略，未对官方 val/test 进行本轮评测，未检验 SLT 的实际收益。',
              '- 所有采样点通过 64 帧 A/B/A 重放；默认固定 8 分支逐框及 Hann 精确复现基准；'
              '全部分片网络参数与 buffers 哈希一致。', '']
    (directory / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(sequences=len(seen), states=all_points,
        report=str(directory / 'report.md')), ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    analyze(parser.parse_args().output)
