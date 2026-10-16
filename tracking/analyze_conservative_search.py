"""Complete-sequence audit of the bounded conservative coverage experiment."""
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
import torch

from tracking.search_fixed8_intervention import file_hash
from tracking.search_policy_experiment import (
    bootstrap_mean_ci, box_iou, load_sequences, sequence_metrics, valid_frames, write_json_atomic,
)
from tracking.train_conservative_search import audit, load_cache


def report_data_root(report):
    if report.get('data_root'):
        return Path(report['data_root']).resolve()
    paths = report.get('source_reports', [])
    if not paths:
        raise ValueError('Missing evaluation data-root provenance')
    roots, names = set(), set()
    for path in paths:
        shard = json.loads(Path(path).read_text())
        if any(shard.get(k) != report.get(k) for k in ('split', 'baseline_sha256', 'protocol')):
            raise ValueError('Legacy source identity differs from merged report')
        if not shard.get('data_root') or names.intersection(shard['sequences']):
            raise ValueError('Missing data root or duplicate legacy sequences')
        roots.add(Path(shard['data_root']).resolve())
        names.update(shard['sequences'])
    if len(roots) != 1 or names != set(report['sequences']):
        raise ValueError('Legacy sources disagree or have incomplete coverage')
    return roots.pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    models = json.loads((root / 'teacher_comparison.json').read_text())
    train, dev, identity = load_cache(root / 'coverage')
    extra, extra_dev, extra_identity = load_cache(root / 'on_policy')
    if extra_identity['policy_sha256'] != file_hash(models['event']['checkpoint']):
        raise ValueError('Aggregated teacher policy mismatch')
    hashes = {json.loads(p.read_text())['network_state_sha256']
              for cache in ('coverage', 'on_policy') for p in (root / cache).glob('shard_*/manifest.json')}
    if len(hashes) != 1:
        raise ValueError('Frozen model state differs between shards')
    sources = dict(sra=ROOT / 'output/search_policy/baseline_val_v2_full.json',
        fixed8=ROOT / 'output/search_policy/learning_pilot_seed0/fixed_8/evaluate_val.json')
    sources.update({arm: root / arm / 'val_selected.json' for arm in models})
    sequences = load_sequences('val', data_root=identity['data_root'])
    names = [s[0] for s in sequences]
    if len(names) != 90:
        raise ValueError('Expected complete official validation')
    summaries, by_method, rows, counts, recovery_curves, action_context = {}, {}, [], {}, {}, {}
    repeated_predictions = {}
    for method, path in sources.items():
        report = json.loads(path.read_text())
        if (set(report['sequences']) != set(names) or report['split'] != 'val'
                or report['baseline_sha256'] != identity['checkpoint_sha256']
                or report_data_root(report) != Path(identity['data_root']).resolve()):
            raise ValueError('Incomplete or mismatched validation: ' + method)
        if method in models and report['policy_sha256'] != file_hash(models[method]['checkpoint']):
            raise ValueError('Evaluation policy hash mismatch')
        values = {}
        for name, frames, boxes, exist in sequences:
            result = report['sequences'][name]
            predictions = np.array(result['predictions'])
            if predictions.shape != boxes.shape or not np.isfinite(predictions).all():
                raise ValueError('Truncated or invalid predictions')
            if method in ('event', 'event_continue'):
                repeated_predictions.setdefault(method, {})[name] = predictions
            metrics = sequence_metrics(predictions, boxes, exist)
            valid = valid_frames(boxes, exist)
            valid[0] = False
            ious = np.array([box_iou(p, b) for p, b in zip(predictions, boxes)])
            metrics['low_iou_rate'] = float((ious[valid] < .2).mean())
            metrics['outside_rate'] = result['diagnostics']['visible_crop_center_outside_rate']
            values[name] = metrics
            rows.append(dict(method=method, sequence=name, visible_iou=metrics['visible_iou'],
                visible_auc=metrics['visible_auc'], mixed_measure=metrics['mixed_measure'],
                low_iou_rate=metrics['low_iou_rate'], outside_rate=metrics['outside_rate'],
                recovered_30=sum(d is not None for d in metrics['recovery_delays']),
                recovery_events=metrics['recovery_events']))
        by_method[method] = values
        delays = [d for v in values.values() for d in v['recovery_delays']]
        summaries[method] = {k: float(np.mean([v[k] for v in values.values()])) for k in
            ('visible_iou', 'visible_auc', 'mixed_measure', 'low_iou_rate', 'outside_rate')}
        summaries[method].update(recovered_30=sum(d is not None for d in delays),
            recovery_events=len(delays), recovery_rate_30=sum(d is not None for d in delays) / max(len(delays), 1),
            recovery_latency=float(np.mean([d for d in delays if d is not None]))
                if any(d is not None for d in delays) else None)
        recovery_curves[method] = [sum(d is not None and d <= t for d in delays) / max(len(delays), 1)
                                   for t in range(30)]
        if method in models:
            counts[method] = {str(a): sum(v.get('policy_action_counts', {}).get(str(a), 0)
                for v in report['sequences'].values()) for a in range(9)}
            expected_actions = sum(len(s[1]) - 2 for s in sequences)
            if sum(counts[method].values()) != expected_actions or any(counts[method][str(a)] for a in range(3, 9)):
                raise ValueError('Incomplete or invalid scale decisions')
            action_context[method] = {}
            action_frames = [f for seq in report['sequences'].values() for f in seq['diagnostics']['frames']
                             if 'policy_action' in f and f['visible']]
            for group, subset in (('healthy', [f for f in action_frames if f['iou'] >= .5]),
                                  ('failed', [f for f in action_frames if f['iou'] < .2])):
                action_context[method][group] = dict(frames=len(subset),
                    intervened=sum(f['policy_action'] != 2 for f in subset),
                    intervention_rate=sum(f['policy_action'] != 2 for f in subset) / max(len(subset), 1))
    parent = torch.load(models['event']['checkpoint'], map_location='cpu')
    continued = torch.load(models['event_continue']['checkpoint'], map_location='cpu')
    identical = all(parent[k] == continued[k] for k in
                    ('backbone_sha256', 'recurrent', 'mlp_depth', 'action_mode', 'decision'))
    identical = identical and parent['policy'].keys() == continued['policy'].keys() and all(
        torch.equal(parent['policy'][k], continued['policy'][k]) for k in parent['policy'])
    max_replay_error = max(float(np.abs(repeated_predictions['event'][n] -
                           repeated_predictions['event_continue'][n]).max()) for n in names)
    if identical and max_replay_error != 0:
        raise RuntimeError('Identical controller failed complete-val exact replay')
    parent_replay = dict(inference_weights_and_configuration_identical=identical,
        compared_sequences=90, max_prediction_error=max_replay_error,
        predictions_bit_identical=max_replay_error == 0)
    pairs = [(arm, ref) for arm in models for ref in ('sra', 'fixed8')]
    pairs += [('event', 'uniform'), ('event_continue', 'event'), ('aggregate', 'event_continue')]
    comparisons = {}
    for method, reference in pairs:
        deltas = {n: by_method[method][n]['visible_iou'] - by_method[reference][n]['visible_iou'] for n in names}
        comparisons[method + '_vs_' + reference] = dict(mean_delta=float(np.mean(list(deltas.values()))),
            ci_95=bootstrap_mean_ci(list(deltas.values())),
            improved=sum(d > 0 for d in deltas.values()), harmed=sum(d < 0 for d in deltas.values()),
            severe_harm_ge_01=sum(d <= -.1 for d in deltas.values()),
            worst_sequences=dict(sorted(deltas.items(), key=lambda x: x[1])[:5]), per_sequence_deltas=deltas)
    teachers = dict(train_uniform=audit([r for r in train if r['uniform']]),
        train_event=audit([r for r in train if r['event']]), train_all=audit(train), development=audit(dev),
        on_policy_train=audit(extra), on_policy_development=audit(extra_dev))
    trajectory_metrics = {}
    for cache in ('coverage', 'on_policy'):
        trajectory_metrics[cache] = {s['name']: s['baseline_metrics']
            for path in (root / cache).glob('shard_*/sequences/*.json')
            for s in [json.loads(path.read_text())]}
        if set(trajectory_metrics[cache]) != set(identity['all_sequences']):
            raise ValueError('Incomplete cached train trajectories')
    closed_loop_train = {}
    dev_names = set(identity['development_sequences'])
    for split, subset in (('train', [n for n in identity['all_sequences'] if n not in dev_names]),
                          ('internal_development', sorted(dev_names))):
        baseline_values = [trajectory_metrics['coverage'][n]['visible_iou'] for n in subset]
        event_values = [trajectory_metrics['on_policy'][n]['visible_iou'] for n in subset]
        deltas = {n: trajectory_metrics['on_policy'][n]['visible_iou'] -
                  trajectory_metrics['coverage'][n]['visible_iou'] for n in subset}
        closed_loop_train[split] = dict(sequences=len(subset), fixed8_iou=float(np.mean(baseline_values)),
            event_iou=float(np.mean(event_values)), mean_delta=float(np.mean(list(deltas.values()))),
            per_sequence_deltas=deltas, used_for_selection=False)
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    with (output / 'per_sequence.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'paired_deltas.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sequence'] + list(comparisons))
        for name in names:
            writer.writerow([name] + [c['per_sequence_deltas'][name] for c in comparisons.values()])
    runtime_path = root / 'isolated_runtime.json'
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else None
    result = dict(protocol=PROTOCOL_NAME, seed=0, sequences=90, frames=sum(len(s[1]) for s in sequences),
        summary=summaries, comparisons=comparisons, action_counts=counts, teachers=teachers,
        validation_action_context=action_context,
        parent_replay_check=parent_replay,
        event_closed_loop_on_training_split=closed_loop_train,
        models=models, recovery_curves=recovery_curves, runtime=runtime,
        frozen_network_state_sha256=next(iter(hashes)), backbone_sha256=identity['checkpoint_sha256'],
        test_used=False, three_seed_gate_evaluated=False,
        sources={k: str(p) for k, p in sources.items()})
    write_json_atomic(str(output / 'results.json'), result)
    labels = list(models)
    colors = {name: plt.get_cmap('tab10')(i) for i, name in enumerate(summaries)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for i, label in enumerate(labels):
        c = comparisons[label + '_vs_fixed8']
        mean, (lo, hi) = c['mean_delta'], c['ci_95']
        axes[0].errorbar(mean, i, xerr=[[max(mean - lo, 0)], [max(hi - mean, 0)]],
                         fmt='o', capsize=3, color=colors[label])
    axes[0].axvline(0, color='black', linewidth=.8)
    axes[0].set(yticks=range(len(labels)), yticklabels=labels, xlabel='Visible IoU delta vs fixed8 (95% paired CI)')
    for label, curve in recovery_curves.items():
        axes[1].step(range(30), curve, where='post', label=label, color=colors[label])
    axes[1].set(xlabel='Frames after valid reappearance', ylabel='Recovered fraction', ylim=(0, 1))
    axes[1].legend(fontsize=8)
    fig.savefig(output / 'paired_recovery.png', dpi=180)
    plt.close(fig)
    lines = ['# 保守尺度策略：状态覆盖与闭环汇聚验证', '',
        '本轮是 seed 0 有界筛查；冻结跟踪器，train 200 条按序列 180/20 划分，完整 val 90 条、94,711 帧，test 未使用。', '',
        '新模型回归因子 4/6 相对默认因子 8 的八帧收益，仅预测收益超过内部开发阈值才干预。'
        '均匀与事件模型使用相同缓存和更新预算；继续监督与闭环汇聚从同一事件检查点出发，增加相同更新次数。', '',
        '| 方法 | 可见 IoU | AUC | Anti-UAV | 低 IoU 比例 | 30 帧恢复 |',
        '|---|---:|---:|---:|---:|---:|']
    for method, s in summaries.items():
        lines.append('| %s | %.6f | %.6f | %.6f | %.2f%% | %d/%d |' % (
            method, s['visible_iou'], s['visible_auc'], s['mixed_measure'], 100 * s['low_iou_rate'],
            s['recovered_30'], s['recovery_events']))
    lines += ['', '| 比较 | ΔIoU | 95% 序列配对区间 | 改善/退化 |', '|---|---:|---|---:|']
    for name, c in comparisons.items():
        lines.append('| %s | %+.6f | [%.6f, %.6f] | %d/%d |' % (
            name, c['mean_delta'], *c['ci_95'], c['improved'], c['harmed']))
    lines += ['', '| 模型 | 选中局部 epoch | 阈值 | val 干预比例 | 开发集健康状态误干预损失率 |',
              '|---|---:|---:|---:|---:|']
    for method, model in models.items():
        selected = model['selected_development']
        total = sum(counts[method].values())
        lines.append('| %s | %s | %s | %.2f%% | %.2f%% |' % (method, model['selected_epoch'],
            selected['margin'], 100 * (counts[method]['0'] + counts[method]['1']) / total,
            100 * selected['healthy_harm_rate']))
    lines += ['', '| 初始事件策略完整闭环 | 序列数 | 固定 8 IoU | 事件模型 IoU | ΔIoU |',
              '|---|---:|---:|---:|---:|']
    for split, s in closed_loop_train.items():
        lines.append('| %s | %d | %.6f | %.6f | %+.6f |' % (
            split, s['sequences'], s['fixed8_iou'], s['event_iou'], s['mean_delta']))
    lines += ['', '此表由采集时完整轨迹复算，未用于选择模型；与候选教师效用不同。']
    if identical:
        lines += ['', '继续监督选回父模型；逐参数及推理配置一致，完整 90 条 val 复跑框最大误差为 %.1f。'
                  '该行是复现检查，不是独立的新模型收益。' % max_replay_error]
    lines += ['', '[配对区间与恢复曲线](paired_recovery.png)']
    if runtime:
        lines += ['', '独立速度：同卡三次轮换、每次 200 帧，去掉前 20 帧，不计解码。', '',
                  '| 方法 | 中位 FPS | 相对固定 8 |', '|---|---:|---:|']
        for method, fps in runtime['median_fps'].items():
            change = runtime.get('relative_to_fixed8_percent', {}).get(method)
            lines.append('| %s | %.2f | %s |' % (method, fps,
                '%+.2f%%' % change if change is not None else '-'))
    lines += ['', '健康状态定义为同状态因子 8 分支八帧 IoU≥0.5；误干预损失为替代动作比它下降超过 0.01。'
              '这一项来自共同内部开发候选，不是验证集上的因果归因。', '',
              '采集审计及收益密度见 results.json 的 teachers；完整逐序列和配对差值见 per_sequence.csv、paired_deltas.csv。', '',
              '结论边界：尺度覆盖来自八帧连续局部 warmup，不等于覆盖完整 4/6 轨迹；'
              '最终选择仍使用共同初始缓存的均匀开发状态，其继续行为为固定 8，实际逐帧部署存在差异。'
              '闭环汇聚已使用当前策略的状态和后续动作，另报告 on_policy_development，但未据此重新选择阈值。', '',
              '新回归门控与上一轮 softmax 教师不同，不能跨轮将变化单独归因于数据修正。'
              '本轮四组之间才是匹配对照；反复使用 val 的探索结果仍需独立 test 确认。', '']
    best = max(models, key=lambda m: summaries[m]['visible_iou'])
    c = comparisons[best + '_vs_fixed8']
    original = comparisons[best + '_vs_sra']
    if (c['mean_delta'] > 0 and c['ci_95'][0] > 0
            and original['mean_delta'] >= .01 and original['ci_95'][0] > 0):
        lines += ['本轮最优 %s 超过固定 8，但仅 seed 0；先复核其他种子，再决定 GRU／SLT。' % best]
    elif c['mean_delta'] > 0:
        lines += ['本轮有数值改善，但尚未同时建立超固定 8 的可靠优势及相对原 SRA 的预定收益条件；'
                  '不据此进入 GRU／SLT，也不将数值改善表述为方法已经有效。']
    else:
        lines += ['本轮尚未证明超过固定 8；不据此扩大 GRU／SLT。'
                  '若没有更具体的可观测信号证据，应暂停当前尺度小模块路线。']
    (output / 'phase_report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(summary=summaries, comparisons={k: {j: v for j, v in c.items()
          if j not in ('per_sequence_deltas', 'worst_sequences')} for k, c in comparisons.items()}), indent=2))


PROTOCOL_NAME = 'conservative_scale_coverage_v1'

if __name__ == '__main__':
    main()
