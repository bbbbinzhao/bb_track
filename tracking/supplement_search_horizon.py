"""Reproducible coverage, concentration and illustrative cases for horizon diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tracking.search_policy_experiment import write_json_atomic


def supplement(root):
    root = Path(root)
    audit = json.loads((root / 'analysis/results.json').read_text())
    if not audit['all_shards_audited'] or audit['split'] != 'train':
        raise ValueError('Require audited train-only horizon results')
    with (root / 'analysis/paired_points.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    coverage = dict(raw_uniform=0, raw_event=0, zero_state_sequences=[],
                    no_visible8=0, long_only=0, no_visible64=0)
    samples = {}
    sequence_names = set()
    for path in root.glob('shard_*/sequences/*.json'):
        payload = json.loads(path.read_text())
        name = payload['name']
        sequence_names.add(name)
        if not payload['samples']:
            coverage['zero_state_sequences'].append(dict(name=name, frames=payload['frame_count']))
        for sample in payload['samples']:
            samples[name, sample['frame']] = sample
            coverage['raw_uniform'] += sample['uniform']
            coverage['raw_event'] += sample['event']
            reward = sample['continuations']['fixed8'][0]['rewards']
            coverage['no_visible8'] += reward['8']['iou'] is None
            coverage['long_only'] += reward['8']['iou'] is None and reward['64']['iou'] is not None
            coverage['no_visible64'] += reward['64']['iou'] is None
    if len(samples) != audit['sampled_states'] or len(sequence_names) != audit['sequences']:
        raise ValueError('Supplemental raw coverage differs from audited results')
    concentration, case_rows, plot_cases = {}, [], []
    for continuation in ('fixed8', 'aggregate'):
        selected = [r for r in rows if r['continuation'] == continuation and r['event'] == 'True']
        by_sequence = {}
        for row in selected:
            by_sequence.setdefault(row['sequence'], []).append(float(row['longer_selection_gain64']))
        values = sorted([(name, float(np.mean(scores))) for name, scores in by_sequence.items()],
                        key=lambda pair: pair[1], reverse=True)
        total = sum(score for _, score in values)
        concentration[continuation] = dict(
            event_points=len(selected), sequences=len(values),
            top5_sequences=[dict(sequence=name, mean_gain=value) for name, value in values[:5]],
            top1_fraction=values[0][1] / total if total else None,
            top5_fraction=sum(value for _, value in values[:5]) / total if total else None,
            mean_excluding_top1=float(np.mean([value for _, value in values[1:]])),
            negative_sequences=sum(value < 0 for _, value in values),
            delayed_opportunity_count=sum(float(row['short_gain8']) < .01
                and float(row['oracle64_gain']) >= .05 for row in selected))
        selections = [('long_selection_gain', sorted(selected,
            key=lambda row: float(row['longer_selection_gain64']), reverse=True)),
            ('short_action_harm', sorted([row for row in selected
                if float(row['short_gain8']) >= .01 and float(row['short_gain64']) < 0],
                key=lambda row: float(row['short_gain64'])))]
        for category, ranked in selections:
            for rank, row in enumerate(ranked[:3], 1):
                sample = samples[row['sequence'], int(row['frame'])]
                candidates = sample['continuations'][continuation]
                case = dict(category=category, rank=rank, continuation=continuation,
                    sequence=row['sequence'], frame=int(row['frame']), split=row['split'],
                    current_iou=sample['current_iou'],
                    short_factor=(4, 6, 8)[int(row['short_action'])],
                    middle_factor=(4, 6, 8)[int(row['middle_action'])],
                    long_factor=(4, 6, 8)[int(row['long_action'])],
                    policy_factor=(4, 6, 8)[sample['policy_action']],
                    short_gain8=float(row['short_gain8']), short_gain64=float(row['short_gain64']),
                    longer_selection_gain64=float(row['longer_selection_gain64']))
                for candidate in candidates:
                    for horizon in (8, 32, 64):
                        case['factor%d_reward%d' % (candidate['factor'], horizon)] = candidate['rewards'][str(horizon)]['iou']
                case_rows.append(case)
                if rank == 1:
                    plot_cases.append((case, candidates))
    result = dict(coverage=coverage, concentration=concentration,
        cases_selection='top three event-state effects per category and continuation; harm requires gain8>=.01 and gain64<0; illustrative only')
    directory = root / 'analysis'
    write_json_atomic(str(directory / 'sensitivity.json'), result)
    with (directory / 'illustrative_cases.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for axis, (case, candidates) in zip(axes.reshape(-1), plot_cases):
        for candidate, color in zip(candidates, ('#127b89', '#c44e52', '#424242')):
            values = np.array([r['iou'] if r['iou'] is not None else np.nan for r in candidate['frames']])
            counts = np.cumsum(np.isfinite(values))
            curve = np.divide(np.nancumsum(values), counts,
                out=np.full(len(values), np.nan), where=counts > 0)
            axis.plot(np.arange(1, 65), curve, color=color,
                      label='First factor %d' % candidate['factor'])
        for horizon in (8, 32):
            axis.axvline(horizon, color='#999999', linewidth=.7, linestyle=':')
        axis.set(title='%s / %s\n%s, t=%d' % (case['continuation'], case['category'],
                     case['sequence'], case['frame']), xlabel='Reward horizon',
                 ylabel='Mean visible IoU of nested prefix', ylim=(0, 1))
        axis.legend(fontsize=8)
    figure.savefig(directory / 'illustrative_trajectories.png', dpi=180)
    plt.close(figure)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    supplement(parser.parse_args().output)
