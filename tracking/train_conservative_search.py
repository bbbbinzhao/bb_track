"""Matched regression gates with uniform, event-balanced and aggregated train data."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from lib.models.search_policy import SearchPolicy, select_search_action
from lib.train.data.search_policy import development_sequences
from tracking.search_conservative_coverage import PROTOCOL
from tracking.search_fixed8_intervention import file_hash
from tracking.search_policy_experiment import write_json_atomic
from tracking.train_fixed8_teacher import per_sequence

MARGINS = [0., .0025, .005, .01, .02, .05, None]


def load_cache(root):
    records, identity, seen, manifests = [], None, set(), []
    for path in sorted(Path(root).glob('shard_*/manifest.json')):
        payload = json.loads(path.read_text())
        current = {k: v for k, v in payload['identity'].items() if k != 'expected_sequences'}
        if current.get('split') != 'train' or current.get('protocol') != PROTOCOL:
            raise ValueError('Only versioned train conservative teachers are allowed')
        if not payload['complete'] or not payload['frozen_weights_and_buffers_unchanged']:
            raise ValueError('Incomplete or mutable tracker teachers')
        if identity is not None and current != identity:
            raise ValueError('Incompatible coverage shards')
        identity = current
        if set(payload['sequences']) != set(payload['identity']['expected_sequences']):
            raise ValueError('Missing shard sequences')
        for name, item in payload['sequences'].items():
            if name in seen:
                raise ValueError('Duplicate sequence')
            seen.add(name)
            sequence = json.loads((path.parent / item['file']).read_text())
            if sequence['name'] != name or not sequence['branch_order_invariant']:
                raise ValueError('Bad branch identity')
            if current['collection'] == 'on_policy' and not sequence['policy_baseline_reproduced']:
                raise ValueError('Current policy branch failed replay')
            for sample in sequence['samples']:
                candidates = sample['candidates']
                if [c['action'] for c in candidates] != [0, 1, 2] or not sample['full_horizon']:
                    raise ValueError('Bad candidate set')
                if candidates[0]['next_iou'] is None:
                    continue
                record = dict(sequence=name, frame=sample['frame'], uniform=sample['uniform'],
                    event=sample['event'], observation=np.asarray(sample['observation'], dtype=np.float32),
                    eight=np.array([c['future_iou'] for c in candidates], dtype=np.float32),
                    one=np.array([c['next_iou'] for c in candidates], dtype=np.float32))
                if not all(np.isfinite(record[k]).all() for k in ('observation', 'eight', 'one')):
                    raise ValueError('Nonfinite teacher')
                records.append(record)
        manifests.append(dict(path=str(path.resolve()), sha256=file_hash(path)))
    if identity is None or seen != set(identity['all_sequences']) or len(seen) != 200:
        raise ValueError('Require exact 200-sequence training coverage')
    dev = development_sequences(sorted(seen))
    return ([r for r in records if r['sequence'] not in dev],
            [r for r in records if r['sequence'] in dev],
            dict(identity, development_sequences=sorted(dev), manifests=manifests))


def audit(records):
    scores = np.array([r['eight'] for r in records])
    gain = scores.max(-1) - scores[:, 2]
    factors = np.array([r['observation'][29] * 8 for r in records])
    return dict(states=len(records), sequences=len({r['sequence'] for r in records}),
        uniform_states=sum(r['uniform'] for r in records), event_states=sum(r['event'] for r in records),
        crop_factors={str(f): int(np.isclose(factors, f).sum()) for f in (4., 6., 8.)},
        oracle_gain_mean=float(gain.mean()), oracle_gain_ge_005=int((gain >= .05).sum()),
        oracle_gain_ge_005_rate=float((gain >= .05).mean()))


def draw_indices(records, count, generator, event_balanced):
    uniform = torch.tensor([i for i, r in enumerate(records) if r['uniform']], dtype=torch.long)
    events = torch.tensor([i for i, r in enumerate(records) if r['event']], dtype=torch.long)
    if not len(uniform):
        raise ValueError('No uniform training states')
    if event_balanced and len(events):
        first = uniform[torch.randint(len(uniform), (count // 2,), generator=generator)]
        second = events[torch.randint(len(events), (count - count // 2,), generator=generator)]
        combined = torch.cat([first, second])
        return combined[torch.randperm(len(combined), generator=generator)]
    return uniform[torch.randint(len(uniform), (count,), generator=generator)]


@torch.no_grad()
def gate_metrics(policy, records, margin):
    obs = torch.tensor(np.array([r['observation'] for r in records]), dtype=torch.float32)
    scores = torch.tensor(np.array([r['eight'] for r in records]), dtype=torch.float32)
    logits, _ = policy(obs)
    decision = dict(rule='advantage_margin_v1', margin=margin)
    actions = select_search_action(logits, torch.tensor([0, 1, 2]), decision)
    chosen = scores.gather(1, actions[:, None]).squeeze(1)
    delta = chosen - scores[:, 2]
    by_sequence = per_sequence(records, delta.tolist())
    healthy = scores[:, 2] >= .5
    intervened = actions != 2
    advantages = scores[:, :2] - scores[:, 2:3]
    opportunities = advantages.max(-1).values >= .05
    return dict(margin=margin, mean_delta=float(np.mean(list(by_sequence.values()))),
        intervention_rate=float(intervened.float().mean()),
        action_counts={str(i): int((actions == i).sum()) for i in range(3)},
        per_sequence_delta=by_sequence,
        intervention_harm_rate=float(((delta < -.01) & intervened).sum() / max(int(intervened.sum()), 1)),
        healthy_states=int(healthy.sum()),
        healthy_harm_rate=float(((delta < -.01) & intervened & healthy).sum() / max(int(healthy.sum()), 1)),
        healthy_net_delta=float(delta[healthy].mean()) if healthy.any() else None,
        prediction_mse=float(((logits[:, :2] - advantages) ** 2).mean()),
        zero_predictor_mse=float((advantages ** 2).mean()),
        positive_gain_precision=float(((delta > 0) & intervened).sum() / max(int(intervened.sum()), 1)),
        opportunities_ge_005=int(opportunities.sum()),
        opportunity_recall_ge_005=float(((delta >= .05) & opportunities).sum() / max(int(opportunities.sum()), 1)))


def selection_key(record):
    return record['mean_delta'], -record['intervention_rate']


def add_diagnostics(report, train, development):
    checkpoint = torch.load(report['checkpoint'], map_location='cpu')
    model = SearchPolicy(recurrent=False, mlp_depth=2).eval()
    model.load_state_dict(checkpoint['policy'])
    margin = checkpoint['decision']['margin']
    report['selected_diagnostics'] = {}
    for split, rows in (('train', train), ('development', development)):
        for group in ('uniform', 'event'):
            subset = [r for r in rows if r[group]]
            report['selected_diagnostics'][split + '_' + group] = gate_metrics(model, subset, margin) if subset else None
    write_json_atomic(str(Path(report['checkpoint']).parent / 'training_report.json'), report)
    return report


def train_arm(args, arm, records, development, identity, additional=None, initial=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    policy = SearchPolicy(recurrent=False, mlp_depth=2)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-4)
    base_updates, base_epoch = 0, 0
    if initial:
        loaded = torch.load(initial, map_location='cpu')
        policy.load_state_dict(loaded['policy'])
        optimizer.load_state_dict(loaded['optimizer'])
        generator.set_state(loaded['rng']['generator'])
        base_updates, base_epoch = loaded['updates'], loaded['epoch']
    pool = records + (additional or [])
    observations = torch.tensor(np.array([r['observation'] for r in pool]), dtype=torch.float32)
    values = torch.tensor(np.array([r['eight'] for r in pool]), dtype=torch.float32)
    target = values[:, :2] - values[:, 2:3]
    output = Path(args.output) / arm
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_identity = dict(protocol=PROTOCOL, recurrent=False, mlp_depth=2, action_mode='scale',
        backbone_sha256=identity['checkpoint_sha256'], cache_identity=identity, seed=args.seed,
        arm=arm, regression='eight-frame advantage to factor8', initial_checkpoint=initial,
        states_per_epoch=args.states_per_epoch, batch_size=args.batch_size)
    history, best, best_epoch, best_path = [], None, None, None
    max_gradient = 0.
    # Epoch zero retains the parent as a candidate under the same dev criterion.
    start_epoch = 0 if initial else 1
    for epoch in range(start_epoch, args.epochs + 1):
        losses = []
        if epoch:
            if additional:
                count = args.states_per_epoch // 2
                old = draw_indices(records, count, generator, True)
                new = draw_indices(additional, args.states_per_epoch - count, generator, True) + len(records)
                selected = torch.cat([old, new])
                selected = selected[torch.randperm(len(selected), generator=generator)]
            else:
                selected = draw_indices(records, args.states_per_epoch, generator, arm != 'uniform')
            policy.train()
            for start in range(0, len(selected), args.batch_size):
                ids = selected[start:start + args.batch_size]
                prediction, _ = policy(observations[ids])
                loss = ((prediction[:, :2] - target[ids]) ** 2).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
                if not torch.isfinite(loss) or not torch.isfinite(gradient):
                    raise RuntimeError('Nonfinite regression loss or gradient')
                max_gradient = max(max_gradient, float(gradient))
                optimizer.step()
                losses.append(float(loss.detach()))
        policy.eval()
        grid = [gate_metrics(policy, development, margin) for margin in MARGINS]
        candidate = max(grid, key=selection_key)
        updates = base_updates + epoch * ((args.states_per_epoch + args.batch_size - 1) // args.batch_size)
        history.append(dict(epoch=epoch, updates=updates, train_loss=float(np.mean(losses)) if losses else None,
                            development_grid=grid, selected_development=candidate))
        path = output / ('policy_seed%d_ep%03d.pth' % (args.seed, epoch))
        torch.save(dict(checkpoint_identity, policy=copy.deepcopy(policy.state_dict()),
            optimizer=optimizer.state_dict(), epoch=base_epoch + epoch, local_epoch=epoch,
            updates=updates, decision=dict(rule='advantage_margin_v1', margin=candidate['margin']),
            rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                     generator=generator.get_state())), str(path))
        if best is None or selection_key(candidate) > selection_key(best):
            best, best_epoch, best_path = candidate, epoch, str(path.resolve())
        print('%s epoch=%d margin=%s dev_delta=%+.6f intervene=%.3f' % (
            arm, epoch, candidate['margin'], candidate['mean_delta'], candidate['intervention_rate']), flush=True)
    if max_gradient <= 0:
        raise RuntimeError('Policy has no nonzero gradients')
    report = dict(identity=checkpoint_identity, train_audit=audit(records),
        additional_audit=audit(additional) if additional else None, development_audit=audit(development),
        history=history, selected_epoch=best_epoch, selected_development=best, checkpoint=best_path,
        initial_checkpoint=initial, max_gradient_norm=max_gradient, base_updates=base_updates,
        additional_updates=args.epochs * ((args.states_per_epoch + args.batch_size - 1) // args.batch_size),
        official_val_used=False, test_used=False,
        selection='initial-cache uniform internal-dev sequence mean; ties fewer interventions, earlier epoch')
    write_json_atomic(str(output / 'training_report.json'), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teachers', required=True)
    parser.add_argument('--on-policy-teachers')
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=('initial', 'aggregate'), default='initial')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--states-per-epoch', type=int, default=8192)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    records, dev, identity = load_cache(args.teachers)
    uniform_development = [r for r in dev if r['uniform']]
    if min(args.epochs, args.states_per_epoch, args.batch_size) < 1 or not uniform_development:
        parser.error('Positive budgets and nonempty uniform development set required')
    report_path = Path(args.output) / 'teacher_comparison.json'
    reports = json.loads(report_path.read_text()) if report_path.exists() else {}
    arms = ('uniform', 'event') if args.stage == 'initial' else ('event_continue', 'aggregate')
    if any(arm in reports for arm in arms):
        raise ValueError('Refusing to overwrite completed arm reports')
    if args.stage == 'initial':
        for arm in arms:
            reports[arm] = add_diagnostics(
                train_arm(args, arm, records, uniform_development, identity), records, dev)
    else:
        extra, extra_dev, extra_identity = load_cache(args.on_policy_teachers)
        parent = reports['event']['checkpoint']
        if (extra_identity['policy_sha256'] != file_hash(parent)
                or extra_identity['checkpoint_sha256'] != identity['checkpoint_sha256']
                or extra_identity['annotations_sha256'] != identity['annotations_sha256']
                or extra_identity['development_sequences'] != identity['development_sequences']):
            raise ValueError('On-policy teacher identity differs')
        for arm in arms:
            reports[arm] = add_diagnostics(train_arm(args, arm, records, uniform_development, identity,
                additional=extra if arm == 'aggregate' else None, initial=parent), records, dev)
            chosen = torch.load(reports[arm]['checkpoint'], map_location='cpu')
            model = SearchPolicy(recurrent=False, mlp_depth=2).eval()
            model.load_state_dict(chosen['policy'])
            reports[arm]['on_policy_development'] = gate_metrics(model,
                [r for r in extra_dev if r['uniform']], chosen['decision']['margin'])
            reports[arm]['on_policy_cache_identity'] = extra_identity
            write_json_atomic(str(Path(args.output) / arm / 'training_report.json'), reports[arm])
    write_json_atomic(str(report_path), reports)


if __name__ == '__main__':
    main()
