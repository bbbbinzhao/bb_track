"""Matched single-step versus eight-step MLP teachers on train-only fixed-eight states."""
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

from lib.models.search_policy import SearchPolicy
from lib.train.data.search_policy import development_sequences
from tracking.search_fixed8_intervention import PROTOCOL, file_hash
from tracking.search_policy_experiment import bootstrap_mean_ci, write_json_atomic


def load_states(root):
    identity, seen, records, manifests = None, set(), [], []
    for path in sorted(root.glob('shard_*/manifest.json')):
        payload = json.loads(path.read_text())
        current = {key: value for key, value in payload['identity'].items() if key != 'expected_sequences'}
        if current['split'] != 'train' or current['protocol'] != PROTOCOL:
            raise ValueError('Only train fixed-eight intervention teachers may train a policy')
        if (not payload['complete'] or not payload['frozen_weights_and_buffers_unchanged']
                or current.get('training_sampling') != 'uniform_full_horizon_v1'):
            raise ValueError('Expected complete uniformly sampled training teachers')
        if identity is not None and identity != current:
            raise ValueError('Teacher shard identities differ')
        identity = current
        if set(payload['sequences']) != set(payload['identity']['expected_sequences']):
            raise ValueError('Incomplete teacher shard')
        for name, item in payload['sequences'].items():
            if name in seen:
                raise ValueError('Duplicate training sequence ' + name)
            seen.add(name)
            sequence = json.loads((path.parent / item['file']).read_text())
            if not sequence['branch_order_invariant'] or sequence['baseline_action_max_box_error'] > 1e-4:
                raise ValueError('Invalid branch verification')
            for sample in sequence['samples']:
                if not sample['full_horizon'] or sample['baseline_next_iou'] is None:
                    continue
                candidates = sample['candidates']
                if [candidate['action'] for candidate in candidates] != list(range(9)):
                    raise ValueError('Incomplete action set')
                record = dict(sequence=name, frame=sample['frame'], observation=sample['observation'],
                    one=np.asarray([candidate['next_iou'] for candidate in candidates], dtype=np.float32),
                    eight=np.asarray([candidate['future_iou'] for candidate in candidates], dtype=np.float32))
                if not all(np.isfinite(record[key]).all() for key in ('observation', 'one', 'eight')):
                    raise ValueError('Nonfinite teacher or observation')
                records.append(record)
        manifests.append(dict(path=str(path.resolve()), sha256=file_hash(path)))
    if identity is None or seen != set(identity['all_sequences']) or len(seen) != 200:
        raise ValueError('Require exact coverage of all 200 training sequences')
    development = development_sequences(sorted(seen))
    train = [record for record in records if record['sequence'] not in development]
    dev = [record for record in records if record['sequence'] in development]
    if not train or not dev:
        raise ValueError('Empty training/development observations')
    return train, dev, dict(identity, development_sequences=sorted(development), manifests=manifests)


def tensors(records):
    return {key: torch.tensor(np.asarray([record[key] for record in records]), dtype=torch.float32)
            for key in ('observation', 'one', 'eight')}


def per_sequence(records, values):
    grouped = {}
    for record, value in zip(records, values):
        grouped.setdefault(record['sequence'], []).append(value)
    return {name: float(np.mean(value)) for name, value in grouped.items()}


@torch.no_grad()
def metrics(policy, data, records, objective, ids, prior, train_constant, temperature=.1):
    logits, _ = policy(data['observation'])
    logits = logits[:, ids]
    probabilities = torch.softmax(logits, -1)
    actions = logits.argmax(-1)
    scores = data[objective][:, ids]
    teachers = torch.softmax(scores / temperature, -1)
    loss = float(-(teachers * torch.log_softmax(logits, -1)).sum(-1).mean())
    entropy = float(-(teachers * teachers.clamp_min(1e-10).log()).sum(-1).mean())
    selected_actions = ids[actions]
    report = dict(loss=loss, teacher_entropy=entropy, kl_to_teacher=loss - entropy,
        train_prior_cross_entropy=float(-(teachers * prior.clamp_min(1e-10).log()).sum(-1).mean()),
        action_counts={str(int(a)): int((selected_actions == a).sum()) for a in ids},
        mean_max_probability=float(probabilities.max(-1).values.mean()), rewards={})
    for reward in ('one', 'eight'):
        values = data[reward][:, ids]
        chosen = values.gather(1, actions[:, None]).squeeze(1)
        deltas = per_sequence(records, (chosen - data[reward][:, 2]).tolist())
        constants = {int(action): float(np.mean(list(per_sequence(records,
            data[reward][:, action].tolist()).values()))) for action in ids}
        fixed = int(train_constant[reward])
        fixed_deltas = per_sequence(records, (chosen - data[reward][:, fixed]).tolist())
        chosen_mean = float(np.mean(list(per_sequence(records, chosen.tolist()).values())))
        report['rewards'][reward] = dict(selected_iou=chosen_mean,
            oracle_iou=float(np.mean(list(per_sequence(records, values.max(-1).values.tolist()).values()))),
            fixed_action_ious=constants, delta_to_best_dev_constant=chosen_mean - max(constants.values()),
            train_selected_constant=fixed, delta_to_train_constant=float(np.mean(list(fixed_deltas.values()))),
            delta_to_fixed8=float(np.mean(list(deltas.values()))), per_sequence_delta_to_fixed8=deltas,
            per_sequence_delta_to_train_constant=fixed_deltas)
    return report


def train_one(args, records, dev_records, identity, objective):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    policy = SearchPolicy(recurrent=False, mlp_depth=2)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-4)
    ids = torch.tensor([0, 1, 2] if args.action_mode == 'scale' else list(range(9)))
    train_data, dev_data = tensors(records), tensors(dev_records)
    temperature = getattr(args, 'teacher_temperature', .1)
    calibration = None
    if getattr(args, 'match_one_step_entropy', False):
        if objective != 'eight':
            raise ValueError('Entropy matching applies only to eight-frame teachers')
        temperature, calibration = match_entropy(train_data['one'][:, ids], train_data['eight'][:, ids])
    teacher = torch.softmax(train_data[objective][:, ids] / temperature, -1)
    prior = teacher.mean(0)
    constants = {}
    for reward in ('one', 'eight'):
        utilities = [float(np.mean(list(per_sequence(records, train_data[reward][:, action].tolist()).values())))
                     for action in ids]
        constants[reward] = int(ids[int(np.argmax(utilities))])
    output = Path(args.output) / objective
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_identity = dict(protocol='fixed8_matched_state_teacher_v1', recurrent=False, mlp_depth=2,
        action_mode=args.action_mode, seed=args.seed, backbone_sha256=identity['checkpoint_sha256'],
        cache_identity=identity, objective=objective, teacher_temperature=temperature,
        entropy_calibration=calibration,
        states_per_epoch=args.states_per_epoch, batch_size=args.batch_size)
    initial = metrics(policy, dev_data, dev_records, objective, ids, prior, constants, temperature)
    best_utility, best_epoch, best_checkpoint = -float('inf'), None, None
    history, max_gradient = [], 0.
    for epoch in range(1, args.epochs + 1):
        policy.train()
        sampled = torch.randint(len(records), (args.states_per_epoch,), generator=generator)
        losses = []
        for start in range(0, len(sampled), args.batch_size):
            selected = sampled[start:start + args.batch_size]
            logits, _ = policy(train_data['observation'][selected])
            loss = -(teacher[selected] * torch.log_softmax(logits[:, ids], -1)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            if not torch.isfinite(loss) or not torch.isfinite(gradient):
                raise RuntimeError('Nonfinite teacher loss/gradient')
            max_gradient = max(max_gradient, float(gradient))
            optimizer.step()
            losses.append(float(loss.detach()))
        policy.eval()
        dev = metrics(policy, dev_data, dev_records, objective, ids, prior, constants, temperature)
        entry = dict(epoch=epoch, updates=epoch * ((args.states_per_epoch + args.batch_size - 1) // args.batch_size),
                     train_loss=float(np.mean(losses)), development=dev)
        history.append(entry)
        checkpoint = output / ('policy_supervised_seed%d_ep%03d.pth' % (args.seed, epoch))
        payload = dict(checkpoint_identity, policy=copy.deepcopy(policy.state_dict()),
            optimizer=optimizer.state_dict(), epoch=epoch, updates=entry['updates'], history=history,
            rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                     generator=generator.get_state()))
        torch.save(payload, str(checkpoint) + '.tmp')
        Path(str(checkpoint) + '.tmp').replace(checkpoint)
        utility = dev['rewards'][objective]['selected_iou']
        if utility > best_utility:
            best_utility, best_epoch, best_checkpoint = utility, epoch, str(checkpoint.resolve())
        print('%s epoch=%d dev_%s_iou=%.6f dev_eight_delta8=%+.6f' % (
            objective, epoch, objective, utility, dev['rewards']['eight']['delta_to_fixed8']), flush=True)
    if max_gradient <= 0:
        raise RuntimeError('No policy gradient')
    chosen = history[best_epoch - 1]['development']
    for reward in ('one', 'eight'):
        value = chosen['rewards'][reward]
        value['ci_to_fixed8'] = bootstrap_mean_ci(list(value['per_sequence_delta_to_fixed8'].values()))
        value['ci_to_train_constant'] = bootstrap_mean_ci(list(value['per_sequence_delta_to_train_constant'].values()))
    report = dict(identity=checkpoint_identity, train_states=len(records), development_states=len(dev_records),
        train_sequences=len({r['sequence'] for r in records}), development_sequences=len({r['sequence'] for r in dev_records}),
        initial_development=initial, history=history, selected_epoch=best_epoch, checkpoint=best_checkpoint,
        selected_development=chosen, max_gradient_norm=max_gradient,
        selection='maximum train-internal development objective IoU; earliest tie',
        official_val_used=False, test_used=False,
        scope='matched single-state MLP screening; not temporal or SLT training')
    write_json_atomic(str(output / 'training_report.json'), report)
    return report


def match_entropy(one_scores, eight_scores):
    def entropy(scores, temperature):
        probabilities = torch.softmax(scores / temperature, -1)
        return float(-(probabilities * probabilities.clamp_min(1e-10).log()).sum(-1).mean())

    target = entropy(one_scores, .1)
    lower, upper = 1e-5, 10.
    if not entropy(eight_scores, lower) <= target <= entropy(eight_scores, upper):
        raise ValueError('Cannot match single-step teacher entropy on training states')
    for _ in range(60):
        middle = (lower + upper) / 2
        if entropy(eight_scores, middle) < target:
            lower = middle
        else:
            upper = middle
    temperature = (lower + upper) / 2
    return temperature, dict(source='train states only', one_temperature=.1,
        target_train_entropy=target, matched_train_entropy=entropy(eight_scores, temperature))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teachers', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--action-mode', choices=('scale', 'joint'), default='scale')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--states-per-epoch', type=int, default=8192)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--objectives', nargs='+', choices=('one', 'eight'), default=['one', 'eight'])
    parser.add_argument('--teacher-temperature', type=float, default=.1)
    parser.add_argument('--match-one-step-entropy', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.states_per_epoch, args.batch_size) < 1 or args.teacher_temperature <= 0:
        parser.error('Positive training budgets required')
    if args.match_one_step_entropy and args.objectives != ['eight']:
        parser.error('Use --objectives eight with entropy matching')
    torch.set_num_threads(1)
    records, dev, identity = load_states(Path(args.teachers))
    reports = {objective: train_one(args, records, dev, identity, objective) for objective in args.objectives}
    write_json_atomic(str(Path(args.output) / 'teacher_comparison.json'), reports)


if __name__ == '__main__':
    main()
