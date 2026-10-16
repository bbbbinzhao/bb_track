"""Train-only matched-state scale interventions with nested 8/32/64 rewards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from lib.train.data.search_policy import development_sequences
from lib.test.utils.search_region import candidate_regions
from tracking.search_conservative_coverage import (
    branch, groups_for_sequence, load_policy, next_policy_region,
)
from tracking.search_fixed8_intervention import crop_metrics, file_hash
from tracking.search_policy_experiment import (
    box_iou, build_runtime, diagnose_sequence, geometry_from_dict, load_sequences, read_rgb,
    run_sequence, sequence_metrics, valid_frames, write_json_atomic,
)
from tracking.verify_search_runtime import state_hash

PROTOCOL = 'matched_state_nested_horizon_v1'
HORIZONS = (8, 32, 64)


def sample_points(boxes, exist, predictions, uniform_points=4, event_points=2):
    groups = groups_for_sequence(boxes, exist, predictions, horizon=64,
                                uniform_points=0, event_points=event_points, minimum_frame=9)
    last = len(boxes) - 65
    if last < 9:
        return {}
    # Interior quantiles avoid spending this bounded experiment at sequence endpoints.
    uniform = np.unique(np.linspace(9, last, uniform_points + 2, dtype=int)[1:-1])
    for t in uniform:
        groups.setdefault(int(t), dict(uniform=False, event=False, failure=False, reappearance=False))
        groups[int(t)]['uniform'] = True
    return dict(sorted(groups.items()))


def prefix_rewards(frame_metrics):
    if len(frame_metrics) != 64:
        raise ValueError('All horizons require the same complete 64-frame rollout')
    result = {}
    for horizon in HORIZONS:
        values = [r['iou'] for r in frame_metrics[:horizon] if r['iou'] is not None]
        result[str(horizon)] = dict(visible_frames=len(values),
            iou=float(np.mean(values)) if values else None)
    return result


def identity_for(args, sequences, expected):
    annotations = hashlib.sha256()
    for name, _, _, _ in sequences:
        annotations.update(name.encode())
        annotations.update((Path(args.data_root) / 'train' / name / 'IR_label.json').read_bytes())
    return dict(protocol=PROTOCOL, split='train', data_root=str(Path(args.data_root).resolve()),
        checkpoint_sha256=file_hash(args.checkpoint), policy_sha256=file_hash(args.policy_checkpoint),
        annotations_sha256=annotations.hexdigest(), horizons=list(HORIZONS),
        state_distribution='full_fixed8_trajectory', minimum_frame=9,
        uniform_points=args.uniform_points, event_points=args.event_points,
        uniform_sampling='interior_linspace_quantiles', event_sampling='failure_reappearance_offsets_v1',
        action_ids=[0, 1, 2], factors=[4, 6, 8], intervention_frames=1,
        continuations=['fixed8', 'aggregate'], all_sequences=[r[0] for r in sequences],
        development_sequences=sorted(development_sequences([r[0] for r in sequences])),
        expected_sequences=[r[0] for r in expected])


@torch.no_grad()
def collect(args):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    sequences = load_sequences('train', data_root=args.data_root)
    expected = sequences[args.sequence_start:args.sequence_end or len(sequences)]
    identity = identity_for(args, sequences, expected)
    directory = Path(args.output)
    manifest = directory / 'manifest.json'
    completed = {}
    if args.resume and manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous['identity'] != identity:
            raise ValueError('Incompatible horizon diagnostic resume')
        completed = previous['sequences']
        if previous['complete']:
            print('Already complete ' + str(manifest), flush=True)
            return
    model = build_runtime(args)
    frozen = state_hash(model)
    policy = load_policy(args.policy_checkpoint, args.device, identity['checkpoint_sha256'])
    started = time.perf_counter()
    for number, (name, frames, boxes, exist) in enumerate(expected, 1):
        if name in completed:
            continue
        baseline = run_sequence(model, frames, boxes, exist, fixed_factor=8.,
            collect_snapshots=True, snapshot_ids=set(range(9, len(frames) - 64)))
        groups = sample_points(boxes, exist, baseline['boxes'], args.uniform_points, args.event_points)
        valid = valid_frames(boxes, exist)
        samples = []
        for t, flags in groups.items():
            snapshot = baseline['snapshots'][t]
            model.restore(snapshot)
            observation = np.array(model.last_diagnostic['observation'], copy=True).tolist()
            shape = read_rgb(frames[t]).shape
            regions = candidate_regions(model.state, model.motion.velocity,
                geometry_from_dict(model.last_diagnostic['crop']), shape)[:3]
            chosen_factor = next_policy_region(model, shape, policy).factor
            chosen_action = int(np.argmin([abs(r.factor - chosen_factor) for r in regions]))
            images = [read_rgb(frame) for frame in frames[t + 1:t + 65]]
            continuations = {}
            for label, continuation in [('fixed8', None), ('aggregate', policy)]:
                candidates = []
                first = None
                for action, region in enumerate(regions):
                    predictions, crops, hann = branch(model, snapshot, images, region, continuation)
                    if action == 0:
                        first = predictions.copy(), list(hann), crops
                    if label == 'fixed8' and action == 2:
                        np.testing.assert_array_equal(predictions, baseline['boxes'][t + 1:t + 65])
                        if hann != [baseline['diagnostics'][j]['use_hann'] for j in range(t + 1, t + 65)]:
                            raise RuntimeError('Fixed8 Hann replay changed')
                    stats = [crop_metrics(boxes[t + j + 1], p, valid[t + j + 1], c)
                             for j, (p, c) in enumerate(zip(predictions, crops))]
                    candidates.append(dict(action=action, factor=region.factor,
                        rewards=prefix_rewards(stats), frames=stats,
                        predictions=predictions.tolist(), hann=hann))
                repeated, repeated_crops, repeated_hann = branch(
                    model, snapshot, images, regions[0], continuation)
                np.testing.assert_array_equal(repeated, first[0])
                if repeated_hann != first[1] or repeated_crops != first[2]:
                    raise RuntimeError('Branch order changed Hann/crop geometry')
                continuations[label] = candidates
            samples.append(dict(frame=t, **flags, observation=observation,
                current_iou=box_iou(baseline['boxes'][t], boxes[t]) if valid[t] else None,
                policy_action=chosen_action, full_horizon=True, continuations=continuations))
        write_json_atomic(str(directory / 'sequences' / (name + '.json')), dict(
            protocol=PROTOCOL, name=name, frame_count=len(frames),
            baseline_metrics=sequence_metrics(baseline['boxes'], boxes, exist),
            fixed8_replayed=True, branch_order_invariant=True, samples=samples))
        completed[name] = dict(file='sequences/' + name + '.json', points=len(samples))
        write_json_atomic(str(manifest), dict(identity=identity, complete=False, sequences=completed))
        print('[%d/%d] %s points=%d elapsed=%.1fs' % (
            number, len(expected), name, len(samples), time.perf_counter() - started), flush=True)
    if state_hash(model) != frozen:
        raise RuntimeError('Frozen network parameters/buffers changed')
    write_json_atomic(str(manifest), dict(identity=identity, complete=True, sequences=completed,
        frozen_weights_and_buffers_unchanged=True, network_state_sha256=frozen,
        elapsed_seconds=time.perf_counter() - started))
    print('COMPLETE ' + str(manifest), flush=True)


@torch.no_grad()
def evaluate_development(args):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    sequences = load_sequences('train', data_root=args.data_root)
    development = development_sequences([r[0] for r in sequences])
    selected = [r for r in sequences if r[0] in development]
    expected = selected[args.sequence_start:args.sequence_end or len(selected)]
    common = identity_for(args, sequences, expected)
    identity = {key: common[key] for key in ('protocol', 'split', 'data_root',
        'checkpoint_sha256', 'policy_sha256', 'annotations_sha256', 'all_sequences',
        'development_sequences', 'expected_sequences')}
    identity.update(experiment='unchanged_aggregate_full_internal_development',
                    controller='aggregate', trajectory='full_sequence_no_gt_resets')
    directory = Path(args.output)
    manifest = directory / 'manifest.json'
    completed = {}
    if args.resume and manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous['identity'] != identity:
            raise ValueError('Incompatible internal development resume')
        completed = previous['sequences']
        if previous['complete']:
            print('Already complete ' + str(manifest), flush=True)
            return
    model = build_runtime(args)
    frozen = state_hash(model)
    policy = load_policy(args.policy_checkpoint, args.device, identity['checkpoint_sha256'])
    started = time.perf_counter()
    for number, (name, frames, boxes, exist) in enumerate(expected, 1):
        if name in completed:
            continue
        result = run_sequence(model, frames, boxes, exist, policy=policy, action_mode='scale')
        metrics = sequence_metrics(result['boxes'], boxes, exist)
        write_json_atomic(str(directory / 'sequences' / (name + '.json')), dict(
            protocol=PROTOCOL, name=name, frame_count=len(frames), metrics=metrics,
            predictions=result['boxes'].tolist(), diagnostics=diagnose_sequence(result, frames, boxes, exist)))
        completed[name] = dict(file='sequences/' + name + '.json')
        write_json_atomic(str(manifest), dict(identity=identity, complete=False, sequences=completed))
        print('[%d/%d] %s IoU=%.6f elapsed=%.1fs' % (
            number, len(expected), name, metrics['visible_iou'], time.perf_counter() - started), flush=True)
    if state_hash(model) != frozen:
        raise RuntimeError('Development evaluation modified frozen network')
    write_json_atomic(str(manifest), dict(identity=identity, complete=True, sequences=completed,
        frozen_weights_and_buffers_unchanged=True, network_state_sha256=frozen,
        elapsed_seconds=time.perf_counter() - started))
    print('COMPLETE ' + str(manifest), flush=True)


def run(args):
    diagnostic_root = Path(args.output).resolve()
    development_only = args.mode == 'run-dev'
    root = diagnostic_root / 'internal_dev' if development_only else diagnostic_root
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    sequences = load_sequences('train', data_root=args.data_root)
    registration = identity_for(args, sequences, sequences)
    registration.update(bootstrap_repetitions=10000, bootstrap_unit='sequence',
        checkpoint_path=str(Path(args.checkpoint).resolve()),
        policy_path=str(Path(args.policy_checkpoint).resolve()),
        reporting=['train180', 'dev20', 'uniform', 'event'],
        selection='none_no_model_training',
        interpretation='one-crop counterfactuals are not full-policy performance bounds')
    protocol_path = diagnostic_root / 'protocol.json'
    if development_only and not protocol_path.exists():
        raise ValueError('Internal development requires the registered horizon experiment')
    if development_only and not (diagnostic_root / 'pipeline_complete.json').exists():
        raise ValueError('Complete horizon collection and analysis before internal development')
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != registration:
        raise ValueError('Pre-registered diagnostic differs')
    if not development_only:
        write_json_atomic(str(protocol_path), registration)
    else:
        development = set(registration['development_sequences'])
        sequences = [r for r in sequences if r[0] in development]
    jobs = [(start, min(start + args.shard_size, len(sequences)))
            for start in range(0, len(sequences), args.shard_size)]
    workers = []
    started = time.perf_counter()
    try:
        while jobs or workers:
            for item in list(workers):
                process, handle, gpu, tag = item
                if process.poll() is not None:
                    handle.close()
                    workers.remove(item)
                    if process.returncode:
                        raise RuntimeError('Horizon diagnostic failed: ' + tag)
                    print('finished ' + tag, flush=True)
            for gpu in args.gpus:
                while jobs and sum(w[2] == gpu for w in workers) < args.workers_per_gpu:
                    start, end = jobs.pop(0)
                    tag = 'shard_%03d_%03d' % (start, end)
                    command = [sys.executable, '-u', __file__, 'dev' if development_only else 'collect',
                        '--output', str(root / tag),
                        '--data-root', args.data_root, '--checkpoint', args.checkpoint,
                        '--policy-checkpoint', args.policy_checkpoint, '--sequence-start', str(start),
                        '--sequence-end', str(end), '--uniform-points', str(args.uniform_points),
                        '--event-points', str(args.event_points), '--resume']
                    handle = (root / 'logs' / (tag + '.log')).open('a')
                    handle.write(json.dumps(dict(gpu=gpu, command=command)) + '\n')
                    handle.flush()
                    process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                                 OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
                    workers.append((process, handle, gpu, tag))
                    print('launch gpu=%d %s' % (gpu, tag), flush=True)
            time.sleep(5)
    finally:
        for process, handle, _, _ in workers:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            handle.close()
    from tracking.analyze_search_horizon import analyze, analyze_development
    (analyze_development if development_only else analyze)(diagnostic_root)
    if not development_only:
        from tracking.supplement_search_horizon import supplement
        supplement(diagnostic_root)
    write_json_atomic(str(root / 'pipeline_complete.json'), dict(
        complete=True, protocol=PROTOCOL, elapsed_seconds=time.perf_counter() - started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('collect', 'run', 'dev', 'run-dev'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', default=str(ROOT / 'data/antiuav410'))
    parser.add_argument('--checkpoint', default=str(ROOT / 'output/checkpoints/train/focustrack/focustrack_stage2/FocusTrack_ep0010.pth.tar'))
    parser.add_argument('--policy-checkpoint', default=str(ROOT / 'output/search_policy/conservative_coverage_seed0/aggregate/policy_seed0_ep009.pth'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--sequence-start', type=int, default=0)
    parser.add_argument('--sequence-end', type=int, default=0)
    parser.add_argument('--uniform-points', type=int, default=4)
    parser.add_argument('--event-points', type=int, default=2)
    parser.add_argument('--shard-size', type=int, default=5)
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.split = 'train'
    if args.uniform_points < 1 or args.event_points < 0:
        parser.error('Positive uniform budget and nonnegative event budget required')
    {'collect': collect, 'run': run, 'dev': evaluate_development, 'run-dev': run}[args.mode](args)


if __name__ == '__main__':
    main()
