"""Train-only mixed-crop and current-policy teachers for a bounded feasibility test."""
import argparse
import copy
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

from lib.models.search_policy import SearchPolicy, select_search_action
from lib.test.utils.search_region import SearchRegion, box_center, candidate_regions
from tracking.search_fixed8_intervention import crop_metrics, file_hash
from tracking.search_policy_experiment import (
    box_iou, build_runtime, failure_starts, geometry_from_dict, load_sequences,
    read_rgb, reappearance_starts, run_sequence, sequence_metrics, valid_frames, write_json_atomic,
)
from tracking.verify_search_runtime import state_hash

PROTOCOL = 'conservative_scale_coverage_v1'


def groups_for_sequence(boxes, exist, predictions, horizon=8, uniform_points=8, event_points=8,
                        minimum_frame=9):
    last = len(boxes) - horizon - 1
    if last < minimum_frame:
        return {}
    uniform = set(np.unique(np.linspace(minimum_frame, last, uniform_points, dtype=int)))
    valid = valid_frames(boxes, exist)
    ious = np.array([box_iou(p, b) if v else 0. for p, b, v in zip(predictions, boxes, valid)])
    failure = {t + offset for t in failure_starts(valid, ious) for offset in (-4, -2, -1, 1)}
    reappear = {t + offset for t in reappearance_starts(exist, valid) for offset in (-2, -1, 1)}
    events = sorted(t for t in failure | reappear if minimum_frame <= t <= last)
    if len(events) > event_points:
        events = [events[i] for i in np.unique(np.linspace(0, len(events) - 1, event_points, dtype=int))]
    events = set(events)
    return {int(t): dict(uniform=t in uniform, event=t in events,
                         failure=t in failure, reappearance=t in reappear)
            for t in sorted(uniform | events)}


def scale_region(box, image_shape, factor):
    height, width = image_shape[:2]
    center = np.clip(box_center(box), [0, 0], [width - 1, height - 1])
    return SearchRegion(tuple(center), float(factor))


def load_policy(path, device, backbone_sha256):
    if not path:
        return None
    payload = torch.load(path, map_location='cpu')
    if (payload['backbone_sha256'] != backbone_sha256 or payload['recurrent']
            or payload['action_mode'] != 'scale' or not payload.get('decision')):
        raise ValueError('Expected matching conservative, memoryless scale controller')
    policy = SearchPolicy(recurrent=False, mlp_depth=payload['mlp_depth']).to(device).eval()
    policy.load_state_dict(payload['policy'])
    policy.decision = payload['decision']
    return policy


def next_policy_region(model, image_shape, policy):
    if policy is None:
        return scale_region(model.state, image_shape, 8.)
    observation = torch.as_tensor(model.last_diagnostic['observation'], dtype=torch.float32,
                                  device=next(policy.parameters()).device).unsqueeze(0)
    logits, _ = policy(observation)
    action = int(select_search_action(logits, torch.tensor([0, 1, 2], device=logits.device),
                                     policy.decision).item())
    return candidate_regions(model.state, model.motion.velocity,
        geometry_from_dict(model.last_diagnostic['crop']), image_shape)[action]


@torch.no_grad()
def branch(model, snapshot, images, first_region, policy=None):
    model.restore(snapshot)
    predictions, crops, hann = [], [], []
    region = first_region
    for image in images:
        model.track(image, search_region=region)
        predictions.append(list(model.state))
        crops.append(copy.deepcopy(model.last_diagnostic['crop']))
        hann.append(model.last_diagnostic['use_hann'])
        region = next_policy_region(model, image.shape, policy)
    return np.array(predictions), crops, hann


@torch.no_grad()
def collect(args):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    sequences = load_sequences('train', data_root=args.data_root)
    expected = sequences[args.sequence_start:args.sequence_end or len(sequences)]
    labels = hashlib.sha256()
    for name, _, _, _ in sequences:
        labels.update(name.encode())
        labels.update((Path(args.data_root) / 'train' / name / 'IR_label.json').read_bytes())
    identity = dict(protocol=PROTOCOL, split='train', data_root=str(Path(args.data_root).resolve()),
        checkpoint_sha256=file_hash(args.checkpoint), annotations_sha256=labels.hexdigest(),
        policy_sha256=file_hash(args.policy_checkpoint) if args.policy_checkpoint else None,
        collection='on_policy' if args.policy_checkpoint else 'mixed_crop_warmup',
        horizon=args.horizon, warmup=args.warmup, uniform_points=args.uniform_points,
        event_points=args.event_points, action_ids=[0, 1, 2],
        continuation='current_policy' if args.policy_checkpoint else 'fixed8',
        all_sequences=[r[0] for r in sequences], expected_sequences=[r[0] for r in expected])
    directory = Path(args.output)
    manifest = directory / 'manifest.json'
    completed = {}
    if args.resume and manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous['identity'] != identity:
            raise ValueError('Incompatible conservative coverage resume')
        if previous['complete']:
            print('Already complete ' + str(manifest), flush=True)
            return
        completed = previous['sequences']
    model = build_runtime(args)
    frozen = state_hash(model)
    policy = load_policy(args.policy_checkpoint, args.device, identity['checkpoint_sha256'])
    started = time.perf_counter()
    global_ids = {r[0]: i for i, r in enumerate(sequences)}
    for number, (name, frames, boxes, exist) in enumerate(expected, 1):
        if name in completed:
            continue
        baseline = run_sequence(model, frames, boxes, exist, policy=policy, action_mode='scale',
            fixed_factor=8. if policy is None else None, collect_snapshots=True,
            snapshot_ids=set(range(1, len(frames))))
        groups = groups_for_sequence(boxes, exist, baseline['boxes'], args.horizon,
            args.uniform_points, args.event_points, args.warmup + 1)
        valid = valid_frames(boxes, exist)
        samples = []
        for point_index, (t, flags) in enumerate(groups.items()):
            if policy is None:
                factor = (4., 6., 8.)[(point_index + global_ids[name]) % 3]
                model.restore(baseline['snapshots'][t - args.warmup])
                for j in range(t - args.warmup + 1, t + 1):
                    image = read_rgb(frames[j])
                    model.track(image, search_region=scale_region(model.state, image.shape, factor))
                snapshot = model.snapshot()
            else:
                factor = None
                snapshot = baseline['snapshots'][t]
                model.restore(snapshot)
            observation = np.array(model.last_diagnostic['observation'], copy=True).tolist()
            current_shape = read_rgb(frames[t]).shape
            regions = candidate_regions(model.state, model.motion.velocity,
                geometry_from_dict(model.last_diagnostic['crop']), current_shape)[:3]
            default_region = next_policy_region(model, current_shape, policy)
            default_action = int(np.argmin([abs(r.factor - default_region.factor) for r in regions]))
            images = [read_rgb(frame) for frame in frames[t + 1:t + args.horizon + 1]]
            candidates = []
            first_result = None
            for action, region in enumerate(regions):
                predictions, crops, hann = branch(model, snapshot, images, region, policy)
                if action == 0:
                    first_result = predictions.copy(), list(hann)
                if policy is not None and action == default_action:
                    np.testing.assert_array_equal(predictions, baseline['boxes'][t + 1:t + args.horizon + 1])
                    if hann != [baseline['diagnostics'][j]['use_hann'] for j in range(t + 1, t + args.horizon + 1)]:
                        raise RuntimeError('Current policy continuation changed Hann')
                stats = [crop_metrics(boxes[t + j + 1], p, valid[t + j + 1], c)
                         for j, (p, c) in enumerate(zip(predictions, crops))]
                rewards = [r['iou'] for r in stats if r['iou'] is not None]
                candidates.append(dict(action=action, next_iou=stats[0]['iou'],
                    future_iou=float(np.mean(rewards)) if rewards else None, frames=stats))
            repeated, _, repeated_hann = branch(model, snapshot, images, regions[0], policy)
            np.testing.assert_array_equal(repeated, first_result[0])
            if repeated_hann != first_result[1]:
                raise RuntimeError('Branch-order Hann contamination')
            samples.append(dict(frame=t, **flags, observation=observation,
                warmup_factor=factor, crop_factor=float(observation[29] * 8),
                full_horizon=True, default_action=default_action, candidates=candidates))
        path = directory / 'sequences' / (name + '.json')
        write_json_atomic(str(path), dict(protocol=PROTOCOL, name=name, frame_count=len(frames),
            branch_order_invariant=True, policy_baseline_reproduced=policy is not None,
            baseline_metrics=sequence_metrics(baseline['boxes'], boxes, exist), samples=samples))
        completed[name] = dict(file=str(path.relative_to(directory)), points=len(samples))
        write_json_atomic(str(manifest), dict(identity=identity, complete=False, sequences=completed))
        print('[%d/%d] %s points=%d elapsed=%.1fs' % (
            number, len(expected), name, len(samples), time.perf_counter() - started), flush=True)
    if state_hash(model) != frozen:
        raise RuntimeError('Frozen weights or buffers changed')
    write_json_atomic(str(manifest), dict(identity=identity, complete=True, sequences=completed,
        frozen_weights_and_buffers_unchanged=True, network_state_sha256=frozen,
        elapsed_seconds=time.perf_counter() - started))
    print('COMPLETE ' + str(manifest), flush=True)


def run(args):
    root = Path(args.output).resolve()
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    count = len(load_sequences('train', data_root=args.data_root))
    jobs = []
    for start in range(0, count, args.shard_size):
        end = min(start + args.shard_size, count)
        directory = root / ('shard_%03d_%03d' % (start, end))
        path = directory / 'manifest.json'
        if not path.exists() or not json.loads(path.read_text()).get('complete'):
            jobs.append((start, end, directory))
    workers = []
    try:
        while jobs or workers:
            for item in list(workers):
                process, handle, gpu, tag = item
                if process.poll() is not None:
                    handle.close()
                    workers.remove(item)
                    if process.returncode:
                        raise RuntimeError('Coverage worker failed: ' + tag)
                    print('finished ' + tag, flush=True)
            for gpu in args.gpus:
                while jobs and sum(w[2] == gpu for w in workers) < args.workers_per_gpu:
                    start, end, directory = jobs.pop(0)
                    command = [sys.executable, '-u', __file__, 'collect', '--output', str(directory),
                        '--data-root', args.data_root, '--checkpoint', args.checkpoint,
                        '--sequence-start', str(start), '--sequence-end', str(end),
                        '--uniform-points', str(args.uniform_points), '--event-points', str(args.event_points),
                        '--horizon', str(args.horizon), '--warmup', str(args.warmup), '--resume']
                    if args.policy_checkpoint:
                        command += ['--policy-checkpoint', args.policy_checkpoint]
                    handle = (root / 'logs' / (directory.name + '.log')).open('a')
                    handle.write(json.dumps(dict(gpu=gpu, command=command)) + '\n')
                    handle.flush()
                    process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
                    workers.append((process, handle, gpu, directory.name))
                    print('launch gpu=%d %s' % (gpu, directory.name), flush=True)
            time.sleep(5)
    finally:
        for process, handle, _, _ in workers:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('collect', 'run'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', default=str(ROOT / 'data/antiuav410'))
    parser.add_argument('--checkpoint', default=str(ROOT / 'output/checkpoints/train/focustrack/focustrack_stage2/FocusTrack_ep0010.pth.tar'))
    parser.add_argument('--policy-checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--sequence-start', type=int, default=0)
    parser.add_argument('--sequence-end', type=int, default=0)
    parser.add_argument('--uniform-points', type=int, default=8)
    parser.add_argument('--event-points', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--shard-size', type=int, default=5)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.split = 'train'
    if min(args.uniform_points, args.event_points, args.horizon, args.warmup) < 1:
        parser.error('Positive collection budgets required')
    (run if args.mode == 'run' else collect)(args)


if __name__ == '__main__':
    main()
