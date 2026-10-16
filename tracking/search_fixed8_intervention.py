"""Same-state, one-crop interventions followed by fixed-factor-eight tracking."""
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

from lib.test.utils.search_region import SearchRegion, box_center, candidate_regions
from tracking.search_policy_experiment import (
    box_iou, build_runtime, failure_starts, geometry_from_dict, load_sequences,
    read_rgb, reappearance_starts, run_sequence, valid_frames, write_json_atomic,
)
from tracking.verify_search_runtime import state_hash


PROTOCOL = 'fixed8_one_crop_then_fixed8_v1'


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sample_groups(valid, ious, exist, uniform_points=32, event_limit=8):
    n = len(valid)
    uniform = set(np.linspace(1, n - 2, min(uniform_points, n - 2), dtype=int)) if n > 2 else set()
    failures = {i - 1 for i in failure_starts(valid, ious)[:event_limit]}
    reappearances = {i - 1 for i in reappearance_starts(exist, valid)[:event_limit]}
    return {int(t): {'uniform_sample': t in uniform, 'failure_sample': t in failures,
                     'reappearance_sample': t in reappearances}
            for t in sorted(uniform | failures | reappearances) if 0 < t < n - 1}


def training_groups(frame_count, points, horizon):
    last = frame_count - horizon - 1
    if last < 1:
        return {}
    return {int(t): dict(uniform_sample=True, failure_sample=False, reappearance_sample=False)
            for t in np.unique(np.linspace(1, last, min(points, last), dtype=int))}


def fixed_region(box, image_shape):
    height, width = image_shape[:2]
    center = np.clip(box_center(box), [0, 0], [width - 1, height - 1])
    return SearchRegion(tuple(center), 8.)


def crop_metrics(box, prediction, valid, crop):
    if not valid:
        return dict(iou=None, coverage=None, center_inside=None, target_pixels=None)
    origin, side = np.asarray(crop['origin_xy']), crop['side']
    intersection = np.maximum(0, np.minimum(box[:2] + box[2:], origin + side)
                              - np.maximum(box[:2], origin)).prod()
    center = box_center(box)
    return dict(iou=box_iou(prediction, box), coverage=float(intersection / box[2:].prod()),
                center_inside=bool(np.all((center >= origin) & (center < origin + side))),
                target_pixels=float(np.sqrt(box[2:].prod()) * crop['output_size'] / side))


@torch.no_grad()
def rollout_candidate(model, snapshot, images, region):
    model.restore(snapshot)
    predictions, crops, hann = [], [], []
    next_region = region
    for image in images:
        model.track(image, search_region=next_region)
        predictions.append(list(model.state))
        crops.append(model.last_diagnostic['crop'])
        hann.append(model.last_diagnostic['use_hann'])
        next_region = fixed_region(model.state, image.shape)
    return np.asarray(predictions), crops, hann


def collect(args):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    records = load_sequences(args.split, data_root=args.data_root)
    expected = records[args.sequence_start:args.sequence_end or len(records)]
    label_hash = hashlib.sha256()
    for name, _, _, _ in records:
        label_hash.update(name.encode())
        label_hash.update((Path(args.data_root) / args.split / name / 'IR_label.json').read_bytes())
    identity = dict(protocol=PROTOCOL, split=args.split,
                    data_root=str(Path(args.data_root).resolve()),
                    checkpoint_sha256=file_hash(args.checkpoint),
                    annotations_sha256=label_hash.hexdigest(), horizon=args.horizon,
                    uniform_points=args.uniform_points, event_limit=args.event_limit,
                    baseline_factor=8., intervention_frames=1, continuation_factor=8.,
                    all_sequences=[record[0] for record in records],
                    expected_sequences=[record[0] for record in expected],
                    reference_sha256=file_hash(args.reference) if args.reference else None)
    if args.split == 'train':
        if args.event_limit:
            raise ValueError('Train teachers require uniform sampling with --event-limit 0')
        identity['training_sampling'] = 'uniform_full_horizon_v1'
    path = Path(args.output) / 'manifest.json'
    completed = {}
    if args.resume and path.exists():
        previous = json.load(path.open())
        if previous['identity'] != identity:
            raise ValueError('Incompatible fixed-eight intervention resume')
        completed = previous['sequences']
        if previous['complete']:
            print('Already complete: ' + str(path), flush=True)
            return
    reference = json.load(open(args.reference)) if args.reference else None
    if reference is not None:
        if (reference['baseline_sha256'] != identity['checkpoint_sha256']
                or reference['split'] != args.split or reference.get('fixed_factor') != 8.
                or reference['protocol'] != 'search_evaluation_v2'
                or Path(reference['data_root']).resolve() != Path(args.data_root).resolve()):
            raise ValueError('Reference must be the matching fixed-eight evaluation')
    model = build_runtime(args)
    frozen_before = state_hash(model)
    started = time.perf_counter()
    for index, (name, frames, boxes, exist) in enumerate(expected, 1):
        if name in completed:
            continue
        valid = valid_frames(boxes, exist)
        if reference is None and args.split == 'train':
            groups = training_groups(len(frames), args.uniform_points, args.horizon)
            baseline = run_sequence(model, frames, boxes, exist, fixed_factor=8.,
                                    collect_snapshots=True, snapshot_ids=set(groups))
            baseline_boxes = baseline['boxes']
        elif reference is None:
            baseline = run_sequence(model, frames, boxes, exist, fixed_factor=8.)
            baseline_boxes = baseline['boxes']
        else:
            baseline_boxes = np.asarray(reference['sequences'][name]['predictions'])
        if baseline_boxes.shape != boxes.shape or not np.isfinite(baseline_boxes).all():
            raise ValueError('Invalid baseline predictions for ' + name)
        ious = np.asarray([box_iou(p, g) if v else 0.
                           for p, g, v in zip(baseline_boxes, boxes, valid)])
        if args.split != 'train' or reference is not None:
            groups = sample_groups(valid, ious, exist, args.uniform_points, args.event_limit)
            baseline = run_sequence(model, frames, boxes, exist, fixed_factor=8.,
                                    collect_snapshots=True, snapshot_ids=set(groups))
        np.testing.assert_allclose(baseline['boxes'], baseline_boxes, rtol=0, atol=1e-4,
                                   err_msg='Fixed-eight baseline replay: ' + name)
        samples, max_error = [], 0.
        for sample_index, (t, flags) in enumerate(groups.items()):
            end = min(t + args.horizon + 1, len(frames))
            images = [read_rgb(frame) for frame in frames[t + 1:end]]
            current_shape = read_rgb(frames[t]).shape
            diag = baseline['diagnostics'][t]
            regions = candidate_regions(baseline_boxes[t], diag['velocity'],
                                        geometry_from_dict(diag['crop']), current_shape)
            snapshot = baseline['snapshots'][t]
            candidate_results = []
            reference_hann = [baseline['diagnostics'][j]['use_hann'] for j in range(t + 1, end)]
            for action, region in enumerate(regions):
                predictions, crops, hann = rollout_candidate(model, snapshot, images, region)
                if action == 2:
                    error = float(np.abs(predictions - baseline_boxes[t + 1:end]).max())
                    max_error = max(max_error, error)
                    np.testing.assert_allclose(predictions, baseline_boxes[t + 1:end], atol=1e-4, rtol=0)
                    if hann != reference_hann:
                        raise RuntimeError('Baseline candidate changed Hann trajectory')
                    baseline_candidate = predictions.copy()
                stats = [crop_metrics(boxes[j], predictions[j - t - 1], valid[j], crops[j - t - 1])
                         for j in range(t + 1, end)]
                values = [value['iou'] for value in stats if value['iou'] is not None]
                candidate_results.append(dict(action=action, region=region.as_dict(),
                    predictions=predictions.tolist(), first_crop=crops[0], hann=hann,
                    next_iou=stats[0]['iou'], future_iou=float(np.mean(values)) if values else None,
                    frames=stats))
            # Repeating A after all other branches catches snapshot/state contamination on CUDA.
            if sample_index == 0:
                repeated, _, repeated_hann = rollout_candidate(model, snapshot, images, regions[2])
                np.testing.assert_array_equal(repeated, baseline_candidate)
                if repeated_hann != reference_hann:
                    raise RuntimeError('Branch-order Hann contamination')
            base = candidate_results[2]
            samples.append(dict(frame=t, **flags, full_horizon=end - t - 1 == args.horizon,
                horizon_frames=end - t - 1, valid_reward_frames=int(valid[t + 1:end].sum()),
                observation=np.asarray(diag['observation']).tolist(),
                baseline_next_iou=base['next_iou'], baseline_future_iou=base['future_iou'],
                candidates=candidate_results))
        seq_path = Path(args.output) / 'sequences' / (name + '.json')
        write_json_atomic(str(seq_path), dict(protocol=PROTOCOL, name=name, frame_count=len(frames),
            baseline_reproduced=True, baseline_action_max_box_error=max_error,
            branch_order_invariant=True, samples=samples))
        completed[name] = dict(file=str(seq_path.relative_to(Path(args.output))), points=len(samples))
        write_json_atomic(str(path), dict(identity=identity, complete=False, sequences=completed))
        print('[%d/%d] %s points=%d baseline_error=%.3g elapsed=%.1fs' % (
            index, len(expected), name, len(samples), max_error, time.perf_counter() - started), flush=True)
    if state_hash(model) != frozen_before:
        raise RuntimeError('Frozen weights or buffers changed')
    write_json_atomic(str(path), dict(identity=identity, complete=True, sequences=completed,
        frozen_weights_and_buffers_unchanged=True, network_state_sha256=frozen_before,
        elapsed_seconds=time.perf_counter() - started))
    print('COMPLETE ' + str(path), flush=True)


def run(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    count = len(load_sequences(args.split, data_root=args.data_root))
    jobs = []
    for start in range(0, count, args.shard_size):
        end = min(start + args.shard_size, count)
        directory = root / ('shard_%03d_%03d' % (start, end))
        manifest = directory / 'manifest.json'
        if not manifest.exists() or not json.load(manifest.open())['complete']:
            jobs.append((start, end, directory))
    processes = []
    try:
        while jobs or processes:
            for item in list(processes):
                process, log, gpu, tag = item
                if process.poll() is not None:
                    log.close()
                    processes.remove(item)
                    if process.returncode:
                        raise RuntimeError('Intervention worker failed: ' + tag)
                    print('finished ' + tag, flush=True)
            for gpu in args.gpus:
                busy = sum(item[2] == gpu for item in processes)
                while jobs and busy < args.workers_per_gpu:
                    start, end, directory = jobs.pop(0)
                    tag = directory.name
                    command = [sys.executable, '-u', __file__, 'collect', '--split', args.split,
                        '--data-root', args.data_root, '--checkpoint', args.checkpoint,
                        '--output', str(directory), '--sequence-start', str(start),
                        '--sequence-end', str(end), '--uniform-points', str(args.uniform_points),
                        '--event-limit', str(args.event_limit), '--horizon', str(args.horizon), '--resume']
                    if args.reference:
                        command.extend(['--reference', args.reference])
                    log = (root / 'logs' / (tag + '.log')).open('a')
                    log.write(json.dumps(dict(gpu=gpu, command=command)) + '\n')
                    log.flush()
                    process = subprocess.Popen(command, cwd=ROOT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                                 MKL_NUM_THREADS='1'), stdout=log, stderr=subprocess.STDOUT)
                    processes.append((process, log, gpu, tag))
                    print('launch gpu=%s %s' % (gpu, tag), flush=True)
                    busy += 1
            time.sleep(5)
    finally:
        for process, log, _, _ in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            log.close()
    print('ALL SHARDS COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('collect', 'run'))
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--data-root', default=str(ROOT / 'data/antiuav410'))
    parser.add_argument('--checkpoint', default=str(ROOT / 'output/checkpoints/train/focustrack/focustrack_stage2/FocusTrack_ep0010.pth.tar'))
    parser.add_argument('--reference')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--sequence-start', type=int, default=0)
    parser.add_argument('--sequence-end', type=int, default=0)
    parser.add_argument('--uniform-points', type=int, default=32)
    parser.add_argument('--event-limit', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--shard-size', type=int, default=6)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.uniform_points, args.horizon, args.shard_size, args.workers_per_gpu) < 1 or args.event_limit < 0:
        parser.error('Invalid sampling/worker counts')
    if args.split == 'val' and not args.reference:
        parser.error('Validation requires the existing fixed-eight reference')
    (collect if args.mode == 'collect' else run)(args)


if __name__ == '__main__':
    main()
