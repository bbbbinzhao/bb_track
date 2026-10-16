"""Continuous train-only teachers and supervised search-policy feasibility runs."""
import argparse
import hashlib
import json
import os
import random
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
import torch

from lib.train.data.search_policy import CACHE_PROTOCOL, WARMUP, SUPERVISION, WINDOW, window_starts
from tracking.search_policy_experiment import (
    box_iou, build_runtime, candidate_regions, geometry_from_dict, load_sequences,
    read_rgb, run_sequence, valid_frames, write_json_atomic,
)
from tracking.verify_search_runtime import state_hash


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('cache', 'train'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', default=os.path.join(ROOT, 'data', 'antiuav410'))
    parser.add_argument('--checkpoint', default=os.path.join(ROOT,
        'output/checkpoints/train/focustrack/focustrack_stage2/FocusTrack_ep0010.pth.tar'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--sequence-start', type=int, default=0)
    parser.add_argument('--sequence-end', type=int, default=0)
    parser.add_argument('--windows-per-sequence', type=int, default=8)
    parser.add_argument('--teacher-temperature', type=float, default=0.1)
    parser.add_argument('--input-manifests', nargs='+')
    parser.add_argument('--policy-type', choices=('mlp', 'gru'), default='mlp')
    parser.add_argument('--action-mode', choices=('scale', 'joint', 'center'), default='joint')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--episodes-per-epoch', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume-checkpoint')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def save_torch_atomic(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(value, path + '.tmp')
    os.replace(path + '.tmp', path)


def collect_cache(args):
    if args.windows_per_sequence < 1 or args.teacher_temperature <= 0:
        raise ValueError('Invalid teacher window/temperature setting')
    args.split = 'train'
    all_records = load_sequences('train', data_root=args.data_root)
    digest = hashlib.sha256()
    for name, _, _, _ in all_records:
        digest.update(name.encode())
        with open(os.path.join(args.data_root, 'train', name, 'IR_label.json'), 'rb') as handle:
            digest.update(handle.read())
    identity = {
        'protocol': CACHE_PROTOCOL, 'split': 'train', 'data_root': os.path.realpath(args.data_root),
        'backbone_sha256': hashlib.sha256(open(args.checkpoint, 'rb').read()).hexdigest(),
        'annotations_sha256': digest.hexdigest(), 'windows_per_sequence': args.windows_per_sequence,
        'warmup': WARMUP, 'supervision': SUPERVISION, 'teacher_temperature': args.teacher_temperature,
        'sequence_start': args.sequence_start, 'sequence_end': args.sequence_end,
        'all_training_sequences': [record[0] for record in all_records],
    }
    records = all_records[args.sequence_start:args.sequence_end or len(all_records)]
    manifest_path = os.path.join(args.output, 'manifest.json')
    completed = {}
    if args.resume and os.path.isfile(manifest_path):
        with open(manifest_path) as handle:
            previous = json.load(handle)
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError('Cannot resume incompatible teacher cache')
        completed = previous['sequences']
    model = build_runtime(args)
    before = state_hash(model)
    started = time.perf_counter()
    for name, frames, boxes, exist in records:
        if name in completed:
            continue
        starts = window_starts(len(frames), args.windows_per_sequence)
        ids = {t for start in starts for t in range(start + WARMUP, start + WINDOW)}
        baseline = run_sequence(model, frames, boxes, exist, collect_snapshots=True,
                                snapshot_ids=ids)
        valid = valid_frames(boxes, exist)
        labels = {}
        for t in sorted(ids):
            if not valid[t + 1]:
                continue
            diag = baseline['diagnostics'][t]
            current_shape = read_rgb(frames[t]).shape
            image = read_rgb(frames[t + 1])
            regions = candidate_regions(baseline['boxes'][t], diag['velocity'],
                                        geometry_from_dict(diag['crop']), current_shape)
            action_ious = []
            for region in regions:
                model.restore(baseline['snapshots'][t])
                prediction = model.track(image, search_region=region)['target_bbox']
                action_ious.append(box_iou(prediction, boxes[t + 1]))
            labels[t] = action_ious
        observations, scores, masks, frame_ids, existence = [], [], [], [], []
        for start in starts:
            indices = np.arange(start, start + WINDOW)
            observations.append(np.stack([baseline['diagnostics'][t]['observation'] for t in indices]))
            scores.append(np.asarray([labels.get(t, [0.] * 9) for t in indices], dtype=np.float32))
            masks.append(np.asarray([t in labels and t >= start + WARMUP for t in indices]))
            frame_ids.append(indices)
            existence.append(exist[indices])
        chunk = {
            'protocol': CACHE_PROTOCOL, 'sequence_name': name,
            'observations': np.asarray(observations, dtype=np.float32).reshape(-1, WINDOW, 32),
            'action_ious': np.asarray(scores, dtype=np.float32).reshape(-1, WINDOW, 9),
            'supervision_mask': np.asarray(masks, dtype=bool).reshape(-1, WINDOW),
            'frame_ids': np.asarray(frame_ids, dtype=np.int64).reshape(-1, WINDOW),
            'exist': np.asarray(existence, dtype=bool).reshape(-1, WINDOW),
        }
        relative_path = os.path.join('sequences', name + '.pt')
        save_torch_atomic(os.path.join(args.output, relative_path), chunk)
        completed[name] = {'file': relative_path, 'windows': len(starts),
                           'supervised_steps': int(chunk['supervision_mask'].sum()),
                           'frames': len(frames)}
        write_json_atomic(manifest_path, dict(identity, complete=False, sequences=completed))
        print('teacher %d/%d %s windows=%d valid_steps=%d elapsed=%.1fs' % (
            len(completed), len(records), name, len(starts), chunk['supervision_mask'].sum(),
            time.perf_counter() - started), flush=True)
    after = state_hash(model)
    if before != after:
        raise RuntimeError('Frozen tracker parameters or buffers changed during teacher collection')
    write_json_atomic(manifest_path, dict(identity, complete=True, sequences=completed,
        network_state_sha256=before, frozen_weights_and_buffers_unchanged=True,
        elapsed_seconds=time.perf_counter() - started))
    print('Wrote ' + manifest_path, flush=True)


def main():
    args = parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.mode == 'cache':
        collect_cache(args)
    else:
        from lib.train.trainers.search_policy_trainer import SearchPolicyTrainer
        SearchPolicyTrainer(args).train()


if __name__ == '__main__':
    main()
