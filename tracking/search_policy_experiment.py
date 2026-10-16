"""Diagnostics, counterfactual search tests, and frozen-tracker policy training."""
import argparse
import hashlib
import json
import os
import random
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lib.models.search_policy import SearchPolicy, select_search_action
from lib.test.evaluation.antiuav410dataset import AntiUAV410Dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.tracker.focustrack import FocusTrack
from lib.test.utils.search_region import (
    CropGeometry, MotionState, SearchRegion, allowed_actions, candidate_regions,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['diagnose', 'intervene', 'cache-teacher',
                                        'train-supervised', 'train-slt', 'evaluate', 'compare',
                                        'merge', 'merge-cache'])
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--data-root', default=os.path.join(PROJECT_ROOT, 'data', 'antiuav410'))
    parser.add_argument('--checkpoint', default='output/checkpoints/train/focustrack/focustrack_stage2/FocusTrack_ep0010.pth.tar')
    parser.add_argument('--policy-checkpoint')
    parser.add_argument('--output', default='output/search_policy')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-sequences', type=int, default=0)
    parser.add_argument('--sequence-start', type=int, default=0)
    parser.add_argument('--sequence-end', type=int, default=0)
    parser.add_argument('--report-suffix', default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--episodes-per-epoch', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-teacher-samples', type=int, default=10000)
    parser.add_argument('--action-mode', choices=['joint', 'center', 'scale'], default='joint')
    parser.add_argument('--policy-type', choices=['gru', 'mlp'], default='gru')
    parser.add_argument('--baseline-report')
    parser.add_argument('--candidate-report')
    parser.add_argument('--input-reports', nargs='+')
    parser.add_argument('--input-caches', nargs='+')
    parser.add_argument('--merged-name', default='merged')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--expanded-candidates', action='store_true')
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--motion-baseline', action='store_true')
    parser.add_argument('--coordinate-control', action='store_true')
    parser.add_argument('--fixed-factor', type=float, choices=(4., 6., 8.))
    parser.add_argument('--save-raw-responses', action='store_true')
    parser.add_argument('--compare-timing', action='store_true')
    return parser.parse_args()


def box_iou(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if (a.size != 4 or b.size != 4 or not np.isfinite(a).all() or not np.isfinite(b).all()
            or a[2] <= 0 or a[3] <= 0 or b[2] <= 0 or b[3] <= 0):
        return 0.0
    left = np.maximum(a[:2], b[:2])
    right = np.minimum(a[:2] + a[2:], b[:2] + b[2:])
    inter = np.maximum(right - left, 0).prod()
    return float(inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-8))


def geometry_from_dict(value):
    return CropGeometry(tuple(value['center_xy']), tuple(value['origin_xy']),
                        int(value['side']), float(value['factor']), int(value['output_size']))


def read_rgb(path):
    image = cv2.imread(path)
    if image is None:
        raise IOError('Unable to read image: ' + path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_runtime(args):
    if not torch.cuda.is_available():
        raise RuntimeError('This tracker implementation requires CUDA; CUDA is not available.')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    base = Tracker('focustrack', 'focustrack_stage2', 'antiuav410_' + args.split, run_id=10)
    params = base.get_parameters()
    params.checkpoint = os.path.abspath(args.checkpoint)
    params.search_diagnostics = True
    params.search_policy_checkpoint = None
    params.debug = 0
    model = FocusTrack(params, 'antiuav410_' + args.split)
    model.network.to(device).eval()
    model.network.requires_grad_(False)
    return model


def load_sequences(split, limit=0, sequence_start=0, sequence_end=0, data_root=None):
    dataset = AntiUAV410Dataset(split, root=data_root)
    records = []
    for seq in dataset.get_sequence_list():
        label_path = os.path.join(dataset.base_path, seq.name, 'IR_label.json')
        with open(label_path, 'r') as handle:
            labels = json.load(handle)
        boxes = np.asarray(labels['gt_rect'], dtype=np.float64).reshape(-1, 4)
        visible = np.asarray(labels['exist'], dtype=bool)
        if len(seq.frames) != len(boxes) or len(seq.frames) != len(visible):
            raise ValueError('Frame/annotation length mismatch for ' + seq.name)
        count = len(seq.frames)
        records.append((seq.name, seq.frames[:count], boxes[:count], visible[:count]))
    start = max(0, int(sequence_start))
    end = int(sequence_end) if sequence_end else len(records)
    end = max(start, min(end, len(records)))
    records = records[start:end]
    if limit:
        records = records[:limit]
    return records


def output_path(args, stem, extension):
    suffix = ('_' + args.report_suffix.lstrip('_')) if args.report_suffix else ''
    return os.path.join(args.output, stem + suffix + extension)


def write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    os.replace(temporary, path)


def valid_frames(boxes, visible):
    return (np.asarray(visible, dtype=bool) & np.isfinite(boxes).all(axis=1)
            & (boxes[:, 2] > 0) & (boxes[:, 3] > 0))


def reappearance_starts(exist, valid):
    pending = False
    starts = []
    for index in range(1, len(exist)):
        if not exist[index - 1]:
            pending = True
        if pending and valid[index]:
            starts.append(index)
            pending = False
    return starts


@torch.no_grad()
def run_sequence(model, frames, boxes, visible, policy=None, sample=False,
                 collect_snapshots=False, snapshot_ids=None, action_mode='joint',
                 motion_baseline=False, coordinate_control=False, fixed_factor=None):
    image = read_rgb(frames[0])
    model.initialize(image, {'init_bbox': boxes[0].tolist()})
    device = next(model.network.parameters()).device
    hidden = None
    next_region = None
    outputs = [boxes[0].tolist()]
    times = [0.0]
    diagnostics = [None]
    snapshots = {}
    for frame_id in range(1, len(frames)):
        image = read_rgb(frames[frame_id])
        start = time.perf_counter()
        model.track(image, search_region=next_region)
        elapsed = time.perf_counter() - start
        output = list(model.state)
        outputs.append(output)
        diag = model.last_diagnostic
        diagnostics.append(diag)
        next_region = None
        if motion_baseline or coordinate_control or fixed_factor is not None:
            geometry = geometry_from_dict(diag['crop'])
            motion_region = candidate_regions(output, model.motion.velocity, geometry,
                                               image.shape)[6 if motion_baseline else 0]
            next_region = SearchRegion(motion_region.center_xy,
                                       fixed_factor if fixed_factor is not None else diag['sra_next_factor'])
        if policy is not None and frame_id < len(frames) - 1:
            observation = torch.as_tensor(diag['observation'], dtype=torch.float32,
                                          device=device).unsqueeze(0)
            logits, hidden = policy(observation, hidden)
            action_ids = torch.as_tensor(allowed_actions(action_mode), device=device)
            action = int(select_search_action(logits, action_ids,
                getattr(policy, 'decision', None), sample=sample).item())
            diag['policy_action'] = action
            geometry = geometry_from_dict(diag['crop'])
            next_region = candidate_regions(output, model.motion.velocity, geometry,
                                             image.shape)[action]
        if collect_snapshots and snapshot_ids and frame_id in snapshot_ids:
            snapshots[frame_id] = model.snapshot()
        times.append(time.perf_counter() - start)
    return {'boxes': np.asarray(outputs), 'visible': visible[:len(outputs)],
            'times': np.asarray(times), 'diagnostics': diagnostics,
            'snapshots': snapshots}


def sequence_metrics(predictions, boxes, visible):
    visible_valid = valid_frames(boxes, visible)
    indices = np.flatnonzero(visible_valid)
    indices = indices[indices != 0]
    ious = np.asarray([box_iou(predictions[i], boxes[i]) for i in indices])
    thresholds = np.linspace(0, 1, 21)
    auc = float(np.mean([np.mean(ious >= threshold) for threshold in thresholds])) if len(ious) else 0.0
    mixed = []
    for i in range(min(len(predictions), len(visible))):
        if not visible[i]:
            mixed.append(0.0)
        elif visible_valid[i]:
            mixed.append(box_iou(predictions[i], boxes[i]))
    recoveries = []
    iou_by_frame = np.full(len(predictions), np.nan)
    for i in range(1, min(len(predictions), len(visible))):
        if visible_valid[i]:
            iou_by_frame[i] = box_iou(predictions[i], boxes[i])
    for first_visible in reappearance_starts(visible, visible_valid):
        end = min(first_visible + 30, len(visible))
        absent_again = np.flatnonzero(~visible[first_visible:end])
        if len(absent_again):
            end = first_visible + int(absent_again[0])
        recovered_at = None
        for i in range(first_visible, max(first_visible, end - 2)):
            if np.all(iou_by_frame[i:i + 3] >= 0.5):
                recovered_at = i + 2
                break
        recoveries.append(None if recovered_at is None else recovered_at - first_visible)
    return {
        'visible_iou': float(ious.mean()) if len(ious) else 0.0,
        'visible_auc': auc,
        'mixed_measure': float(np.mean(mixed)) if mixed else 0.0,
        'fps': None,
        'recovery_rate_30': float(np.mean([v is not None for v in recoveries])) if recoveries else None,
        'recovery_latency': float(np.mean([v for v in recoveries if v is not None])) if any(v is not None for v in recoveries) else None,
        'recovery_unrecovered': float(np.mean([v is None for v in recoveries])) if recoveries else None,
        'recovery_events': len(recoveries),
        'recovery_delays': recoveries,
        'visible_valid_frames': len(indices),
    }


def failure_starts(visible, ious, threshold=0.2, duration=5):
    """Return one start index for each continuous visible low-IoU episode."""
    starts = []
    active = False
    last_start = len(ious) - duration
    for index in range(1, max(1, last_start + 1)):
        failed = bool(visible[index:index + duration].all() and
                      np.all(ious[index:index + duration] < threshold))
        if failed and not active:
            starts.append(index)
        active = failed
    return starts


def diagnose_sequence(result, frames, boxes, visible):
    exist = np.asarray(visible, dtype=bool)
    visible = valid_frames(boxes, visible)
    records = []
    ious = np.asarray([box_iou(result['boxes'][i], boxes[i]) if visible[i] else 0.0
                       for i in range(len(boxes))])
    for i in range(1, min(len(boxes), len(result['diagnostics']))):
        diag = result['diagnostics'][i]
        if diag is None:
            continue
        gt = boxes[i]
        crop = diag['crop']
        x0, y0 = crop['origin_xy']
        side = crop['side']
        cx, cy = gt[:2] + gt[2:] / 2
        center_outside = not (x0 <= cx <= x0 + side and y0 <= cy <= y0 + side)
        intersection = max(0.0, min(gt[0] + gt[2], x0 + side) - max(gt[0], x0)) * \
                       max(0.0, min(gt[1] + gt[3], y0 + side) - max(gt[1], y0))
        coverage = intersection / max(gt[2] * gt[3], 1e-8)
        target_pixels = float(np.sqrt(max(gt[2] * gt[3], 0.0)) *
                               diag['crop']['output_size'] / max(side, 1))
        records.append({'frame': i, 'visible': bool(visible[i]), 'exist': bool(exist[i]), 'iou': float(ious[i]),
                        'center_outside_crop': bool(center_outside), 'bbox_coverage': float(coverage),
                        'target_pixels': target_pixels, 'crop': crop,
                        'presence': diag['presence'], 'max_score': diag['max_score'],
                        'search_factor': diag['crop']['factor'], 'next_factor': diag['next_factor'],
                        'velocity': diag['velocity'], 'use_hann': diag['use_hann'],
                        'elapsed': diag['elapsed']})
        if 'policy_action' in diag:
            records[-1]['policy_action'] = diag['policy_action']
    failures = failure_starts(visible, ious)
    visible_records = [r for r in records if r['visible']]
    return {
        'frames': records,
        'failure_starts': failures,
        'visible_crop_center_outside_rate': float(np.mean([r['center_outside_crop'] for r in visible_records])) if visible_records else None,
        'visible_mean_bbox_coverage': float(np.mean([r['bbox_coverage'] for r in visible_records])) if visible_records else None,
        'visible_mean_target_pixels': float(np.mean([r['target_pixels'] for r in visible_records])) if visible_records else None,
        'failure_count': len(failures),
        'invisible_frames': int(np.sum(~exist[1:])),
        'invalid_visible_frames': int(np.sum(exist[1:] & ~visible[1:])),
    }


def bootstrap_mean_ci(values, repetitions=10000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return [0.0, 0.0]
    rng = np.random.RandomState(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return np.percentile(means, [2.5, 97.5]).tolist()


def aggregate_reports(records):
    keys = ('visible_iou', 'visible_auc', 'mixed_measure', 'fps',
            'recovery_rate_30', 'recovery_latency', 'recovery_unrecovered')
    output = {}
    for key in keys:
        vals = [r[key] for r in records if r.get(key) is not None]
        output[key] = float(np.mean(vals)) if vals else None
    return output


def evaluate(args):
    if sum((args.motion_baseline, args.coordinate_control, bool(args.policy_checkpoint),
            args.fixed_factor is not None)) > 1:
        raise ValueError('Choose one search controller')
    model = build_runtime(args)
    policy = None
    if args.mode == 'evaluate' and args.policy_checkpoint:
        checkpoint = torch.load(args.policy_checkpoint, map_location='cpu')
        policy = SearchPolicy(recurrent=checkpoint.get('recurrent', True),
                              mlp_depth=checkpoint.get('mlp_depth', 3)).to(args.device)
        policy.load_state_dict(checkpoint['policy'])
        policy.decision = checkpoint.get('decision')
        policy.eval()
        if checkpoint.get('backbone_sha256') and checkpoint['backbone_sha256'] != hashlib.sha256(
                open(args.checkpoint, 'rb').read()).hexdigest():
            raise ValueError('Policy was trained with a different frozen tracker')
        policy_action_mode = checkpoint.get('action_mode', args.action_mode)
    else:
        policy_action_mode = args.action_mode
    sequences = load_sequences(args.split, args.max_sequences,
                               args.sequence_start, args.sequence_end, args.data_root)
    reports = []
    details = {}
    weight_sha = hashlib.sha256(open(args.checkpoint, 'rb').read()).hexdigest()
    report_path = output_path(args, '%s_%s' % (args.mode, args.split), '.json')
    metadata = {'split': args.split, 'baseline_checkpoint': os.path.abspath(args.checkpoint),
                'baseline_sha256': weight_sha, 'policy_checkpoint': args.policy_checkpoint,
                'policy_sha256': hashlib.sha256(open(args.policy_checkpoint, 'rb').read()).hexdigest()
                    if args.policy_checkpoint else None,
                'policy_action_mode': policy_action_mode if args.policy_checkpoint else None,
                'motion_baseline': args.motion_baseline, 'data_root': os.path.abspath(args.data_root),
                'coordinate_control': args.coordinate_control,
                'fixed_factor': args.fixed_factor,
                'protocol': 'search_evaluation_v2', 'expected_sequences': len(sequences)}
    if args.resume and os.path.isfile(report_path):
        with open(report_path) as handle:
            previous = json.load(handle)
        if any(previous.get(key) != value for key, value in metadata.items()):
            raise ValueError('Cannot resume incompatible evaluation report')
        details = previous['sequences']
        reports = [value['metrics'] for value in details.values()]
    for number, (name, frames, boxes, visible) in enumerate(sequences, 1):
        if name in details:
            continue
        result = run_sequence(model, frames, boxes, visible, policy=policy,
                              action_mode=policy_action_mode, motion_baseline=args.motion_baseline,
                              coordinate_control=args.coordinate_control, fixed_factor=args.fixed_factor)
        metrics = sequence_metrics(result['boxes'], boxes, visible)
        metrics['fps'] = float((len(result['times']) - 1) / max(result['times'].sum(), 1e-9))
        reports.append(metrics)
        details[name] = {'metrics': metrics, 'predictions': result['boxes'].tolist(),
                         'visible_iou': [None if not visible[i] else box_iou(result['boxes'][i], boxes[i])
                                         for i in range(len(boxes))],
                         'diagnostics': diagnose_sequence(result, frames, boxes, visible)}
        details[name]['policy_action_counts'] = {
            str(action): sum(diag is not None and diag.get('policy_action') == action
                             for diag in result['diagnostics']) for action in range(9)}
        details[name]['crop_factor_counts'] = {
            str(factor): sum(diag is not None and diag['crop']['factor'] == factor
                             for diag in result['diagnostics']) for factor in (4., 6., 8.)}
        if args.save_raw_responses:
            raw_dir = os.path.join(args.output, 'raw_responses')
            os.makedirs(raw_dir, exist_ok=True)
            np.savez_compressed(os.path.join(raw_dir, name + '.npz'),
                                frame_ids=np.arange(1, len(frames)),
                                score_maps=np.stack([d['score_map'] for d in result['diagnostics'][1:]]),
                                observations=np.stack([d['observation'] for d in result['diagnostics'][1:]]))
            details[name]['raw_responses'] = os.path.join(raw_dir, name + '.npz')
        write_json_atomic(report_path, dict(metadata, complete=False, sequences=details))
        print('[%d/%d] %s visible IoU %.4f, AUC %.4f, %.1f FPS' %
              (number, len(sequences), name, metrics['visible_iou'], metrics['visible_auc'], metrics['fps']))
    summary = aggregate_reports(reports)
    summary_ci = {}
    for key in ('visible_iou', 'visible_auc', 'mixed_measure', 'fps'):
        values = [r[key] for r in reports if r.get(key) is not None]
        summary_ci[key] = bootstrap_mean_ci(values, seed=args.seed) if values else None
    diagnostic_records = [details[name]['diagnostics'] for name in details]
    diagnostic_summary = {
        'sequences': len(diagnostic_records),
        'failure_events': int(sum(r['failure_count'] for r in diagnostic_records)),
        'mean_visible_crop_center_outside_rate': float(np.mean([
            r['visible_crop_center_outside_rate'] for r in diagnostic_records
            if r['visible_crop_center_outside_rate'] is not None])) if any(
                r['visible_crop_center_outside_rate'] is not None for r in diagnostic_records) else None,
        'mean_visible_bbox_coverage': float(np.mean([
            r['visible_mean_bbox_coverage'] for r in diagnostic_records
            if r['visible_mean_bbox_coverage'] is not None])) if any(
                r['visible_mean_bbox_coverage'] is not None for r in diagnostic_records) else None,
        'mean_visible_target_pixels': float(np.mean([
            r['visible_mean_target_pixels'] for r in diagnostic_records
            if r['visible_mean_target_pixels'] is not None])) if any(
                r['visible_mean_target_pixels'] is not None for r in diagnostic_records) else None,
    }
    os.makedirs(args.output, exist_ok=True)
    payload = dict(metadata, complete=True, summary=summary, summary_ci=summary_ci,
                   diagnostic_summary=diagnostic_summary, sequences=details)
    write_json_atomic(report_path, payload)
    print(json.dumps(summary, indent=2))
    print('Wrote', report_path)


def intervention_samples(visible, ious, limit_uniform=32, event_limit=8, exist=None):
    n = len(visible)
    uniform = np.linspace(1, max(1, n - 2), min(limit_uniform, max(n - 2, 1)), dtype=int)
    failures = [max(0, index - 1) for index in failure_starts(visible, ious)]
    reappear = [i - 1 for i in reappearance_starts(visible if exist is None else exist, visible)]
    return sorted(set(uniform.tolist() + failures[:event_limit] + reappear[:event_limit]))


def run_intervention(args):
    model = build_runtime(args)
    sequences = load_sequences('val', args.max_sequences,
                               args.sequence_start, args.sequence_end, args.data_root)
    output = {}
    protocol = 'search_intervention_v2_hann_integer_geometry'
    path = output_path(args, 'intervention_val', '.json')
    weight_sha = hashlib.sha256(open(args.checkpoint, 'rb').read()).hexdigest()
    reference = None
    if args.baseline_report:
        with open(args.baseline_report) as handle:
            reference = json.load(handle)
        if reference.get('baseline_sha256') != weight_sha or reference.get('split') != 'val':
            raise ValueError('Baseline reference checkpoint or split does not match')
    if args.resume and os.path.isfile(path):
        with open(path) as handle:
            previous = json.load(handle)
        if (previous.get('protocol') != protocol or previous.get('checkpoint_sha256') != weight_sha
                or previous.get('expanded_candidates') != args.expanded_candidates):
            raise ValueError('Cannot resume incompatible intervention report')
        output = previous['sequences']
    started = time.perf_counter()
    for seq_id, (name, frames, boxes, visible) in enumerate(sequences, 1):
        if name in output:
            continue
        exist = visible.copy()
        visible = valid_frames(boxes, visible)
        if reference is None:
            baseline = run_sequence(model, frames, boxes, visible)
        else:
            baseline = {'boxes': np.asarray(reference['sequences'][name]['predictions'])}
        ious = np.asarray([box_iou(p, g) if visible[i] else 0.0
                           for i, (p, g) in enumerate(zip(baseline['boxes'], boxes))])
        sample_ids = intervention_samples(visible, ious, exist=exist)
        failure_ids = {i - 1 for i in failure_starts(visible, ious)}
        reappear_ids = {i - 1 for i in reappearance_starts(exist, visible)}
        snapshots_run = run_sequence(model, frames, boxes, visible,
                                     collect_snapshots=True, snapshot_ids=set(sample_ids))
        if not np.allclose(snapshots_run['boxes'], baseline['boxes'], atol=1e-4, rtol=0):
            raise RuntimeError('Frozen baseline could not be reproduced for ' + name)
        samples = []
        for t in sample_ids:
            if t <= 0 or t + 1 >= len(frames) or t not in snapshots_run['snapshots']:
                continue
            model.restore(snapshots_run['snapshots'][t])
            current_diag = snapshots_run['diagnostics'][t]
            height, width = cv2.imread(frames[t]).shape[:2]
            geometry = geometry_from_dict(current_diag['crop'])
            actions = candidate_regions(baseline['boxes'][t], current_diag['velocity'],
                                        geometry, (height, width, 3), expanded=args.expanded_candidates)
            future_images = [read_rgb(frames[j]) for j in range(t + 1, min(t + 9, len(frames)))]
            candidate_results = []
            for action_id in range(len(actions)):
                model.restore(snapshots_run['snapshots'][t])
                next_region = actions[action_id]
                pred = []
                for frame_id in range(t + 1, min(t + 9, len(frames))):
                    image = future_images[frame_id - t - 1]
                    model.track(image, search_region=next_region if frame_id == t + 1 else None)
                    if frame_id == t + 1:
                        first_crop = model.last_diagnostic['crop']
                        model.search_factor = model.last_diagnostic['sra_next_factor']
                    next_region = None
                    pred.append(list(model.state))
                first = box_iou(pred[0], boxes[t + 1]) if visible[t + 1] else None
                future = [box_iou(p, boxes[t + 1 + k]) for k, p in enumerate(pred)
                          if t + 1 + k < len(visible) and visible[t + 1 + k]]
                target = boxes[t + 1]
                candidate_side = first_crop['side']
                origin = np.asarray(first_crop['origin_xy'])
                intersection = np.maximum(0, np.minimum(target[:2] + target[2:], origin + candidate_side)
                                          - np.maximum(target[:2], origin)).prod()
                coverage = float(intersection / (target[2] * target[3])) if visible[t + 1] else None
                center_inside = bool(np.all((target[:2] + target[2:] / 2 >= origin) &
                                            (target[:2] + target[2:] / 2 < origin + candidate_side)))
                candidate_results.append({'action': action_id, 'region': actions[action_id].as_dict(),
                                          'next_iou': first,
                                          'future_iou': float(np.mean(future)) if future else None,
                                          'covered': coverage,
                                          'center_inside': center_inside if visible[t + 1] else None,
                                          'crop': first_crop,
                                          'target_pixels': float(np.sqrt(max(target[2] * target[3], 0)) *
                                                                 model.params.search_size / candidate_side)
                                          if visible[t + 1] else None})
            base_iou = [box_iou(baseline['boxes'][j], boxes[j]) for j in range(t + 1, min(t + 9, len(frames)))
                        if visible[j]]
            samples.append({'frame': t, 'uniform_sample': t in np.linspace(1, max(1, len(frames) - 2),
                                                                           min(32, max(len(frames) - 2, 1)), dtype=int),
                            'baseline_future_iou': float(np.mean(base_iou)) if base_iou else None,
                            'failure_sample': t in failure_ids,
                            'reappearance_sample': t in reappear_ids,
                            'full_horizon': t + 8 < len(frames),
                            'candidates': candidate_results})
        output[name] = samples
        write_json_atomic(path, {'protocol': protocol, 'checkpoint_sha256': weight_sha,
                                'expanded_candidates': args.expanded_candidates,
                                'complete': len(output) == len(sequences),
                                'expected_sequences': len(sequences),
                                'elapsed_seconds': time.perf_counter() - started,
                                'sequences': output})
        print('[%d/%d] %s: %d intervention points' % (seq_id, len(sequences), name, len(samples)))
    all_samples = [sample for samples in output.values() for sample in samples]
    paired = []
    for sample in all_samples:
        candidates = [c for c in sample['candidates'] if c['future_iou'] is not None]
        if candidates and sample['baseline_future_iou'] is not None:
            paired.append((sample['baseline_future_iou'], max(c['future_iou'] for c in candidates)))
    intervention_summary = summarize_intervention(output, args.seed)
    os.makedirs(args.output, exist_ok=True)
    write_json_atomic(path, {'protocol': protocol, 'checkpoint_sha256': weight_sha,
                            'expanded_candidates': args.expanded_candidates,
                            'complete': True, 'expected_sequences': len(sequences),
                            'elapsed_seconds': time.perf_counter() - started,
                            'summary': intervention_summary, 'sequences': output})
    print('Wrote', path)


def summarize_intervention(sequences, seed=0, include_groups=True):
    samples = [sample for sequence_samples in sequences.values()
               for sample in sequence_samples]
    paired = []
    sequence_deltas = {}
    for name, sequence_samples in sequences.items():
        local_pairs = []
        for sample in sequence_samples:
            candidates = [c for c in sample['candidates'] if c.get('future_iou') is not None]
            baseline = sample.get('baseline_future_iou')
            if candidates and baseline is not None:
                local_pairs.append((baseline, max(c['future_iou'] for c in candidates)))
        if local_pairs:
            sequence_deltas[name] = float(np.mean([candidate - baseline
                                                    for baseline, candidate in local_pairs]))
    for sample in samples:
        candidates = [c for c in sample['candidates'] if c.get('future_iou') is not None]
        baseline = sample.get('baseline_future_iou')
        if candidates and baseline is not None:
            paired.append((baseline, max(c['future_iou'] for c in candidates)))
    deltas = np.asarray([candidate - baseline for baseline, candidate in paired], dtype=np.float64)
    baselines = np.asarray([baseline for baseline, _ in paired], dtype=np.float64)
    candidates = np.asarray([candidate for _, candidate in paired], dtype=np.float64)
    result = {
        'points': len(samples),
        'usable_points': len(paired),
        'mean_baseline_future_iou': float(baselines.mean()) if len(baselines) else None,
        'mean_best_candidate_future_iou': float(candidates.mean()) if len(candidates) else None,
        'mean_best_minus_baseline': float(deltas.mean()) if len(deltas) else None,
        'positive_delta_005_rate': float(np.mean(deltas >= 0.05)) if len(deltas) else None,
        'best_delta_ci': bootstrap_mean_ci(deltas, seed=seed) if len(deltas) else None,
        'usable_sequence_count': len(sequence_deltas),
        'sequences_with_positive_delta': int(sum(delta > 0 for delta in sequence_deltas.values())),
        'sequences_with_delta_005': int(sum(delta >= 0.05 for delta in sequence_deltas.values())),
        'sequence_delta_mean': float(np.mean(list(sequence_deltas.values()))) if sequence_deltas else None,
        'sequence_delta_ci': bootstrap_mean_ci(list(sequence_deltas.values()), seed=seed)
        if sequence_deltas else None,
    }
    if include_groups:
        for group in ('uniform_sample', 'failure_sample', 'reappearance_sample', 'full_horizon'):
            selected = {name: [sample for sample in values if sample.get(group, False)]
                        for name, values in sequences.items()}
            result[group] = summarize_intervention(selected, seed, include_groups=False)
        full_failures = {name: [sample for sample in values if sample.get('failure_sample')
                               and sample.get('full_horizon')] for name, values in sequences.items()}
        result['failure_full_horizon'] = summarize_intervention(full_failures, seed, include_groups=False)
        result['sequences_with_improvement_event'] = sum(
            any(sample.get('baseline_future_iou') is not None and any(
                candidate.get('future_iou') is not None and
                candidate['future_iou'] - sample['baseline_future_iou'] > 1e-6
                for candidate in sample['candidates']) for sample in values)
            for values in sequences.values())
        result['sequences_with_improvement_event_005'] = sum(
            any(sample.get('baseline_future_iou') is not None and any(
                candidate.get('future_iou') is not None and
                candidate['future_iou'] - sample['baseline_future_iou'] >= 0.05
                for candidate in sample['candidates']) for sample in values)
            for values in sequences.values())
        failure_rate = result['failure_full_horizon']['positive_delta_005_rate']
        result['engineering_gate_passed'] = bool(
            result['sequences_with_improvement_event'] >= 20 and
            failure_rate is not None and failure_rate >= 0.2)
    return result


def merge_reports(args):
    if not args.input_reports:
        raise ValueError('merge requires --input-reports')
    payloads = []
    for path in args.input_reports:
        with open(path, 'r') as handle:
            payloads.append(json.load(handle))
    if not payloads:
        raise ValueError('No reports supplied')
    sequences = {}
    for payload in payloads:
        for name, value in payload.get('sequences', {}).items():
            if name in sequences:
                raise ValueError('Duplicate sequence in reports: ' + name)
            sequences[name] = value

    first = payloads[0]
    if first.get('protocol') in ('search_evaluation_v2', 'search_intervention_v2_hann_integer_geometry'):
        if any(not payload.get('complete', False) for payload in payloads):
            raise ValueError('Cannot merge unfinished experiment shards')
    for payload in payloads[1:]:
        for key in ('protocol', 'split', 'baseline_sha256', 'checkpoint_sha256',
                    'policy_checkpoint', 'expanded_candidates', 'motion_baseline', 'coordinate_control',
                    'fixed_factor', 'data_root', 'policy_sha256', 'policy_action_mode'):
            if payload.get(key) != first.get(key):
                raise ValueError('Incompatible report metadata: ' + key)
    if all(isinstance(value, dict) and 'metrics' in value for value in sequences.values()):
        records = [value['metrics'] for value in sequences.values()]
        summary = aggregate_reports(records)
        summary_ci = {}
        for key in ('visible_iou', 'visible_auc', 'mixed_measure', 'fps'):
            values = [record[key] for record in records if record.get(key) is not None]
            summary_ci[key] = bootstrap_mean_ci(values, seed=args.seed) if values else None
        diagnostic_records = [value.get('diagnostics', {}) for value in sequences.values()]
        outside = [value['visible_crop_center_outside_rate'] for value in diagnostic_records
                    if value.get('visible_crop_center_outside_rate') is not None]
        coverage = [value['visible_mean_bbox_coverage'] for value in diagnostic_records
                    if value.get('visible_mean_bbox_coverage') is not None]
        pixels = [value['visible_mean_target_pixels'] for value in diagnostic_records
                  if value.get('visible_mean_target_pixels') is not None]
        diagnostic_summary = {
            'sequences': len(diagnostic_records),
            'failure_events': int(sum(value.get('failure_count', 0) for value in diagnostic_records)),
            'mean_visible_crop_center_outside_rate': float(np.mean(outside)) if outside else None,
            'mean_visible_bbox_coverage': float(np.mean(coverage)) if coverage else None,
            'mean_visible_target_pixels': float(np.mean(pixels)) if pixels else None,
        }
        merged = {
            'protocol': first.get('protocol'),
            'expanded_candidates': first.get('expanded_candidates'),
            'split': first.get('split'),
            'baseline_checkpoint': first.get('baseline_checkpoint'),
            'baseline_sha256': first.get('baseline_sha256'),
            'policy_checkpoint': first.get('policy_checkpoint'),
            'policy_sha256': first.get('policy_sha256'),
            'policy_action_mode': first.get('policy_action_mode'),
            'data_root': first.get('data_root'),
            'motion_baseline': first.get('motion_baseline', False),
            'coordinate_control': first.get('coordinate_control', False),
            'fixed_factor': first.get('fixed_factor'),
            'source_reports': [os.path.abspath(path) for path in args.input_reports],
            'summary': summary,
            'summary_ci': summary_ci,
            'diagnostic_summary': diagnostic_summary,
            'sequences': sequences,
        }
    elif all(isinstance(value, list) for value in sequences.values()):
        merged = {
            'protocol': first.get('protocol'),
            'expanded_candidates': first.get('expanded_candidates'),
            'complete': all(payload.get('complete', False) for payload in payloads),
            'worker_elapsed_seconds': [payload.get('elapsed_seconds') for payload in payloads],
            'checkpoint_sha256': first.get('checkpoint_sha256'),
            'source_reports': [os.path.abspath(path) for path in args.input_reports],
            'summary': summarize_intervention(sequences, args.seed),
            'sequences': sequences,
        }
    else:
        raise ValueError('Reports have incompatible sequence payloads')
    os.makedirs(args.output, exist_ok=True)
    path = output_path(args, args.merged_name, '.json')
    write_json_atomic(path, merged)
    print(json.dumps(merged.get('summary', {}), indent=2))
    print('Wrote', path)


def merge_caches(args):
    if not args.input_caches:
        raise ValueError('merge-cache requires --input-caches')
    chunks = [torch.load(path, map_location='cpu') for path in args.input_caches]
    keys = ('observations', 'targets', 'sequence_ids')
    merged = {}
    for key in keys:
        arrays = [np.asarray(chunk[key]) for chunk in chunks if key in chunk]
        if arrays:
            merged[key] = np.concatenate(arrays, axis=0)
    merged['source_caches'] = [os.path.abspath(path) for path in args.input_caches]
    os.makedirs(args.output, exist_ok=True)
    path = output_path(args, args.merged_name, '.pt')
    torch.save(merged, path)
    print('Merged %d samples from %d caches' % (len(merged.get('observations', [])), len(chunks)))
    print('Wrote', path)


def compare_reports(args):
    if not args.baseline_report or not args.candidate_report:
        raise ValueError('compare requires --baseline-report and --candidate-report')
    with open(args.baseline_report, 'r') as handle:
        baseline = json.load(handle)
    with open(args.candidate_report, 'r') as handle:
        candidate = json.load(handle)
    baseline_sequences = baseline['sequences']
    candidate_sequences = candidate['sequences']
    if baseline.get('baseline_sha256') != candidate.get('baseline_sha256'):
        raise ValueError('Paired reports use different frozen tracker checkpoints')
    common = sorted(set(baseline_sequences) & set(candidate_sequences))
    if set(baseline_sequences) != set(candidate_sequences):
        raise ValueError('Paired comparison requires identical complete sequence sets')
    metrics = ('visible_iou', 'visible_auc', 'mixed_measure', 'fps',
               'recovery_rate_30', 'recovery_latency')
    if not args.compare_timing:
        metrics = tuple(metric for metric in metrics if metric != 'fps')
    report = {'baseline_report': os.path.abspath(args.baseline_report),
              'candidate_report': os.path.abspath(args.candidate_report),
              'sequences': len(common), 'metrics': {}, 'per_sequence_deltas': {}}
    report['timing_comparable'] = args.compare_timing
    for name in common:
        report['per_sequence_deltas'][name] = {
            metric: (candidate_sequences[name]['metrics'][metric] - baseline_sequences[name]['metrics'][metric])
            if candidate_sequences[name]['metrics'].get(metric) is not None
            and baseline_sequences[name]['metrics'].get(metric) is not None else None
            for metric in metrics}
    for metric in metrics:
        pairs = [(baseline_sequences[name]['metrics'].get(metric),
                  candidate_sequences[name]['metrics'].get(metric)) for name in common]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        deltas = np.asarray([b - a for a, b in pairs], dtype=np.float64)
        report['metrics'][metric] = {
            'baseline_mean': float(np.mean([a for a, _ in pairs])) if pairs else None,
            'candidate_mean': float(np.mean([b for _, b in pairs])) if pairs else None,
            'delta_mean': float(deltas.mean()) if len(deltas) else None,
            'delta_ci_95': bootstrap_mean_ci(deltas, seed=args.seed) if len(deltas) else None,
            'positive_sequence_rate': float(np.mean(deltas > 0)) if len(deltas) else None,
        }
    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, 'comparison.json')
    with open(path, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print('Wrote', path)


def collect_teacher_cache(args):
    """Build one-step counterfactual teacher labels from frozen baseline states."""
    model = build_runtime(args)
    sequences = load_sequences('train', args.max_sequences,
                               args.sequence_start, args.sequence_end, args.data_root)
    observations, target_probs, sequence_ids = [], [], []
    for seq_id, (name, frames, boxes, visible) in enumerate(sequences, 1):
        if len(observations) >= args.max_teacher_samples:
            break
        stride = max(1, (len(frames) - 2) // 48)
        candidate_times = np.arange(1, max(2, len(frames) - 1), stride, dtype=int)
        candidate_times = candidate_times[candidate_times + 1 < len(frames)]
        baseline = run_sequence(model, frames, boxes, visible, collect_snapshots=True,
                                snapshot_ids=set(candidate_times.tolist()))
        for t in candidate_times:
            if len(observations) >= args.max_teacher_samples:
                break
            if not visible[t + 1] or t not in baseline['snapshots']:
                continue
            diag = baseline['diagnostics'][t]
            image = read_rgb(frames[t])
            geometry = geometry_from_dict(diag['crop'])
            candidates = candidate_regions(baseline['boxes'][t], diag['velocity'], geometry, image.shape)
            action_ious = []
            for candidate in candidates:
                model.restore(baseline['snapshots'][t])
                pred = model.track(read_rgb(frames[t + 1]), search_region=candidate)['target_bbox']
                action_ious.append(box_iou(pred, boxes[t + 1]))
            scores = torch.tensor(action_ious, dtype=torch.float32)
            observations.append(diag['observation'])
            target_probs.append(torch.softmax(scores / 0.1, dim=0).numpy())
            sequence_ids.append(seq_id)
        print('teacher cache %d/%d %s (%d states)' %
              (seq_id, len(sequences), name, len(observations)))
    os.makedirs(args.output, exist_ok=True)
    path = output_path(args, 'teacher_cache_%s' % args.seed, '.pt')
    torch.save({'observations': np.asarray(observations, dtype=np.float32),
                'targets': np.asarray(target_probs, dtype=np.float32),
                'sequence_ids': np.asarray(sequence_ids, dtype=np.int32)}, path)
    print('Wrote', path)


def train_supervised(args):
    cache_path = output_path(args, 'teacher_cache_%s' % args.seed, '.pt')
    cache = torch.load(cache_path, map_location='cpu')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    recurrent = args.policy_type == 'gru'
    policy = SearchPolicy(recurrent=recurrent).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-4)
    observations = torch.as_tensor(cache['observations'], dtype=torch.float32)
    targets = torch.as_tensor(cache['targets'], dtype=torch.float32)
    generator = torch.Generator().manual_seed(args.seed)
    sequence_ids = np.asarray(cache.get('sequence_ids', np.arange(len(observations))))
    sequence_samples = {int(sid): np.flatnonzero(sequence_ids == sid) for sid in np.unique(sequence_ids)}
    windows = [(indices[start:start + 12]) for indices in sequence_samples.values()
               for start in range(0, max(len(indices) - 11, 1), 8)
               if len(indices[start:start + 12]) == 12]
    if not windows:
        windows = [np.asarray([index]) for index in range(len(observations))]
    for epoch in range(args.epochs):
        permutation = torch.randperm(len(windows), generator=generator).tolist()
        losses = []
        policy.train()
        for start in range(0, len(permutation), args.batch_size):
            batch_windows = [windows[i] for i in permutation[start:start + args.batch_size]]
            x = torch.as_tensor(np.stack([observations[w] for w in batch_windows]), device=device)
            y = torch.as_tensor(np.stack([targets[w] for w in batch_windows]), device=device)
            hidden = None
            time_logits = []
            for time_index in range(x.shape[1]):
                logits, hidden = policy(x[:, time_index], hidden)
                time_logits.append(logits)
            logits = torch.stack(time_logits, dim=1)
            first_loss_step = 4 if recurrent and x.shape[1] > 4 else 0
            loss = -(y[:, first_loss_step:] * F.log_softmax(logits[:, first_loss_step:], dim=-1)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        print('epoch %d/%d loss %.6f' % (epoch + 1, args.epochs, np.mean(losses)))
        save_policy(args, policy, optimizer, epoch + 1, 'supervised')


def save_policy(args, policy, optimizer, epoch, stage):
    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, 'policy_%s_seed%d_ep%03d.pth' % (stage, args.seed, epoch))
    torch.save({'policy': policy.state_dict(), 'optimizer': optimizer.state_dict(),
                'epoch': epoch, 'seed': args.seed, 'recurrent': policy.recurrent,
                'mlp_depth': policy.mlp_depth,
                'action_mode': args.action_mode,
                'backbone_checkpoint': os.path.abspath(args.checkpoint),
                'backbone_sha256': hashlib.sha256(open(args.checkpoint, 'rb').read()).hexdigest()}, path)


def policy_rollout(model, policy, record, sample, horizon, start_frame, action_mode='joint'):
    name, frames, boxes, visible = record
    warmup = 4
    if start_frame + warmup + horizon >= len(frames):
        return [], [], None
    image = read_rgb(frames[start_frame])
    model.initialize(image, {'init_bbox': boxes[start_frame].tolist()})
    device = next(model.network.parameters()).device
    hidden = None
    next_region = None
    observations, actions = [], []
    for offset in range(1, warmup + 1):
        model.track(read_rgb(frames[start_frame + offset]), search_region=next_region)
    reward_values = []
    for step in range(horizon):
        current_frame = start_frame + warmup + step
        diag = model.last_diagnostic
        obs = torch.as_tensor(diag['observation'], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, hidden = policy(obs, hidden)
            action_ids = torch.as_tensor(allowed_actions(action_mode), device=device)
            valid_logits = logits.index_select(-1, action_ids)
            selected = (torch.distributions.Categorical(logits=valid_logits).sample()
                        if sample else valid_logits.argmax(-1))
            action = action_ids[selected]
        observations.append(obs.squeeze(0).cpu())
        actions.append(int(action.item()))
        current_image = read_rgb(frames[current_frame])
        geometry = geometry_from_dict(model.last_diagnostic['crop'])
        next_region = candidate_regions(model.state, model.motion.velocity, geometry,
                                         current_image.shape)[actions[-1]]
        model.track(read_rgb(frames[current_frame + 1]), search_region=next_region)
        if visible[current_frame + 1] and boxes[current_frame + 1, 2] > 0 and boxes[current_frame + 1, 3] > 0:
            reward_values.append(box_iou(model.state, boxes[current_frame + 1]))
    return observations, actions, float(np.mean(reward_values)) if reward_values else None


def train_slt(args):
    sequences = load_sequences('train', args.max_sequences,
                               args.sequence_start, args.sequence_end, args.data_root)
    if not sequences:
        raise RuntimeError('No AntiUAV training sequences found.')
    model = build_runtime(args)
    for parameter in model.network.parameters():
        parameter.requires_grad_(False)
    model.network.eval()
    policy = SearchPolicy(recurrent=True).to(args.device)
    supervised_path = args.policy_checkpoint or os.path.join(args.output,
        'policy_supervised_seed%d_ep%03d.pth' % (args.seed, args.epochs))
    if not os.path.isfile(supervised_path):
        raise FileNotFoundError('Run train-supervised first or pass --policy-checkpoint.')
    policy.load_state_dict(torch.load(supervised_path, map_location=args.device)['policy'])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        horizon = 8 if epoch < args.epochs // 2 else 16
        losses, sample_rewards, greedy_rewards = [], [], []
        for episode in range(args.episodes_per_epoch):
            record = rng.choice(sequences)
            max_start = len(record[1]) - horizon - 5
            if max_start < 0:
                continue
            starts = [i for i in range(max_start + 1)
                      if record[3][i] and record[2][i, 2] > 0 and record[2][i, 3] > 0]
            if not starts:
                continue
            start_frame = rng.choice(starts)
            sampled_obs, sampled_actions, sampled_reward = policy_rollout(model, policy, record, True,
                                                                          horizon, start_frame, args.action_mode)
            _, _, greedy_reward = policy_rollout(model, policy, record, False, horizon,
                                                 start_frame, args.action_mode)
            if not sampled_obs or sampled_reward is None or greedy_reward is None:
                continue
            obs = torch.stack(sampled_obs).to(args.device)
            hidden = None
            log_probs, entropies = [], []
            action_ids = torch.as_tensor(allowed_actions(args.action_mode), device=args.device)
            for step, action in enumerate(sampled_actions):
                logits, hidden = policy(obs[step:step + 1], hidden)
                valid_logits = logits.index_select(-1, action_ids)
                distribution = torch.distributions.Categorical(logits=valid_logits)
                local_action = int((action_ids == action).nonzero(as_tuple=False).item())
                log_probs.append(distribution.log_prob(torch.tensor([local_action], device=args.device)))
                entropies.append(distribution.entropy())
            advantage = torch.tensor(sampled_reward - greedy_reward, device=args.device).detach()
            loss = -advantage * torch.cat(log_probs).sum() - 0.001 * torch.cat(entropies).mean()
            if episode % args.batch_size == 0:
                optimizer.zero_grad(set_to_none=True)
            (loss / args.batch_size).backward()
            losses.append(float(loss.detach()))
            sample_rewards.append(sampled_reward)
            greedy_rewards.append(greedy_reward)
            if (episode + 1) % 32 == 0:
                print('epoch %d episode %d/%d policy_loss %.5f sample %.4f greedy %.4f' %
                      (epoch + 1, episode + 1, args.episodes_per_epoch,
                       np.mean(losses), np.mean(sample_rewards), np.mean(greedy_rewards)))
            if (episode + 1) % args.batch_size == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
        if args.episodes_per_epoch % args.batch_size:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        save_policy(args, policy, optimizer, epoch + 1, 'slt')


def main():
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.mode in ('diagnose', 'evaluate'):
        evaluate(args)
    elif args.mode == 'intervene':
        run_intervention(args)
    elif args.mode == 'cache-teacher':
        collect_teacher_cache(args)
    elif args.mode == 'train-supervised':
        train_supervised(args)
    elif args.mode == 'train-slt':
        train_slt(args)
    elif args.mode == 'compare':
        compare_reports(args)
    elif args.mode == 'merge':
        merge_reports(args)
    elif args.mode == 'merge-cache':
        merge_caches(args)


if __name__ == '__main__':
    main()
