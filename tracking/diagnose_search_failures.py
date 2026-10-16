"""Audit baseline crop failures and replay same-state, same-center shrink tests."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPECTED_WEIGHT = 'c259c341eca7179ff5612a034898678337f9d10c2c8d456ca4aae184de7a0564'


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError('Unsupported JSON value: ' + type(value).__name__)

    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=scalar) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def iou(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    inter = np.maximum(0, np.minimum(a[:2] + a[2:], b[:2] + b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(inter / max(1e-12, a[2:].prod() + b[2:].prod() - inter))


def geometry(box, factor):
    x, y, w, h = map(float, box)
    side = math.ceil(math.sqrt(w * h) * factor)
    return [round(x + w / 2 - side / 2), round(y + h / 2 - side / 2), side]


def coverage(crop, box):
    x, y, side = crop
    gx, gy, w, h = box
    inter = max(0, min(x + side, gx + w) - max(x, gx)) * max(0, min(y + side, gy + h) - max(y, gy))
    return float(inter / max(w * h, 1e-12))


def center_inside(crop, box):
    x, y, side = crop
    cx, cy = np.asarray(box[:2]) + np.asarray(box[2:]) / 2
    return bool(x <= cx < x + side and y <= cy < y + side)


def starts(mask):
    mask = np.asarray(mask, dtype=bool)
    return np.flatnonzero(mask & ~np.r_[False, mask[:-1]])


def confirmation(good, start, end, consecutive=3):
    for index in range(start, end - consecutive + 1):
        if np.all(good[index:index + consecutive]):
            return index + consecutive - 1
    return None


def history_flags(predictions, records, ious, valid, frame, window=5):
    """Both histories stop at frame-1; only the audit flag accesses GT."""
    if frame < window + 1:
        return False, False
    ids = list(range(frame - window, frame))
    oracle = bool(np.all(valid[ids]) and np.all(ious[ids] >= 0.5))
    confident = all(records[j]['presence'] >= 0.8 and records[j]['max_score'] >= 0.5 for j in ids)
    boxes = predictions[ids]
    centers = boxes[:, :2] + boxes[:, 2:] / 2
    side = geometry(predictions[frame - 1], records[frame]['search_factor'])[2]
    small_motion = bool(np.max(np.linalg.norm(np.diff(centers, axis=0), axis=1)) <= side * 0.25)
    areas = np.sqrt(boxes[:, 2] * boxes[:, 3])
    stable_size = bool(areas.max() / max(areas.min(), 1e-12) <= 1.5)
    return oracle, bool(confident and small_motion and stable_size)


def sequence_arrays(args, name, sequence):
    labels = json.loads((args.data_root / 'val' / name / 'IR_label.json').read_text())
    boxes = np.asarray(labels['gt_rect'], dtype=float)
    exist = np.asarray(labels['exist'], dtype=bool)
    predictions = np.asarray(sequence['predictions'], dtype=float)
    assert boxes.shape == predictions.shape and len(exist) == len(boxes), name
    valid = exist & np.isfinite(boxes).all(1) & (boxes[:, 2:] > 0).all(1)
    records = {r['frame']: r for r in sequence['diagnostics']['frames']}
    assert set(records) == set(range(1, len(boxes))), name
    ious = np.array([iou(p, b) if ok else 0.0 for p, b, ok in zip(predictions, boxes, valid)])
    return labels, boxes, exist, valid, predictions, records, ious


def visible_end(valid, start):
    invalid = np.flatnonzero(~valid[start:])
    return start + int(invalid[0]) if len(invalid) else len(valid)


def event_record(name, kind, frame, valid, ious, cov, records):
    end = visible_end(valid, frame)
    recovered = confirmation(ious >= 0.5, frame, end)
    stop = recovered + 1 if recovered is not None else end
    ids = list(range(frame, stop))
    expanded = [j for j in ids if records[j]['next_factor'] > records[j]['search_factor']]
    triggered = [j for j in ids if records[j]['presence'] < 0.8 and records[j]['max_score'] < 0.5]
    covered = [j for j in ids if cov[j] >= 0.999]
    first_cover = covered[0] if covered else None
    first_good = next((j for j in ids if ious[j] >= 0.5), None)
    first_good_after_cover = next((j for j in ids if first_cover is not None and j >= first_cover and ious[j] >= 0.5), None)
    cover_confirm = confirmation(cov >= 0.999, frame, stop)
    return {
        'sequence': name, 'kind': kind, 'frame': frame,
        'observation_end': end - 1, 'censor_reason': 'sequence_end' if end == len(valid) else 'visibility_or_invalid_gt',
        'first_trigger_delay': triggered[0] - frame if triggered else None,
        'first_actual_expansion_delay': expanded[0] - frame if expanded else None,
        'first_full_coverage_delay': first_cover - frame if first_cover is not None else None,
        'first_good_iou_delay': first_good - frame if first_good is not None else None,
        'recovery_confirmation_delay': recovered - frame if recovered is not None else None,
        'recovered_within_30': recovered is not None and recovered - frame < 30,
        'coverage_to_first_good_delay': first_good_after_cover - first_cover if first_good_after_cover is not None else None,
        'coverage_to_recovery_delay': recovered - first_cover if recovered is not None and first_cover is not None else None,
        'stable_coverage_confirmation_delay': cover_confirm - frame if cover_confirm is not None else None,
        'covered_but_low_frames': sum(cov[j] >= 0.999 and ious[j] < 0.2 for j in ids),
        'trigger_frames': len(triggered), 'actual_expansions': len(expanded),
        'trigger_at_max_frames': sum(records[j]['search_factor'] >= 8 for j in triggered),
        'max_factor': max(records[j]['search_factor'] for j in ids),
        'followup_frames': end - frame,
        'minimum_actual_crop_side': min(records[j]['crop']['side'] for j in ids),
        'maximum_actual_crop_side': max(records[j]['crop']['side'] for j in ids),
    }


def prepare(args):
    baseline = json.loads(args.baseline.read_text())
    assert baseline['baseline_sha256'] == EXPECTED_WEIGHT
    assert baseline['split'] == 'val' and len(baseline['sequences']) == 90
    expected = {p.parent.name for p in (args.data_root / 'val').glob('*/IR_label.json')}
    assert set(baseline['sequences']) == expected
    all_rows, events, tasks, sequence_rows = [], [], [], []
    for name, sequence in sorted(baseline['sequences'].items()):
        labels, boxes, exist, valid, predictions, records, ious = sequence_arrays(args, name, sequence)
        cov = np.zeros(len(boxes))
        outside = np.zeros(len(boxes), dtype=bool)
        local_rows = []
        for frame in range(1, len(boxes)):
            record = records[frame]
            factor = record['search_factor']
            crop = geometry(predictions[frame - 1], factor)
            assert crop == record['crop']['origin_xy'] + [record['crop']['side']], (name, frame)
            cov[frame] = coverage(crop, boxes[frame]) if valid[frame] else 0.0
            outside[frame] = valid[frame] and not center_inside(crop, boxes[frame])
            low_p = record['presence'] < 0.8
            low_s = record['max_score'] < 0.5
            expected_factor = min(8, factor + 1) if low_p and low_s else 6
            assert record['next_factor'] == expected_factor, (name, frame, 'SRA transition')
            if frame + 1 < len(boxes):
                assert records[frame + 1]['search_factor'] == expected_factor
                assert records[frame + 1]['use_hann'] == (not (low_p and low_s))
            history_gt, history_online = history_flags(predictions, records, ious, valid, frame)
            gtcenter = boxes[frame, :2] + boxes[frame, 2:] / 2
            prevcenter = predictions[frame - 1, :2] + predictions[frame - 1, 2:] / 2
            previous_valid = bool(valid[frame - 1])
            error = np.zeros(2)
            motion = np.zeros(2)
            cf_center = cf_box = None
            if previous_valid and valid[frame]:
                previous_gt_center = boxes[frame - 1, :2] + boxes[frame - 1, 2:] / 2
                error = prevcenter - previous_gt_center
                motion = gtcenter - previous_gt_center
                corrected = predictions[frame - 1].copy()
                corrected[:2] = previous_gt_center - corrected[2:] / 2
                cf_center = center_inside(geometry(corrected, factor), boxes[frame])
                cf_box = center_inside(geometry(boxes[frame - 1], factor), boxes[frame])
                assert np.allclose(gtcenter - prevcenter, motion - error)
            category = ('outside_center' if outside[frame] else
                        'fully_covered' if cov[frame] >= 0.999 else 'partly_covered') if valid[frame] else 'invalid_or_absent'
            row = {
                'sequence': name, 'frame': frame, 'exist': bool(exist[frame]), 'valid': bool(valid[frame]),
                'iou': float(ious[frame]), 'coverage': float(cov[frame]), 'category': category,
                'presence': record['presence'], 'max_score': record['max_score'],
                'presence_below_threshold': low_p, 'response_below_threshold': low_s,
                'sra_trigger': low_p and low_s, 'factor': factor, 'next_factor': record['next_factor'],
                'crop_side': crop[2], 'hann_used': record['use_hann'],
                'previous_valid': previous_valid, 'previous_iou': float(ious[frame - 1]) if previous_valid else None,
                'previous5_gt_reliable': history_gt, 'previous5_online_gate': history_online,
                'previous_center_error_in_halfcrop': float(np.linalg.norm(error) / (crop[2] / 2)) if previous_valid and valid[frame] else None,
                'gt_displacement_in_halfcrop': float(np.linalg.norm(motion) / (crop[2] / 2)) if previous_valid and valid[frame] else None,
                'previous_gt_center_would_cover_center': cf_center,
                'previous_gt_box_would_cover_center': cf_box,
                'target_pixels': float(np.sqrt(boxes[frame, 2:].prod()) * 256 / crop[2]) if valid[frame] else None,
            }
            for attr in ('OC', 'FM', 'TS', 'LR', 'IC', 'DBC'):
                value = labels.get(attr)
                row['attribute_' + attr] = int(value[frame]) if isinstance(value, list) and len(value) == len(boxes) else None
            local_rows.append(row)
        by_frame = {r['frame']: r for r in local_rows}
        for frame in starts(outside):
            if frame == 0:
                continue
            event = event_record(name, 'crop_exit', int(frame), valid, ious, cov, records)
            event.update({k: by_frame[frame][k] for k in (
                'previous_valid', 'previous_iou', 'previous5_gt_reliable', 'previous5_online_gate',
                'previous_gt_center_would_cover_center', 'previous_gt_box_would_cover_center',
                'previous_center_error_in_halfcrop', 'gt_displacement_in_halfcrop')})
            events.append(event)
        # Non-overlapping sustained-loss events, ending at three consecutive good frames.
        frame = 1
        while frame + 4 < len(boxes):
            if np.all(valid[frame:frame + 5]) and np.all(ious[frame:frame + 5] < 0.2):
                event = event_record(name, 'sustained_loss', frame, valid, ious, cov, records)
                event['initial_category'] = by_frame[frame]['category']
                event['initial_presence_low'] = by_frame[frame]['presence_below_threshold']
                event['initial_response_low'] = by_frame[frame]['response_below_threshold']
                events.append(event)
                delay = event['recovery_confirmation_delay']
                frame = frame + delay + 1 if delay is not None else event['observation_end'] + 1
            else:
                frame += 1
        eligible = [r['frame'] for r in local_rows if r['frame'] >= 6]
        uniform = set(np.asarray(eligible)[np.linspace(0, len(eligible) - 1, min(32, len(eligible))).astype(int)].tolist()) if eligible else set()
        gate_ids = [r['frame'] for r in local_rows if r['previous5_online_gate']]
        gate_sample = set(np.asarray(gate_ids)[np.linspace(0, len(gate_ids) - 1, min(32, len(gate_ids))).astype(int)].tolist()) if gate_ids else set()
        covered_loss = {r['frame'] for r in local_rows if r['valid'] and r['coverage'] >= 0.999 and r['iou'] < 0.2}
        for frame in sorted(uniform | gate_sample | covered_loss):
            row = by_frame[frame]
            tasks.append({'sequence': name, 'frame': frame, 'uniform': frame in uniform,
                          'online_gate_sample': frame in gate_sample, 'covered_failure': frame in covered_loss,
                          'previous5_gt_reliable': row['previous5_gt_reliable'],
                          'previous5_online_gate': row['previous5_online_gate']})
        all_rows += local_rows
        sequence_rows.append({'sequence': name, 'frames': len(boxes),
                              'valid_frames': sum(r['valid'] for r in local_rows),
                              'visible_iou': float(ious[1:][valid[1:]].mean()),
                              'outside_frames': int(outside.sum()), 'covered_failure_frames': len(covered_loss),
                              'has_attribute_OC': 'OC' in labels})
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'frames.csv', all_rows)
    write_csv(args.output / 'events.csv', events)
    write_csv(args.output / 'sequences.csv', sequence_rows)
    write_json(args.output / 'events.json', events)
    write_json(args.output / 'tasks.json', tasks)
    write_json(args.output / 'protocol.json', {
        'version': 1, 'split': 'val', 'sequences': len(sequence_rows), 'baseline': str(args.baseline.resolve()),
        'baseline_report_sha256': sha256(args.baseline), 'frozen_checkpoint_sha256': EXPECTED_WEIGHT,
        'full_coverage_threshold': 0.999, 'failure_iou': 0.2, 'recovery_iou': 0.5,
        'recovery_consecutive_frames': 3, 'failure_consecutive_frames': 5,
        'frame_index': 'zero-based; initialization excluded', 'shrink_factors': args.factors,
        'task_counts': {k: sum(t[k] for t in tasks) for k in ('uniform', 'online_gate_sample', 'covered_failure')},
        'task_total_deduplicated': len(tasks), 'history_window': 5,
        'online_gate': 'previous five presence>=0.8 and peak>=0.5; center steps<=0.25 current baseline crop side; sqrt-area max/min<=1.5; no current-frame output or GT',
        'causal_interpretation': 'descriptive decomposition and controlled crop intervention; no attribution of motion to camera/target or occlusion without additional evidence',
    })
    print('Prepared', len(all_rows), 'frames,', len(events), 'events,', len(tasks), 'tasks', flush=True)


def replay_state(model, sequence, records, frame, factor):
    # In the original tracker the template is fixed and these are all temporal inputs.
    model.state = list(sequence['predictions'][frame - 1])
    model.former_box = list(model.state)
    model.frame_id = frame - 1
    model.search_factor = factor
    model.use_hann = bool(records[frame]['use_hann'])
    model.flag = records[frame - 1]['presence'] if frame > 1 else 1.0


def tensor_state_hash(network):
    h = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def intervene(args):
    import cv2
    import torch
    from lib.test.parameter.focustrack import parameters
    from lib.test.tracker.focustrack import FocusTrack

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.cuda.set_device(args.device)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    params = parameters('focustrack_stage2', test_epoch=10)
    params.debug = 0
    params.checkpoint = str(args.checkpoint.resolve())
    assert sha256(args.checkpoint) == EXPECTED_WEIGHT
    model = FocusTrack(params, 'antiuav410_val')
    assert not model.use_grid and model.cfg.TEST.USE_REGION_ADJUST
    model.network.requires_grad_(False).eval()
    frozen_before = tensor_state_hash(model.network)
    baseline = json.loads(args.baseline.read_text())
    protocol = json.loads((args.output / 'protocol.json').read_text())
    assert sha256(args.baseline) == protocol['baseline_report_sha256']
    assert args.factors == protocol['shrink_factors']
    tasks = json.loads((args.output / 'tasks.json').read_text())
    names = sorted({t['sequence'] for t in tasks})[args.shard::args.shards]
    destination = args.output / 'interventions' / (('smoke_' if args.max_tasks else '') + 'shard_%02d.json' % args.shard)
    assert not destination.exists(), 'Existing results must not be overwritten; use a fresh output directory'
    outputs, audits = [], []
    captured = {}
    cls_hook = model.network.cls_head.register_forward_hook(
        lambda module, inputs, output: captured.update(presence=float(output.softmax(1)[0, -1])))
    box_hook = model.network.box_head.register_forward_hook(
        lambda module, inputs, output: captured.update(peak=float(output[0].max())))

    def read_image(path):
        image = cv2.imread(str(path))
        if image is None:
            raise IOError(str(path))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        for number, name in enumerate(names, 1):
            sequence = baseline['sequences'][name]
            _, boxes, _, valid, _, records, _ = sequence_arrays(args, name, sequence)
            frames = sorted((args.data_root / 'val' / name).glob('*.jpg'), key=lambda p: int(p.stem))
            assert len(frames) == len(boxes)
            model.initialize(read_image(frames[0]), {'init_bbox': boxes[0].tolist()})
            sequence_tasks = [t for t in tasks if t['sequence'] == name]
            if args.max_tasks:
                sequence_tasks = sequence_tasks[:args.max_tasks]
            max_error = 0.0
            for task_number, task in enumerate(sequence_tasks):
                frame = task['frame']
                record = records[frame]
                image = read_image(frames[frame])
                factor = record['search_factor']
                replay_state(model, sequence, records, frame, factor)
                reference = model.track(image)['target_bbox']
                error = float(np.max(np.abs(np.asarray(reference) - sequence['predictions'][frame])))
                max_error = max(max_error, error)
                assert error <= 1e-4, (name, frame, 'baseline replay mismatch', error)
                assert abs(captured['presence'] - record['presence']) < 1e-5
                assert abs(captured['peak'] - record['max_score']) < 1e-5
                assert model.search_factor == record['next_factor']
                result = dict(task, baseline_iou=iou(reference, boxes[frame]) if valid[frame] else None,
                              baseline_factor=factor, baseline_coverage=coverage(geometry(sequence['predictions'][frame - 1], factor), boxes[frame]) if valid[frame] else None,
                              valid=bool(valid[frame]), candidates=[])
                for candidate_factor in args.factors:
                    if candidate_factor >= factor:
                        continue
                    replay_state(model, sequence, records, frame, candidate_factor)
                    prediction = model.track(image)['target_bbox']
                    crop = geometry(sequence['predictions'][frame - 1], candidate_factor)
                    result['candidates'].append({
                        'factor': candidate_factor, 'prediction': list(prediction), 'crop': crop,
                        'iou': iou(prediction, boxes[frame]) if valid[frame] else None,
                        'coverage': coverage(crop, boxes[frame]) if valid[frame] else None,
                        'target_pixels': float(np.sqrt(boxes[frame, 2:].prod()) * 256 / crop[2]) if valid[frame] else None,
                        'presence': captured['presence'], 'peak': captured['peak'],
                        'hann_used': bool(record['use_hann']),
                    })
                if task_number == 0:
                    replay_state(model, sequence, records, frame, factor)
                    repeated = model.track(image)['target_bbox']
                    assert np.allclose(reference, repeated, atol=1e-7, rtol=0), 'A/B/A state contamination'
                outputs.append(result)
            audits.append({'sequence': name, 'replayed_tasks': len(sequence_tasks), 'max_bbox_error': max_error,
                           'a_b_a_check': True})
            print('shard', args.shard, number, '/', len(names), name, 'tasks', len(sequence_tasks), 'error', max_error, flush=True)
    cls_hook.remove()
    box_hook.remove()
    frozen_after = tensor_state_hash(model.network)
    assert frozen_before == frozen_after
    write_json(destination, {'complete': not args.max_tasks, 'shard': args.shard, 'shards': args.shards,
                             'protocol_sha256': sha256(args.output / 'protocol.json'),
                             'network_state_unchanged': True, 'network_state_sha256': frozen_after,
                             'sequence_audits': audits, 'results': outputs})


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def event_summary(events):
    recovered = [e for e in events if e['recovery_confirmation_delay'] is not None]
    return {'events': len(events), 'sequences': len({e['sequence'] for e in events}),
            'recovery_rate_30': mean([e['recovered_within_30'] for e in events]),
            'rate30_is_observed_lower_bound_if_early_censored': True,
            'unrecovered_censored_before_30': sum(e['followup_frames'] < 30 and not e['recovered_within_30'] for e in events),
            'recovered_before_censoring': len(recovered),
            'recovery_confirmation_mean_success_only': mean([e['recovery_confirmation_delay'] for e in recovered]),
            'recovery_confirmation_median_success_only': float(np.median([e['recovery_confirmation_delay'] for e in recovered])) if recovered else None,
            'trigger_first_frame_rate': mean([e['first_trigger_delay'] == 0 for e in events]),
            'never_triggered_before_recovery_or_censoring': sum(e['first_trigger_delay'] is None for e in events),
            'actual_expansion_first_frame_rate': mean([e['first_actual_expansion_delay'] == 0 for e in events]),
            'first_trigger_mean_detected_only': mean([e['first_trigger_delay'] for e in events]),
            'first_full_coverage_mean_observed_only': mean([e['first_full_coverage_delay'] for e in events]),
            'ever_full_coverage': sum(e['first_full_coverage_delay'] is not None for e in events),
            'fully_covered_at_least_once_but_not_recovered': sum(e['first_full_coverage_delay'] is not None and e['recovery_confirmation_delay'] is None for e in events),
            'coverage_to_first_good_mean_observed_only': mean([e['coverage_to_first_good_delay'] for e in events]),
            'coverage_to_recovery_mean_recovered_only': mean([e['coverage_to_recovery_delay'] for e in events]),
            'covered_but_low_events': sum(e['covered_but_low_frames'] > 0 for e in events)}


def paired_sequence_ci(rows, key, repetitions=10000):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['sequence'], []).append(row[key])
    means = np.array([np.mean(values) for values in grouped.values()])
    if not len(means):
        return {'sequences': 0, 'mean': None, 'ci95': None}
    rng = np.random.RandomState(0)
    samples = np.array([rng.choice(means, len(means), replace=True).mean() for _ in range(repetitions)])
    return {'sequences': len(means), 'mean': float(means.mean()), 'ci95': np.percentile(samples, [2.5, 97.5]).tolist()}


def load_rows(args):
    with (args.output / 'frames.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    bool_keys = ('valid', 'exist', 'previous_valid', 'previous5_gt_reliable', 'previous5_online_gate',
                 'presence_below_threshold', 'response_below_threshold', 'sra_trigger',
                 'previous_gt_center_would_cover_center', 'previous_gt_box_would_cover_center')
    for row in rows:
        for key in bool_keys:
            row[key] = {'True': True, 'False': False, '': None}[row[key]]
        for key in ('iou', 'coverage', 'previous_iou', 'factor', 'next_factor', 'target_pixels', 'presence', 'max_score'):
            row[key] = float(row[key]) if row[key] else None
        row['frame'] = int(row['frame'])
    return rows


def report(args):
    rows = load_rows(args)
    events = json.loads((args.output / 'events.json').read_text())
    valid = [r for r in rows if r['valid']]
    low = [r for r in valid if r['iou'] < 0.2]
    covered = [r for r in low if r['coverage'] >= 0.999]
    exits = [e for e in events if e['kind'] == 'crop_exit']
    by_id = {(r['sequence'], r['frame']): r for r in rows}
    failed_visible_exits = [e for e in exits if e['previous_valid'] and by_id[e['sequence'], e['frame']]['iou'] < 0.2]
    failed_exit_good = [e for e in failed_visible_exits if e['previous_iou'] >= 0.5]
    failed_exit_bad = [e for e in failed_visible_exits if e['previous_iou'] < 0.2]
    prev_good = [e for e in exits if e['previous_iou'] is not None and e['previous_iou'] >= 0.5]
    prev_bad = [e for e in exits if e['previous_iou'] is not None and e['previous_iou'] < 0.2]
    summary = {
        'valid_frames': len(valid), 'low_iou_frames': len(low),
        'low_iou_category_counts': dict(Counter(r['category'] for r in low)),
        'covered_low_frames': len(covered), 'covered_low_95percent': sum(r['coverage'] >= 0.95 for r in low),
        'covered_low_binary_presence_correct_at_05': sum(r['presence'] >= 0.5 for r in covered),
        'covered_low_binary_presence_wrong_at_05': sum(r['presence'] < 0.5 for r in covered),
        'covered_low_presence_between_05_and_08': sum(0.5 <= r['presence'] < 0.8 for r in covered),
        'covered_low_signal_table': dict(Counter('presence_%s_peak_%s' % (
            'low' if r['presence_below_threshold'] else 'high', 'low' if r['response_below_threshold'] else 'high') for r in covered)),
        'outside_low_signal_table': dict(Counter('presence_%s_peak_%s' % (
            'low' if r['presence_below_threshold'] else 'high', 'low' if r['response_below_threshold'] else 'high') for r in low if r['category'] == 'outside_center')),
        'crop_exit_events': event_summary(exits),
        'failed_crop_exits_with_valid_previous_frame': event_summary(failed_visible_exits),
        'failed_crop_exits_previous_good': event_summary(failed_exit_good),
        'failed_crop_exits_previous_bad': event_summary(failed_exit_bad),
        'failed_crop_exits_previous_good_gt_previous_box_still_misses': sum(e['previous_gt_box_would_cover_center'] is False for e in failed_exit_good),
        'failed_crop_exits_previous_bad_gt_previous_center_covers': sum(e['previous_gt_center_would_cover_center'] is True for e in failed_exit_bad),
        'crop_exit_previous_good': event_summary(prev_good),
        'crop_exit_previous_bad': event_summary(prev_bad),
        'crop_exit_previous_missing': sum(not e['previous_valid'] for e in exits),
        'previous_bad_exits_corrected_previous_center_covers': sum(e['previous_gt_center_would_cover_center'] is True for e in prev_bad),
        'previous_good_exits_corrected_previous_box_still_misses': sum(e['previous_gt_box_would_cover_center'] is False for e in prev_good),
        'covered_low_previous5_gt_reliable': sum(r['previous5_gt_reliable'] for r in covered),
        'covered_low_previous5_online_gate': sum(r['previous5_online_gate'] for r in covered),
        'sustained_losses': event_summary([e for e in events if e['kind'] == 'sustained_loss']),
        'covered_sustained_losses': event_summary([e for e in events if e['kind'] == 'sustained_loss' and e['initial_category'] == 'fully_covered']),
        'covered_sustained_losses_both_signals_low': event_summary([e for e in events if e['kind'] == 'sustained_loss' and e['initial_category'] == 'fully_covered' and e['initial_presence_low'] and e['initial_response_low']]),
    }
    attributes = {}
    for attr in ('OC', 'FM', 'TS', 'LR', 'IC', 'DBC'):
        for value in ('0', '1'):
            group = [r for r in valid if r['attribute_' + attr] == value]
            attributes[attr + '=' + value] = {'frames': len(group), 'sequences': len({r['sequence'] for r in group}),
                'low_iou_rate': mean([r['iou'] < 0.2 for r in group]),
                'covered_low_rate': mean([r['iou'] < 0.2 and r['coverage'] >= 0.999 for r in group])}
    summary['attribute_associations_in_available_annotations'] = attributes
    covered_by_sequence = []
    for name in sorted({r['sequence'] for r in rows}):
        local = [r for r in valid if r['sequence'] == name]
        signals = [r for r in covered if r['sequence'] == name]
        covered_by_sequence.append({'sequence': name, 'valid_frames': len(local),
            'covered_low_frames': len(signals), 'covered_low_rate': len(signals) / len(local) if local else 0,
            'covered_low_trigger_frames': sum(r['sra_trigger'] for r in signals),
            'covered_low_previous5_gt_reliable': sum(r['previous5_gt_reliable'] for r in signals),
            'covered_low_previous5_online_gate': sum(r['previous5_online_gate'] for r in signals)})
    write_csv(args.output / 'covered_failures_per_sequence.csv', covered_by_sequence)
    write_json(args.output / 'summary.json', summary)
    files = sorted((args.output / 'interventions').glob('shard_*.json'))
    interventions = []
    if files:
        payloads = [json.loads(p.read_text()) for p in files]
        assert all(p['complete'] and p['network_state_unchanged'] for p in payloads)
        assert {p['shard'] for p in payloads} == set(range(payloads[0]['shards']))
        assert all(p['protocol_sha256'] == sha256(args.output / 'protocol.json') for p in payloads)
        interventions = [r for p in payloads for r in p['results']]
        expected = json.loads((args.output / 'tasks.json').read_text())
        assert len(interventions) == len(expected)
        assert {(r['sequence'], r['frame']) for r in interventions} == {(r['sequence'], r['frame']) for r in expected}
        assert len({(r['sequence'], r['frame']) for r in interventions}) == len(expected)
        opportunities = []
        candidate_rows = []
        for result in interventions:
            if not result['valid']:
                continue
            for candidate in result['candidates']:
                candidate_rows.append({k: result[k] for k in ('sequence', 'frame', 'uniform', 'online_gate_sample', 'covered_failure', 'previous5_gt_reliable', 'previous5_online_gate')} | {
                    'factor': candidate['factor'], 'baseline_iou': result['baseline_iou'], 'candidate_iou': candidate['iou'],
                    'delta': candidate['iou'] - result['baseline_iou'], 'candidate_coverage': candidate['coverage'],
                    'baseline_coverage': result['baseline_coverage'], 'presence': candidate['presence'], 'peak': candidate['peak']})
            if result['covered_failure']:
                feasible = [c for c in result['candidates'] if c['coverage'] >= 0.999]
                best = max(feasible, key=lambda c: c['iou']) if feasible else None
                opportunities.append({k: result[k] for k in ('sequence', 'frame', 'previous5_gt_reliable', 'previous5_online_gate')} | {
                    'baseline_iou': result['baseline_iou'], 'has_smaller_full_cover_crop': bool(feasible),
                    'best_factor': best['factor'] if best else None, 'best_iou': best['iou'] if best else None,
                    'gain_005': best is not None and best['iou'] - result['baseline_iou'] >= 0.05,
                    'successful_localization': best is not None and best['iou'] >= 0.5})
        shrink = {'all_covered_failures': len(opportunities), 'feasible_shrink': sum(r['has_smaller_full_cover_crop'] for r in opportunities),
                  'oracle_gain_005': sum(r['gain_005'] for r in opportunities),
                  'oracle_success_iou05': sum(r['successful_localization'] for r in opportunities),
                  'oracle_success_sequences': len({r['sequence'] for r in opportunities if r['successful_localization']}),
                  'oracle_success_previous5_gt_reliable': sum(r['successful_localization'] and r['previous5_gt_reliable'] for r in opportunities),
                  'oracle_success_previous5_online_gate': sum(r['successful_localization'] and r['previous5_online_gate'] for r in opportunities),
                  'oracle_success_both_histories': sum(r['successful_localization'] and r['previous5_gt_reliable'] and r['previous5_online_gate'] for r in opportunities),
                  'oracle_success_previous5_gt_reliable_sequences': sorted({r['sequence'] for r in opportunities if r['successful_localization'] and r['previous5_gt_reliable']}),
                  'oracle_success_previous5_online_gate_sequences': sorted({r['sequence'] for r in opportunities if r['successful_localization'] and r['previous5_online_gate']}),
                  'covered_failure_previous5_gt_reliable_total': sum(r['previous5_gt_reliable'] for r in opportunities),
                  'covered_failure_previous5_online_gate_total': sum(r['previous5_online_gate'] for r in opportunities),
                  'history_window_sensitivity_descriptive': {},
                  'uniform_and_gate_sample_actions': {}}
        for window in (1, 3, 5):
            counts = Counter()
            for result in opportunities:
                sequence_rows = [by_id.get((result['sequence'], t)) for t in range(result['frame'] - window, result['frame'])]
                if not all(r is not None for r in sequence_rows):
                    continue
                gt_ok = all(r['valid'] and r['iou'] >= 0.5 for r in sequence_rows)
                confident = all(r['presence'] >= 0.8 and r['max_score'] >= 0.5 for r in sequence_rows)
                counts['covered_failures_gt_reliable'] += gt_ok
                counts['rescued_gt_reliable'] += gt_ok and result['successful_localization']
                counts['covered_failures_both_confident'] += confident
                counts['rescued_both_confident'] += confident and result['successful_localization']
            shrink['history_window_sensitivity_descriptive'][str(window)] = dict(counts)
        for group_key in ('uniform', 'online_gate_sample'):
            for factor in args.factors:
                group = [r for r in candidate_rows if r[group_key] and r['factor'] == factor]
                shrink['uniform_and_gate_sample_actions'][group_key + '_factor' + str(factor)] = {
                    'frames': len(group), 'sequence_paired_delta': paired_sequence_ci(group, 'delta'),
                    'harm_ge005_rate': mean([r['delta'] <= -0.05 for r in group]),
                    'gain_ge005_rate': mean([r['delta'] >= 0.05 for r in group]),
                    'coverage_lost_rate': mean([r['baseline_coverage'] >= 0.999 and r['candidate_coverage'] < 0.999 for r in group]),
                    'good_to_low_iou_rate': mean([r['baseline_iou'] >= 0.5 and r['candidate_iou'] < 0.2 for r in group])}
                retained = [r for r in group if r['baseline_coverage'] >= 0.999 and r['candidate_coverage'] >= 0.999]
                shrink['uniform_and_gate_sample_actions'][group_key + '_factor' + str(factor)]['both_crops_fully_cover'] = {
                    'frames': len(retained), 'sequence_paired_delta': paired_sequence_ci(retained, 'delta'),
                    'harm_ge005_rate': mean([r['delta'] <= -0.05 for r in retained])}
        write_csv(args.output / 'shrink_candidates.csv', candidate_rows)
        write_csv(args.output / 'shrink_opportunities.csv', opportunities)
        write_json(args.output / 'shrink_summary.json', shrink)
        summary['shrink'] = shrink
    text = '# 搜索裁剪失败、SRA 响应与缩小裁剪诊断\n\n'
    text += '完整官方 val：90 条序列；复用已审计的原 SRA 轨迹，使用原始 exist/GT 重新计算。初始化帧排除。\n\n'
    text += '当前帧失败 IoU<0.2；完整覆盖为框覆盖率≥0.999；连续五帧失败触发失效事件，连续三帧 IoU≥0.5 确认恢复。恢复观察在目标再次消失、GT 非法或序列结束时截断。\n\n'
    text += 'SRA 同时要求 presence<0.8 和原始峰值<0.5；触发不等于实际扩大，因子可能已到 8。因子上升也不保证实际像素边长上升，因为预测框始终更新。\n\n'
    text += '## 完整统计\n\n```json\n' + json.dumps(summary, indent=2, ensure_ascii=False) + '\n```\n\n'
    text += '## 解释边界\n\n'
    text += '- 裁剪出界起点的前一帧 IoU 与 GT 位移用于分解误差；GT 修正裁剪是离线几何参照，不是在线策略。前帧 IoU 高也不等于中心、尺度完全正确。\n'
    text += '- 图像坐标位移不能单独区分目标运动与相机运动。遮挡等属性仅在存在标注的帧上分析，属性与失败共现不证明因果。\n'
    text += '- 中心出界事件中含未发生定位失败、以及重新出现的情况；针对问题一应优先看 failed_crop_exits_with_valid_previous_frame 及其前帧好/坏分层。事件可能来自相同失效轨迹，不能作为独立序列样本。\n'
    text += '- 存在概率高说明分支判定搜索图有目标，不证明选中了正确目标；低于 0.8 是控制阈值，不等于二分类 argmax 错误。\n'
    text += '- 缩小实验保持原预测中心、模板、原坐标还原与当前 Hann 状态。每个分支先复原上一帧状态；原裁剪逐任务复现、A/B/A、冻结参数和缓冲哈希均检查。\n'
    text += '- GT 选择缩小后仍完整覆盖且 IoU 最高的候选为 oracle。缩小同时改变背景干扰和目标像素大小，不能据此单独归因于分辨率。\n'
    text += '- 前五帧 GT IoU≥0.5 是事后可靠性；在线 gate 仅用截至前一帧的置信度和框变化。在线 gate 通过也可能稳定跟错。\n'
    text += '- 正常帧对照按每序列最多 32 个均匀时刻和最多 32 个 gate 时刻取样，分开报告；失效帧全量枚举。不能将事件样本收益当作整体指标或闭环收益。\n'
    (args.output / 'report.md').write_text(text)
    print(json.dumps({k: v for k, v in summary.items() if k != 'attribute_associations_in_available_annotations'}, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'intervene', 'report'))
    parser.add_argument('--baseline', type=Path, default=ROOT / 'output/search_policy/baseline_val_v2_full.json')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/antiuav410')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'output/checkpoints/train/focustrack/focustrack_stage2/FocusTrack_ep0010.pth.tar')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/search_failure_diagnosis/20261016')
    parser.add_argument('--factors', type=float, nargs='+', default=[2., 3., 4., 5.])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--max-tasks', type=int, default=0, help='Smoke checks only; report refuses incomplete results')
    args = parser.parse_args()
    assert 0 <= args.shard < args.shards
    {'prepare': prepare, 'intervene': intervene, 'report': report}[args.mode](args)


if __name__ == '__main__':
    main()
