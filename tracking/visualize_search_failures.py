"""Produce diagnostic figures and qualified background-motion estimates."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.diagnose_search_failures import geometry, load_rows, write_csv, write_json


def frames_for(root, name):
    return sorted((root / 'val' / name).glob('*.jpg'), key=lambda p: int(p.stem))


def background_motion(previous, current, previous_box, current_box):
    images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in (previous, current)]
    if any(image is None for image in images):
        raise IOError('Unable to decode image pair')
    detector = cv2.ORB_create(nfeatures=2000, fastThreshold=5)
    keys, desc = [], []
    for image, box in zip(images, (previous_box, current_box)):
        mask = np.full(image.shape, 255, dtype=np.uint8)
        x, y, w, h = box
        left, top = max(0, int(x - max(w, 20))), max(0, int(y - max(h, 20)))
        right, bottom = min(image.shape[1], int(x + w + max(w, 20))), min(image.shape[0], int(y + h + max(h, 20)))
        mask[top:bottom, left:right] = 0
        keypoints, descriptors = detector.detectAndCompute(image, mask)
        keys.append(keypoints)
        desc.append(descriptors)
    result = {'quality_pass': False, 'matches': 0, 'inliers': 0, 'inlier_ratio': 0.0,
              'background_consistent': None, 'camera_dx': None, 'camera_dy': None,
              'target_residual_dx': None, 'target_residual_dy': None}
    if any(d is None or len(d) < 2 for d in desc):
        return result
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = [pair[0] for pair in matcher.knnMatch(desc[0], desc[1], k=2)
               if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    result['matches'] = len(matches)
    if len(matches) < 20:
        return result
    source = np.float32([keys[0][m.queryIdx].pt for m in matches])
    destination = np.float32([keys[1][m.trainIdx].pt for m in matches])
    affine, inliers = cv2.estimateAffinePartial2D(source, destination, method=cv2.RANSAC,
                                               ransacReprojThreshold=3, maxIters=2000, confidence=0.99)
    if affine is None or inliers is None:
        return result
    good = inliers.ravel().astype(bool)
    error = np.linalg.norm(source @ affine[:, :2].T + affine[:, 2] - destination, axis=1)
    spatial_coverage = float(cv2.contourArea(cv2.convexHull(source[good])) / images[0].size) if good.sum() >= 3 else 0.0
    result.update(inliers=int(good.sum()), inlier_ratio=float(good.mean()),
                  median_inlier_error=float(np.median(error[good])), spatial_coverage=spatial_coverage)
    result['quality_pass'] = bool(good.sum() >= 20 and good.mean() >= 0.5 and np.median(error[good]) <= 2 and spatial_coverage >= 0.05)
    if not result['quality_pass']:
        return result
    previous_center = np.asarray(previous_box[:2]) + np.asarray(previous_box[2:]) / 2
    current_center = np.asarray(current_box[:2]) + np.asarray(current_box[2:]) / 2
    camera = affine[:, :2] @ previous_center + affine[:, 2] - previous_center
    target_motion = current_center - previous_center
    residual = target_motion - camera
    result.update(camera_dx=float(camera[0]), camera_dy=float(camera[1]),
                  target_residual_dx=float(residual[0]), target_residual_dy=float(residual[1]),
                  gt_motion_norm=float(np.linalg.norm(target_motion)),
                  background_motion_norm=float(np.linalg.norm(camera)),
                  residual_motion_norm=float(np.linalg.norm(residual)),
                  background_consistent=bool(np.linalg.norm(residual) <= 0.3 * max(np.linalg.norm(target_motion), 1)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/search_failure_diagnosis/20261016')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'output/search_policy/baseline_val_v2_full.json')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/antiuav410')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    summary = json.loads((args.output / 'summary.json').read_text())
    events = json.loads((args.output / 'events.json').read_text())
    rows = load_rows(args)
    by_id = {(r['sequence'], r['frame']): r for r in rows}
    baseline = json.loads(args.baseline.read_text())
    exits = [e for e in events if e['kind'] == 'crop_exit' and e['previous_valid'] and by_id[e['sequence'], e['frame']]['iou'] < 0.2]
    previous_good = [e for e in exits if e['previous_iou'] >= 0.5]
    motion = []
    for event in previous_good:
        name, frame = event['sequence'], event['frame']
        frames = frames_for(args.data_root, name)
        boxes = json.loads((args.data_root / 'val' / name / 'IR_label.json').read_text())['gt_rect']
        estimate = background_motion(frames[frame - 1], frames[frame], boxes[frame - 1], boxes[frame])
        motion.append({'sequence': name, 'frame': frame, **estimate})
    write_csv(args.output / 'background_motion.csv', motion)
    write_json(args.output / 'background_motion_summary.json', {
        'evaluated_previous_good_failed_crop_exits': len(motion),
        'quality_pass': sum(m['quality_pass'] for m in motion),
        'background_consistent_within_30percent_gt_displacement': sum(m['background_consistent'] is True for m in motion),
        'method': 'ORB background features excluding expanded GT boxes; ratio test0.75; partial-affine RANSAC3px; >=20inliers, ratio>=0.5, median error<=2px, inlier convex hull>=5percent image area',
        'limits': 'Offline evidence only. Background may contain independently moving objects, parallax or low texture. Quality rejection is unknown, not target-only motion. Consistency is not proof of camera causality.'})

    plots = args.output / 'figures'
    plots.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    counts = [sum(e['previous_iou'] < 0.2 for e in exits),
              sum(0.2 <= e['previous_iou'] < 0.5 for e in exits), len(previous_good)]
    axes[0, 0].bar(['Prev IoU < .2', 'Prev .2-.5', 'Prev >= .5'], counts, color=['#c34b45', '#b28b22', '#31866b'])
    axes[0, 0].set(title='Failed crop exits, valid previous frame', ylabel='Events')
    table = summary['covered_low_signal_table']
    keys = ['presence_high_peak_high', 'presence_high_peak_low', 'presence_low_peak_high', 'presence_low_peak_low']
    axes[0, 1].bar(['P high / S high', 'P high / S low', 'P low / S high', 'P low / S low'], [table.get(k, 0) for k in keys], color=['#31866b', '#4f70ad', '#b28b22', '#c34b45'])
    axes[0, 1].tick_params(axis='x', labelrotation=20)
    axes[0, 1].set(title='Fully covered low-IoU frames', ylabel='Frames')
    groups = [('Previous IoU >= .5', previous_good), ('Previous IoU < .2', [e for e in exits if e['previous_iou'] < 0.2])]
    for label, group in groups:
        if group:
            axes[1, 0].plot(range(30), [sum(e['recovery_confirmation_delay'] is not None and e['recovery_confirmation_delay'] <= k for e in group) / len(group) for k in range(30)], label=label)
    axes[1, 0].set(title='Observed stable-recovery fraction (censoring retained)', xlabel='Frames since crop exit', ylabel='Fraction', ylim=(0, 1))
    axes[1, 0].legend()
    shrink_path = args.output / 'shrink_summary.json'
    if shrink_path.exists():
        shrink = json.loads(shrink_path.read_text())
        for group_key, offset, color in [('uniform', -0.05, '#4f70ad'), ('online_gate_sample', 0.05, '#31866b')]:
            for factor in (2., 3., 4., 5.):
                value = shrink['uniform_and_gate_sample_actions'][group_key + '_factor' + str(factor)]['sequence_paired_delta']
                avg, ci = value['mean'], value['ci95']
                axes[1, 1].errorbar(factor + offset, avg, yerr=[[avg - ci[0]], [ci[1] - avg]], fmt='o', color=color)
        axes[1, 1].axhline(0, color='black', linewidth=0.8)
        axes[1, 1].set(title='Single-step shrink delta, sequence bootstrap CI', xlabel='Factor (blue: uniform; green: online gate)', ylabel='IoU delta')
    fig.savefig(plots / 'diagnostic_overview.png', dpi=170)
    plt.close(fig)

    selected = []
    pools = [('exit_after_good', previous_good),
             ('coverage_before_recovery', sorted([e for e in exits if e['covered_but_low_frames'] > 0], key=lambda e: e['covered_but_low_frames'], reverse=True)),
             ('covered_loss', [e for e in events if e['kind'] == 'sustained_loss' and e['initial_category'] == 'fully_covered'])]
    opportunity_path = args.output / 'shrink_opportunities.csv'
    if opportunity_path.exists():
        import csv
        with opportunity_path.open() as stream:
            opportunities = list(csv.DictReader(stream))
        reliable = [r for r in opportunities if r['successful_localization'] == 'True' and r['previous5_gt_reliable'] == 'True']
        pools.append(('shrink_with_reliable_history', [{'sequence': r['sequence'], 'frame': int(r['frame'])} for r in reliable]))
    for kind, pool in pools:
        used = set()
        for event in pool:
            if event['sequence'] in used:
                continue
            selected.append((kind, event))
            used.add(event['sequence'])
            if len(used) >= 2:
                break
    for number, (kind, event) in enumerate(selected):
        name, frame = event['sequence'], event['frame']
        sequence = baseline['sequences'][name]
        frames = frames_for(args.data_root, name)
        boxes = json.loads((args.data_root / 'val' / name / 'IR_label.json').read_text())['gt_rect']
        predictions = sequence['predictions']
        ids = sorted({max(1, min(len(frames) - 1, frame + d)) for d in (-2, -1, 0, 1, 5, 15)})
        fig, axes = plt.subplots(2, len(ids), figsize=(3 * len(ids), 6), constrained_layout=True)
        for column, index in enumerate(ids):
            image = cv2.cvtColor(cv2.imread(str(frames[index])), cv2.COLOR_BGR2RGB)
            row = by_id[name, index]
            crop = geometry(predictions[index - 1], row['factor'])
            gt, pred = boxes[index], predictions[index]
            for band in (0, 1):
                axis = axes[band, column]
                axis.imshow(image)
                for box, color, label in [(gt, '#42e055', 'GT'), (pred, '#ff4747', 'Prediction'), (crop[:2] + [crop[2], crop[2]], '#32c7ff', 'Crop')]:
                    axis.add_patch(Rectangle(box[:2], box[2], box[3], fill=False, edgecolor=color, linewidth=1.2, label=label))
                if band == 1:
                    x, y, side = crop
                    x0, y0 = min(x, gt[0]) - 10, min(y, gt[1]) - 10
                    x1, y1 = max(x + side, gt[0] + gt[2]) + 10, max(y + side, gt[1] + gt[3]) + 10
                    axis.set_xlim(max(0, x0), min(image.shape[1], x1))
                    axis.set_ylim(min(image.shape[0], y1), max(0, y0))
                axis.set_xticks([])
                axis.set_yticks([])
            axes[0, column].set_title('t=%d IoU=%.2f\ncover=%.2f factor=%.0f' % (index, row['iou'], row['coverage'], row['factor']), fontsize=10)
        axes[1, 0].legend(fontsize=8, loc='lower left')
        fig.suptitle('%s | %s | event t=%d' % (kind, name, frame), fontsize=11)
        filename = '%02d_%s.png' % (number, kind)
        fig.savefig(plots / filename, dpi=130)
        plt.close(fig)
    write_json(args.output / 'illustrative_cases.json', [{'kind': kind, 'sequence': e['sequence'], 'frame': e['frame']} for kind, e in selected])
    print('Saved motion checks and', len(selected) + 1, 'figures', flush=True)


if __name__ == '__main__':
    main()
