"""Focused checks for experimental validity; no dataset or CUDA required."""
import unittest

import numpy as np
import torch

from tracking.search_policy_experiment import (
    box_iou, failure_starts, reappearance_starts, sequence_metrics, summarize_intervention,
)
from lib.test.utils.search_region import CropGeometry, MotionState, candidate_regions
from lib.train.data.processing_utils import sample_target


class ProtocolTests(unittest.TestCase):
    def test_integer_crop_roundtrip_with_padding(self):
        image = np.zeros((25, 31, 3), dtype=np.uint8)
        image[0, 0] = 255
        box = [-2.3, -1.2, 7.1, 5.6]
        geometry = CropGeometry.from_box(box, 4, 32)
        patch, resize, mask = sample_target(image, box, 4, output_sz=32)
        self.assertEqual(patch.shape, (32, 32, 3))
        self.assertTrue(mask.any())
        self.assertAlmostEqual(resize, 32 / geometry.side)
        self.assertEqual(geometry.origin_xy, (round(1.25 - geometry.side / 2),
                                            round(1.6 - geometry.side / 2)))
        np.testing.assert_allclose(geometry.crop_to_image(geometry.image_to_crop(box)), box)

    def test_failed_runs_are_events(self):
        ious = np.array([1.] + [0.] * 8 + [0.8] + [0.] * 5)
        self.assertEqual(failure_starts(np.ones(len(ious), dtype=bool), ious), [1, 10])

    def test_initialization_and_invalid_gt_excluded(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 0, 0]], dtype=float)
        metrics = sequence_metrics(boxes.copy(), boxes, np.array([True, True]))
        self.assertEqual(metrics['visible_valid_frames'], 0)
        self.assertEqual(metrics['visible_iou'], 0)
        self.assertEqual(box_iou([np.nan, 0, 10, 10], boxes[0]), 0)

    def test_recovery_requires_three_frames_and_respects_new_absence(self):
        boxes = np.tile([0., 0., 10., 10.], (8, 1))
        visible = np.array([True, False, True, False, True, True, True, True])
        metrics = sequence_metrics(boxes, boxes, visible)
        self.assertEqual(metrics['recovery_delays'], [None, 2])

    def test_motion_is_causal_and_displacement_is_bounded(self):
        motion = MotionState()
        motion.update([0, 0, 10, 10], True)
        motion.update([1000, 0, 10, 10], True)
        previous = motion.velocity.copy()
        motion.update([2000, 0, 10, 10], False)
        np.testing.assert_allclose(motion.velocity, previous * 0.8)
        geometry = CropGeometry.from_box([0, 0, 10, 10], 4, 256)
        actions = candidate_regions([0, 0, 10, 10], [1000, 0], geometry, (100, 100, 3))
        self.assertEqual(len(actions), 9)
        self.assertLessEqual(np.linalg.norm(np.asarray(actions[-1].center_xy) - [5, 5]), 10)

    def test_invalid_gt_is_not_a_disappearance(self):
        self.assertEqual(reappearance_starts([True] * 4, [True, False, True, True]), [])
        self.assertEqual(reappearance_starts([True, False, True, True],
                                            [True, False, False, True]), [3])

    def test_gate_uses_failure_samples(self):
        sample = {'baseline_future_iou': 0.1, 'candidates': [{'future_iou': 0.2}],
                  'uniform_sample': True, 'failure_sample': False, 'full_horizon': True}
        summary = summarize_intervention({str(i): [sample] for i in range(20)})
        self.assertFalse(summary['engineering_gate_passed'])
        sample['failure_sample'] = True
        self.assertTrue(summarize_intervention({str(i): [sample] for i in range(20)})[
            'engineering_gate_passed'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
