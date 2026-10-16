"""Tests for the diagnostic distinctions that affect the research conclusions."""
import unittest

import numpy as np

from tracking.diagnose_search_failures import (
    center_inside, confirmation, coverage, event_record, geometry, history_flags, starts,
)


class SearchFailureDiagnosisTests(unittest.TestCase):
    def test_partial_coverage_is_not_center_outside(self):
        crop = [0, 0, 10]
        box = [8, 3, 3, 3]
        self.assertTrue(center_inside(crop, box))
        self.assertLess(coverage(crop, box), 0.999)
        self.assertFalse(center_inside(crop, [9, 3, 2, 3]))

    def test_crop_uses_original_integer_rounding(self):
        self.assertEqual(geometry([1, 3, 4, 4], 2), [-1, 1, 8])

    def test_recovery_needs_three_good_frames_before_censoring(self):
        good = np.array([False, True, True, False, True, True, True])
        self.assertIsNone(confirmation(good, 0, 6))
        self.assertEqual(confirmation(good, 0, 7), 6)

    def test_event_trigger_at_cap_is_not_actual_expansion(self):
        n = 9
        valid = np.ones(n, dtype=bool)
        valid[7] = False
        records = {i: {'presence': 0.1, 'max_score': 0.1, 'search_factor': 8,
                       'next_factor': 8, 'crop': {'side': 80}} for i in range(n)}
        event = event_record('x', 'crop_exit', 1, valid, np.zeros(n), np.ones(n), records)
        self.assertEqual(event['first_trigger_delay'], 0)
        self.assertIsNone(event['first_actual_expansion_delay'])
        self.assertEqual(event['observation_end'], 6)
        self.assertIsNone(event['recovery_confirmation_delay'])

    def test_coverage_is_not_recovery(self):
        n = 8
        records = {i: {'presence': 1, 'max_score': 1, 'search_factor': 6,
                       'next_factor': 6, 'crop': {'side': 60}} for i in range(n)}
        event = event_record('x', 'sustained_loss', 1, np.ones(n, dtype=bool), np.zeros(n), np.ones(n), records)
        self.assertEqual(event['first_full_coverage_delay'], 0)
        self.assertEqual(event['covered_but_low_frames'], 7)
        self.assertIsNone(event['recovery_confirmation_delay'])

    def test_online_history_does_not_access_current_observation_or_gt(self):
        predictions = np.tile([20., 20., 10., 10.], (9, 1))
        records = {i: {'presence': 0.99, 'max_score': 0.9, 'search_factor': 6} for i in range(9)}
        records[6]['presence'] = 0.0
        valid = np.zeros(9, dtype=bool)
        oracle, online = history_flags(predictions, records, np.zeros(9), valid, 6)
        self.assertFalse(oracle)
        self.assertTrue(online)
        records[5]['presence'] = 0.0
        self.assertFalse(history_flags(predictions, records, np.zeros(9), valid, 6)[1])

    def test_online_gate_rejects_unstable_motion_even_when_confident(self):
        predictions = np.tile([20., 20., 10., 10.], (9, 1))
        predictions[5, 0] = 500
        records = {i: {'presence': 0.99, 'max_score': 0.9, 'search_factor': 6} for i in range(9)}
        self.assertFalse(history_flags(predictions, records, np.ones(9), np.ones(9, dtype=bool), 6)[1])

    def test_segment_starts_do_not_count_each_frame_as_event(self):
        self.assertEqual(starts([False, True, True, False, True]).tolist(), [1, 4])

    def test_background_motion_recovers_known_translation(self):
        import cv2
        from pathlib import Path
        import tempfile
        from tracking.visualize_search_failures import background_motion

        image = np.random.RandomState(2).randint(0, 256, (480, 640), dtype=np.uint8)
        translated = cv2.warpAffine(image, np.float32([[1, 0, 15], [0, 1, -8]]), (640, 480))
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / 'a.png', Path(directory) / 'b.png'
            cv2.imwrite(str(first), image)
            cv2.imwrite(str(second), translated)
            result = background_motion(first, second, [200, 200, 20, 10], [215, 192, 20, 10])
        self.assertTrue(result['quality_pass'])
        self.assertAlmostEqual(result['camera_dx'], 15, delta=0.5)
        self.assertAlmostEqual(result['camera_dy'], -8, delta=0.5)
        self.assertTrue(result['background_consistent'])

    def test_textureless_background_is_unknown(self):
        import cv2
        from pathlib import Path
        import tempfile
        from tracking.visualize_search_failures import background_motion

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'blank.png'
            cv2.imwrite(str(path), np.zeros((160, 160), dtype=np.uint8))
            result = background_motion(path, path, [10, 10, 5, 5], [20, 20, 5, 5])
        self.assertFalse(result['quality_pass'])
        self.assertIsNone(result['background_consistent'])


if __name__ == '__main__':
    unittest.main()
