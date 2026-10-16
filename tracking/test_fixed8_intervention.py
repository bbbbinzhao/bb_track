"""Check the experimental contract without a CUDA tracker."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from lib.test.utils.search_region import SearchRegion
from tracking.search_fixed8_intervention import crop_metrics, rollout_candidate, sample_groups, training_groups
from tracking.analyze_fixed8_intervention import choose, group_statistics
from tracking.train_fixed8_teacher import load_states, match_entropy, train_one


class FakeTracker:
    def restore(self, snapshot):
        self.state = copy.deepcopy(snapshot['state'])
        self.regions = []

    def track(self, image, search_region):
        self.regions.append(search_region)
        self.state[0] += search_region.factor
        self.last_diagnostic = dict(crop=dict(factor=search_region.factor), use_hann=True)


class FixedEightProtocolTests(unittest.TestCase):
    def test_only_first_crop_is_intervened(self):
        model = FakeTracker()
        snapshot = dict(state=[2., 3., 4., 6.])
        images = [np.zeros((100, 100, 3))] * 8
        first, _, _ = rollout_candidate(model, snapshot, images, SearchRegion((5., 6.), 4.))
        self.assertEqual([region.factor for region in model.regions], [4.] + [8.] * 7)
        self.assertEqual(model.regions[1].center_xy, (8., 6.))
        rollout_candidate(model, snapshot, images, SearchRegion((40., 50.), 6.))
        repeated, _, _ = rollout_candidate(model, snapshot, images, SearchRegion((5., 6.), 4.))
        np.testing.assert_array_equal(first, repeated)
        self.assertEqual(snapshot['state'], [2., 3., 4., 6.])

    def test_groups_retain_overlap_and_limit_events(self):
        visible = np.ones(30, dtype=bool)
        visible[15:17] = False
        ious = np.ones(30)
        ious[5:12] = 0
        ious[19:26] = 0
        groups = sample_groups(visible, ious, visible, uniform_points=28, event_limit=1)
        self.assertTrue(groups[4]['failure_sample'])
        self.assertTrue(groups[4]['uniform_sample'])
        self.assertFalse(groups[18]['failure_sample'])
        self.assertTrue(groups[16]['reappearance_sample'])
        self.assertNotIn(0, groups)

    def test_invisible_box_has_no_localization_metrics(self):
        crop = dict(origin_xy=[0, 0], side=20, output_size=256)
        self.assertTrue(all(value is None for value in crop_metrics(
            np.zeros(4), np.ones(4), False, crop).values()))
        result = crop_metrics(np.array([5., 5., 10., 10.]), [5, 5, 10, 10], True, crop)
        self.assertEqual(result['coverage'], 1.)
        self.assertEqual(result['target_pixels'], 128.)

    def test_ties_prefer_unchanged_action(self):
        self.assertEqual(choose([.4] * 9), 2)
        self.assertEqual(choose([.4, .5, .4] + [.3] * 6), 1)

    def test_oracle_statistics_weight_sequences_equally(self):
        row = dict(sequence='a', oracle_delta=.1, baseline_iou8=.4, oracle_iou8=.5,
            one_step_chosen_delta8=.01, one_step_regret8=.09, motion_increment=.01,
            scale_oracle_delta=.09, baseline_coverage=1., oracle_coverage=1.,
            baseline_pixels=10., oracle_pixels=12., next_vs_eight_disjoint=False, best_action=0)
        rows = [row] * 10 + [dict(row, sequence='b', oracle_delta=.3)]
        stats = group_statistics(rows)
        self.assertAlmostEqual(stats['oracle_sequence_mean_delta'], .2)
        self.assertAlmostEqual(stats['oracle_point_mean_delta'], 1.3 / 11)

    def test_train_points_have_a_full_future_without_event_labels(self):
        groups = training_groups(40, 8, 8)
        self.assertEqual(len(groups), 8)
        for frame, flags in groups.items():
            self.assertGreater(frame, 0)
            self.assertLess(frame + 8, 40)
            self.assertFalse(flags['failure_sample'] or flags['reappearance_sample'])
        self.assertEqual(training_groups(9, 8, 8), {})

    def test_multiframe_teacher_loader_rejects_validation_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / 'shard_000_001'
            shard.mkdir()
            (shard / 'manifest.json').write_text(json.dumps(dict(identity=dict(split='val'))))
            with self.assertRaisesRegex(ValueError, 'Only train'):
                load_states(Path(directory))

    def test_matched_teachers_share_sampling_but_change_the_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.set_num_threads(1)
            row = dict(sequence='train', observation=np.ones(32),
                       one=np.asarray([.9, .5, .1] * 3), eight=np.asarray([.1, .5, .9] * 3))
            dev = [dict(row, sequence='development')]
            args = SimpleNamespace(seed=0, action_mode='scale', output=directory,
                                   states_per_epoch=16, batch_size=4, epochs=1)
            identity = dict(checkpoint_sha256='synthetic')
            first = train_one(args, [row] * 8, dev, identity, 'one')
            second = train_one(args, [row] * 8, dev, identity, 'eight')
            a = torch.load(first['checkpoint'], map_location='cpu')
            b = torch.load(second['checkpoint'], map_location='cpu')
            torch.testing.assert_close(a['rng']['generator'], b['rng']['generator'], rtol=0, atol=0)
            self.assertEqual(a['updates'], b['updates'])
            self.assertTrue(any(not torch.equal(a['policy'][key], b['policy'][key]) for key in a['policy']))
            self.assertGreater(first['max_gradient_norm'], 0.)
            np.testing.assert_array_equal(row['observation'], np.ones(32))

    def test_train_entropy_calibration_compensates_reward_averaging(self):
        one = torch.tensor([[.2, .5, .9], [.3, .31, .34]])
        temperature, record = match_entropy(one, one / 8)
        self.assertAlmostEqual(temperature, .1 / 8, places=6)
        self.assertAlmostEqual(record['target_train_entropy'], record['matched_train_entropy'], places=6)


if __name__ == '__main__':
    unittest.main()
