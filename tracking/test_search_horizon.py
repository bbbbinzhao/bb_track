"""Nested visibility, counterfactual reversals and sequence weighting contracts."""
import unittest

import numpy as np

from tracking.analyze_search_horizon import long_only_effects, point_effects, preferred_action, summarize
from tracking.search_horizon_diagnostic import prefix_rewards, sample_points


def candidate(values):
    frames = [dict(iou=value) for value in values]
    return dict(frames=frames, rewards=prefix_rewards(frames))


class HorizonTests(unittest.TestCase):
    def test_rewards_use_nested_prefixes_and_visible_only(self):
        rewards = prefix_rewards([dict(iou=None)] * 8 + [dict(iou=.6)] * 24 + [dict(iou=.2)] * 32)
        self.assertIsNone(rewards['8']['iou'])
        self.assertEqual(rewards['8']['visible_frames'], 0)
        self.assertAlmostEqual(rewards['32']['iou'], .6)
        self.assertAlmostEqual(rewards['64']['iou'], (24 * .6 + 32 * .2) / 56)
        with self.assertRaisesRegex(ValueError, 'complete 64'):
            prefix_rewards([dict(iou=.5)] * 8)

    def test_positive_short_reward_can_have_negative_long_effect(self):
        candidates = [candidate([.8] * 8 + [.1] * 56), candidate([.4] * 64), candidate([.5] * 64)]
        effect = point_effects(candidates, 0)
        self.assertEqual(effect['short_action'], 0)
        self.assertEqual(effect['long_action'], 2)
        self.assertAlmostEqual(effect['short_gain8'], .3)
        self.assertLess(effect['short_gain64'], -.3)
        self.assertEqual(effect['reversal_05'], 1.)
        self.assertGreater(effect['short_regret64'], .3)

    def test_ranking_change_is_not_automatically_harm(self):
        candidates = [candidate([.8] * 8 + [.6] * 56),
                      candidate([.7] * 8 + [.9] * 56), candidate([.5] * 64)]
        effect = point_effects(candidates, 2)
        self.assertEqual(effect['mismatch'], 1.)
        self.assertGreater(effect['short_gain64'], 0.)
        self.assertEqual(effect['short_reversal'], 0.)
        self.assertEqual(effect['policy_gain64'], 0.)
        self.assertGreater(effect['longer_selection_gain64'], 0.)
        self.assertLess(effect['middle_regret64'], effect['short_regret64'])

    def test_no_visible_short_reward_is_excluded_from_paired_comparison(self):
        candidates = [candidate([None] * 8 + [.5] * 56) for _ in range(3)]
        self.assertIsNone(point_effects(candidates, 2))
        self.assertEqual(long_only_effects(candidates, 2), dict(policy_gain64=0., oracle64_gain=0.))
        self.assertIsNone(long_only_effects([candidate([None] * 64)] * 3, 2))

    def test_ties_do_not_count_as_scale_intervention(self):
        self.assertEqual(preferred_action([.5, .5, .5]), 2)
        self.assertEqual(preferred_action([.5, .6, .4]), 1)

    def test_group_statistics_weight_sequences_not_points(self):
        effect = point_effects([candidate([.8] * 64), candidate([.4] * 64), candidate([.5] * 64)], 0)
        rows = [dict(sequence='many', **effect)] * 9
        zero = point_effects([candidate([.4] * 64), candidate([.4] * 64), candidate([.5] * 64)], 2)
        result = summarize(rows + [dict(sequence='one', **zero)])
        self.assertAlmostEqual(result['metrics']['short_gain64']['mean'], .15)
        self.assertEqual(result['points'], 10)
        self.assertEqual(result['sequences'], 2)

    def test_points_share_full_horizon_and_preserve_event_labels(self):
        boxes = np.tile([10., 10., 5., 5.], (240, 1))
        predictions = boxes.copy()
        predictions[60:90, :2] += 80
        groups = sample_points(boxes, np.ones(240, dtype=bool), predictions)
        self.assertEqual(sum(r['uniform'] for r in groups.values()), 4)
        self.assertLessEqual(sum(r['event'] for r in groups.values()), 2)
        self.assertTrue(all(9 <= t and t + 64 < 240 for t in groups))
        self.assertTrue(any(r['event'] and r['failure'] for r in groups.values()))
        self.assertEqual(sample_points(boxes[:60], np.ones(60), predictions[:60]), {})


if __name__ == '__main__':
    unittest.main()
