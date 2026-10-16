"""Causal continuation, conservative defaults and matched-budget sampling contracts."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lib.models.search_policy import SearchPolicy, select_search_action
from lib.test.utils.search_region import SearchRegion
from tracking.search_conservative_coverage import branch, groups_for_sequence
from tracking.train_conservative_search import draw_indices, load_cache, train_arm
from tracking.analyze_conservative_search import report_data_root


class FakeTracker:
    def restore(self, snapshot):
        self.state = copy.deepcopy(snapshot['state'])
        self.regions = []

    def track(self, image, search_region):
        self.regions.append(search_region)
        self.state[0] += search_region.factor
        self.last_diagnostic = dict(crop=dict(factor=search_region.factor), use_hann=True)


class ConservativeTests(unittest.TestCase):
    def test_legacy_merged_root_is_resolved_from_verified_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = dict(split='val', baseline_sha256='frozen', protocol='evaluation')
            sources = []
            for i in range(2):
                path = root / ('shard%d.json' % i)
                path.write_text(json.dumps(dict(identity, data_root=str(root), sequences={str(i): {}})))
                sources.append(str(path))
            report = dict(identity, source_reports=sources, sequences={'0': {}, '1': {}})
            self.assertEqual(report_data_root(report), root.resolve())
            self.assertEqual(report_data_root(dict(data_root=str(root))), root.resolve())

    def test_legacy_root_requires_consistent_identity_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'shard.json'
            identity = dict(split='val', baseline_sha256='frozen', protocol='evaluation')
            path.write_text(json.dumps(dict(identity, data_root=str(root), sequences={'0': {}})))
            with self.assertRaisesRegex(ValueError, 'incomplete coverage'):
                report_data_root(dict(identity, source_reports=[str(path)], sequences={'0': {}, '1': {}}))
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                report_data_root(dict(identity, split='test', source_reports=[str(path)], sequences={'0': {}}))

    def test_gate_defaults_and_ties_retain_eight(self):
        ids = torch.tensor([0, 1, 2])
        logits = torch.tensor([[.02, .01, 900.], [-.1, -.2, -900.], [.03, .04, 0.]])
        action = select_search_action(logits, ids, dict(rule='advantage_margin_v1', margin=.02))
        self.assertEqual(action.tolist(), [2, 2, 1])
        disabled = select_search_action(logits, ids, dict(rule='advantage_margin_v1', margin=None))
        self.assertEqual(disabled.tolist(), [2, 2, 2])

    def test_legacy_and_unsupported_sampling(self):
        logits = torch.tensor([[.1, .9, .2]])
        self.assertEqual(select_search_action(logits, torch.tensor([0, 1, 2])).item(), 1)
        with self.assertRaisesRegex(ValueError, 'deterministic'):
            select_search_action(logits, torch.tensor([0, 1, 2]),
                dict(rule='advantage_margin_v1', margin=0.), sample=True)

    def test_branch_continues_current_policy_and_restores_state(self):
        model = FakeTracker()
        snapshot = dict(state=[2., 3., 4., 6.])
        images = [np.zeros((100, 100, 3))] * 8
        with patch('tracking.search_conservative_coverage.next_policy_region',
                   return_value=SearchRegion((5., 6.), 6.)) as select:
            first, _, _ = branch(model, snapshot, images, SearchRegion((5., 6.), 4.), object())
            self.assertEqual([r.factor for r in model.regions], [4.] + [6.] * 7)
            self.assertEqual(select.call_count, 8)
            branch(model, snapshot, images, SearchRegion((5., 6.), 8.), object())
            repeated, _, _ = branch(model, snapshot, images, SearchRegion((5., 6.), 4.), object())
        np.testing.assert_array_equal(first, repeated)
        self.assertEqual(snapshot['state'], [2., 3., 4., 6.])

    def test_event_locations_respect_warmup_and_future(self):
        boxes = np.tile([10., 10., 5., 5.], (60, 1))
        predictions = boxes.copy()
        predictions[25:33, :2] = 70
        exist = np.ones(60, dtype=bool)
        groups = groups_for_sequence(boxes, exist, predictions, event_points=8)
        self.assertTrue(any(f['event'] and f['failure'] for f in groups.values()))
        self.assertLessEqual(sum(f['event'] for f in groups.values()), 8)
        self.assertTrue(all(t >= 9 and t + 8 < 60 for t in groups))
        self.assertEqual(sum(f['uniform'] for f in groups.values()), 8)

    def test_matched_sampling_and_event_balance(self):
        rows = [dict(uniform=True, event=False)] * 8 + [dict(uniform=False, event=True)] * 2
        uniform = draw_indices(rows, 100, torch.Generator().manual_seed(0), False)
        events = draw_indices(rows, 100, torch.Generator().manual_seed(0), True)
        self.assertEqual(len(uniform), len(events))
        self.assertTrue((uniform < 8).all())
        self.assertEqual(int((events >= 8).sum()), 50)

    def test_validation_teachers_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / 'shard_000_001'
            shard.mkdir()
            (shard / 'manifest.json').write_text(json.dumps(dict(identity=dict(split='val'))))
            with self.assertRaisesRegex(ValueError, 'Only versioned train'):
                load_cache(directory)

    def test_regression_gradient_and_checkpoint_gate_roundtrip(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            row = dict(sequence='train', frame=10, uniform=True, event=True,
                observation=np.ones(32, dtype=np.float32), eight=np.array([.9, .5, .6]),
                one=np.array([.9, .5, .6]))
            args = SimpleNamespace(seed=0, output=directory, epochs=1, states_per_epoch=16, batch_size=4)
            report = train_arm(args, 'event', [row] * 8, [dict(row, sequence='dev')],
                               dict(checkpoint_sha256='synthetic'))
            payload = torch.load(report['checkpoint'], map_location='cpu')
            self.assertEqual(payload['updates'], 4)
            self.assertEqual(payload['decision']['rule'], 'advantage_margin_v1')
            self.assertGreater(report['max_gradient_norm'], 0.)
            policy = SearchPolicy(recurrent=False, mlp_depth=2)
            policy.load_state_dict(payload['policy'])
            self.assertTrue(all(torch.isfinite(p).all() for p in policy.parameters()))

    def test_development_labels_do_not_change_training_weights(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            row = dict(sequence='train', frame=10, uniform=True, event=True,
                observation=np.ones(32, dtype=np.float32), eight=np.array([.9, .5, .6]),
                one=np.array([.9, .5, .6]))
            args = SimpleNamespace(seed=0, output=directory, epochs=1, states_per_epoch=16, batch_size=4)
            identity = dict(checkpoint_sha256='synthetic')
            first = train_arm(args, 'first', [row] * 8, [dict(row, sequence='dev')], identity)
            second = train_arm(args, 'second', [row] * 8,
                [dict(row, sequence='dev', eight=np.array([.1, .3, .9]))], identity)
            a = torch.load(first['checkpoint'], map_location='cpu')
            b = torch.load(second['checkpoint'], map_location='cpu')
            for key in a['policy']:
                torch.testing.assert_close(a['policy'][key], b['policy'][key], rtol=0, atol=0)

    def test_continuations_match_optimizer_and_update_budget(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            row = dict(sequence='train', frame=10, uniform=True, event=True,
                observation=np.ones(32, dtype=np.float32), eight=np.array([.9, .5, .6]),
                one=np.array([.9, .5, .6]))
            rows = [row] * 8
            dev = [dict(row, sequence='dev')]
            args = SimpleNamespace(seed=0, output=directory, epochs=1, states_per_epoch=16, batch_size=4)
            identity = dict(checkpoint_sha256='synthetic')
            first = train_arm(args, 'event', rows, dev, identity)
            train_arm(args, 'event_continue', rows, dev, identity, initial=first['checkpoint'])
            train_arm(args, 'aggregate', rows, dev, identity, additional=rows, initial=first['checkpoint'])
            a = torch.load(Path(directory) / 'event_continue/policy_seed0_ep001.pth', map_location='cpu')
            b = torch.load(Path(directory) / 'aggregate/policy_seed0_ep001.pth', map_location='cpu')
            self.assertEqual(a['updates'], 8)
            self.assertEqual(a['updates'], b['updates'])
            for key in a['policy']:
                torch.testing.assert_close(a['policy'][key], b['policy'][key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
