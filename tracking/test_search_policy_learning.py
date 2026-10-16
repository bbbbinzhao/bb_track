"""Check temporal masks, action restrictions and train/cache boundaries."""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import hashlib

import numpy as np
import torch

from lib.models.search_policy import SearchPolicy
from lib.train.data.search_policy import (
    CACHE_PROTOCOL, WINDOW, WARMUP, load_teacher_windows,
    supervised_policy_loss, window_starts,
)


class LearningProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def batch(self):
        return {'observations': torch.randn(2, WINDOW, 32),
                'action_ious': torch.rand(2, WINDOW, 9),
                'supervision_mask': torch.ones(2, WINDOW, dtype=torch.bool)}

    def test_windows_are_contiguous_and_have_a_next_frame(self):
        self.assertEqual(window_starts(13), [])
        for n in (14, 28, 1000):
            for start in window_starts(n):
                self.assertGreaterEqual(start, 1)
                self.assertLess(start + WINDOW, n)

    def test_invisible_and_warmup_targets_do_not_affect_loss(self):
        data = self.batch()
        data['supervision_mask'][0, 7] = False
        policy = SearchPolicy(recurrent=True)
        ids = torch.tensor([0, 1, 2])
        loss = supervised_policy_loss(policy, data, ids)[0]
        changed = dict(data, action_ious=data['action_ious'].clone())
        changed['action_ious'][:, :WARMUP] = 100
        changed['action_ious'][0, 7] = -100
        other = supervised_policy_loss(policy, changed, ids)[0]
        torch.testing.assert_close(loss, other)

    def test_restricted_teacher_ignores_excluded_actions(self):
        data = self.batch()
        policy = SearchPolicy(recurrent=False)
        ids = torch.tensor([0, 1, 2])
        first = supervised_policy_loss(policy, data, ids)[0]
        data['action_ious'][:, :, 3:] = 1000
        second = supervised_policy_loss(policy, data, ids)[0]
        torch.testing.assert_close(first, second)

    def test_gru_warmup_advances_state_with_gradients(self):
        data = self.batch()
        data['observations'].requires_grad_()
        policy = SearchPolicy(recurrent=True)
        loss = supervised_policy_loss(policy, data, torch.arange(9))[0]
        loss.backward()
        self.assertGreater(float(data['observations'].grad[:, :WARMUP].abs().sum()), 0.)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in policy.parameters()), 0.)

    def test_all_invisible_window_skips_loss(self):
        data = self.batch()
        data['supervision_mask'].zero_()
        self.assertIsNone(supervised_policy_loss(SearchPolicy(), data, torch.arange(9))[0])

    def test_loader_rejects_duplicate_global_sequence_names_and_sparse_time(self):
        with tempfile.TemporaryDirectory() as root:
            names = ['sequence_%d' % index for index in range(10)]
            manifest = {'protocol': CACHE_PROTOCOL, 'complete': True, 'split': 'train',
                        'data_root': root, 'backbone_sha256': 'frozen', 'annotations_sha256': 'annotations',
                        'windows_per_sequence': 1, 'warmup': WARMUP, 'supervision': 8,
                        'teacher_temperature': .1, 'all_training_sequences': names, 'sequences': {}}
            for name in names:
                torch.save({'protocol': CACHE_PROTOCOL, 'sequence_name': name,
                            'observations': np.zeros((1, WINDOW, 32), np.float32),
                            'action_ious': np.ones((1, WINDOW, 9), np.float32),
                            'supervision_mask': np.ones((1, WINDOW), bool),
                            'frame_ids': np.arange(1, WINDOW + 1)[None]}, os.path.join(root, name + '.pt'))
                manifest['sequences'][name] = {'file': name + '.pt'}
            path = os.path.join(root, 'manifest.json')
            with open(path, 'w') as handle:
                json.dump(manifest, handle)
            train, dev, _ = load_teacher_windows([path])
            self.assertFalse({r['sequence_name'] for r in train} & {r['sequence_name'] for r in dev})
            self.assertEqual(len(train) + len(dev), 10)
            with self.assertRaisesRegex(ValueError, 'Duplicate global'):
                load_teacher_windows([path, path])
            sample = torch.load(os.path.join(root, names[0] + '.pt'))
            sample['frame_ids'][0, 5:] += 1
            torch.save(sample, os.path.join(root, names[0] + '.pt'))
            with self.assertRaisesRegex(ValueError, 'consecutive'):
                load_teacher_windows([path])
            manifest['split'] = 'val'
            with open(path, 'w') as handle:
                json.dump(manifest, handle)
            with self.assertRaisesRegex(ValueError, 'cannot train'):
                load_teacher_windows([path])

    def test_training_checkpoint_resume_matches_uninterrupted_updates(self):
        from lib.train.trainers.search_policy_trainer import SearchPolicyTrainer

        with tempfile.TemporaryDirectory() as root:
            checkpoint = os.path.join(root, 'frozen.pth')
            torch.save({'frozen': True}, checkpoint)
            with open(checkpoint, 'rb') as handle:
                weight_hash = hashlib.sha256(handle.read()).hexdigest()
            identity = {'backbone_sha256': weight_hash, 'data_root': root}
            record = {'sequence_name': 'train', 'observations': np.ones((WINDOW, 32), np.float32),
                      'action_ious': np.tile(np.linspace(0, 1, 9, dtype=np.float32), (WINDOW, 1)),
                      'supervision_mask': np.array([False] * WARMUP + [True] * 8),
                      'frame_ids': np.arange(WINDOW)}
            dev = dict(record, sequence_name='development')

            def settings(output, epochs, resume=None):
                return SimpleNamespace(input_manifests=['synthetic'], checkpoint=checkpoint,
                    data_root=root, device='cpu', policy_type='mlp', action_mode='scale',
                    teacher_temperature=.1, seed=0, batch_size=2, episodes_per_epoch=4,
                    output=os.path.join(root, output), epochs=epochs, resume_checkpoint=resume)

            with patch('lib.train.trainers.search_policy_trainer.load_teacher_windows',
                       return_value=([record], [dev], identity)):
                torch.manual_seed(123)
                whole = SearchPolicyTrainer(settings('whole', 2))
                whole.train()
                torch.manual_seed(123)
                first = SearchPolicyTrainer(settings('split', 1))
                first.train()
                resume = os.path.join(root, 'split', 'policy_supervised_seed0_ep001.pth')
                resumed = SearchPolicyTrainer(settings('split', 2, resume))
                resumed.train()
                self.assertEqual(whole.updates, 4)
                self.assertEqual(whole.updates, resumed.updates)
                self.assertGreater(resumed.max_gradient_norm, 0.)
                for name, value in whole.policy.state_dict().items():
                    torch.testing.assert_close(value, resumed.policy.state_dict()[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
