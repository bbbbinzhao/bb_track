"""Small policy trainer with independent checkpoints and train-only holdout."""
import hashlib
import json
import os
import random

import numpy as np
import torch

from lib.models.search_policy import SearchPolicy
from lib.train.actors.search_policy import SearchPolicyActor
from lib.train.admin.stats import AverageMeter
from lib.train.data.search_policy import (
    collate_windows, load_teacher_windows, supervised_policy_loss,
)
from lib.test.utils.search_region import allowed_actions


class SearchPolicyTrainer:
    """Unlike LTRTrainer, batches contain causal observations, not template_images."""

    def __init__(self, settings):
        self.settings = settings
        if not settings.input_manifests:
            raise ValueError('Training requires complete train-only cache manifests')
        self.records, self.development, self.cache_identity = load_teacher_windows(settings.input_manifests)
        if not self.records or not self.development:
            raise ValueError('Both train and sequence-disjoint development windows are required')
        if settings.episodes_per_epoch < 1 or settings.batch_size < 1:
            raise ValueError('Invalid training batch/episode count')
        with open(settings.checkpoint, 'rb') as handle:
            self.backbone_sha256 = hashlib.sha256(handle.read()).hexdigest()
        if self.backbone_sha256 != self.cache_identity['backbone_sha256']:
            raise ValueError('Frozen tracker differs from teacher cache')
        if os.path.realpath(settings.data_root) != self.cache_identity['data_root']:
            raise ValueError('Training data root differs from teacher cache')
        self.device = torch.device(settings.device)
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        self.policy = SearchPolicy(recurrent=settings.policy_type == 'gru', mlp_depth=2).to(self.device)
        self.actor = SearchPolicyActor(self.policy, settings.action_mode, settings.teacher_temperature)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=1e-4, weight_decay=1e-4)
        self.generator = torch.Generator().manual_seed(settings.seed)
        self.epoch = 0
        self.updates = 0
        self.history = []
        self.max_gradient_norm = 0.
        self.action_ids = torch.tensor(allowed_actions(settings.action_mode), device=self.device)
        # No valid teacher in a clip means no update, but those frames remain in the cache.
        self.trainable = [record for record in self.records if record['supervision_mask'].any()]
        if not self.trainable:
            raise ValueError('No visible localization teachers in training windows')
        if settings.resume_checkpoint:
            self.restore(settings.resume_checkpoint)

    def checkpoint_identity(self):
        return {'recurrent': self.policy.recurrent, 'mlp_depth': self.policy.mlp_depth,
                'action_mode': self.settings.action_mode, 'seed': self.settings.seed,
                'backbone_sha256': self.backbone_sha256,
                'cache_identity': self.cache_identity,
                'episodes_per_epoch': self.settings.episodes_per_epoch,
                'batch_size': self.settings.batch_size,
                'teacher_temperature': self.settings.teacher_temperature}

    def save(self):
        path = os.path.join(self.settings.output, 'policy_supervised_seed%d_ep%03d.pth' % (
            self.settings.seed, self.epoch))
        payload = dict(self.checkpoint_identity(),
            protocol='supervised_search_policy_v1', policy=self.policy.state_dict(),
            optimizer=self.optimizer.state_dict(), epoch=self.epoch, updates=self.updates,
            history=self.history, max_gradient_norm=self.max_gradient_norm,
            backbone_checkpoint=os.path.abspath(self.settings.checkpoint),
            rng={'python': random.getstate(), 'numpy': np.random.get_state(),
                 'torch': torch.get_rng_state(), 'generator': self.generator.get_state(),
                 'cuda': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else []})
        torch.save(payload, path + '.tmp')
        os.replace(path + '.tmp', path)
        return path

    def restore(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        if checkpoint.get('protocol') != 'supervised_search_policy_v1':
            raise ValueError('Cannot resume a legacy smoke checkpoint')
        if any(checkpoint.get(key) != value for key, value in self.checkpoint_identity().items()):
            raise ValueError('Checkpoint training identity differs')
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch, self.updates = checkpoint['epoch'], checkpoint['updates']
        self.history = checkpoint['history']
        self.max_gradient_norm = checkpoint['max_gradient_norm']
        rng = checkpoint['rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        self.generator.set_state(rng['generator'])
        if self.device.type == 'cuda' and rng['cuda']:
            torch.cuda.set_rng_state_all(rng['cuda'])

    @torch.no_grad()
    def development_metrics(self):
        self.policy.eval()
        scores, predicted, losses = [], [], []
        for start in range(0, len(self.development), 32):
            records = self.development[start:start + 32]
            data = collate_windows(records, self.device)
            loss, logits, values, mask = supervised_policy_loss(
                self.policy, data, self.action_ids, self.settings.teacher_temperature)
            if loss is None:
                continue
            scores.append(values[mask].cpu())
            predicted.append(logits.argmax(-1)[mask].cpu())
            losses.extend([float(loss)] * int(mask.sum()))
        if not scores:
            return {'valid_steps': 0}
        scores, predicted = torch.cat(scores), torch.cat(predicted)
        chosen = scores.gather(1, predicted[:, None]).squeeze(1)
        oracle = scores.max(-1).values
        fixed_means = scores.mean(0)
        selected_actions = self.action_ids.cpu()[predicted]
        return {'valid_steps': len(chosen), 'loss': float(np.mean(losses)),
                'selected_iou': float(chosen.mean()), 'oracle_iou': float(oracle.mean()),
                'regret': float((oracle - chosen).mean()),
                'fixed_action_ious': {str(int(action)): float(mean)
                                     for action, mean in zip(self.action_ids.cpu(), fixed_means)},
                'delta_to_best_fixed_action': float(chosen.mean() - fixed_means.max()),
                'action_counts': {str(int(action)): int((selected_actions == action).sum())
                                  for action in self.action_ids.cpu()}}

    def train(self):
        os.makedirs(self.settings.output, exist_ok=True)
        initial = self.development_metrics()
        print('development before training: ' + json.dumps(initial), flush=True)
        for epoch in range(self.epoch + 1, self.settings.epochs + 1):
            self.policy.train()
            loss_meter = AverageMeter()
            sampled = torch.randint(len(self.trainable), (self.settings.episodes_per_epoch,),
                                    generator=self.generator).tolist()
            for start in range(0, len(sampled), self.settings.batch_size):
                records = [self.trainable[index] for index in sampled[start:start + self.settings.batch_size]]
                batch = collate_windows(records, self.device)
                loss, stats = self.actor(batch)
                if loss is None:
                    continue
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite policy loss')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite policy gradient')
                self.max_gradient_norm = max(self.max_gradient_norm, float(norm))
                self.optimizer.step()
                self.updates += 1
                loss_meter.update(float(loss.detach()), stats['valid_steps'])
            self.epoch = epoch
            development = self.development_metrics()
            entry = {'epoch': epoch, 'train_loss': loss_meter.avg, 'updates': self.updates,
                     'development': development}
            self.history.append(entry)
            path = self.save()
            payload = {'protocol': 'supervised_search_policy_v1',
                       'identity': self.checkpoint_identity(),
                       'train_windows': len(self.records), 'development_windows': len(self.development),
                       'trainable_windows': len(self.trainable), 'initial_development': initial,
                       'history': self.history, 'checkpoint': path,
                       'max_gradient_norm': self.max_gradient_norm,
                       'scope': 'train-only teacher evaluation; full-val closed-loop results are separate'}
            report_path = os.path.join(self.settings.output, 'training_report.json')
            with open(report_path + '.tmp', 'w') as handle:
                json.dump(payload, handle, indent=2, allow_nan=False)
            os.replace(report_path + '.tmp', report_path)
            print(json.dumps(entry), flush=True)
            print('Wrote ' + path, flush=True)
