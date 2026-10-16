"""Supervision for crop decisions; tracker weights are outside this actor."""
import torch

from lib.train.actors.base_actor import BaseActor
from lib.train.data.search_policy import supervised_policy_loss
from lib.test.utils.search_region import allowed_actions


class SearchPolicyActor(BaseActor):
    def __init__(self, policy, action_mode, temperature=0.1):
        super().__init__(policy, None)
        self.action_mode = action_mode
        self.temperature = temperature

    def __call__(self, data):
        action_ids = torch.tensor(allowed_actions(self.action_mode), device=data['observations'].device)
        loss, logits, scores, mask = supervised_policy_loss(
            self.net, data, action_ids, self.temperature)
        if loss is None:
            return None, {'valid_steps': 0}
        selected = logits.argmax(-1)
        selected_iou = scores.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
        oracle_iou = scores.max(-1).values
        return loss, {
            'valid_steps': int(mask.sum()), 'loss': float(loss.detach()),
            'teacher_selected_iou': float(selected_iou[mask].mean()),
            'teacher_oracle_iou': float(oracle_iou[mask].mean()),
            'teacher_regret': float((oracle_iou - selected_iou)[mask].mean()),
        }
