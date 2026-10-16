import torch
from torch import nn


def select_search_action(logits, action_ids, decision=None, sample=False):
    """Optional advantage gate; ties and disabled interventions retain factor eight."""
    if decision is not None:
        if decision.get('rule') != 'advantage_margin_v1' or action_ids.tolist() != [0, 1, 2]:
            raise ValueError('Advantage gating requires scale-only actions')
        if sample:
            raise ValueError('Regression advantage gate is deterministic')
        best, index = logits[..., :2].max(-1)
        margin = decision.get('margin')
        enabled = torch.zeros_like(best, dtype=torch.bool) if margin is None else best > margin
        return torch.where(enabled, index, torch.full_like(index, 2))
    valid = logits.index_select(-1, action_ids)
    selected = torch.distributions.Categorical(logits=valid).sample() if sample else valid.argmax(-1)
    return action_ids[selected]


class SearchPolicy(nn.Module):
    """Small recurrent controller for choosing the next search crop."""

    observation_dim = 32
    num_actions = 9

    def __init__(self, hidden_dim=64, recurrent=True, mlp_depth=2):
        super().__init__()
        self.recurrent = recurrent
        self.mlp_depth = mlp_depth
        self.encoder = nn.Sequential(
            nn.Linear(self.observation_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        if recurrent:
            self.temporal = nn.GRUCell(hidden_dim, hidden_dim)
            self.head = nn.Linear(hidden_dim, self.num_actions)
        else:
            layers = []
            for _ in range(mlp_depth - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])
            self.temporal = nn.Sequential(*layers)
            self.head = nn.Linear(hidden_dim, self.num_actions)

    def forward(self, observation, hidden=None):
        encoded = self.encoder(observation)
        if self.recurrent:
            if hidden is None:
                hidden = encoded.new_zeros(encoded.shape[:-1] + (encoded.shape[-1],))
            hidden = self.temporal(encoded, hidden)
            features = hidden
        else:
            features = self.temporal(encoded)
        return self.head(features), hidden

