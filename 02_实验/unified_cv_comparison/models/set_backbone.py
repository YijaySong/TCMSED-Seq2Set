# -*- coding: utf-8 -*-
"""Set-conditioned Q networks and optional auxiliary-logit controls."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))
        bound = 1.0 / math.sqrt(in_features)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_weight.data.fill_(sigma_init / math.sqrt(in_features))
        self.sigma_bias.data.fill_(sigma_init / math.sqrt(out_features))

    @staticmethod
    def _noise(size, device):
        noise = torch.randn(size, device=device)
        return noise.sign() * noise.abs().sqrt()

    def forward(self, inputs):
        if not self.training:
            return functional.linear(inputs, self.mu_weight, self.mu_bias)
        eps_in = self._noise(self.in_features, inputs.device)
        eps_out = self._noise(self.out_features, inputs.device)
        weight = self.mu_weight + self.sigma_weight * eps_out.outer(eps_in)
        bias = self.mu_bias + self.sigma_bias * eps_out
        return functional.linear(inputs, weight, bias)


def linear_layer(in_features, out_features, noisy=False):
    return NoisyLinear(in_features, out_features) if noisy else nn.Linear(in_features, out_features)


class SupervisedQNetwork(nn.Module):
    def _init_supervision(self, width, n_actions, supervision_head):
        if supervision_head not in ('q', 'separate'):
            raise ValueError('supervision_head must be q or separate')
        self.supervision_head = supervision_head
        if supervision_head == 'separate':
            self.aux_head = nn.Linear(width, n_actions)

    def supervised_logits(self, inputs):
        if self.supervision_head == 'separate':
            return self.aux_head(self.forward_features(inputs))
        return self(inputs)


class ConcatQNetwork(SupervisedQNetwork):
    """Direct concatenation of symptoms and the selected-label multi-hot set."""

    def __init__(self, state_vector_len, n_actions, n_units=256, n_units2=128,
                 dropout=0.15, supervision_head='q'):
        super().__init__()
        if n_actions <= 1 or state_vector_len <= n_actions - 1:
            raise ValueError('State must contain symptoms and selected labels')
        self.symptom_count = state_vector_len - (n_actions - 1)
        self.encoder = nn.Sequential(
            nn.Linear(state_vector_len, n_units), nn.LayerNorm(n_units), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(n_units, n_units2), nn.LayerNorm(n_units2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.q_head = nn.Linear(n_units2, n_actions)
        self._init_supervision(n_units2, n_actions, supervision_head)

    def forward_features(self, inputs):
        return self.encoder(inputs)

    def forward(self, inputs):
        return self.q_head(self.forward_features(inputs))


class SymptomOnlyQNetwork(ConcatQNetwork):
    """H1 control: ignores selected-set features while action masking remains active."""

    def forward_features(self, inputs):
        symptom_only = inputs[:, :self.symptom_count]
        padded = torch.cat([symptom_only, torch.zeros_like(inputs[:, self.symptom_count:])], dim=1)
        return super().forward_features(padded)


class SetAwareDuelingDQN(SupervisedQNetwork):
    """症状集合、已选标签集合和逐元素交互的轻量 Dueling 网络。"""

    def __init__(self, state_vector_len, n_actions, n_units=256, n_units2=128, dropout=0.15,
                 noisy=False, supervision_head='q', dueling=True):
        super().__init__()
        self.label_count = n_actions - 1
        self.symptom_count = state_vector_len - self.label_count
        if self.label_count <= 0 or self.symptom_count <= 0:
            raise ValueError('状态必须由症状块、已选标签块和STOP动作组成')
        self.symptom_encoder = nn.Sequential(
            linear_layer(self.symptom_count, n_units, noisy), nn.LayerNorm(n_units), nn.ReLU(), nn.Dropout(dropout),
        )
        self.selected_encoder = nn.Sequential(
            linear_layer(self.label_count, n_units, noisy), nn.LayerNorm(n_units), nn.ReLU(), nn.Dropout(dropout),
        )
        self.joint = nn.Sequential(
            linear_layer(n_units * 3, n_units2, noisy), nn.LayerNorm(n_units2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.dueling = dueling
        if dueling:
            self.value_stream = nn.Sequential(
                linear_layer(n_units2, n_units2, noisy), nn.ReLU(), linear_layer(n_units2, 1, noisy),
            )
            self.advantage_stream = nn.Sequential(
                linear_layer(n_units2, n_units2, noisy), nn.ReLU(), linear_layer(n_units2, n_actions, noisy),
            )
        else:
            self.q_head = nn.Linear(n_units2, n_actions)
        self._init_supervision(n_units2, n_actions, supervision_head)

    def forward_features(self, inputs):
        symptom = self.symptom_encoder(inputs[:, :self.symptom_count])
        selected = self.selected_encoder(inputs[:, self.symptom_count:self.symptom_count + self.label_count])
        return self.joint(torch.cat([symptom, selected, symptom * selected], dim=1))

    def forward(self, inputs):
        features = self.forward_features(inputs)
        if not self.dueling:
            return self.q_head(features)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def build_q_network(model_type, state_vector_len, n_actions, n_units=256, n_units2=128,
                    dropout=0.15, supervision_head='q'):
    if model_type == 'concat_q':
        return ConcatQNetwork(state_vector_len, n_actions, n_units, n_units2, dropout, supervision_head)
    if model_type == 'symptom_only_q':
        return SymptomOnlyQNetwork(state_vector_len, n_actions, n_units, n_units2, dropout, supervision_head)
    if model_type in ('set_q', 'set_dueling', 'set_dueling_noisy'):
        return SetAwareDuelingDQN(
            state_vector_len, n_actions, n_units, n_units2, dropout,
            noisy=model_type == 'set_dueling_noisy', supervision_head=supervision_head,
            dueling=model_type != 'set_q',
        )
    raise ValueError(f'Unknown model_type: {model_type}')
