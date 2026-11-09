"""Matched symptom-mean encoders for direct and chain controls."""

import torch
from torch import nn
from torch.nn import functional as F


class MeanSymptomEncoder(nn.Module):
    def __init__(self, vocab, embedding=128, hidden=256):
        super().__init__()
        self.symptoms = nn.Embedding(vocab, embedding, padding_idx=0)
        self.context = nn.Sequential(nn.Linear(embedding, hidden), nn.Tanh())

    def encode(self, tokens):
        mask = tokens.ne(0).unsqueeze(-1)
        mean = (self.symptoms(tokens) * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.context(mean)


class SigmoidModel(MeanSymptomEncoder):
    def __init__(self, vocab, labels, matched=False):
        super().__init__(vocab)
        self.dropout = nn.Dropout(0.15)
        if matched:
            self.head = nn.Sequential(
                nn.Linear(256, 964), nn.ReLU(),
                nn.Linear(964, 256), nn.ReLU(), nn.Linear(256, labels))
        else:
            self.head = nn.Linear(256, labels)

    def forward(self, tokens):
        return self.head(self.dropout(self.encode(tokens)))

    def loss(self, tokens, truth):
        return F.binary_cross_entropy_with_logits(self(tokens), truth)


class ClassifierChain(MeanSymptomEncoder):
    """Fixed label order, teacher-forced binary training, free-running decoding."""

    def __init__(self, vocab, labels):
        super().__init__(vocab)
        self.labels = labels
        self.dropout = nn.Dropout(0.15)
        self.head = nn.Linear(256 + labels, labels)

    def _label_logit(self, context, prefix, label):
        state = torch.cat((context, prefix), dim=1)
        return F.linear(state, self.head.weight[label:label + 1],
                        self.head.bias[label:label + 1]).squeeze(1)

    def loss(self, tokens, truth):
        context = self.dropout(self.encode(tokens))
        prefix = torch.zeros_like(truth)
        losses = []
        for label in range(self.labels):
            logit = self._label_logit(context, prefix, label)
            losses.append(F.binary_cross_entropy_with_logits(logit, truth[:, label]))
            prefix = prefix.clone()
            prefix[:, label] = truth[:, label]
        return torch.stack(losses).mean()

    @torch.no_grad()
    def predict_thresholds(self, tokens, thresholds):
        context = self.encode(tokens)
        count = len(thresholds)
        batch = len(tokens)
        context = context.unsqueeze(0).expand(count, -1, -1).reshape(count * batch, -1)
        cuts = torch.as_tensor(thresholds, dtype=context.dtype, device=context.device)
        cuts = cuts[:, None].expand(-1, batch).reshape(-1)
        prefix = context.new_zeros((count * batch, self.labels))
        conditional = context.new_zeros((count * batch, self.labels))
        for label in range(self.labels):
            probability = self._label_logit(context, prefix, label).sigmoid()
            conditional[:, label] = probability
            prefix[:, label] = (probability >= cuts).to(prefix.dtype)
        empty = prefix.sum(1).eq(0)
        if empty.any():
            row = empty.nonzero(as_tuple=True)[0]
            prefix[row, conditional[row].argmax(1)] = 1
        return prefix.reshape(count, batch, self.labels).to(torch.int64)
