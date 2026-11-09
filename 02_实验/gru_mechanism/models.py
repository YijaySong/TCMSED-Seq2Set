"""Controlled variants of the locked GRU label-set decoder."""

import torch
from torch import nn
from torch.nn import functional as F

from gru_sequence import SymptomGRU


def fixed_targets(y):
    rows = []
    k = y.shape[1]
    for row in y:
        labels = torch.nonzero(row, as_tuple=False).flatten()
        if len(labels) == 0:
            raise ValueError("Empty reference set")
        rows.append(torch.cat((labels, labels.new_tensor([k]))))
    targets = torch.full((len(rows), max(map(len, rows))), -100, dtype=torch.long)
    for i, row in enumerate(rows):
        targets[i, :len(row)] = row
    return targets


class GRUNoLabelHistory(SymptomGRU):
    """Hide the previous action identity while retaining recurrent state."""

    def step(self, previous, hidden, context, chosen):
        bos = torch.full_like(previous, self.k + 1)
        hidden = self.cell(torch.cat((self.labels(bos), context), dim=1), hidden)
        logits = self.head(self.dropout(hidden))
        illegal = torch.cat((chosen, ~chosen.any(1, keepdim=True)), dim=1)
        return logits.masked_fill(illegal, float("-inf")), hidden


class NoStopCardinalityGRU(SymptomGRU):
    """Predict set size from symptoms, then select that many distinct labels."""

    def __init__(self, vocab, k, embedding=128, hidden=256):
        super().__init__(vocab, k, embedding=embedding, hidden=hidden)
        self.head = nn.Linear(hidden, k)
        self.cardinality_head = nn.Linear(hidden, k)

    def step(self, previous, hidden, context, chosen):
        hidden = self.cell(torch.cat((self.labels(previous), context), dim=1), hidden)
        logits = self.head(self.dropout(hidden))
        return logits.masked_fill(chosen, float("-inf")), hidden

    def loss(self, x, targets):
        context = self.encode(x)
        cardinality = targets.ne(-100).sum(1) - 1
        cardinality_loss = F.cross_entropy(self.cardinality_head(context), cardinality - 1)
        hidden = context
        chosen = torch.zeros((len(x), self.k), dtype=torch.bool, device=x.device)
        previous = torch.full((len(x),), self.k + 1, dtype=torch.long, device=x.device)
        total = context.new_zeros(len(x))
        for t in range(targets.shape[1] - 1):
            logits, hidden = self.step(previous, hidden, context, chosen)
            target = targets[:, t]
            active = target.ge(0) & target.lt(self.k)
            label_target = target.masked_fill(~active, -100)
            total += F.cross_entropy(logits, label_target,
                                     ignore_index=-100, reduction="none")
            chosen = chosen.clone()
            chosen[active, target[active]] = True
            previous = torch.where(active, target, previous)
        label_loss = (total / cardinality).mean()
        return label_loss + cardinality_loss

    @torch.no_grad()
    def generate(self, x):
        context = self.encode(x)
        count = self.cardinality_head(context).argmax(1) + 1
        hidden = context
        chosen = torch.zeros((len(x), self.k), dtype=torch.bool, device=x.device)
        previous = torch.full((len(x),), self.k + 1, dtype=torch.long, device=x.device)
        traces = [[] for _ in range(len(x))]
        for t in range(int(count.max())):
            logits, hidden = self.step(previous, hidden, context, chosen)
            action = logits.argmax(1)
            active = t < count
            for i in torch.where(active)[0].tolist():
                traces[i].append(int(action[i]))
            chosen[active, action[active]] = True
            previous = torch.where(active, action, previous)
        return chosen.to(torch.int64), traces
