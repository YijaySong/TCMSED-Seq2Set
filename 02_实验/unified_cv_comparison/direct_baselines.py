"""Legacy direct multi-label comparators on the locked cross-validation folds."""

import torch
from torch import nn
from torch.nn import functional as F

from models.cnn import SymptomCNN
from models.modern_baselines import (
    AsymmetricMLP,
    ClassifierChain,
    LabelGraphMLP,
    SymptomSetAttention,
    asymmetric_loss,
)


METHODS = (
    "mlp",
    "cnn",
    "asl_mlp",
    "classifier_chain_legacy",
    "label_graph",
    "set_attention",
)


class BinaryMLP(nn.Module):
    """The MLP from method_selection_v2_4200/run_all.py."""

    def __init__(self, vocab, labels, width=256):
        super().__init__()
        self.width = vocab - 2
        self.net = nn.Sequential(
            nn.Linear(self.width, width),
            nn.LayerNorm(width),
            nn.ReLU(),
            nn.Linear(width, labels),
        )

    def forward(self, tokens):
        features = torch.zeros(len(tokens), self.width, device=tokens.device)
        valid = tokens.ge(2)
        rows = torch.arange(len(tokens), device=tokens.device)[:, None].expand_as(tokens)
        features[rows[valid], tokens[valid] - 2] = 1
        return self.net(features)


def build(method, vocab, labels, train_y):
    """Build an old direct baseline; graph edges use training labels only."""
    if method == "mlp":
        return BinaryMLP(vocab, labels)
    if method == "cnn":
        return SymptomCNN(vocab, labels, embedding=128, channels=128)
    if method == "asl_mlp":
        return AsymmetricMLP(vocab, labels, width=256)
    if method == "classifier_chain_legacy":
        return ClassifierChain(vocab, labels, width=256)
    if method == "label_graph":
        targets = torch.as_tensor(train_y, dtype=torch.float32)
        if targets.ndim != 2 or targets.shape[1] != labels:
            raise ValueError("Training label matrix has the wrong shape")
        cooccurrence = targets.T @ targets
        cooccurrence.fill_diagonal_(0)
        return LabelGraphMLP(vocab, labels, cooccurrence, width=256)
    if method == "set_attention":
        return SymptomSetAttention(vocab, labels, embedding=128,
                                   heads=4, layers=2, width=256)
    raise ValueError(f"Unknown direct baseline: {method}")


def loss(model, method, tokens, target):
    if method == "asl_mlp":
        return asymmetric_loss(model(tokens), target)
    if method == "classifier_chain_legacy":
        return model.loss(tokens, target)
    if method in METHODS:
        return F.binary_cross_entropy_with_logits(model(tokens), target)
    raise ValueError(f"Unknown direct baseline: {method}")


@torch.no_grad()
def predict_thresholds(model, method, tokens, thresholds):
    """Return [threshold, record, label] predictions, with nonempty sets.

    The caller places the model in eval mode. For the classifier chain, each
    threshold is applied during autoregressive decoding, not to a teacher-
    forced or unconditional logit matrix.
    """
    values = torch.as_tensor(thresholds, dtype=torch.float32, device=tokens.device)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Thresholds must be a nonempty one-dimensional vector")
    if method == "classifier_chain_legacy":
        features = model.features(tokens)
        labels = model.k
        predictions = torch.zeros((len(values), len(tokens), labels),
                                  dtype=torch.float32, device=tokens.device)
        probabilities = torch.empty_like(predictions)
        expanded = features.unsqueeze(0).expand(len(values), -1, -1)
        for label in range(labels):
            state = torch.cat((expanded, predictions), dim=2)
            logit = F.linear(state, model.head.weight[label:label + 1],
                             model.head.bias[label:label + 1]).squeeze(-1)
            probabilities[:, :, label] = logit.sigmoid()
            predictions[:, :, label] = (probabilities[:, :, label]
                                         >= values[:, None]).float()
    elif method in METHODS:
        probabilities = model(tokens).sigmoid().unsqueeze(0)
        predictions = (probabilities >= values[:, None, None]).float()
        probabilities = probabilities.expand(len(values), -1, -1)
    else:
        raise ValueError(f"Unknown direct baseline: {method}")

    threshold_idx, record_idx = torch.where(predictions.sum(dim=2) == 0)
    if len(threshold_idx):
        label_idx = probabilities[threshold_idx, record_idx].argmax(dim=1)
        predictions[threshold_idx, record_idx, label_idx] = 1
    return predictions.to(torch.int64)
