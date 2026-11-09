"""Additional multi-label baselines for binary symptom sets."""
import torch
from torch import nn
from torch.nn import functional as F


class SymptomSetAttention(nn.Module):
    """Permutation-invariant attention pooling followed by a multi-label head."""
    def __init__(self, vocab, labels, embedding=128, heads=4, layers=2, width=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embedding, padding_idx=0)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embedding, heads, width, dropout=.15,
                                       activation='gelu', batch_first=True,
                                       norm_first=True)
            for _ in range(layers)
        ])
        self.query = nn.Parameter(torch.zeros(1, 1, embedding))
        nn.init.normal_(self.query, std=.02)
        self.pool = nn.MultiheadAttention(embedding, heads, dropout=.15, batch_first=True)
        self.norm = nn.LayerNorm(embedding)
        self.head = nn.Linear(embedding, labels)

    def forward(self, tokens):
        valid = tokens.ne(0)
        h = self.embedding(tokens)
        for block in self.blocks:
            h = block(h, src_key_padding_mask=~valid)
        q = self.query.expand(len(tokens), -1, -1)
        pooled, _ = self.pool(q, h, h, key_padding_mask=~valid)
        return self.head(self.norm(pooled[:, 0]))


class AsymmetricMLP(nn.Module):
    """Asymmetric focal loss is supplied by the runner; this is the shared head."""
    def __init__(self, vocab, labels, width=256):
        super().__init__()
        self.width = vocab - 2
        self.net = nn.Sequential(nn.Linear(self.width, width), nn.LayerNorm(width),
                                 nn.ReLU(), nn.Linear(width, labels))

    def forward(self, tokens):
        z = torch.zeros(len(tokens), self.width, device=tokens.device)
        valid = tokens.ge(2)
        rows = torch.arange(len(tokens), device=tokens.device)[:, None].expand_as(tokens)
        z[rows[valid], tokens[valid] - 2] = 1
        return self.net(z)


class ClassifierChain(nn.Module):
    """Fixed-order conditional multi-label chain; order is recorded in the protocol."""
    def __init__(self, vocab, labels, width=256):
        super().__init__()
        self.k = labels
        self.width = vocab - 2
        self.encoder = nn.Sequential(nn.Linear(self.width, width), nn.LayerNorm(width), nn.ReLU())
        self.head = nn.Linear(width + labels, labels)

    def features(self, tokens):
        z = torch.zeros(len(tokens), self.width, device=tokens.device)
        valid = tokens.ge(2)
        rows = torch.arange(len(tokens), device=tokens.device)[:, None].expand_as(tokens)
        z[rows[valid], tokens[valid] - 2] = 1
        return self.encoder(z)

    def forward(self, tokens, prefix=None):
        h = self.features(tokens)
        if prefix is None:
            prefix = torch.zeros(len(tokens), self.k, device=tokens.device)
        return self.head(torch.cat((h, prefix), 1))

    def loss(self, tokens, target):
        prefix = torch.zeros_like(target)
        losses = []
        for j in range(self.k):
            logits = self.forward(tokens, prefix)
            losses.append(F.binary_cross_entropy_with_logits(logits[:, j], target[:, j]))
            prefix[:, j] = target[:, j]
        return torch.stack(losses).mean()

    @torch.no_grad()
    def predict(self, tokens):
        prefix = torch.zeros(len(tokens), self.k, device=tokens.device)
        for j in range(self.k):
            logits = self.forward(tokens, prefix)
            prefix[:, j] = (logits[:, j].sigmoid() >= .5).float()
        empty = prefix.sum(1).eq(0)
        if empty.any():
            prefix[empty, logits[empty].argmax(1)] = 1
        return prefix


class LabelGraphMLP(nn.Module):
    """Train-only co-occurrence graph propagation over label logits."""
    def __init__(self, vocab, labels, adjacency, width=256):
        super().__init__()
        self.width = vocab - 2
        self.net = nn.Sequential(nn.Linear(self.width, width), nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, labels))
        a = adjacency.float() + torch.eye(labels)
        a = a / a.sum(1, keepdim=True).clamp_min(1)
        self.register_buffer('adjacency', a)

    def forward(self, tokens):
        z = torch.zeros(len(tokens), self.width, device=tokens.device)
        valid = tokens.ge(2)
        rows = torch.arange(len(tokens), device=tokens.device)[:, None].expand_as(tokens)
        z[rows[valid], tokens[valid] - 2] = 1
        logits = self.net(z)
        return logits + .5 * (logits @ self.adjacency.T)


def asymmetric_loss(logits, target, gamma_pos=0., gamma_neg=2., clip=0.05):
    xs_pos = logits.sigmoid()
    xs_neg = (1 - logits.sigmoid()).clamp_min(1e-8)
    if clip:
        xs_neg = (xs_neg + clip).clamp_max(1)
    loss_pos = target * torch.log(xs_pos.clamp_min(1e-8))
    loss_neg = (1 - target) * torch.log(xs_neg)
    if gamma_pos or gamma_neg:
        pt = xs_pos * target + xs_neg * (1 - target)
        one_sided = (1 - pt).pow(gamma_pos * target + gamma_neg * (1 - target))
        loss_pos = loss_pos * one_sided
        loss_neg = loss_neg * one_sided
    return -(loss_pos + loss_neg).mean()
