"""Adapters for the fixed-set-likelihood and self-critical policy baselines."""

import numpy as np
import torch

from models.supervised_set import SupervisedSet
from rl_engine import rollout, set_distribution


MAP_BATCH = 8
BEAM_WIDTH = 5


def build(vocab, k):
    return SupervisedSet(vocab, k, hidden=256, hidden2=128, max_subsets=4096)


def train_step(model, x, y, stage, sample_rng=None, samples=4):
    """Backpropagate one minibatch and return its detached mean loss.

    The exact-NLL implementation backpropagates one record at a time to bound
    memory. Its backward_batch result is a float, not a tensor to backpropagate
    again. The RL stage uses the same model after supervised initialization.
    """
    model.train()
    if stage == "nll":
        value = float(model.backward_batch(x, y))
    elif stage == "rl":
        if sample_rng is None:
            raise ValueError("RL sampling requires a torch.Generator")
        loss = rollout(model, x, y, sample_rng, samples=samples)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite RL loss")
        loss.backward()
        value = float(loss.detach())
    else:
        raise ValueError(f"Unknown set training stage: {stage}")
    if not np.isfinite(value):
        raise FloatingPointError(f"Nonfinite {stage} loss")
    return value


def _beam_one(model, x):
    symptoms = model.encode(x)
    k = model.k
    beams = [(torch.zeros(k, device=x.device), 0.0, False)]
    for _ in range(k + 1):
        candidates = []
        for selected, score, done in beams:
            if done:
                candidates.append((selected, score, True))
                continue
            illegal = torch.cat((selected.bool()[None],
                                 ~selected.bool().any().reshape(1, 1)), dim=1)
            logp = model(symptoms, selected[None]).masked_fill(
                illegal, -torch.inf).log_softmax(dim=1)[0]
            legal = torch.where(torch.isfinite(logp))[0]
            ranked = legal[torch.argsort(logp[legal], descending=True)[:BEAM_WIDTH]]
            for action in ranked.tolist():
                proposed = selected.clone()
                finished = action == k
                if not finished:
                    proposed[action] = 1
                candidates.append((proposed, score + float(logp[action]), finished))
        candidates.sort(key=lambda entry: entry[1], reverse=True)
        beams = candidates[:BEAM_WIDTH]
        if all(entry[2] for entry in beams):
            break
    completed = [entry for entry in beams if entry[2]] or beams
    return max(completed, key=lambda entry: entry[1])[0].to(torch.int64)


@torch.no_grad()
def decode(model, tokens, mode, batch_tokens, batch_size=64):
    """Return predictions in input order for greedy, beam-5, or exact set MAP."""
    if mode not in ("greedy", "beam5", "map"):
        raise ValueError(f"Unknown set decoder: {mode}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not tokens:
        raise ValueError("tokens must contain at least one record")
    model.eval()
    predicted = []
    traces = [] if mode == "greedy" else None
    step = min(batch_size, MAP_BATCH) if mode == "map" else batch_size
    for start in range(0, len(tokens), step):
        x = batch_tokens(tokens, range(start, min(start + step, len(tokens))))
        if mode == "greedy":
            sets, actions = model.generate(x)
            predicted.append(sets.to(torch.int64).cpu().numpy())
            traces.extend(actions)
        elif mode == "map":
            probabilities, bits = set_distribution(model, x)
            predicted.append(bits[probabilities.argmax(dim=1)].to(torch.int64).cpu().numpy())
        else:
            predicted.append(torch.stack([_beam_one(model, row[None]) for row in x])
                             .cpu().numpy())
    return np.concatenate(predicted, axis=0), traces
