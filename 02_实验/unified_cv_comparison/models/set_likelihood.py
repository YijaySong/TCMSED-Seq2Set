"""Exact target-set likelihood for a deterministic, set-conditioned policy.

Labels are 0..K-1 and STOP is K. No repeats, no empty output, STOP is forced
at K labels. This is an independent objective implementation, not a trainer.
The subset forward algorithm is an existing probability-summing technique.
"""

from numbers import Integral

import torch


def _target_labels(target, label_count, max_subsets):
    if isinstance(label_count, bool) or not isinstance(label_count, Integral) or label_count < 1:
        raise ValueError('label_count must be a positive integer')
    if isinstance(max_subsets, bool) or not isinstance(max_subsets, Integral) or max_subsets < 2:
        raise ValueError('max_subsets must be an integer >= 2')
    try:
        labels = tuple(target)
    except TypeError as error:
        raise ValueError('target must be an iterable of integer label indices') from error
    if not labels:
        raise ValueError('empty target is invalid under the nonempty-output protocol')
    if any(isinstance(a, bool) or not isinstance(a, Integral) for a in labels):
        raise ValueError('target labels must be integer indices')
    if len(set(labels)) != len(labels):
        raise ValueError('duplicate target labels')
    if any(a < 0 or a >= label_count for a in labels):
        raise ValueError('target label outside 0..K-1 (STOP is not a label)')
    if (1 << len(labels)) > max_subsets:
        raise ValueError('target exceeds max_subsets; exact cost is exponential in target size')
    return tuple(sorted(int(a) for a in labels))


def _subset_bits(size, device):
    masks = torch.arange(1 << size, device=device)
    bits = (masks[:, None] & (1 << torch.arange(size, device=device))) != 0
    return masks, bits


def selected_subsets(target, label_count, *, dtype=torch.float64, device=None, max_subsets=4096):
    """Rows encode subsets of sorted(target), in increasing local bitmask order.

All K label coordinates are retained, including labels outside the target.
The default resource guard permits at most 4096 rows (12 target labels).
"""
    labels = _target_labels(target, label_count, max_subsets)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('only float32 and float64 are supported')
    _, bits = _subset_bits(len(labels), device)
    selected = torch.zeros((len(bits), label_count), dtype=dtype, device=device)
    selected[:, list(labels)] = bits.to(dtype)
    return selected


def set_log_probability_from_logits(logits, target, *, max_subsets=4096):
    """Sum probabilities of all target permutations followed by STOP, in log space.

logits has shape [2**len(target), K+1], with rows as in selected_subsets.
Logits must come from one fixed policy, with no hidden order dependence.
Softmax uses ALL legal actions, never only the remaining true labels.
"""
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError('logits must have shape [2**m, K+1], K >= 1')
    labels = _target_labels(target, logits.shape[1] - 1, max_subsets)
    if logits.shape[0] != 1 << len(labels):
        raise ValueError('logits row count must equal 2**len(target)')
    if logits.dtype not in (torch.float32, torch.float64):
        raise ValueError('logits must use float32 or float64')
    if not torch.isfinite(logits).all().item():
        raise FloatingPointError('all unmasked logits must be finite')
    k, m = logits.shape[1] - 1, len(labels)
    masks, bits = _subset_bits(m, logits.device)
    selected = torch.zeros((len(masks), k), dtype=torch.bool, device=logits.device)
    selected[:, list(labels)] = bits
    illegal = torch.cat((selected, (masks == 0)[:, None]), dim=1)
    logp = torch.log_softmax(logits.masked_fill(illegal, -torch.inf), dim=1)

    # Merge paths only at identical selected sets. Cardinality layers avoid
    # in-place autograd updates and a separate Python graph operation per subset.
    counts = bits.sum(dim=1)
    previous_masks = masks[:1]
    previous_alpha = logits.new_zeros(1)
    label_tensor = torch.tensor(labels, device=logits.device)
    for depth in range(1, m + 1):
        current_masks = masks[counts == depth]
        local_labels = bits[current_masks].nonzero(as_tuple=True)[1].reshape(-1, depth)
        predecessors = current_masks[:, None] ^ (1 << local_labels)
        predecessor_rows = torch.searchsorted(previous_masks, predecessors)
        transitions = logp[predecessors, label_tensor[local_labels]]
        previous_alpha = torch.logsumexp(previous_alpha[predecessor_rows] + transitions, dim=1)
        previous_masks = current_masks
    # At the full vocabulary, STOP is the only legal action: logp is exactly 0.
    return previous_alpha[0] + logp[-1, k]


def set_nll_from_logits(logits, target, *, normalization='none', max_subsets=4096):
    """Raw set NLL, or explicit per-record scaling by target size + one STOP.

Use normalization='target_steps' only when that record weighting is intended.
This scaling is NOT a constant shared by records of different cardinalities.
"""
    if normalization not in ('none', 'target_steps'):
        raise ValueError("normalization must be 'none' or 'target_steps'")
    target = tuple(target)
    loss = -set_log_probability_from_logits(logits, target, max_subsets=max_subsets)
    return loss if normalization == 'none' else loss / (len(target) + 1)


def model_set_log_probability(model, symptoms, target, label_count, *, max_subsets=4096):
    """Differentiable single-record adapter for model(concat(symptoms, selected)).

Require eval mode (including child modules) to disable standard dropout and
batch statistics; eval does not disable gradients. Caller must also ensure the
model has no custom stochastic behavior or hidden history dependence. This
does not accept a recurrent history decoder or load any data/checkpoint.
"""
    if not isinstance(symptoms, torch.Tensor) or symptoms.ndim != 1 or not symptoms.numel():
        raise ValueError('symptoms must be a nonempty 1-D tensor')
    if symptoms.dtype not in (torch.float32, torch.float64):
        raise ValueError('symptoms must use float32 or float64')
    if not torch.isfinite(symptoms).all().item():
        raise FloatingPointError('symptoms must be finite')
    if any(module.training for module in model.modules()):
        raise ValueError('model and all child modules must be in eval mode; gradients remain enabled')
    labels = _target_labels(target, label_count, max_subsets)
    selected = selected_subsets(labels, label_count, dtype=symptoms.dtype,
                                device=symptoms.device, max_subsets=max_subsets)
    states = torch.cat((symptoms.expand(len(selected), -1), selected), dim=1)
    logits = model(states)
    if not isinstance(logits, torch.Tensor) or logits.shape != (len(selected), label_count + 1):
        raise ValueError('model must return one K+1 action-logit vector per subset')
    return set_log_probability_from_logits(logits, labels, max_subsets=max_subsets)
