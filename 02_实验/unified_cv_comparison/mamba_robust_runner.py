"""Permutation-robust Mamba-style encoder check."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

import run_modern_encoders as base

ROOT = Path(__file__).resolve().parent
SEEDS = (9, 17, 29)


def fixed_orders(row):
    # Canonicalize the symptom set before applying deterministic transforms.
    # This makes the averaged context invariant to the input token order.
    values = sorted(int(v) for v in row if int(v) != 0)
    if len(values) < 2:
        return [values] * 4
    orders = [values, list(reversed(values))]
    orders.append(values[1:] + values[:1])
    orders.append(values[::2] + values[1::2])
    return orders


class PermutationAveragedMamba(nn.Module):
    """Average four deterministic symptom orders before set decoding."""

    def __init__(self, vocab, embedding=96, hidden=256):
        super().__init__()
        self.base = base.MambaStyleEncoder(vocab, embedding, hidden)

    def forward(self, x):
        rows = [fixed_orders(row.tolist()) for row in x]
        flat = [order for variants in rows for order in variants]
        length = max(map(len, flat))
        tokens = torch.zeros(len(flat), length, dtype=x.dtype, device=x.device)
        for i, row in enumerate(flat):
            if row:
                tokens[i, :len(row)] = torch.tensor(row, dtype=x.dtype, device=x.device)
        contexts = self.base(tokens).reshape(len(rows), 4, -1)
        return contexts.mean(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    base.EPOCHS = 2 if args.smoke_test else 100
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    sets, vocab, k, schema = base.prepare()
    out = ROOT / ("mamba_robust_v2_smoke_results" if args.smoke_test else "mamba_robust_v2_results")
    out.mkdir(exist_ok=True)
    protocol = {
        "method": "mamba_style_sequence_permutation_averaged",
        "seeds": list(SEEDS), "epochs": base.EPOCHS, "batch_size": 64,
        "primary_metric": "exact", "secondary_metric": "sample_f1",
        "split": "record-level 70/15/15", "context_orders": 4,
        "orders": "canonical sorted order, reverse, left rotation, even-odd interleave",
        "schema": schema,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    for seed in ((SEEDS[0],) if args.smoke_test else SEEDS):
        folder = out / "mamba_style_sequence_permutation_averaged" / f"seed_{seed}"
        if (folder / "summary.json").exists():
            continue
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        encoder = PermutationAveragedMamba(vocab)
        model = base.SequenceSetModel(encoder, k)
        summary, history, pred, traces = base.fit(model, sets, seed, protocol["method"])
        summary["protocol"] = protocol
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "selection": summary["selection"]}, folder / "best.pt")
        (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "predictions.json").write_text(json.dumps({
            "ids": sets["test"][2], "truth": sets["test"][1].astype(int).tolist(),
            "predicted": pred.astype(int).tolist(), "traces": traces,
        }, ensure_ascii=False), encoding="utf-8")
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*/seed_*/summary.json"))]
    (out / "all_summaries.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print("COMPLETE", len(summaries), "runs", flush=True)


if __name__ == "__main__":
    main()
