"""TCN, Mamba-style SSM, and PatchTST-style encoders for set construction."""
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
SEEDS = (9, 17, 29)
BATCH = 64
EPOCHS = 100


def read_records():
    rows = []
    with (ROOT / "广安门心绞痛4200例.txt").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f, delimiter=" "):
            rows.append((r["id"], tuple(x for x in r["symptom"].split(",") if x),
                         tuple(x for x in r["Se"].split(",") if x)))
    if len(rows) != 4200 or [int(r[0]) for r in rows] != list(range(1, 4201)):
        raise ValueError("Expected 4200 records with ids 1..4200")
    return rows


def prepare():
    rows = read_records()
    rng = np.random.default_rng(20260923)
    order = rng.permutation(len(rows))
    parts = {"train": order[:2940], "val": order[2940:3570], "test": order[3570:]}
    symptoms = sorted({s for _, ss, _ in rows for s in ss})
    labels = sorted({s for _, _, yy in rows for s in yy})
    sid = {s: i + 2 for i, s in enumerate(symptoms)}
    lid = {s: i for i, s in enumerate(labels)}
    sets = {}
    for name, ix in parts.items():
        toks, y, ids = [], np.zeros((len(ix), len(labels)), dtype=np.float32), []
        for j in ix:
            rid, ss, yy = rows[int(j)]
            ids.append(rid)
            # A deterministic order is used at evaluation; training adds order augmentation.
            toks.append(sorted(sid[s] for s in ss))
            for lab in yy:
                y[len(ids) - 1, lid[lab]] = 1
        sets[name] = (toks, y, ids)
    schema = {"symptoms": {str(i): s for s, i in sid.items()},
              "labels": {str(i): s for s, i in lid.items()},
              "split_seed": 20260923,
              "split_sizes": {k: len(v) for k, v in parts.items()}}
    return sets, len(sid) + 2, len(labels), schema


def batch_tokens(seq, ix):
    rows = [seq[int(i)] for i in ix]
    out = torch.zeros((len(rows), max(map(len, rows))), dtype=torch.long)
    for i, row in enumerate(rows):
        out[i, :len(row)] = torch.tensor(row)
    return out


def metrics(y, p):
    tp = (y * p).sum(1)
    den = y.sum(1) + p.sum(1)
    sf = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    ld = y.sum(0) + p.sum(0)
    lf = np.divide(2 * (y * p).sum(0), ld, out=np.zeros_like(ld), where=ld > 0)
    return {"exact": float((y == p).all(1).mean()),
            "sample_f1": float(sf.mean()),
            "micro_f1": float(2 * tp.sum() / max(1, den.sum())),
            "macro_f1": float(lf.mean()),
            "cardinality_mae": float(abs(y.sum(1) - p.sum(1)).mean())}


def random_targets(y, generator):
    rows, k = [], y.shape[1]
    for row in y:
        labels = torch.nonzero(row, as_tuple=False).flatten()
        if not len(labels):
            raise ValueError("Empty reference set")
        rows.append(torch.cat((labels[torch.randperm(len(labels), generator=generator)],
                               torch.tensor([k]))))
    out = torch.full((len(rows), max(map(len, rows))), -100, dtype=torch.long)
    for i, row in enumerate(rows):
        out[i, :len(row)] = row
    return out


class TCNEncoder(nn.Module):
    def __init__(self, vocab, emb=96, hidden=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=0)
        blocks = []
        for dilation in (1, 2, 4, 8):
            blocks += [nn.Conv1d(emb if not blocks else hidden, hidden, 3,
                                 padding=dilation, dilation=dilation), nn.ReLU(),
                       nn.BatchNorm1d(hidden)]
        self.net = nn.Sequential(*blocks)
        self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())

    def forward(self, x):
        mask = x.ne(0)
        h = self.embedding(x).transpose(1, 2)
        h = self.net(h).transpose(1, 2)
        return self.proj((h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1))


class MambaStyleEncoder(nn.Module):
    """Pure-PyTorch selective state-space encoder, inspired by Mamba."""
    def __init__(self, vocab, emb=96, hidden=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=0)
        self.in_proj = nn.Linear(emb, hidden * 2)
        self.delta = nn.Linear(emb, hidden)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.log_decay = nn.Parameter(torch.zeros(hidden))

    def forward(self, x):
        mask = x.ne(0)
        e = self.embedding(x)
        u, gate = self.in_proj(e).chunk(2, dim=-1)
        decay = torch.sigmoid(self.log_decay)[None, None, :] * torch.sigmoid(self.delta(e))
        state = e.new_zeros(e.shape[0], e.shape[2] * 2)
        # Project input to state width before the recurrent scan.
        state = e.new_zeros(e.shape[0], u.shape[-1])
        outputs = []
        for t in range(x.shape[1]):
            state = decay[:, t] * state + (1 - decay[:, t]) * torch.tanh(u[:, t])
            outputs.append(state * torch.sigmoid(gate[:, t]))
        h = torch.stack(outputs, dim=1)
        return self.out((h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1))


class PatchTSTStyleEncoder(nn.Module):
    """Patch embedding adaptation; the input remains symptom tokens, not time."""
    def __init__(self, vocab, emb=96, hidden=256, patch=4, stride=2):
        super().__init__()
        self.patch, self.stride = patch, stride
        self.embedding = nn.Embedding(vocab, emb, padding_idx=0)
        self.patch_proj = nn.Linear(patch * emb, hidden)
        layer = nn.TransformerEncoderLayer(hidden, 4, 512, dropout=.15,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        mask = x.ne(0)
        e = self.embedding(x)
        if e.shape[1] < self.patch:
            e = F.pad(e, (0, 0, 0, self.patch - e.shape[1]))
            mask = F.pad(mask, (0, self.patch - mask.shape[1]), value=False)
        n = 1 + (e.shape[1] - self.patch) // self.stride
        patches = torch.stack([e[:, i * self.stride:i * self.stride + self.patch].reshape(len(x), -1)
                               for i in range(n)], dim=1)
        pmask = torch.stack([mask[:, i * self.stride:i * self.stride + self.patch].any(1)
                             for i in range(n)], dim=1)
        h = self.encoder(self.patch_proj(patches), src_key_padding_mask=~pmask)
        return self.norm((h * pmask.unsqueeze(-1)).sum(1) / pmask.sum(1, keepdim=True).clamp_min(1))


class SequenceSetModel(nn.Module):
    def __init__(self, encoder, k, hidden=256, emb=96, dropout=.15):
        super().__init__()
        self.encoder, self.k = encoder, k
        self.labels = nn.Embedding(k + 2, emb)
        self.context = nn.Linear(hidden, hidden)
        self.cell = nn.GRUCell(emb + hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, k + 1)

    def step(self, previous, hidden, context, chosen):
        hidden = self.cell(torch.cat((self.labels(previous), context), dim=1), hidden)
        logits = self.head(self.dropout(hidden))
        illegal = torch.cat((chosen, ~chosen.any(1, keepdim=True)), dim=1)
        return logits.masked_fill(illegal, float("-inf")), hidden

    def loss(self, x, targets):
        context = self.context(self.encoder(x)); hidden = context
        chosen = torch.zeros(len(x), self.k, dtype=torch.bool, device=x.device)
        previous = torch.full((len(x),), self.k + 1, dtype=torch.long, device=x.device)
        sums = context.new_zeros(len(x)); counts = context.new_zeros(len(x))
        for t in range(targets.shape[1]):
            logits, hidden = self.step(previous, hidden, context, chosen)
            target = targets[:, t]; valid = target.ne(-100); active = valid & ~chosen.all(1)
            ce = F.cross_entropy(logits, target, ignore_index=-100, reduction="none")
            sums = sums + ce * active; counts = counts + active
            selected = valid & target.lt(self.k); chosen = chosen.clone()
            chosen[selected, target[selected]] = True
            previous = torch.where(valid, target, torch.full_like(target, self.k))
        return (sums / counts.clamp_min(1)).mean()

    @torch.no_grad()
    def generate(self, x):
        context = self.context(self.encoder(x)); hidden = context
        chosen = torch.zeros(len(x), self.k, dtype=torch.bool, device=x.device)
        previous = torch.full((len(x),), self.k + 1, dtype=torch.long, device=x.device)
        done = torch.zeros(len(x), dtype=torch.bool, device=x.device)
        traces = [[] for _ in range(len(x))]
        for _ in range(self.k + 1):
            logits, hidden = self.step(previous, hidden, context, chosen)
            action = logits.argmax(1)
            for i in range(len(x)):
                if not done[i]: traces[i].append(int(action[i]))
            selected = ~done & action.lt(self.k); chosen[selected, action[selected]] = True
            done = done | action.eq(self.k); previous = action
            if done.all(): break
        return chosen.to(torch.int64), traces


def make_model(name, vocab, k):
    if name == "tcn_sequence": encoder = TCNEncoder(vocab)
    elif name == "mamba_style_sequence": encoder = MambaStyleEncoder(vocab)
    elif name == "patchtst_style_sequence": encoder = PatchTSTStyleEncoder(vocab)
    else: raise ValueError(name)
    return SequenceSetModel(encoder, k)


def decode(model, tokens):
    model.eval(); out, traces = [], []
    for st in range(0, len(tokens), BATCH):
        p, t = model.generate(batch_tokens(tokens, range(st, min(st + BATCH, len(tokens))))); out.append(p.numpy()); traces.extend(t)
    return np.concatenate(out), traces


def fit(model, sets, seed, method):
    tt, y, _ = sets["train"]; vt, vy, _ = sets["val"]; xt, ty, _ = sets["test"]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, amsgrad=True)
    rng = torch.Generator().manual_seed(seed); best = None; best_state = None; history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); order = torch.randperm(len(tt), generator=rng); total = 0.
        for st in range(0, len(tt), BATCH):
            ix = order[st:st + BATCH]; x = batch_tokens(tt, ix)
            # Random input ordering prevents TCN/SSM from learning one arbitrary order.
            permuted = []
            for row in x.tolist():
                row = [z for z in row if z]
                random.shuffle(row); permuted.append(row)
            x = batch_tokens(permuted, range(len(permuted)))
            target = random_targets(torch.tensor(y[ix.numpy()]), rng)
            opt.zero_grad(set_to_none=True); loss = model.loss(x, target); loss.backward(); opt.step(); total += float(loss.detach()) * len(ix)
        pred, _ = decode(model, vt); row = {"epoch": epoch, "loss": total / len(tt), **metrics(vy, pred)}; history.append(row)
        rank = (row["exact"], row["sample_f1"], -epoch)
        if best is None or rank > (best["exact"], best["sample_f1"], -best["epoch"]):
            best = row; best_state = {n: v.detach().clone() for n, v in model.state_dict().items()}
    model.load_state_dict(best_state); pred, traces = decode(model, xt)
    return {"method": method, "seed": seed, "group": "modern_encoder_sequence_set",
            "selection": best, "test": metrics(ty, pred),
            "parameters": sum(p.numel() for p in model.parameters())}, history, pred, traces


def main():
    global EPOCHS
    ap = argparse.ArgumentParser(); ap.add_argument("--cpu-threads", type=int, default=6); ap.add_argument("--smoke-test", action="store_true"); args = ap.parse_args()
    EPOCHS = 2 if args.smoke_test else 100
    torch.set_num_threads(args.cpu_threads); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    sets, vocab, k, schema = prepare(); out = ROOT / ("smoke_results" if args.smoke_test else "results"); out.mkdir(exist_ok=True)
    protocol = {"methods": ["tcn_sequence", "mamba_style_sequence", "patchtst_style_sequence"], "seeds": list(SEEDS), "epochs": EPOCHS, "batch_size": BATCH, "primary_metric": "exact", "secondary_metric": "sample_f1", "split": "record-level 70/15/15", "adaptation": "symptom-token encoder with random input permutation augmentation; shared GRU ADD/STOP decoder", "schema": schema}
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    for method in protocol["methods"]:
        for seed in ((SEEDS[0],) if args.smoke_test else SEEDS):
            folder = out / method / f"seed_{seed}";
            if (folder / "summary.json").exists(): continue
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); model = make_model(method, vocab, k)
            summary, history, pred, traces = fit(model, sets, seed, method); summary["protocol"] = protocol; folder.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "selection": summary["selection"]}, folder / "best.pt")
            (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            (folder / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            (folder / "predictions.json").write_text(json.dumps({"ids": sets["test"][2], "truth": sets["test"][1].astype(int).tolist(), "predicted": pred.astype(int).tolist(), "traces": traces}, ensure_ascii=False), encoding="utf-8")
    all_s = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*/seed_*/summary.json"))]
    (out / "all_summaries.json").write_text(json.dumps(all_s, ensure_ascii=False, indent=2), encoding="utf-8"); print("COMPLETE", len(all_s), "runs", flush=True)


if __name__ == "__main__": main()
