"""Run the locked matched-encoder comparison on three internal CV folds."""

import argparse
import copy
import csv
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from gru_sequence import SymptomGRU, random_targets
from models import ClassifierChain, SigmoidModel


ROOT = Path(__file__).resolve().parent
METHODS = ("sigmoid_small", "sigmoid_matched", "classifier_chain", "gru_stop")
SEEDS = (9, 17, 29)
FOLDS = (1, 2, 3)
BATCH = 64
EPOCHS = 100
THRESHOLDS = np.arange(5, 96, dtype=np.float64) / 100.0
EMA_DECAY = 0.995


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def load_data():
    manifest = json.loads((ROOT / "splits" / "protocol.json").read_text(encoding="utf-8"))
    data_path = ROOT / manifest["data_file"]
    if sha256(data_path) != manifest["data_sha256"]:
        raise ValueError("Data file differs from locked split protocol")
    if manifest["record_count"] != 4200 or manifest["outer_folds"] != 3:
        raise ValueError("Unexpected split protocol")
    if (manifest["training_seeds"], manifest["epochs"], manifest["batch_size"]) != (
            list(SEEDS), EPOCHS, BATCH):
        raise ValueError("Training budget differs from locked split protocol")

    symptoms = manifest["schema"]["symptoms"]
    labels = manifest["schema"]["labels"]
    symptom_ids = {name: i + 2 for i, name in enumerate(symptoms)}
    label_ids = {name: i for i, name in enumerate(labels)}
    records = {}
    with data_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=" ")
        if reader.fieldnames != ["id", "symptom", "Se"]:
            raise ValueError("Unexpected data columns")
        for row in reader:
            record_id = int(row["id"])
            if record_id in records:
                raise ValueError(f"Repeated record id {record_id}")
            tokens = sorted(symptom_ids[name] for name in row["symptom"].split(","))
            target = np.zeros(len(labels), dtype=np.int64)
            for name in row["Se"].split(","):
                target[label_ids[name]] = 1
            records[record_id] = (tokens, target)
    if sorted(records) != list(range(1, 4201)):
        raise ValueError("Expected record ids 1..4200")

    folds = {}
    for info in manifest["folds"]:
        path = ROOT / "splits" / info["file"]
        if sha256(path) != info["sha256"]:
            raise ValueError(f"Split file differs: {path.name}")
        ids = {role: [] for role in ("train", "val", "test")}
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                ids[row["split"]].append(int(row["id"]))
        if (sorted(i for group in ids.values() for i in group) != sorted(records)
                or any(len(ids[role]) != info["sizes"][role] for role in ids)):
            raise ValueError(f"Invalid partitions in {path.name}")
        folds[info["fold"]] = {
            role: (ids[role], [records[i][0] for i in ids[role]],
                   np.stack([records[i][1] for i in ids[role]]))
            for role in ids
        }
    if set(folds) != set(FOLDS):
        raise ValueError("Expected three folds")
    test_ids = [i for fold in FOLDS for i in folds[fold]["test"][0]]
    if sorted(test_ids) != sorted(records):
        raise ValueError("Outer test folds must cover each id exactly once")
    return manifest, folds, len(symptoms) + 2, len(labels)


def batch_tokens(rows, indices):
    selected = [rows[int(i)] for i in indices]
    tokens = torch.zeros((len(selected), max(map(len, selected))), dtype=torch.long)
    for row, values in enumerate(selected):
        tokens[row, :len(values)] = torch.tensor(values, dtype=torch.long)
    return tokens


def metrics(truth, predicted):
    truth = np.asarray(truth, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    tp = (truth * predicted).sum(1)
    denominator = truth.sum(1) + predicted.sum(1)
    sample_f1 = np.divide(2 * tp, denominator, out=np.zeros(len(tp), dtype=float),
                          where=denominator > 0)
    label_tp = (truth * predicted).sum(0)
    label_den = truth.sum(0) + predicted.sum(0)
    label_f1 = np.divide(2 * label_tp, label_den,
                         out=np.zeros(len(label_tp), dtype=float), where=label_den > 0)
    return {
        "exact": float((truth == predicted).all(1).mean()),
        "sample_f1": float(sample_f1.mean()),
        "micro_f1": float(2 * tp.sum() / max(1, denominator.sum())),
        "macro_f1": float(label_f1.mean()),
        "cardinality_mae": float(abs(truth.sum(1) - predicted.sum(1)).mean()),
    }


def model_for(method, vocab, labels):
    if method == "sigmoid_small":
        return SigmoidModel(vocab, labels)
    if method == "sigmoid_matched":
        return SigmoidModel(vocab, labels, matched=True)
    if method == "classifier_chain":
        return ClassifierChain(vocab, labels)
    if method == "gru_stop":
        return SymptomGRU(vocab, labels, embedding=128, hidden=256)
    raise ValueError(f"Unknown method {method}")


@torch.no_grad()
def validation_predictions(model, method, tokens):
    model.eval()
    x = batch_tokens(tokens, range(len(tokens)))
    if method == "gru_stop":
        predicted, _ = model.generate(x)
        return predicted.numpy()[None, :, :]
    if method == "classifier_chain":
        return model.predict_thresholds(x, THRESHOLDS).numpy()
    scores = model(x).sigmoid().numpy()
    candidates = (scores[None, :, :] >= THRESHOLDS[:, None, None]).astype(np.int64)
    empty_threshold, empty_record = np.where(candidates.sum(2) == 0)
    candidates[empty_threshold, empty_record, scores[empty_record].argmax(1)] = 1
    return candidates


def select_validation(model, method, tokens, truth):
    candidates = validation_predictions(model, method, tokens)
    if method == "gru_stop":
        return metrics(truth, candidates[0])
    ranked = []
    for index, threshold in enumerate(THRESHOLDS):
        row = metrics(truth, candidates[index])
        rank = (row["exact"], row["sample_f1"], row["micro_f1"], -threshold)
        ranked.append((rank, {"threshold": float(threshold), **row}))
    return max(ranked, key=lambda item: item[0])[1]


@torch.no_grad()
def test_predictions(model, method, tokens, threshold):
    model.eval()
    x = batch_tokens(tokens, range(len(tokens)))
    if method == "gru_stop":
        predicted, traces = model.generate(x)
        return predicted.numpy(), traces
    if method == "classifier_chain":
        return model.predict_thresholds(x, [threshold])[0].numpy(), None
    scores = model(x).sigmoid().numpy()
    predicted = (scores >= threshold).astype(np.int64)
    empty = predicted.sum(1) == 0
    if empty.any():
        predicted[empty, scores[empty].argmax(1)] = 1
    return predicted, None


def update_ema(ema, model, step):
    decay = min(EMA_DECAY, 1.0 - 1.0 / (step + 1))
    with torch.no_grad():
        for averaged, current in zip(ema.parameters(), model.parameters()):
            averaged.lerp_(current, 1.0 - decay)
        for averaged, current in zip(ema.buffers(), model.buffers()):
            averaged.copy_(current)


def train_one(method, fold, seed, folds, vocab, labels, output, protocol_hash):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = model_for(method, vocab, labels)
    ema = copy.deepcopy(model)
    ema.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                  weight_decay=1e-4, amsgrad=True)
    order_generator = torch.Generator().manual_seed(seed)
    target_generator = torch.Generator().manual_seed(100000 + seed)
    train_ids, train_tokens, train_y = folds[fold]["train"]
    _, val_tokens, val_y = folds[fold]["val"]
    test_ids, test_tokens, test_y = folds[fold]["test"]
    history = []
    best = None
    best_state = None
    step = 0
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_ids), generator=order_generator)
        total_loss = 0.0
        for start in range(0, len(train_ids), BATCH):
            indices = order[start:start + BATCH]
            x = batch_tokens(train_tokens, indices)
            truth = torch.as_tensor(train_y[indices.numpy()], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            if method == "gru_stop":
                targets = random_targets(truth, target_generator)
                loss = model.loss(x, targets)
            else:
                loss = model.loss(x, truth)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: {method}, fold {fold}, seed {seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                           error_if_nonfinite=True)
            optimizer.step()
            step += 1
            update_ema(ema, model, step)
            total_loss += float(loss.detach()) * len(indices)
        row = {"epoch": epoch, "train_loss": total_loss / len(train_ids),
               **select_validation(ema, method, val_tokens, val_y)}
        history.append(row)
        rank = (row["exact"], row["sample_f1"], -epoch)
        if best is None or rank > (best["exact"], best["sample_f1"], -best["epoch"]):
            best = row
            best_state = {name: tensor.detach().clone()
                          for name, tensor in ema.state_dict().items()}

    training_seconds = time.perf_counter() - started
    ema.load_state_dict(best_state)
    threshold = best.get("threshold")
    started = time.perf_counter()
    predicted, traces = test_predictions(ema, method, test_tokens, threshold)
    inference_seconds = time.perf_counter() - started
    summary = {
        "method": method, "fold": fold, "seed": seed,
        "selection": best, "test": metrics(test_y, predicted),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": training_seconds,
        "test_inference_seconds": inference_seconds,
        "test_records": len(test_ids),
        "protocol_sha256": protocol_hash,
    }
    predictions = {"ids": test_ids, "truth": test_y.astype(int).tolist(),
                   "predicted": predicted.astype(int).tolist()}
    if traces is not None:
        predictions["traces"] = traces
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "selection": best}, output / "best.pt")
    save_json(output / "history.json", history)
    save_json(output / "predictions.json", predictions)
    save_json(output / "summary.json", summary)
    return summary


def run_smoke(folds, vocab, labels):
    ids, rows, y = folds[1]["train"]
    _, val_rows, val_y = folds[1]["val"]
    for method in METHODS:
        torch.manual_seed(9)
        model = model_for(method, vocab, labels)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        x = batch_tokens(rows, range(BATCH))
        truth = torch.as_tensor(y[:BATCH], dtype=torch.float32)
        targets = random_targets(truth, torch.Generator().manual_seed(9))
        loss = model.loss(x, targets) if method == "gru_stop" else model.loss(x, truth)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = select_validation(model, method, val_rows, val_y)
        print(f"SMOKE {method} params={sum(p.numel() for p in model.parameters())} "
              f"loss={float(loss.detach()):.4f} val_exact={row['exact']:.4f}", flush=True)
    print("SMOKE COMPLETE; outer test records were not evaluated", flush=True)


def make_run_protocol(split_protocol):
    sources = [ROOT / "run_all.py", ROOT / "models.py", ROOT / "gru_sequence.py",
               ROOT / "广安门心绞痛4200例.txt", ROOT / "splits" / "protocol.json"]
    sources += [ROOT / "splits" / f"fold_{fold}.csv" for fold in FOLDS]
    return {
        "study": "matched mean-encoder comparison, internal CV",
        "methods": list(METHODS), "folds": list(FOLDS), "seeds": list(SEEDS),
        "epochs": EPOCHS, "batch_size": BATCH, "device": "cpu",
        "encoder": "Embedding(42,128,pad=0) -> symptom mean -> Linear(128,256) -> Tanh; independently trained, same seed initialization",
        "randomness": "same seed controls initialization and batch order; GRU target permutations use independent seed 100000+seed",
        "sigmoid_small": "single Linear(256,K); BCE; sigmoid threshold on validation",
        "sigmoid_matched": "Linear(256,964)-ReLU-Linear(964,256)-ReLU-Linear(256,K); BCE",
        "classifier_chain": "vocabulary label order; teacher-forced BCE; free-running threshold recalibration on validation",
        "gru_stop": "unchanged SymptomGRU source; random target order; joint K+1 ADD/STOP action softmax",
        "thresholds": [float(x) for x in THRESHOLDS],
        "threshold_ties": ["validation exact", "validation sample F1",
                           "validation micro F1", "smaller threshold"],
        "checkpoint_ties": ["validation exact", "validation sample F1",
                            "earlier epoch"],
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4,
                      "amsgrad": True, "gradient_clip_norm": 1.0,
                      "ema_decay_cap": EMA_DECAY},
        "split_protocol_sha256": sha256(ROOT / "splits" / "protocol.json"),
        "data_sha256": split_protocol["data_sha256"],
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__, "numpy": np.__version__},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.smoke_test and args.init_only:
        parser.error("--smoke-test and --init-only cannot be combined")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    split_protocol, folds, vocab, labels = load_data()
    if args.smoke_test:
        run_smoke(folds, vocab, labels)
        return

    protocol = make_run_protocol(split_protocol)
    output = ROOT / "results"
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous != protocol:
            raise ValueError("Existing results use a different protocol or source")
    else:
        output.mkdir(exist_ok=True)
        save_json(protocol_path, protocol)
    protocol_hash = sha256(protocol_path)
    if args.init_only:
        print(f"INITIALIZED {protocol_path}", flush=True)
        return
    selected_methods = [args.method] if args.method else METHODS
    selected_folds = [args.fold] if args.fold else FOLDS
    selected_seeds = [args.seed] if args.seed else SEEDS
    for method in selected_methods:
        for fold in selected_folds:
            for seed in selected_seeds:
                folder = output / method / f"fold_{fold}" / f"seed_{seed}"
                required = [folder / name for name in
                            ("summary.json", "predictions.json", "history.json", "best.pt")]
                if all(path.exists() for path in required):
                    summary = json.loads(required[0].read_text(encoding="utf-8"))
                    if summary["protocol_sha256"] != protocol_hash:
                        raise ValueError(f"Completed run has different protocol: {folder}")
                    print(f"SKIP {method} fold={fold} seed={seed}", flush=True)
                    continue
                summary = train_one(method, fold, seed, folds, vocab, labels,
                                    folder, protocol_hash)
                print(f"DONE {method} fold={fold} seed={seed} "
                      f"exact={summary['test']['exact']:.4f}", flush=True)
    complete = len(list(output.glob("*/fold_*/seed_*/summary.json")))
    print(f"COMPLETE {complete}/36 runs", flush=True)


if __name__ == "__main__":
    main()
