#!/usr/bin/env python3
"""Train the standalone GRU sequence-to-set model using the bundled 3-fold protocol."""

import argparse
import csv
import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from gru_sequence import SymptomGRU, random_targets


ROOT = Path(__file__).resolve().parent
FOLDS = (1, 2, 3)
SEEDS = (9, 17, 29)
EPOCHS = 100
BATCH_SIZE = 64
EMA_DECAY = 0.995


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def read_dataset(protocol):
    data_path = ROOT / protocol["data_file"]
    if sha256(data_path) != protocol["data_sha256"]:
        raise ValueError("Bundled dataset does not match the locked protocol")
    symptom_ids = {name: index + 2
                   for index, name in enumerate(protocol["schema"]["symptoms"])}
    label_ids = {name: index
                 for index, name in enumerate(protocol["schema"]["labels"])}
    records = {}
    with data_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=" ")
        if reader.fieldnames != ["id", "symptom", "Se"]:
            raise ValueError("Dataset columns must be id, symptom, Se")
        for row in reader:
            record_id = int(row["id"])
            if record_id in records:
                raise ValueError("Repeated record id: {}".format(record_id))
            symptoms = [symptom_ids[name] for name in row["symptom"].split(",")]
            target = np.zeros(len(label_ids), dtype=np.float32)
            for label in row["Se"].split(","):
                target[label_ids[label]] = 1.0
            records[record_id] = (symptoms, target)
    if sorted(records) != list(range(1, protocol["record_count"] + 1)):
        raise ValueError("Expected record IDs 1 through {}".format(protocol["record_count"]))
    return records, symptom_ids, label_ids


def read_folds(protocol):
    folds = {}
    all_test_ids = []
    for info in protocol["folds"]:
        fold = info["fold"]
        path = ROOT / "splits" / info["file"]
        if sha256(path) != info["sha256"]:
            raise ValueError("Split file hash mismatch: {}".format(path.name))
        groups = {name: [] for name in ("train", "val", "test")}
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                groups[row["split"]].append(int(row["id"]))
        if any(len(groups[name]) != info["sizes"][name] for name in groups):
            raise ValueError("Split sizes differ from locked protocol")
        folds[fold] = groups
        all_test_ids.extend(groups["test"])
    if sorted(all_test_ids) != list(range(1, protocol["record_count"] + 1)):
        raise ValueError("Outer test folds must cover every record once")
    return folds


def batch_tokens(rows, indices):
    selected = [rows[int(index)] for index in indices]
    result = torch.zeros((len(selected), max(map(len, selected))), dtype=torch.long)
    for i, row in enumerate(selected):
        result[i, :len(row)] = torch.tensor(row, dtype=torch.long)
    return result


def metrics(truth, predicted):
    truth = np.asarray(truth, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    overlap = (truth * predicted).sum(1)
    denominator = truth.sum(1) + predicted.sum(1)
    sample = np.divide(2 * overlap, denominator,
                       out=np.zeros(len(truth), dtype=float), where=denominator > 0)
    tp = (truth * predicted).sum(0)
    label_denominator = truth.sum(0) + predicted.sum(0)
    label_f1 = np.divide(2 * tp, label_denominator,
                         out=np.zeros(len(tp), dtype=float), where=label_denominator > 0)
    return {
        "exact": float((truth == predicted).all(1).mean()),
        "sample_f1": float(sample.mean()),
        "micro_f1": float(2 * overlap.sum() / max(1, denominator.sum())),
        "macro_f1": float(label_f1.mean()),
        "cardinality_mae": float(abs(truth.sum(1) - predicted.sum(1)).mean()),
    }


@torch.no_grad()
def predict(model, tokens):
    model.eval()
    outputs = []
    traces = []
    for start in range(0, len(tokens), BATCH_SIZE):
        batch = batch_tokens(tokens, range(start, min(start + BATCH_SIZE, len(tokens))))
        prediction, batch_traces = model.generate(batch)
        outputs.append(prediction.cpu().numpy())
        traces.extend(batch_traces)
    return np.concatenate(outputs, axis=0), traces


def update_ema(ema, model, step):
    decay = min(EMA_DECAY, 1.0 - 1.0 / (step + 1))
    with torch.no_grad():
        for averaged, current in zip(ema.parameters(), model.parameters()):
            averaged.lerp_(current, 1.0 - decay)
        for averaged, current in zip(ema.buffers(), model.buffers()):
            averaged.copy_(current)


def snapshot(model):
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()}


def train_one(fold, seed, groups, records, protocol, output, epochs):
    folder = output / "gru_stop" / "fold_{}".format(fold) / "seed_{}".format(seed)
    folder.mkdir(parents=True, exist_ok=True)
    best_path = folder / "best.pt"
    if best_path.exists() and (folder / "summary.json").exists():
        saved = torch.load(best_path, map_location="cpu", weights_only=False)
        if saved.get("protocol_sha256") == sha256(output / "protocol.json"):
            print("SKIP fold {} seed {}".format(fold, seed), flush=True)
            return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    schema = protocol["schema"]
    vocab_size = len(schema["symptoms"]) + 2
    label_count = len(schema["labels"])
    model = SymptomGRU(vocab_size, label_count, embedding=128, hidden=256)
    ema = deepcopy(model)
    ema.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                  weight_decay=1e-4, amsgrad=True)
    order_generator = torch.Generator().manual_seed(seed)
    target_generator = torch.Generator().manual_seed(100000 + seed)

    train_ids = groups[fold]["train"]
    val_ids = groups[fold]["val"]
    test_ids = groups[fold]["test"]
    train_tokens = [records[index][0] for index in train_ids]
    train_y = np.stack([records[index][1] for index in train_ids])
    val_tokens = [records[index][0] for index in val_ids]
    val_y = np.stack([records[index][1] for index in val_ids])
    history = []
    best = None
    best_state = None
    step = 0
    first_epoch = 1

    last_path = folder / "last.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("protocol_sha256") != sha256(output / "protocol.json"):
            raise ValueError("Saved checkpoint uses a different protocol")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        order_generator.set_state(state["order_rng"])
        target_generator.set_state(state["target_rng"])
        torch.set_rng_state(state["torch_rng"])
        history = state["history"]
        best = state["best"]
        best_state = state["best_state"]
        step = state["step"]
        first_epoch = state["epoch"] + 1

    for epoch in range(first_epoch, epochs + 1):
        model.train()
        order = torch.randperm(len(train_ids), generator=order_generator)
        total_loss = 0.0
        for start in range(0, len(train_ids), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            x = batch_tokens(train_tokens, indices)
            y = torch.as_tensor(train_y[indices.numpy()], dtype=torch.float32)
            targets = random_targets(y, target_generator)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite GRU training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                           error_if_nonfinite=True)
            optimizer.step()
            step += 1
            update_ema(ema, model, step)
            total_loss += float(loss.detach()) * len(indices)

        val_prediction, _ = predict(ema, val_tokens)
        row = {"epoch": epoch, "train_loss": total_loss / len(train_ids),
               **metrics(val_y, val_prediction)}
        history.append(row)
        rank = (row["exact"], row["sample_f1"], -epoch)
        if best is None or rank > (best["exact"], best["sample_f1"], -best["epoch"]):
            best = row
            best_state = snapshot(ema)

        torch.save({
            "model": snapshot(model), "ema": snapshot(ema),
            "optimizer": optimizer.state_dict(),
            "order_rng": order_generator.get_state(),
            "target_rng": target_generator.get_state(),
            "torch_rng": torch.get_rng_state(), "epoch": epoch, "step": step,
            "history": history, "best": best, "best_state": best_state,
            "protocol_sha256": sha256(output / "protocol.json"),
        }, last_path)
        save_json(folder / "history.json", history)
        print("fold {} seed {} epoch {}/{}: val EM={:.4f}, sample-F1={:.4f}".format(
            fold, seed, epoch, epochs, row["exact"], row["sample_f1"]), flush=True)

    ema.load_state_dict(best_state)
    test_tokens = [records[index][0] for index in test_ids]
    test_y = np.stack([records[index][1] for index in test_ids])
    test_prediction, traces = predict(ema, test_tokens)
    checkpoint = {
        "model": best_state,
        "model_config": {"vocab_size": vocab_size, "label_count": label_count,
                         "embedding": 128, "hidden": 256, "dropout": 0.15},
        "schema": schema, "fold": fold, "seed": seed, "selection": best,
        "test_ids": test_ids, "protocol_sha256": sha256(output / "protocol.json"),
        "data_sha256": protocol["data_sha256"],
    }
    torch.save(checkpoint, best_path)
    save_json(folder / "history.json", history)
    save_json(folder / "predictions.json", {
        "ids": test_ids, "truth": test_y.astype(int).tolist(),
        "predicted": test_prediction.astype(int).tolist(), "traces": traces,
    })
    save_json(folder / "summary.json", {
        "method": "gru_stop", "fold": fold, "seed": seed,
        "selection": best, "test": metrics(test_y, test_prediction),
        "test_records": len(test_ids),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "protocol_sha256": sha256(output / "protocol.json"),
    })
    print("DONE fold {} seed {}: test EM={:.4f}".format(
        fold, seed, metrics(test_y, test_prediction)["exact"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, choices=FOLDS, default=list(FOLDS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.cpu_threads < 1:
        parser.error("epochs and cpu threads must be positive")

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    source = json.loads((ROOT / "splits" / "protocol.json").read_text(encoding="utf-8"))
    if args.smoke_test:
        records, _, _ = read_dataset(source)
        rows = list(records.values())[:16]
        tokens = [row[0] for row in rows]
        y = torch.as_tensor(np.stack([row[1] for row in rows]), dtype=torch.float32)
        model = SymptomGRU(len(source["schema"]["symptoms"]) + 2,
                           len(source["schema"]["labels"]))
        targets = random_targets(y, torch.Generator().manual_seed(9))
        loss = model.loss(batch_tokens(tokens, range(len(tokens))), targets)
        loss.backward()
        predicted, _ = predict(model.eval(), tokens)
        assert torch.isfinite(loss) and predicted.shape == y.shape
        assert (predicted.sum(1) > 0).all()
        print("SMOKE PASS: GRU loss, backpropagation and set decoding")
        return
    if args.epochs != source["epochs"]:
        raise ValueError("Training epochs differ from the bundled locked protocol")
    if not set(args.seeds).issubset(set(source["training_seeds"])):
        raise ValueError("Seeds must be selected from the bundled locked protocol")
    records, _, _ = read_dataset(source)
    groups = read_folds(source)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "model": "SymptomGRU variable-cardinality sequence-to-set",
        "folds": list(args.folds), "seeds": list(args.seeds),
        "epochs": args.epochs, "batch_size": BATCH_SIZE,
        "primary_metric": "exact match", "checkpoint_selection": [
            "validation exact match", "validation sample F1", "earlier epoch"],
        "data_sha256": source["data_sha256"],
        "split_protocol_sha256": sha256(ROOT / "splits" / "protocol.json"),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted(ROOT.rglob("*.py"))
        },
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Output directory uses a different protocol; choose another --output")
    save_json(protocol_path, protocol)

    for fold in args.folds:
        for seed in args.seeds:
            train_one(fold, seed, groups, records, source, output, args.epochs)
    print("COMPLETE {} run(s)".format(len(args.folds) * len(args.seeds)), flush=True)


if __name__ == "__main__":
    main()
