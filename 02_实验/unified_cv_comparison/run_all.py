"""Run additional model comparisons on the locked three-fold protocol."""

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
from torch.nn import functional as F

import direct_baselines
import modern_extensions
import set_baselines
from gru_sequence import random_targets
from mamba_robust_runner import PermutationAveragedMamba
from models.transformer_sequence import SymptomTransformerSequence
from models.transformer_set import SymptomTransformer
from run_modern_encoders import SequenceSetModel


ROOT = Path(__file__).resolve().parent
DIRECT_METHODS = ("mlp", "cnn", "symptom_transformer", "asl_mlp",
                  "classifier_chain_legacy", "label_graph", "set_attention")
SEQUENCE_METHODS = ("transformer_sequence_matched", "tcn_sequence",
                    "patchtst_style_sequence", "mamba_style_v3")
SET_METHODS = ("set_nll", "set_rl")
METHODS = DIRECT_METHODS + SEQUENCE_METHODS + SET_METHODS
OUTPUT_METHODS = DIRECT_METHODS + SEQUENCE_METHODS + (
    "set_nll_greedy", "set_nll_beam5", "set_nll_map", "set_rl")
SEEDS = (9, 17, 29)
FOLDS = (1, 2, 3)
BATCH = 64
EPOCHS = 100
THRESHOLDS = np.arange(5, 96, dtype=np.float64) / 100.0
EMA_DECAY = 0.995
RL_PRETRAIN_EPOCHS = 50
ORDER_AUGMENTED = ("tcn_sequence", "patchtst_style_sequence")


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


def batch_tokens(rows, indices, pad_to=None):
    selected = [rows[int(i)] for i in indices]
    width = max(map(len, selected))
    if pad_to is not None:
        if pad_to < width:
            raise ValueError("Fixed token width is shorter than a symptom set")
        width = pad_to
    tokens = torch.zeros((len(selected), width), dtype=torch.long)
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


def model_for(method, vocab, labels, train_y=None):
    if method in direct_baselines.METHODS:
        return direct_baselines.build(method, vocab, labels, train_y)
    if method == "symptom_transformer":
        return SymptomTransformer(vocab, labels, embedding=128, heads=4,
                                  layers=2, feedforward=256)
    if method == "transformer_sequence_matched":
        return SymptomTransformerSequence(vocab, labels, embedding=160,
                                          hidden=192, heads=4, layers=1,
                                          feedforward=384)
    if method == "mamba_style_v3":
        return SequenceSetModel(PermutationAveragedMamba(vocab), labels)
    if method in ("tcn_sequence", "patchtst_style_sequence"):
        return modern_extensions.build(method, vocab, labels)
    if method in SET_METHODS:
        return set_baselines.build(vocab, labels)
    raise ValueError(f"Unknown method {method}")


@torch.no_grad()
def direct_scores(model, tokens):
    model.eval()
    chunks = []
    for start in range(0, len(tokens), BATCH):
        x = batch_tokens(tokens, range(start, min(start + BATCH, len(tokens))))
        chunks.append(model(x).sigmoid().numpy())
    return np.concatenate(chunks)


@torch.no_grad()
def decode_sequence(model, tokens, pad_to=None):
    model.eval()
    predictions, traces = [], []
    for start in range(0, len(tokens), BATCH):
        x = batch_tokens(tokens, range(start, min(start + BATCH, len(tokens))),
                         pad_to=pad_to)
        predicted, trace = model.generate(x)
        predictions.append(predicted.numpy())
        traces.extend(trace)
    return np.concatenate(predictions), traces


def validation_predictions(model, method, tokens):
    if method in SEQUENCE_METHODS:
        predicted, _ = decode_sequence(
            model, tokens, pad_to=model.encoder.embedding.num_embeddings - 2
            if method == "tcn_sequence" else None)
        return predicted[None, :, :]
    if method in SET_METHODS:
        predicted, _ = set_baselines.decode(model, tokens, "greedy", batch_tokens, BATCH)
        return predicted[None, :, :]
    if method == "classifier_chain_legacy":
        model.eval()
        chunks = []
        for start in range(0, len(tokens), BATCH):
            x = batch_tokens(tokens, range(start, min(start + BATCH, len(tokens))))
            chunks.append(direct_baselines.predict_thresholds(
                model, method, x, THRESHOLDS).numpy())
        return np.concatenate(chunks, axis=1)
    scores = direct_scores(model, tokens)
    candidates = (scores[None, :, :] >= THRESHOLDS[:, None, None]).astype(np.int64)
    empty_threshold, empty_record = np.where(candidates.sum(2) == 0)
    candidates[empty_threshold, empty_record, scores[empty_record].argmax(1)] = 1
    return candidates


def select_validation(model, method, tokens, truth):
    candidates = validation_predictions(model, method, tokens)
    if method in SEQUENCE_METHODS + SET_METHODS:
        return metrics(truth, candidates[0])
    ranked = []
    for index, threshold in enumerate(THRESHOLDS):
        row = metrics(truth, candidates[index])
        rank = (row["exact"], row["sample_f1"], row["micro_f1"], -threshold)
        ranked.append((rank, {"threshold": float(threshold), **row}))
    return max(ranked, key=lambda item: item[0])[1]


@torch.no_grad()
def test_predictions(model, method, tokens, threshold):
    if method in SEQUENCE_METHODS:
        return decode_sequence(
            model, tokens, pad_to=model.encoder.embedding.num_embeddings - 2
            if method == "tcn_sequence" else None)
    if method in SET_METHODS:
        return set_baselines.decode(model, tokens, "greedy", batch_tokens, BATCH)
    if method == "classifier_chain_legacy":
        model.eval()
        chunks = []
        for start in range(0, len(tokens), BATCH):
            x = batch_tokens(tokens, range(start, min(start + BATCH, len(tokens))))
            chunks.append(direct_baselines.predict_thresholds(
                model, method, x, [threshold])[0].numpy())
        return np.concatenate(chunks), None
    scores = direct_scores(model, tokens)
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


def permuted_batch(rows, indices, generator, pad_to=None):
    selected = [list(rows[int(index)]) for index in indices]
    for row in selected:
        generator.shuffle(row)
    return batch_tokens(selected, range(len(selected)), pad_to=pad_to)


def output_methods(method):
    if method == "set_nll":
        return ("set_nll_greedy", "set_nll_beam5", "set_nll_map")
    return (method,)


def run_folder(results, method, fold, seed):
    return results / method / f"fold_{fold}" / f"seed_{seed}"


def train_one(method, fold, seed, folds, vocab, labels, results, protocol_hash):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_ids, train_tokens, train_y = folds[fold]["train"]
    _, val_tokens, val_y = folds[fold]["val"]
    test_ids, test_tokens, test_y = folds[fold]["test"]
    model = model_for(method, vocab, labels, train_y)
    ema = copy.deepcopy(model)
    ema.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                  weight_decay=1e-4, amsgrad=True)
    order_generator = torch.Generator().manual_seed(seed)
    target_generator = torch.Generator().manual_seed(100000 + seed)
    sample_generator = torch.Generator().manual_seed(30018 + seed)
    symptom_order_generator = random.Random(200000 + seed)
    history = []
    best = None
    best_state = None
    pretraining_best = None
    pretraining_state = None
    step = 0
    checkpoint_path = (results / "_checkpoints" / method /
                       f"fold_{fold}" / f"seed_{seed}" / "last.pt")
    first_epoch = 1
    elapsed_before_resume = 0.0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (saved["protocol_sha256"] != protocol_hash
                or (saved["method"], saved["fold"], saved["seed"]) !=
                (method, fold, seed)):
            raise ValueError(f"Epoch checkpoint uses a different protocol: {checkpoint_path}")
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        order_generator.set_state(saved["order_generator"])
        target_generator.set_state(saved["target_generator"])
        sample_generator.set_state(saved["sample_generator"])
        symptom_order_generator.setstate(saved["symptom_order_generator"])
        torch.set_rng_state(saved["torch_rng"])
        history = saved["history"]
        best, best_state = saved["best"], saved["best_state"]
        pretraining_best = saved["pretraining_best"]
        pretraining_state = saved["pretraining_state"]
        step = saved["step"]
        first_epoch = saved["epoch"] + 1
        elapsed_before_resume = saved["training_seconds"]
    started = time.perf_counter()

    for epoch in range(first_epoch, EPOCHS + 1):
        if method == "set_rl" and epoch == RL_PRETRAIN_EPOCHS + 1:
            if pretraining_state is None:
                raise RuntimeError("Set-RL requires a selected supervised checkpoint")
            model.load_state_dict(pretraining_state)
            ema = copy.deepcopy(model)
            ema.requires_grad_(False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5,
                                          weight_decay=1e-4, amsgrad=True)
            step = 0
        model.train()
        order = torch.randperm(len(train_ids), generator=order_generator)
        total_loss = 0.0
        stage = ("rl" if method == "set_rl" and epoch > RL_PRETRAIN_EPOCHS
                 else "nll" if method in SET_METHODS else "supervised")
        for start in range(0, len(train_ids), BATCH):
            indices = order[start:start + BATCH]
            x = (permuted_batch(train_tokens, indices, symptom_order_generator,
                                pad_to=vocab - 2 if method == "tcn_sequence" else None)
                 if method in ORDER_AUGMENTED else batch_tokens(train_tokens, indices))
            truth = torch.as_tensor(train_y[indices.numpy()], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            if method in SET_METHODS:
                batch_loss = set_baselines.train_step(
                    model, x, truth, stage, sample_generator)
            elif method in SEQUENCE_METHODS:
                targets = random_targets(truth, target_generator)
                loss = model.loss(x, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss: {method}, fold {fold}, seed {seed}")
                loss.backward()
                batch_loss = float(loss.detach())
            elif method in direct_baselines.METHODS:
                loss = direct_baselines.loss(model, method, x, truth)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss: {method}, fold {fold}, seed {seed}")
                loss.backward()
                batch_loss = float(loss.detach())
            else:
                loss = F.binary_cross_entropy_with_logits(model(x), truth)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss: {method}, fold {fold}, seed {seed}")
                loss.backward()
                batch_loss = float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                           error_if_nonfinite=True)
            optimizer.step()
            step += 1
            update_ema(ema, model, step)
            total_loss += batch_loss * len(indices)
        row = {"epoch": epoch, "stage": stage,
               "train_loss": total_loss / len(train_ids),
               **select_validation(ema, method, val_tokens, val_y)}
        history.append(row)
        rank = (row["exact"], row["sample_f1"], -epoch)
        if method == "set_rl" and epoch <= RL_PRETRAIN_EPOCHS:
            if (pretraining_best is None or rank >
                    (pretraining_best["exact"], pretraining_best["sample_f1"],
                     -pretraining_best["epoch"])):
                pretraining_best = row
                pretraining_state = {name: tensor.detach().clone()
                                     for name, tensor in ema.state_dict().items()}
        elif best is None or rank > (best["exact"], best["sample_f1"], -best["epoch"]):
            best = row
            best_state = {name: tensor.detach().clone()
                          for name, tensor in ema.state_dict().items()}

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "protocol_sha256": protocol_hash, "method": method,
            "fold": fold, "seed": seed, "epoch": epoch,
            "model": model.state_dict(), "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "order_generator": order_generator.get_state(),
            "target_generator": target_generator.get_state(),
            "sample_generator": sample_generator.get_state(),
            "symptom_order_generator": symptom_order_generator.getstate(),
            "torch_rng": torch.get_rng_state(), "history": history,
            "best": best, "best_state": best_state,
            "pretraining_best": pretraining_best,
            "pretraining_state": pretraining_state, "step": step,
            "training_seconds": elapsed_before_resume + time.perf_counter() - started,
        }
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(checkpoint_path)
        if method in SET_METHODS or epoch == 1 or epoch % 10 == 0:
            selected = pretraining_best if method == "set_rl" and stage == "nll" else best
            print(f"EPOCH {method} fold={fold} seed={seed} "
                  f"{epoch}/{EPOCHS} stage={stage} "
                  f"val_exact={row['exact']:.4f} best={selected['epoch']}", flush=True)

    training_seconds = elapsed_before_resume + time.perf_counter() - started
    if best_state is None or (method == "set_rl" and best["epoch"] <= RL_PRETRAIN_EPOCHS):
        raise RuntimeError(f"No eligible checkpoint for {method}, fold {fold}, seed {seed}")
    pretraining_predicted = None
    pretraining_traces = None
    pretraining_inference_seconds = None
    if method == "set_rl":
        pretrained = copy.deepcopy(ema)
        pretrained.load_state_dict(pretraining_state)
        inference_started = time.perf_counter()
        pretraining_predicted, pretraining_traces = set_baselines.decode(
            pretrained, test_tokens, "greedy", batch_tokens, BATCH)
        pretraining_inference_seconds = time.perf_counter() - inference_started
    ema.load_state_dict(best_state)
    threshold = best.get("threshold")
    summaries = []
    for output_method in output_methods(method):
        decoder = output_method.removeprefix("set_nll_") if method == "set_nll" else None
        inference_started = time.perf_counter()
        if decoder is not None:
            predicted, traces = set_baselines.decode(
                ema, test_tokens, decoder, batch_tokens, BATCH)
        else:
            predicted, traces = test_predictions(ema, method, test_tokens, threshold)
        inference_seconds = time.perf_counter() - inference_started
        summary = {
            "method": output_method, "training_method": method,
            "fold": fold, "seed": seed, "selection": best,
            "test": metrics(test_y, predicted),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "training_seconds": training_seconds,
            "test_inference_seconds": inference_seconds,
            "test_records": len(test_ids),
            "protocol_sha256": protocol_hash,
        }
        if decoder is not None:
            summary["decoder"] = decoder
        if method == "set_rl":
            summary["pretraining_selection"] = pretraining_best
            summary["rl_stage_start_epoch"] = RL_PRETRAIN_EPOCHS + 1
            summary["pretraining_test"] = metrics(test_y, pretraining_predicted)
            summary["pretraining_inference_seconds"] = pretraining_inference_seconds
        predictions = {"ids": test_ids, "truth": test_y.astype(int).tolist(),
                       "predicted": predicted.astype(int).tolist()}
        if traces is not None:
            predictions["traces"] = traces
        folder = run_folder(results, output_method, fold, seed)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({"model": best_state, "selection": best}, folder / "best.pt")
        save_json(folder / "history.json", history)
        save_json(folder / "predictions.json", predictions)
        if method == "set_rl":
            torch.save({"model": pretraining_state,
                        "selection": pretraining_best},
                       folder / "pretraining_best.pt")
            save_json(folder / "pretraining_predictions.json", {
                "ids": test_ids, "truth": test_y.astype(int).tolist(),
                "predicted": pretraining_predicted.astype(int).tolist(),
                "traces": pretraining_traces,
            })
        artifact_names = ["best.pt", "history.json", "predictions.json"]
        if method == "set_rl":
            artifact_names += ["pretraining_best.pt", "pretraining_predictions.json"]
        summary["artifact_sha256"] = {
            name: sha256(folder / name)
            for name in artifact_names
        }
        save_json(folder / "summary.json", summary)
        summaries.append(summary)
    return summaries


def run_smoke(folds, vocab, labels, methods):
    _, rows, y = folds[1]["train"]
    _, val_rows, val_y = folds[1]["val"]
    for method in methods:
        torch.manual_seed(9)
        model = model_for(method, vocab, labels, y)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        x = (permuted_batch(rows, range(8), random.Random(200009),
                            pad_to=vocab - 2 if method == "tcn_sequence" else None)
             if method in ORDER_AUGMENTED else batch_tokens(rows, range(8)))
        truth = torch.as_tensor(y[:8], dtype=torch.float32)
        targets = random_targets(truth, torch.Generator().manual_seed(9))
        optimizer.zero_grad(set_to_none=True)
        if method in SET_METHODS:
            loss_value = set_baselines.train_step(model, x, truth, "nll")
        elif method in SEQUENCE_METHODS:
            loss = model.loss(x, targets)
            loss.backward()
            loss_value = float(loss.detach())
        elif method in direct_baselines.METHODS:
            loss = direct_baselines.loss(model, method, x, truth)
            loss.backward()
            loss_value = float(loss.detach())
        else:
            loss = F.binary_cross_entropy_with_logits(model(x), truth)
            loss.backward()
            loss_value = float(loss.detach())
        optimizer.step()
        if method == "set_rl":
            optimizer.zero_grad(set_to_none=True)
            set_baselines.train_step(model, x, truth, "rl",
                                     torch.Generator().manual_seed(17))
            optimizer.step()
        row = select_validation(model, method, val_rows[:8], val_y[:8])
        print(f"SMOKE {method} params={sum(p.numel() for p in model.parameters())} "
              f"loss={loss_value:.4f} val_exact={row['exact']:.4f}", flush=True)
    print("SMOKE COMPLETE; outer test records were not evaluated", flush=True)


def make_run_protocol(split_protocol):
    sources = [ROOT / "run_all.py", ROOT / "gru_sequence.py",
               ROOT / "run_modern_encoders.py", ROOT / "mamba_robust_runner.py",
               ROOT / "direct_baselines.py", ROOT / "modern_extensions.py",
               ROOT / "set_baselines.py", ROOT / "rl_engine.py",
               ROOT / "广安门心绞痛4200例.txt", ROOT / "splits" / "protocol.json"]
    sources += sorted((ROOT / "models").glob("*.py"))
    sources += [ROOT / "splits" / f"fold_{fold}.csv" for fold in FOLDS]
    return {
        "study": "current manuscript comparison baselines on post-development internal CV",
        "methods": list(METHODS), "folds": list(FOLDS), "seeds": list(SEEDS),
        "output_methods": list(OUTPUT_METHODS),
        "reused_reference_methods": ["sigmoid_small", "sigmoid_matched",
                                     "classifier_chain", "gru_stop"],
        "epochs": EPOCHS, "batch_size": BATCH, "device": "cpu",
        "randomness": "seed controls initialization and batch order; sequence target permutations use independent 100000+seed; TCN/PatchTST symptom permutations use independent 200000+seed",
        "symptom_order": "canonical vocabulary-ID order for all evaluation; TCN and PatchTST training batches shuffle the symptom order",
        "tcn_fixed_token_width": "all TCN train/validation/test rows padded to the 40-symptom vocabulary width so predictions do not depend on batch neighbors",
        "direct_methods": list(DIRECT_METHODS),
        "legacy_chain": "original 13678-parameter classifier-chain architecture; threshold recalibrated by validation-only autoregressive decoding",
        "label_graph": "label co-occurrence adjacency fitted on the corresponding training fold only",
        "symptom_transformer": "position-free symptom self-attention and BCE multilabel head; validation-selected global threshold",
        "transformer_sequence_matched": "mean symptom encoder, one-layer 192-wide Transformer label decoder, joint ADD/STOP; 520715 parameters",
        "mamba_style_v3": "four deterministic symptom orders averaged by a pure-PyTorch Mamba-style encoder, followed by GRU ADD/STOP decoder; not official Mamba or Mamba-2",
        "set_nll": "one exact set-likelihood training run per fold/seed; validation greedy selects checkpoint; test greedy, beam-5, exact MAP share its weights",
        "set_rl": {"supervised_epochs": RL_PRETRAIN_EPOCHS,
                   "rl_epochs": EPOCHS - RL_PRETRAIN_EPOCHS,
                   "rl_learning_rate": 1e-5,
                   "initialization": "best supervised EMA checkpoint within epochs 1..50",
                   "checkpoint_eligible_epochs": [RL_PRETRAIN_EPOCHS + 1, EPOCHS],
                   "reward": "sample F1 with self-critical greedy baseline and four samples"},
        "thresholds": [float(x) for x in THRESHOLDS],
        "threshold_ties": ["validation exact", "validation sample F1",
                           "validation micro F1", "smaller threshold"],
        "checkpoint_ties": ["validation exact", "validation sample F1",
                            "earlier epoch"],
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4,
                      "amsgrad": True, "gradient_clip_norm": 1.0,
                      "ema_decay_cap": EMA_DECAY,
                      "rl_phase_resets_optimizer_and_ema": True},
        "split_protocol_sha256": sha256(ROOT / "splits" / "protocol.json"),
        "data_sha256": split_protocol["data_sha256"],
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__, "numpy": np.__version__},
    }


def completed_run(folder, protocol_hash):
    required = ("summary.json", "predictions.json", "history.json", "best.pt")
    if not all((folder / name).exists() for name in required):
        return False
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["protocol_sha256"] != protocol_hash:
        raise ValueError(f"Completed run has a different protocol: {folder}")
    digests = summary.get("artifact_sha256", {})
    expected_artifacts = {"best.pt", "history.json", "predictions.json"}
    if summary["training_method"] == "set_rl":
        expected_artifacts.update(("pretraining_best.pt", "pretraining_predictions.json"))
    if set(digests) != expected_artifacts:
        raise ValueError(f"Completed run lacks artifact hashes: {folder}")
    for name, digest in digests.items():
        if sha256(folder / name) != digest:
            raise ValueError(f"Completed run artifact changed: {folder / name}")
    if summary["training_method"] == "set_rl" and summary["selection"]["epoch"] <= RL_PRETRAIN_EPOCHS:
        raise ValueError(f"Set-RL checkpoint predates RL training: {folder}")
    return True


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
        run_smoke(folds, vocab, labels, [args.method] if args.method else METHODS)
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
                folders = [run_folder(output, name, fold, seed)
                           for name in output_methods(method)]
                if all(completed_run(folder, protocol_hash) for folder in folders):
                    print(f"SKIP {method} fold={fold} seed={seed}", flush=True)
                    continue
                summaries = train_one(method, fold, seed, folds, vocab, labels,
                                      output, protocol_hash)
                for summary in summaries:
                    print(f"DONE {summary['method']} fold={fold} seed={seed} "
                          f"exact={summary['test']['exact']:.4f}", flush=True)
    complete = len(list(output.glob("*/fold_*/seed_*/summary.json")))
    expected = len(OUTPUT_METHODS) * len(FOLDS) * len(SEEDS)
    print(f"COMPLETE {complete}/{expected} outputs "
          f"({len(METHODS) * len(FOLDS) * len(SEEDS)} trainings)", flush=True)


if __name__ == "__main__":
    main()
