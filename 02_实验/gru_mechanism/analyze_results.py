"""Verify 36 GRU mechanism runs and summarize paired out-of-fold results."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from run_all import (FOLDS, METHODS, ROOT, SEEDS, load_data, metrics,
                     model_for, test_predictions)


def per_record(truth, predicted):
    truth = np.asarray(truth, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    overlap = (truth * predicted).sum(1)
    total = truth.sum(1) + predicted.sum(1)
    return {
        "exact": (truth == predicted).all(1).astype(float),
        "sample_f1": np.divide(2 * overlap, total,
                                out=np.zeros(len(truth), dtype=float), where=total > 0),
        "cardinality_mae": abs(truth.sum(1) - predicted.sum(1)).astype(float),
    }


def bootstrap_interval(difference, draws=10000, seed=20261001):
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for start in range(0, draws, 250):
        end = min(start + 250, draws)
        sampled = rng.integers(0, len(difference), size=(end - start, len(difference)))
        means[start:end] = difference[sampled].mean(1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def collect(results):
    protocol_path = results / "protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if (protocol["methods"] != list(METHODS) or
            protocol["seeds"] != list(SEEDS) or
            protocol["folds"] != list(FOLDS) or
            protocol["epochs"] != 100 or protocol["batch_size"] != 64):
        raise ValueError("Unexpected M04 run protocol")
    for relative_path, digest in protocol["source_sha256"].items():
        path = ROOT / relative_path
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Package source or data differs from run: {relative_path}")
    if (hashlib.sha256((ROOT / "splits" / "protocol.json").read_bytes()).hexdigest()
            != protocol["split_protocol_sha256"]):
        raise ValueError("Split protocol differs from run")
    _, folds, _, labels = load_data()
    expected_ids = set(range(1, 4201))
    runs = {}
    for method in METHODS:
        for seed in SEEDS:
            for fold in FOLDS:
                folder = results / method / f"fold_{fold}" / f"seed_{seed}"
                summary_path = folder / "summary.json"
                predictions_path = folder / "predictions.json"
                history_path = folder / "history.json"
                checkpoint_path = folder / "best.pt"
                if not all(path.exists() for path in (
                        summary_path, predictions_path, history_path, checkpoint_path)):
                    raise FileNotFoundError(f"Missing run in {folder}")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                pred = json.loads(predictions_path.read_text(encoding="utf-8"))
                history = json.loads(history_path.read_text(encoding="utf-8"))
                if (summary["method"], summary["fold"], summary["seed"]) != (method, fold, seed):
                    raise ValueError(f"Wrong run identity in {folder}")
                if summary["protocol_sha256"] != protocol_hash:
                    raise ValueError(f"Wrong protocol hash in {folder}")
                for name, digest in summary["artifact_sha256"].items():
                    path = folder / name
                    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                        raise ValueError(f"Artifact hash differs in {folder}: {name}")
                if (set(summary["artifact_sha256"]) !=
                        {"best.pt", "history.json", "predictions.json"} or
                        [row["epoch"] for row in history] != list(range(1, 101)) or
                        summary["selection"]["epoch"] not in range(1, 101)):
                    raise ValueError(f"Incomplete 100-epoch run in {folder}")
                test_ids, _, test_y = folds[fold]["test"]
                truth = np.asarray(pred["truth"], dtype=np.int64)
                predicted = np.asarray(pred["predicted"], dtype=np.int64)
                if (pred["ids"] != test_ids or truth.shape != (len(test_ids), labels)
                        or predicted.shape != truth.shape
                        or not np.array_equal(truth, test_y)):
                    raise ValueError(f"Test IDs or labels differ in {folder}")
                if not np.isin(predicted, [0, 1]).all():
                    raise ValueError(f"Invalid prediction in {folder}")
                if (predicted.sum(1) == 0).any():
                    raise ValueError(f"Empty prediction in {folder}")
                traces = pred.get("traces")
                if not isinstance(traces, list) or len(traces) != len(test_ids):
                    raise ValueError(f"Missing action traces in {folder}")
                for index, trace in enumerate(traces):
                    actions = (trace if method == "gru_no_stop_cardinality"
                               else trace[:-1])
                    if (method != "gru_no_stop_cardinality" and
                            (not trace or trace[-1] != labels)):
                        raise ValueError(f"Invalid STOP trace in {folder}")
                    if (len(actions) != len(set(actions)) or
                            any(action not in range(labels) for action in actions) or
                            len(actions) != int(predicted[index].sum()) or
                            set(actions) != set(np.flatnonzero(predicted[index]))):
                        raise ValueError(f"Trace differs from prediction in {folder}")
                recomputed = metrics(truth, predicted)
                for key, value in recomputed.items():
                    if not np.isclose(value, summary["test"][key], atol=1e-10):
                        raise ValueError(f"Metric {key} differs in {folder}")
                runs[method, seed, fold] = (summary, pred)

    rows = {}
    for method in METHODS:
        for seed in SEEDS:
            by_id = {}
            for fold in FOLDS:
                _, pred = runs[method, seed, fold]
                for record_id, truth, prediction in zip(
                        pred["ids"], pred["truth"], pred["predicted"]):
                    if record_id in by_id:
                        raise ValueError(f"Repeated test id {record_id}")
                    by_id[record_id] = (truth, prediction)
            if set(by_id) != expected_ids:
                raise ValueError("Out-of-fold predictions do not cover all records")
            truth = np.asarray([by_id[i][0] for i in sorted(expected_ids)], dtype=np.int64)
            predicted = np.asarray([by_id[i][1] for i in sorted(expected_ids)], dtype=np.int64)
            rows[method, seed] = (metrics(truth, predicted), per_record(truth, predicted))
    return runs, rows


def render(runs, rows, m03_comparison=None, checkpoint_check=None):
    lines = [
        "# GRU 机制对照：三折内部交叉验证结果",
        "",
        "三折中每条记录各获得一次折外预测，每种方法使用训练种子 9、17、29。",
        "主表为 4200 条记录的折外指标在三个种子间的均值±样本标准差。",
        "这是开发后的记录级内部评价；既有 4200 条记录此前已参与方法探索。",
        "旧版 2940/630/630 单次划分的消融结果不与此处数值拼接。",
        "",
        "| 方法 | EM | 样本 F1 | Micro-F1 | Macro-F1 | 基数 MAE | 参数量 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    measures = ("exact", "sample_f1", "micro_f1", "macro_f1", "cardinality_mae")
    for method in METHODS:
        values = np.asarray([[rows[method, seed][0][name] for name in measures]
                             for seed in SEEDS])
        formatted = [f"{column.mean():.4f} ± {column.std(ddof=1):.4f}"
                     for column in values.T]
        params = runs[method, SEEDS[0], FOLDS[0]][0]["parameters"]
        lines.append(f"| {method} | " + " | ".join(formatted) + f" | {params:,} |")

    lines.extend(["", "## 折别与计算成本", "",
                  "| 方法 | 折 | EM均值±标准差 | 训练秒数均值 | 1400条推理秒数均值 |",
                  "|---|---:|---:|---:|---:|"])
    for method in METHODS:
        for fold in FOLDS:
            subset = [runs[method, seed, fold][0] for seed in SEEDS]
            exact = np.asarray([item["test"]["exact"] for item in subset])
            train_seconds = np.mean([item["training_seconds"] for item in subset])
            infer_seconds = np.mean([item["test_inference_seconds"] for item in subset])
            lines.append(f"| {method} | {fold} | {exact.mean():.4f} ± "
                         f"{exact.std(ddof=1):.4f} | {train_seconds:.1f} | {infer_seconds:.2f} |")

    lines.extend(["", "## 相对完整 GRU 的配对差值", "",
                  "先对同一记录的三个种子差值取均值，再按 4200 个记录有放回抽样 10,000 次。",
                  "EM 与样本 F1 取完整 GRU 减变体，基数 MAE 取变体减完整 GRU，正值均有利于完整 GRU。",
                  "区间条件于这批记录、划分、方法和三个已训练实例；不覆盖重新选模或患者内相关。",
                  "", "| 对照 | EM差值及95%区间 | 样本F1差值及95%区间 | 基数MAE差值及95%区间 |",
                  "|---|---:|---:|---:|"])
    for method in METHODS:
        if method == "gru_stop":
            continue
        cells = []
        for measure in ("exact", "sample_f1", "cardinality_mae"):
            differences = np.stack([
                rows["gru_stop", seed][1][measure] - rows[method, seed][1][measure]
                for seed in SEEDS])
            if measure == "cardinality_mae":
                differences *= -1
            per_record_difference = differences.mean(0)
            lower, upper = bootstrap_interval(per_record_difference)
            cells.append(f"{per_record_difference.mean():+.4f} [{lower:+.4f}, {upper:+.4f}]")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.extend(["", "## 改动范围", "",
                  "| 方法 | 症状编码器 | 解码器输入与输出 | 训练目标 | 终止规则 |",
                  "|---|---|---|---|---|",
                  "| 完整 GRU | 均值编码 | 前一动作嵌入，GRU，标签/STOP 联合头 | 随机目标顺序逐步交叉熵 | 逐步 STOP |",
                  "| 固定目标顺序 | 相同 | 相同 | 词表顺序逐步交叉熵 | 相同；推理不强制固定顺序 |",
                  "| 去前一标签身份 | 相同 | 每一步输入固定 BOS，保留 GRU 隐状态和合法动作掩码 | 随机目标顺序逐步交叉熵 | 逐步 STOP |",
                  "| 无 STOP、独立基数头 | 相同 | GRU 标签头及症状上下文基数头 | 平均标签交叉熵＋基数交叉熵 | 先预测数量，再输出相应数量标签 |",
                  "", "无 STOP 方案同时更改输出头、损失和终止规则，差值不能单独归于 STOP。",
                  "固定顺序只改变训练标签顺序；去前一标签身份仍保留循环隐状态。",
                  "区间条件于既定划分和三个训练实例；没有患者标识，记录级区间不等于患者级泛化区间。", ""])
    if m03_comparison is not None:
        lines.extend([
            f"与另行提供的 M03 结果相比，完整 GRU 的 {m03_comparison['runs']} 份折/种子预测集合逐条相同。",
            f"两份结果有 {m03_comparison['trace_mismatches']} 条动作轨迹顺序不同；轨迹差异不改变上述集合比较。", ""])
    if checkpoint_check is not None:
        lines.extend([
            f"重新载入 {checkpoint_check['runs']} 个检查点后，保存的测试预测集合均可重现。",
            f"检查点重推理有 {checkpoint_check['trace_mismatches']} 条动作轨迹顺序与文件不同。", ""])
    return "\n".join(lines)


def compare_m03_reference(runs, reference):
    matched = 0
    trace_mismatches = 0
    for fold in FOLDS:
        for seed in SEEDS:
            path = reference / "gru_stop" / f"fold_{fold}" / f"seed_{seed}" / "predictions.json"
            if not path.exists():
                raise FileNotFoundError(path)
            old = json.loads(path.read_text(encoding="utf-8"))
            new = runs["gru_stop", seed, fold][1]
            for field in ("ids", "truth", "predicted"):
                if old[field] != new[field]:
                    raise ValueError(f"Full GRU differs from M03 for {field}, fold {fold}, seed {seed}")
            trace_mismatches += sum(left != right for left, right in
                                    zip(old["traces"], new["traces"]))
            matched += 1
    return {"runs": matched, "trace_mismatches": trace_mismatches}


def replay_checkpoint(folder, method, vocab, labels, tokens, expected):
    saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model = model_for(method, vocab, labels)
    model.load_state_dict(saved["model"])
    predicted, traces = test_predictions(model, method, tokens)
    if not np.array_equal(predicted, np.asarray(expected["predicted"], dtype=np.int64)):
        raise ValueError(f"Checkpoint does not reproduce predicted sets in {folder}")
    return sum(left != right for left, right in zip(traces, expected["traces"]))


def verify_checkpoints(runs, results):
    _, folds, vocab, labels = load_data()
    trace_mismatches = 0
    for method in METHODS:
        for seed in SEEDS:
            for fold in FOLDS:
                folder = results / method / f"fold_{fold}" / f"seed_{seed}"
                _, tokens, _ = folds[fold]["test"]
                trace_mismatches += replay_checkpoint(
                    folder, method, vocab, labels, tokens,
                    runs[method, seed, fold][1])
    return {"runs": len(runs), "trace_mismatches": trace_mismatches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--m03-results", type=Path,
                        help="Optional M03 results directory for exact full-GRU set check")
    parser.add_argument("--verify-checkpoints", action="store_true",
                        help="Reload all checkpoints and reproduce saved test sets")
    parser.add_argument("--cpu-threads", type=int, default=6)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    runs, rows = collect(args.results)
    m03_comparison = (compare_m03_reference(runs, args.m03_results)
                      if args.m03_results else None)
    checkpoint_check = None
    if args.verify_checkpoints:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        checkpoint_check = verify_checkpoints(runs, args.results)
    path = args.results / "analysis_report.md"
    path.write_text(render(runs, rows, m03_comparison, checkpoint_check),
                    encoding="utf-8")
    print(f"VERIFIED {len(runs)}/36 runs; wrote {path}")


if __name__ == "__main__":
    main()
