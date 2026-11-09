"""Verify the 36 locked runs and summarize out-of-fold comparisons."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_all import FOLDS, METHODS, ROOT, SEEDS, load_data, metrics


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
    if protocol["methods"] != list(METHODS) or protocol["seeds"] != list(SEEDS):
        raise ValueError("Unexpected method or seed protocol")
    _, folds, _, labels = load_data()
    expected_ids = set(range(1, 4201))
    runs = {}
    for method in METHODS:
        for seed in SEEDS:
            for fold in FOLDS:
                folder = results / method / f"fold_{fold}" / f"seed_{seed}"
                summary_path = folder / "summary.json"
                predictions_path = folder / "predictions.json"
                if not summary_path.exists() or not predictions_path.exists():
                    raise FileNotFoundError(f"Missing run in {folder}")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                pred = json.loads(predictions_path.read_text(encoding="utf-8"))
                if (summary["method"], summary["fold"], summary["seed"]) != (method, fold, seed):
                    raise ValueError(f"Wrong run identity in {folder}")
                if summary["protocol_sha256"] != protocol_hash:
                    raise ValueError(f"Wrong protocol hash in {folder}")
                test_ids, _, test_y = folds[fold]["test"]
                truth = np.asarray(pred["truth"], dtype=np.int64)
                predicted = np.asarray(pred["predicted"], dtype=np.int64)
                if (pred["ids"] != test_ids or truth.shape != (len(test_ids), labels)
                        or predicted.shape != truth.shape
                        or not np.array_equal(truth, test_y)):
                    raise ValueError(f"Test IDs or labels differ in {folder}")
                if not np.isin(predicted, [0, 1]).all():
                    raise ValueError(f"Invalid prediction in {folder}")
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


def render(runs, rows):
    lines = [
        "# 同编码器对照：三折内部交叉验证结果",
        "",
        "三折中每条记录各获得一次折外预测，每种方法使用训练种子 9、17、29。",
        "主表为 4200 条记录的折外指标在三个种子间的均值±样本标准差。",
        "这是开发后的记录级内部评价；既有 4200 条记录此前已参与方法探索。",
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

    lines.extend(["", "## 相对 GRU 的配对差值", "",
                  "先对同一记录的三个种子差值取均值，再按 4200 个记录有放回抽样 10,000 次。",
                  "EM 与样本 F1 取 GRU−对照，基数 MAE 取对照−GRU，正值均有利于 GRU。",
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
    lines.extend(["", "## 解释范围", "",
                  "四种方法的症状编码结构与输入词表相同，但独立端到端训练；解码器和损失不同。",
                  "容量接近的直接预测头控制参数量，不能控制全部训练和推理计算量。",
                  "Classifier Chain 采用固定词表顺序并在验证集搜索阈值，本身也是标签依赖的逐步方法。",
                  "比较结果不单独识别任意一种序贯或集合机制的因果效应。",
                  "没有患者标识，记录级抽样区间不能解释为患者级泛化区间。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    runs, rows = collect(args.results)
    path = args.results / "analysis_report.md"
    path.write_text(render(runs, rows), encoding="utf-8")
    print(f"VERIFIED {len(runs)}/36 runs; wrote {path}")


if __name__ == "__main__":
    main()
