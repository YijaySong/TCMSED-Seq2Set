"""Read-only M05 analysis of the locked M03 and M04 out-of-fold predictions."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SEEDS = (9, 17, 29)
FOLDS = (1, 2, 3)
M04_METHODS = (
    "gru_stop",
    "gru_fixed_order",
    "gru_no_label_history",
    "gru_no_stop_cardinality",
)
METHODS = ("gru_stop", "sigmoid_matched", *M04_METHODS[1:])
COMPARATORS = tuple(method for method in METHODS if method != "gru_stop")
MEASURES = ("exact", "sample_f1", "cardinality_mae")
GROUPS = (("all", (1, 2, 3, 4, 5, 6)),
          *((str(n), (n,)) for n in range(1, 7)),
          ("1-2", (1, 2)), ("3-4", (3, 4)),
          ("1-5", (1, 2, 3, 4, 5)), ("5-6", (5, 6)))
ILLUSTRATIVE_SCREEN_PAIRS = (
    ("食欲不振", "食欲旺盛"),
    ("口渴喜冷饮", "口渴喜热饮"),
    ("胸痛遇寒加重", "胸痛遇热加重"),
    ("苔厚腻", "苔薄白"),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def validate_existing_runs(package, results):
    # Each package's collect() checks its own saved metrics, fold IDs and labels.
    code = ("from pathlib import Path; from analyze_results import collect; "
            "runs, _ = collect(Path(__import__('sys').argv[1])); "
            "assert len(runs) == 36, len(runs); print(len(runs))")
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code, str(results.resolve())],
        cwd=package, check=True, capture_output=True, text=True,
    )
    if completed.stdout.strip() != "36":
        raise ValueError(f"Unexpected verifier output for {package}: {completed.stdout}")


def load_inputs(m03_package, m04_package, m03_results, m04_results):
    source_protocols = [read_json(package / "splits" / "protocol.json")
                        for package in (m03_package, m04_package)]
    if digest(m03_package / "splits" / "protocol.json") != digest(
            m04_package / "splits" / "protocol.json"):
        raise ValueError("M03 and M04 split protocols differ")
    source = source_protocols[0]
    result_protocols = [read_json(folder / "protocol.json")
                        for folder in (m03_results, m04_results)]
    for package, results, result_protocol in zip(
            (m03_package, m04_package), (m03_results, m04_results), result_protocols):
        if (result_protocol["seeds"] != list(SEEDS) or
                result_protocol["folds"] != list(FOLDS) or
                result_protocol["epochs"] != 100 or
                result_protocol["batch_size"] != 64 or
                result_protocol["split_protocol_sha256"] != digest(
                    package / "splits" / "protocol.json")):
            raise ValueError(f"Unexpected run protocol in {results}")
        if digest(package / source["data_file"]) != source["data_sha256"]:
            raise ValueError(f"Data file changed in {package}")
        for relative, expected_hash in result_protocol["source_sha256"].items():
            if digest(package / relative) != expected_hash:
                raise ValueError(f"Source differs from run: {package / relative}")
        validate_existing_runs(package, results)

    labels = source["schema"]["labels"]
    if len(labels) != 10:
        raise ValueError("Unexpected label vocabulary")
    label_index = {label: i for i, label in enumerate(labels)}
    records = {}
    with (m04_package / source["data_file"]).open(
            encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=" ")
        if reader.fieldnames != ["id", "symptom", "Se"]:
            raise ValueError("Unexpected data columns")
        for row in reader:
            record_id = int(row["id"])
            if record_id in records:
                raise ValueError(f"Duplicate record {record_id}")
            records[record_id] = {
                "symptoms": row["symptom"].split(","),
                "truth": tuple(sorted(label_index[name] for name in row["Se"].split(","))),
            }
    if sorted(records) != list(range(1, 4201)):
        raise ValueError("Expected record IDs 1..4200")

    partitions = {}
    outer_test_ids = []
    for fold in FOLDS:
        path = m04_package / "splits" / f"fold_{fold}.csv"
        expected_hash = source["folds"][fold - 1]["sha256"]
        if digest(path) != expected_hash:
            raise ValueError(f"Split file changed: {path}")
        split = defaultdict(list)
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                split[row["split"]].append(int(row["id"]))
        if (set(split) != {"train", "val", "test"} or
                sorted(i for ids in split.values() for i in ids) != sorted(records) or
                any(len(split[role]) != source["folds"][fold - 1]["sizes"][role]
                    for role in split)):
            raise ValueError(f"Invalid fold {fold}")
        partitions[fold] = split
        outer_test_ids.extend(split["test"])
    if sorted(outer_test_ids) != sorted(records):
        raise ValueError("Outer test folds do not cover each record exactly once")
    return source, records, partitions


def load_predictions(m03_results, m04_results, records, partitions, label_count):
    predictions = {method: {seed: {} for seed in SEEDS} for method in METHODS}
    traces = {seed: {} for seed in SEEDS}
    for fold in FOLDS:
        expected_ids = partitions[fold]["test"]
        for seed in SEEDS:
            for method in ("gru_stop", "sigmoid_matched"):
                path = (m03_results / method / f"fold_{fold}" /
                        f"seed_{seed}" / "predictions.json")
                saved = read_json(path)
                if saved["ids"] != expected_ids:
                    raise ValueError(f"M03 test IDs differ: {path}")
                if method == "gru_stop":
                    m03_gru = saved
                else:
                    add_predictions(predictions[method][seed], saved, records, label_count)
            for method in M04_METHODS:
                path = (m04_results / method / f"fold_{fold}" /
                        f"seed_{seed}" / "predictions.json")
                saved = read_json(path)
                if saved["ids"] != expected_ids:
                    raise ValueError(f"M04 test IDs differ: {path}")
                add_predictions(predictions[method][seed], saved, records, label_count)
                if method == "gru_stop":
                    for field in ("ids", "truth", "predicted"):
                        if saved[field] != m03_gru[field]:
                            raise ValueError(f"M03/M04 GRU {field} differs in fold {fold}, seed {seed}")
                    for record_id, trace in zip(saved["ids"], saved["traces"]):
                        if (not trace or trace[-1] != label_count or
                                tuple(sorted(trace[:-1])) != predictions[method][seed][record_id]):
                            raise ValueError(f"Invalid GRU action trace for record {record_id}")
                        traces[seed][record_id] = tuple(trace)
    for method in METHODS:
        for seed in SEEDS:
            if sorted(predictions[method][seed]) != sorted(records):
                raise ValueError(f"Incomplete out-of-fold predictions: {method}, {seed}")
    return predictions, traces


def add_predictions(target, saved, records, label_count):
    if not (len(saved["ids"]) == len(saved["truth"]) == len(saved["predicted"])):
        raise ValueError("Prediction arrays differ in length")
    for record_id, truth, predicted in zip(
            saved["ids"], saved["truth"], saved["predicted"]):
        if record_id in target or record_id not in records:
            raise ValueError(f"Duplicate or unknown record {record_id}")
        if (len(truth) != label_count or len(predicted) != label_count or
                set(truth) - {0, 1} or set(predicted) - {0, 1}):
            raise ValueError(f"Invalid label vector for record {record_id}")
        true_set = tuple(i for i, value in enumerate(truth) if value)
        pred_set = tuple(i for i, value in enumerate(predicted) if value)
        if true_set != records[record_id]["truth"] or not pred_set:
            raise ValueError(f"Reference labels or prediction differ for record {record_id}")
        target[record_id] = pred_set


def fixed_baselines(records, partitions, label_count):
    predictions = {"top_six_marginal": {}, "most_common_full_set": {}}
    definitions = []
    for fold in FOLDS:
        train_ids = partitions[fold]["train"]
        marginal = Counter(label for record_id in train_ids
                           for label in records[record_id]["truth"])
        top_six = tuple(sorted(sorted(range(label_count),
                                      key=lambda label: (-marginal[label], label))[:6]))
        sets = Counter(records[record_id]["truth"] for record_id in train_ids)
        most_common = min(sets, key=lambda labels: (-sets[labels], labels))
        for name, chosen in (("top_six_marginal", top_six),
                             ("most_common_full_set", most_common)):
            for record_id in partitions[fold]["test"]:
                predictions[name][record_id] = chosen
            definitions.append({"fold": fold, "method": name,
                                "train_support": sets[chosen] if name ==
                                "most_common_full_set" else None,
                                "label_indices": list(chosen)})
    if any(sorted(row) != sorted(records) for row in predictions.values()):
        raise ValueError("Diagnostic baseline does not cover all test records")
    return predictions, definitions


def per_record(truth, predicted):
    true_set, pred_set = set(truth), set(predicted)
    return (float(true_set == pred_set),
            2 * len(true_set & pred_set) / (len(true_set) + len(pred_set)),
            float(abs(len(true_set) - len(pred_set))))


def group_ids(records, cardinalities):
    return [record_id for record_id in sorted(records)
            if len(records[record_id]["truth"]) in cardinalities]


def summarize(predictions, records):
    rows = []
    for method in METHODS:
        for group, cardinalities in GROUPS:
            ids = group_ids(records, cardinalities)
            by_seed = []
            for seed in SEEDS:
                values = np.asarray([per_record(records[i]["truth"],
                                                predictions[method][seed][i])
                                     for i in ids])
                by_seed.append(values.mean(axis=0))
            by_seed = np.asarray(by_seed)
            row = {"method": method, "true_cardinality": group, "n": len(ids)}
            for column, measure in enumerate(MEASURES):
                row[f"{measure}_mean"] = float(by_seed[:, column].mean())
                row[f"{measure}_seed_sd"] = float(by_seed[:, column].std(ddof=1))
            rows.append(row)
    return rows


def baseline_summary(baselines, records):
    rows = []
    for method, predictions in baselines.items():
        for group, cardinalities in GROUPS:
            ids = group_ids(records, cardinalities)
            values = np.asarray([per_record(records[i]["truth"], predictions[i])
                                 for i in ids])
            row = {"method": method, "true_cardinality": group, "n": len(ids)}
            row.update({measure: float(values[:, column].mean())
                        for column, measure in enumerate(MEASURES)})
            rows.append(row)
    return rows


def bootstrap_interval(differences, draws=10000, seed=20261001):
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for start in range(0, draws, 250):
        end = min(start + 250, draws)
        indices = rng.integers(0, len(differences), size=(end - start, len(differences)))
        means[start:end] = differences[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def paired_differences(predictions, records, draws):
    rows = []
    for method in COMPARATORS:
        for group, cardinalities in GROUPS:
            ids = group_ids(records, cardinalities)
            differences = []
            for record_id in ids:
                true_set = records[record_id]["truth"]
                seed_differences = []
                for seed in SEEDS:
                    primary = per_record(true_set, predictions["gru_stop"][seed][record_id])
                    other = per_record(true_set, predictions[method][seed][record_id])
                    seed_differences.append((primary[0] - other[0],
                                             primary[1] - other[1],
                                             other[2] - primary[2]))
                differences.append(np.mean(seed_differences, axis=0))
            differences = np.asarray(differences)
            for column, measure in enumerate(MEASURES):
                values = differences[:, column]
                lower, upper = (bootstrap_interval(values, draws)
                                if group == "all" else (None, None))
                rows.append({"comparator": method, "true_cardinality": group,
                             "measure": measure, "n": len(ids),
                             "gru_advantage": float(values.mean()),
                             "ci_lower": lower, "ci_upper": upper})
    return rows


def cardinality_matrix(predictions, records, methods=("gru_stop", "gru_no_stop_cardinality")):
    rows = []
    for method in methods:
        counts = Counter((len(records[i]["truth"]), len(predictions[method][seed][i]))
                         for seed in SEEDS for i in sorted(records))
        for true_count in range(1, 7):
            for predicted_count in range(1, 11):
                rows.append({"method": method, "true_cardinality": true_count,
                             "predicted_cardinality": predicted_count,
                             "count_across_seeds": counts[true_count, predicted_count],
                             "mean_count_per_seed": counts[true_count, predicted_count] / 3})
    return rows


def per_label_metrics(predictions, records, labels,
                      methods=("gru_stop", "sigmoid_matched")):
    seed_rows = []
    summary_rows = []
    ids = sorted(records)
    for method in methods:
        for label_index, label in enumerate(labels):
            support = sum(label_index in records[i]["truth"] for i in ids)
            for seed in SEEDS:
                tp = sum(label_index in records[i]["truth"] and
                         label_index in predictions[method][seed][i] for i in ids)
                fp = sum(label_index not in records[i]["truth"] and
                         label_index in predictions[method][seed][i] for i in ids)
                fn = support - tp
                seed_rows.append({
                    "method": method, "label": label, "seed": seed,
                    "support": support, "tp": tp, "fp": fp, "fn": fn,
                    "precision": tp / (tp + fp) if tp + fp else 0.0,
                    "recall": tp / support if support else 0.0,
                    "f1": 2 * tp / (2 * tp + fp + fn)
                    if 2 * tp + fp + fn else 0.0,
                })
            seed_values = seed_rows[-len(SEEDS):]
            row = {"method": method, "label": label, "support": support}
            for field in ("tp", "fp", "fn", "precision", "recall", "f1"):
                values = [item[field] for item in seed_values]
                row[f"{field}_mean"] = statistics.mean(values)
                row[f"{field}_seed_sd"] = statistics.stdev(values)
            summary_rows.append(row)
    return seed_rows, summary_rows


def cardinality_summary(predictions, records,
                        methods=("gru_stop", "sigmoid_matched", "gru_no_stop_cardinality")):
    groups = (("all", (1, 2, 3, 4, 5, 6)),
              *((str(n), (n,)) for n in range(1, 7)),
              ("1-3", (1, 2, 3)), ("4-5", (4, 5)))
    rows = []
    for method in methods:
        for group, cardinalities in groups:
            ids = group_ids(records, cardinalities)
            row = {"method": method, "true_cardinality": group, "n": len(ids)}
            by_seed = []
            for seed in SEEDS:
                differences = [len(predictions[method][seed][i]) -
                               len(records[i]["truth"]) for i in ids]
                by_seed.append({
                    "cardinality_exact": sum(d == 0 for d in differences) / len(ids),
                    "underprediction": sum(d < 0 for d in differences) / len(ids),
                    "overprediction": sum(d > 0 for d in differences) / len(ids),
                    "cardinality_mae": sum(abs(d) for d in differences) / len(ids),
                })
            for field in by_seed[0]:
                values = [item[field] for item in by_seed]
                row[f"{field}_mean"] = statistics.mean(values)
                row[f"{field}_seed_sd"] = statistics.stdev(values)
            rows.append(row)
    return rows


def error_type(truth, predicted):
    truth, predicted = set(truth), set(predicted)
    if truth == predicted:
        return "exact"
    if predicted < truth:
        return "omission_only"
    if truth < predicted:
        return "addition_only"
    return "mixed"


def has_screen_pair(symptoms):
    terms = set(symptoms)
    return any(first in terms and second in terms
               for first, second in ILLUSTRATIVE_SCREEN_PAIRS)


def select_cases(records, predictions, traces, partitions, labels):
    median_symptom_count = statistics.median(len(row["symptoms"]) for row in records.values())
    cases = []
    counts_before = {}
    counts_after = {}
    for category in ("exact", "omission_only", "addition_only"):
        stable = []
        for record_id in sorted(records):
            truth = records[record_id]["truth"]
            selected = [predictions["gru_stop"][seed][record_id] for seed in SEEDS]
            if (len(set(selected)) == 1 and
                    all(error_type(truth, item) == category for item in selected)):
                stable.append(record_id)
        eligible = [record_id for record_id in stable
                    if not has_screen_pair(records[record_id]["symptoms"])]
        counts_before[category] = len(stable)
        counts_after[category] = len(eligible)
        if not eligible:
            raise ValueError(f"No illustrative case remains after the four-pair screen "
                             f"for {category}; {len(stable)} before screening")
        chosen = min(eligible, key=lambda i:
                     (abs(len(records[i]["symptoms"]) - median_symptom_count), i))
        truth = records[chosen]["truth"]
        predicted = predictions["gru_stop"][SEEDS[0]][chosen]
        cases.append({
            "record_id": chosen,
            "fold": next(fold for fold in FOLDS if chosen in partitions[fold]["test"]),
            "category": category,
            "symptoms": records[chosen]["symptoms"],
            "reference_labels": [labels[index] for index in truth],
            "predicted_labels": [labels[index] for index in predicted],
            "omitted_labels": [labels[index] for index in truth if index not in predicted],
            "extra_labels": [labels[index] for index in predicted if index not in truth],
            "traces": {str(seed): [labels[action] if action < len(labels) else "STOP"
                                   for action in traces[seed][chosen]] for seed in SEEDS},
        })
    return {"rule": "The predicted set and error type must agree across all three seeds. "
            "For displayed cases only, exclude records containing both terms of any "
            "listed pair. Within each type choose symptom count closest to the "
            "dataset median, breaking ties by lowest record ID.",
            "illustrative_screen_pairs": [list(pair) for pair in ILLUSTRATIVE_SCREEN_PAIRS],
            "median_symptom_count": median_symptom_count,
            "eligible_count_before_screen": counts_before,
            "eligible_count_after_screen": counts_after,
            "cases": cases}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def format_mean_sd(row, name):
    return f"{row[f'{name}_mean']:.4f} ± {row[f'{name}_seed_sd']:.4f}"


def format_count_sd(row, name):
    return f"{row[f'{name}_mean']:.1f} ± {row[f'{name}_seed_sd']:.1f}"


def report_text(strata, baseline_rows, baseline_definitions, paired, matrix,
                cases, labels, records, draws, label_rows, cardinality_rows):
    by_method_group = {(row["method"], row["true_cardinality"]): row for row in strata}
    six_predictions = sum(row["count_across_seeds"] for row in matrix
                          if row["method"] == "gru_no_stop_cardinality" and
                          row["predicted_cardinality"] == 6)
    distinct_sets = {row["truth"] for row in records.values()}
    six_label_sets = {labels for labels in distinct_sets if len(labels) == 6}
    lines = ["# M05 数量与病例分析", "",
             "M03 和 M04 的 72 份运行预测通过各自的 `collect()` 校验；两包使用相同数据和三折划分。",
             "每个方法在三个训练种子下均覆盖全部 4,200 条折外记录。M03 与 M04 的完整 GRU 预测集合逐条一致。",
             "表中均值±标准差描述三次训练的波动。1、2 标签组仅有 33、97 条记录，分层结果只作描述。",
             "既有记录参与过方法探索；这里属于开发后的记录级内部评价。", "",
             "## 按真实标签数量", "",
             "| 方法 | 标签数 | 记录数 | 完全匹配率 | 样本 F1 | 数量 MAE |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in strata:
        if row["true_cardinality"] in {"all", "1-2", "3-4", "1-5", "5-6"}:
            continue
        lines.append(f"| {row['method']} | {row['true_cardinality']} | {row['n']} | "
                     f"{format_mean_sd(row, 'exact')} | {format_mean_sd(row, 'sample_f1')} | "
                     f"{format_mean_sd(row, 'cardinality_mae')} |")
    primary_short = by_method_group["gru_stop", "1-5"]["exact_mean"]
    fixed_short = by_method_group["gru_fixed_order", "1-5"]["exact_mean"]
    primary_six = by_method_group["gru_stop", "6"]["exact_mean"]
    fixed_six = by_method_group["gru_fixed_order", "6"]["exact_mean"]
    no_stop_six = by_method_group["gru_no_stop_cardinality", "6"]["exact_mean"]
    lines.extend(["", "5–6 标签合组会被六标签记录主导；逐个标签数的结果保留在上表。",
                  f"1–5 标签合计，完整 GRU 与固定顺序版的完全匹配率分别为 "
                  f"{primary_short:.4f} 与 {fixed_short:.4f}；六标签组分别为 "
                  f"{primary_six:.4f} 与 {fixed_six:.4f}。总体差值不能推广到所有标签数量。",
                  f"无 STOP 独立基数头在三个种子的 {six_predictions / (len(records) * len(SEEDS)):.2%} "
                  f"输出中预测六个标签；六标签组完全匹配率为 {no_stop_six:.4f}，"
                  f"完整 GRU 为 {primary_six:.4f}。较短集合中的基数误差见上表。"
                  "该方案还改变输出头和训练损失，差距不能单独归于终止动作。",
                  f"参考记录有 {len(distinct_sets)} 种完整组合，其中六标签组合 "
                  f"{len(six_label_sets)} 种；六标签记录并非同一固定集合。",
                  "", "## 固定预测诊断基线", "",
                  "每折仅用 2,380 条训练记录确定标签，测试折统一预测相同集合；验证和测试标签均未参与选择。",
                  "边际前六标签按训练频数排序；最常见完整集合按训练中完整组合出现次数排序。频数并列时按词表索引确定。",
                  "", "| 方法 | 完全匹配率 | 样本 F1 | 数量 MAE |",
                  "|---|---:|---:|---:|"])
    for row in baseline_rows:
        if row["true_cardinality"] == "all":
            lines.append(f"| {row['method']} | {row['exact']:.4f} | "
                         f"{row['sample_f1']:.4f} | {row['cardinality_mae']:.4f} |")
    lines.extend(["", "| 折 | 规则 | 固定预测集合 | 训练中该完整集合的例数 |",
                  "|---:|---|---|---:|"])
    for item in baseline_definitions:
        names = ", ".join(labels[i] for i in item["label_indices"])
        support = item["train_support"] if item["train_support"] is not None else "不适用"
        lines.append(f"| {item['fold']} | {item['method']} | {names} | {support} |")
    lines.extend(["", "两个固定集合均含六个标签，因此数量 MAE 相同。"
                  "它们用于检查标签分布集中程度，不是可与训练模型等量比较的可学习方法。",
                  "", "## 完整 GRU 的配对差值", "",
                  f"每条记录先对三个种子的差值求平均，再按记录配对重采样 {draws:,} 次。"
                  "表内差值为有利于 GRU 的方向：完全匹配率和样本 F1 取 GRU 减对照，"
                  "数量 MAE 取对照减 GRU。", "",
                  "| 对照 | 完全匹配率差值 [95%区间] | 样本 F1 差值 [95%区间] | 数量 MAE 差值 [95%区间] |",
                  "|---|---:|---:|---:|"])
    for method in COMPARATORS:
        cells = []
        for measure in MEASURES:
            row = next(x for x in paired if x["comparator"] == method and
                       x["true_cardinality"] == "all" and x["measure"] == measure)
            cells.append(f"{row['gru_advantage']:+.4f} "
                         f"[{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}]")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.extend(["", "上述区间条件于这三折、现有训练实例及记录级抽样；"
                  "它不涵盖重新选择模型、重新训练或患者内相关。"
                  "各真实标签数组的配对均值写在 `paired_differences.csv`；"
                  "小样本分组不报告重采样区间。", "",
                  "## 真实数量与预测数量", "",
                  "下表的单元格为三个种子的预测次数除以三，因此允许出现小数。"
                  "每行未舍入的总数等于该真实数量的记录数，显示值可能有 0.1 的舍入差。", ""])
    max_pred = max(row["predicted_cardinality"] for row in matrix
                   if row["count_across_seeds"])
    for method in ("gru_stop", "gru_no_stop_cardinality"):
        lines.extend([f"### {method}", "",
                      "| 真实\\预测 | " + " | ".join(map(str, range(1, max_pred + 1))) + " |",
                      "|---:" + "|---:" * max_pred + "|"])
        for true_count in range(1, 7):
            cells = [next(row["mean_count_per_seed"] for row in matrix
                          if row["method"] == method and
                          row["true_cardinality"] == true_count and
                          row["predicted_cardinality"] == pred_count)
                     for pred_count in range(1, max_pred + 1)]
            lines.append(f"| {true_count} | " + " | ".join(f"{value:.1f}" for value in cells) + " |")
        lines.append("")
    lines.extend(["## 各证候要素的识别", "",
                  "阳性记录数按 4,200 条不同记录计算；TP、FP、FN 和比率均先逐种子计算，"
                  "表中为三种子的均值±样本标准差。完整逐种子计数见 `per_label_seed_metrics.csv`。",
                  "逐标签准确率把大量阴性判断一并计入，这里用精确率、召回率和 F1 展示阳性标签识别。",
                  "", "| 方法 | 证候要素 | 阳性记录数 | TP | FP | FN | 精确率 | 召回率 | F1 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in label_rows:
        lines.append(f"| {row['method']} | {row['label']} | {row['support']} | "
                     f"{format_count_sd(row, 'tp')} | {format_count_sd(row, 'fp')} | "
                     f"{format_count_sd(row, 'fn')} | {format_mean_sd(row, 'precision')} | "
                     f"{format_mean_sd(row, 'recall')} | {format_mean_sd(row, 'f1')} |")
    gru_by_label = {row["label"]: row for row in label_rows if row["method"] == "gru_stop"}
    sigmoid_by_label = {row["label"]: row for row in label_rows
                        if row["method"] == "sigmoid_matched"}
    better_f1 = sum(gru_by_label[name]["f1_mean"] > sigmoid_by_label[name]["f1_mean"]
                    for name in labels)
    better_precision = sum(gru_by_label[name]["precision_mean"] >
                           sigmoid_by_label[name]["precision_mean"] for name in labels)
    better_recall = sum(gru_by_label[name]["recall_mean"] >
                        sigmoid_by_label[name]["recall_mean"] for name in labels)
    extra_fn = sum(gru_by_label[name]["fn_mean"] - sigmoid_by_label[name]["fn_mean"]
                   for name in labels)
    fewer_fp = sum(sigmoid_by_label[name]["fp_mean"] - gru_by_label[name]["fp_mean"]
                   for name in labels)
    lines.extend(["",
                  f"完整 GRU 在 {better_f1}/10 个标签上的 F1、"
                  f"{better_precision}/10 个标签上的精确率高于近容量 sigmoid；"
                  f"召回率仅在 {better_recall}/10 个标签上较高。"
                  f"按每个种子的 4,200 条记录计，GRU 平均少发生 {fewer_fp:.1f} 次标签误加，"
                  f"同时多发生 {extra_fn:.1f} 次标签漏选。",
                  "", "## 预测集合数量", "",
                  "数量正确、低估和高估的比例以本行真实数量组的记录数为分母；"
                  "数量 MAE 为预测数与真实数之差的绝对值均值。"
                  "每个比例先在一个种子的相应折外记录组计算，再汇总三个种子。",
                  "数量正确不等于完整集合匹配：标签个数相同，具体证候要素仍可能选错。",
                  "", "| 方法 | 真实标签数 | 记录数 | 数量正确率 | 低估率 | 高估率 | 数量 MAE |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for method in ("gru_stop", "sigmoid_matched", "gru_no_stop_cardinality"):
        for group in ("all", "1-3", "4-5", "6"):
            row = next(item for item in cardinality_rows
                       if item["method"] == method and item["true_cardinality"] == group)
            lines.append(f"| {row['method']} | {row['true_cardinality']} | {row['n']} | "
                         f"{format_mean_sd(row, 'cardinality_exact')} | "
                         f"{format_mean_sd(row, 'underprediction')} | "
                         f"{format_mean_sd(row, 'overprediction')} | "
                         f"{format_mean_sd(row, 'cardinality_mae')} |")
    lines.extend(["", "逐个真实标签数的完整统计见 `cardinality_summary.csv`；"
                  "1–3 标签组仅有 375 条，六标签组有 2,657 条，合组结果不能代替逐个数量分析。",
                  ""])
    lines.extend(["## 按固定规则抽取的病例", "",
                  "展示病例先要求三个种子的预测集合及错误类型相同。为使样例更便于阅读，"
                  "仅在选例时排除同时出现以下成对描述的记录："
                  "食欲不振/食欲旺盛、口渴喜冷饮/口渴喜热饮、"
                  "胸痛遇寒加重/胸痛遇热加重、苔厚腻/苔薄白。"
                  "这项展示筛选不改数据、模型结果或任何统计量，也不判断这些共现是否临床无效。",
                  "",
                  "| 样例类型 | 筛选前候选数 | 筛选后候选数 |",
                  "|---|---:|---:|",
                  *[f"| {name} | {cases['eligible_count_before_screen'][key]} | "
                    f"{cases['eligible_count_after_screen'][key]} |"
                    for key, name in (("exact", "完整匹配"),
                                      ("omission_only", "仅漏选"),
                                      ("addition_only", "仅多选"))],
                  "",
                  f"全数据的症状数量中位数为 {cases['median_symptom_count']:g}。"
                  "在筛选后候选中，选症状数最接近该中位数的记录；并列时选最小记录 ID。",
                  "病例用于展示模型输出，不把动作次序解释成医生的辨证推理。", ""])
    category_names = {"exact": "完整匹配", "omission_only": "仅漏选",
                      "addition_only": "仅多选"}
    for item in cases["cases"]:
        lines.extend([f"### 记录 {item['record_id']}：{category_names[item['category']]}", "",
                      f"折：{item['fold']}；症状：{'、'.join(item['symptoms'])}。",
                      f"参考集合：{'、'.join(item['reference_labels'])}。",
                      f"预测集合：{'、'.join(item['predicted_labels'])}。",
                      f"种子 9 动作：{' → '.join(item['traces']['9'])}。", ""])
    lines.extend(["所有三个种子的动作轨迹、漏选和多选标签见 `case_examples.json`。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m03-package", type=Path,
                        default=PROJECT / "matched_encoder_comparison")
    parser.add_argument("--m04-package", type=Path,
                        default=PROJECT / "gru_mechanism")
    parser.add_argument("--m03-results", type=Path)
    parser.add_argument("--m04-results", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        parser.error("--bootstrap-draws must be at least 100")
    m03_package, m04_package = args.m03_package.resolve(), args.m04_package.resolve()
    m03_results = (args.m03_results or m03_package / "results").resolve()
    m04_results = (args.m04_results or m04_package / "results").resolve()
    source, records, partitions = load_inputs(
        m03_package, m04_package, m03_results, m04_results)
    labels = source["schema"]["labels"]
    predictions, traces = load_predictions(
        m03_results, m04_results, records, partitions, len(labels))
    baselines, definitions = fixed_baselines(records, partitions, len(labels))
    strata = summarize(predictions, records)
    baseline_rows = baseline_summary(baselines, records)
    paired = paired_differences(predictions, records, args.bootstrap_draws)
    matrix = cardinality_matrix(predictions, records)
    label_seed_rows, label_rows = per_label_metrics(predictions, records, labels)
    cardinality_rows = cardinality_summary(predictions, records)
    cases = select_cases(records, predictions, traces, partitions, labels)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "stratified_metrics.csv", strata)
    write_csv(output / "diagnostic_baselines.csv", baseline_rows)
    write_csv(output / "paired_differences.csv", paired)
    write_csv(output / "cardinality_matrix.csv", matrix)
    write_csv(output / "per_label_seed_metrics.csv", label_seed_rows)
    write_csv(output / "per_label_metrics.csv", label_rows)
    write_csv(output / "cardinality_summary.csv", cardinality_rows)
    write_json(output / "baseline_definitions.json", definitions)
    write_json(output / "case_examples.json", cases)
    write_json(output / "analysis_manifest.json", {
        "m03_results_protocol_sha256": digest(m03_results / "protocol.json"),
        "m04_results_protocol_sha256": digest(m04_results / "protocol.json"),
        "shared_split_protocol_sha256": digest(m04_package / "splits" / "protocol.json"),
        "data_sha256": source["data_sha256"],
        "seeds": list(SEEDS), "folds": list(FOLDS),
        "bootstrap_draws": args.bootstrap_draws,
        "methods": list(METHODS),
    })
    (output / "analysis_report.md").write_text(
        report_text(strata, baseline_rows, definitions, paired, matrix,
                    cases, labels, records, args.bootstrap_draws,
                    label_rows, cardinality_rows),
        encoding="utf-8")
    print(f"VERIFIED 72/72 source runs; wrote {output / 'analysis_report.md'}")


if __name__ == "__main__":
    main()
