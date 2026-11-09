"""Audit unified three-fold predictions and compare methods on paired records."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_all import FOLDS, ROOT, SEEDS, load_data, metrics


REFERENCE_METHODS = ("sigmoid_small", "sigmoid_matched", "classifier_chain", "gru_stop")
NEW_TRAIN_METHODS = (
    "mlp", "cnn", "symptom_transformer", "asl_mlp", "classifier_chain_legacy",
    "label_graph", "set_attention", "transformer_sequence_matched", "tcn_sequence",
    "patchtst_style_sequence", "mamba_style_v3", "set_nll", "set_rl",
)
NEW_METHODS = (
    "mlp", "cnn", "symptom_transformer", "asl_mlp", "classifier_chain_legacy",
    "label_graph", "set_attention", "transformer_sequence_matched", "tcn_sequence",
    "patchtst_style_sequence", "mamba_style_v3", "set_nll_greedy", "set_nll_beam5",
    "set_nll_map", "set_rl",
)
METHODS = REFERENCE_METHODS + NEW_METHODS
SET_NLL_OUTPUTS = ("set_nll_greedy", "set_nll_beam5", "set_nll_map")
RL_PRETRAINING = "set_rl_pretraining"
MEASURES = ("exact", "sample_f1", "micro_f1", "macro_f1", "cardinality_mae")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def check_protocol(folder, expected_methods, split_hash, data_hash,
                   expected_outputs=None):
    path = folder / "protocol.json"
    protocol = read_json(path)
    if (protocol.get("methods") != list(expected_methods)
            or protocol.get("folds") != list(FOLDS)
            or protocol.get("seeds") != list(SEEDS)
            or protocol.get("epochs") != 100
            or protocol.get("batch_size") != 64):
        raise ValueError(f"Unexpected evaluation protocol: {path}")
    if (protocol.get("split_protocol_sha256") != split_hash
            or protocol.get("data_sha256") != data_hash):
        raise ValueError(f"Data or fold protocol differs: {path}")
    if expected_outputs is not None and protocol.get("output_methods") != list(expected_outputs):
        raise ValueError(f"Output method list differs: {path}")
    return sha256(path)


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
    difference = np.asarray(difference, dtype=float)
    if difference.ndim != 1 or not len(difference) or draws < 1:
        raise ValueError("Expected a nonempty per-record difference and positive draws")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for start in range(0, draws, 250):
        end = min(start + 250, draws)
        sampled = rng.integers(0, len(difference), size=(end - start, len(difference)))
        means[start:end] = difference[sampled].mean(1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def checked_predictions(prediction, test_ids, test_y, labels, saved_metrics, folder):
    truth = np.asarray(prediction.get("truth"), dtype=np.int64)
    predicted = np.asarray(prediction.get("predicted"), dtype=np.int64)
    if (prediction.get("ids") != test_ids
            or truth.shape != (len(test_ids), labels)
            or predicted.shape != truth.shape
            or not np.array_equal(truth, test_y)):
        raise ValueError(f"Test IDs or truth differ: {folder}")
    if not np.isin(predicted, (0, 1)).all():
        raise ValueError(f"Predictions must be binary: {folder}")
    recomputed = metrics(truth, predicted)
    for measure in MEASURES:
        if not np.isclose(recomputed[measure], saved_metrics[measure], atol=1e-10):
            raise ValueError(f"Saved {measure} differs: {folder}")
    return truth, predicted


def collect(reference_results, results):
    split_path = ROOT / "splits" / "protocol.json"
    split_protocol = read_json(split_path)
    split_hash = sha256(split_path)
    data_path = ROOT / split_protocol["data_file"]
    data_hash = sha256(data_path)
    if data_hash != split_protocol["data_sha256"]:
        raise ValueError("Data file differs from locked folds")
    protocol_hashes = {
        "reference": check_protocol(reference_results, REFERENCE_METHODS, split_hash, data_hash),
        "new": check_protocol(results, NEW_TRAIN_METHODS, split_hash, data_hash,
                              NEW_METHODS),
    }
    _, folds, _, labels = load_data()
    expected_ids = {record_id for fold in FOLDS for record_id in folds[fold]["test"][0]}
    if len(expected_ids) != split_protocol["record_count"]:
        raise ValueError("Outer test folds do not cover every record once")

    runs = {}
    rows = {}
    shared_set_nll = {}
    pretraining_by_seed = ({seed: {} for seed in SEEDS}
                           if "set_rl" in METHODS else {})
    for method in METHODS:
        source = "reference" if method in REFERENCE_METHODS else "new"
        base = reference_results if source == "reference" else results
        for seed in SEEDS:
            by_id = {}
            for fold in FOLDS:
                folder = base / method / f"fold_{fold}" / f"seed_{seed}"
                summary = read_json(folder / "summary.json")
                prediction = read_json(folder / "predictions.json")
                if (summary.get("method"), summary.get("fold"), summary.get("seed")) != (
                        method, fold, seed):
                    raise ValueError(f"Run identity differs: {folder}")
                if summary.get("protocol_sha256") != protocol_hashes[source]:
                    raise ValueError(f"Run protocol hash differs: {folder}")
                if source == "new":
                    history = read_json(folder / "history.json")
                    if (not (folder / "best.pt").exists()
                            or [row.get("epoch") for row in history] != list(range(1, 101))
                            or summary["selection"]["epoch"] not in range(1, 101)):
                        raise ValueError(f"Incomplete 100-epoch run: {folder}")
                    if summary["selection"] != history[summary["selection"]["epoch"] - 1]:
                        raise ValueError(f"Selected validation row differs from history: {folder}")
                    training_method = ("set_nll" if method in SET_NLL_OUTPUTS else method)
                    if summary.get("training_method") != training_method:
                        raise ValueError(f"Training method differs: {folder}")
                    if method in SET_NLL_OUTPUTS:
                        if (summary.get("decoder") != method.removeprefix("set_nll_")
                                or any(row.get("stage") != "nll" for row in history)):
                            raise ValueError(f"Set-NLL decoding or history differs: {folder}")
                    if method == "set_rl":
                        pretraining = summary.get("pretraining_selection", {})
                        if (summary["selection"]["epoch"] not in range(51, 101)
                                or pretraining.get("epoch") not in range(1, 51)
                                or summary.get("rl_stage_start_epoch") != 51
                                or [row.get("stage") for row in history] !=
                                ["nll"] * 50 + ["rl"] * 50):
                            raise ValueError(f"Set-RL stages or checkpoint differ: {folder}")
                        if pretraining != history[pretraining["epoch"] - 1]:
                            raise ValueError(f"Set-RL pretraining row differs: {folder}")
                    required_artifacts = {"best.pt", "history.json", "predictions.json"}
                    if method == "set_rl":
                        required_artifacts |= {"pretraining_best.pt",
                                               "pretraining_predictions.json"}
                    if set(summary.get("artifact_sha256", {})) != required_artifacts:
                        raise ValueError(f"Missing new-run artifact hashes: {folder}")
                for filename, digest in summary.get("artifact_sha256", {}).items():
                    path = folder / filename
                    if not path.exists() or sha256(path) != digest:
                        raise ValueError(f"Run artifact differs: {path}")
                if method in SET_NLL_OUTPUTS:
                    shared = (summary["selection"], summary["parameters"],
                              summary["training_seconds"],
                              summary["artifact_sha256"]["best.pt"],
                              summary["artifact_sha256"]["history.json"])
                    key = seed, fold
                    if key in shared_set_nll and shared_set_nll[key] != shared:
                        raise ValueError(f"Set-NLL decoders do not share training: {folder}")
                    shared_set_nll[key] = shared
                test_ids, _, test_y = folds[fold]["test"]
                if summary.get("test_records") != len(test_ids):
                    raise ValueError(f"Test record count differs: {folder}")
                truth, predicted = checked_predictions(
                    prediction, test_ids, test_y, labels, summary["test"], folder)
                if method == "set_rl":
                    pretraining_prediction = read_json(folder / "pretraining_predictions.json")
                    if ("pretraining_test" not in summary
                            or not isinstance(summary.get("pretraining_inference_seconds"),
                                              (int, float))
                            or summary["pretraining_inference_seconds"] < 0):
                        raise ValueError(f"Missing Set-RL pretraining evaluation: {folder}")
                    pre_truth, pre_predicted = checked_predictions(
                        pretraining_prediction, test_ids, test_y, labels,
                        summary["pretraining_test"], folder / "pretraining_predictions.json")
                    for record_id, y, y_hat in zip(test_ids, pre_truth, pre_predicted):
                        if record_id in pretraining_by_seed[seed]:
                            raise ValueError(f"Repeated Set-RL pretraining record {record_id}")
                        pretraining_by_seed[seed][record_id] = (y, y_hat)
                runs[method, seed, fold] = summary
                for record_id, y, y_hat in zip(test_ids, truth, predicted):
                    if record_id in by_id:
                        raise ValueError(f"Repeated test record {record_id}: {method}, {seed}")
                    by_id[record_id] = (y, y_hat)
            if set(by_id) != expected_ids:
                raise ValueError(f"Incomplete out-of-fold coverage: {method}, {seed}")
            ordered_ids = sorted(expected_ids)
            truth = np.stack([by_id[i][0] for i in ordered_ids])
            predicted = np.stack([by_id[i][1] for i in ordered_ids])
            rows[method, seed] = {"metrics": metrics(truth, predicted),
                                  "per_record": per_record(truth, predicted)}
    for seed, by_id in pretraining_by_seed.items():
        if set(by_id) != expected_ids:
            raise ValueError(f"Incomplete Set-RL pretraining OOF coverage: {seed}")
        ordered_ids = sorted(expected_ids)
        truth = np.stack([by_id[i][0] for i in ordered_ids])
        predicted = np.stack([by_id[i][1] for i in ordered_ids])
        rows[RL_PRETRAINING, seed] = {"metrics": metrics(truth, predicted),
                                      "per_record": per_record(truth, predicted)}
    return runs, rows


def summarize(runs, rows, draws=10000):
    methods = {}
    for method in METHODS:
        values = np.asarray([[rows[method, seed]["metrics"][name] for name in MEASURES]
                             for seed in SEEDS])
        methods[method] = {
            "oof_mean": dict(zip(MEASURES, values.mean(0).tolist())),
            "oof_sample_sd": dict(zip(MEASURES, values.std(0, ddof=1).tolist())),
            "parameters": [runs[method, seed, fold]["parameters"]
                           for fold in FOLDS for seed in SEEDS],
            "training_seconds_mean": float(np.mean([
                runs[method, seed, fold]["training_seconds"]
                for fold in FOLDS for seed in SEEDS])),
            "test_inference_seconds_mean": float(np.mean([
                runs[method, seed, fold]["test_inference_seconds"]
                for fold in FOLDS for seed in SEEDS])),
        }
    differences = {}
    for method in METHODS:
        if method == "gru_stop":
            continue
        differences[method] = {}
        for measure in ("exact", "sample_f1", "cardinality_mae"):
            paired = np.stack([
                rows["gru_stop", seed]["per_record"][measure]
                - rows[method, seed]["per_record"][measure]
                for seed in SEEDS]).mean(0)
            if measure == "cardinality_mae":
                paired = -paired
            lower, upper = bootstrap_interval(paired, draws=draws)
            differences[method][measure] = {
                "gru_advantage": float(paired.mean()),
                "ci95": [lower, upper],
            }
    pretraining_values = np.asarray([
        [rows[RL_PRETRAINING, seed]["metrics"][name] for name in MEASURES]
        for seed in SEEDS])
    rl_pretraining = {
        "oof_mean": dict(zip(MEASURES, pretraining_values.mean(0).tolist())),
        "oof_sample_sd": dict(zip(MEASURES, pretraining_values.std(0, ddof=1).tolist())),
        "paired_final_minus_pretraining": {},
    }
    for measure in ("exact", "sample_f1", "cardinality_mae"):
        paired = np.stack([
            rows["set_rl", seed]["per_record"][measure]
            - rows[RL_PRETRAINING, seed]["per_record"][measure]
            for seed in SEEDS]).mean(0)
        if measure == "cardinality_mae":
            paired = -paired
        lower, upper = bootstrap_interval(paired, draws=draws)
        rl_pretraining["paired_final_minus_pretraining"][measure] = {
            "rl_advantage": float(paired.mean()), "ci95": [lower, upper],
        }
    return {"methods": methods, "paired_vs_gru": differences,
            "rl_vs_pretraining": rl_pretraining,
            "folds": list(FOLDS), "seeds": list(SEEDS), "record_count": 4200,
            "new_training_runs": len(NEW_TRAIN_METHODS) * len(FOLDS) * len(SEEDS),
            "new_output_runs": len(NEW_METHODS) * len(FOLDS) * len(SEEDS),
            "bootstrap_draws": draws}


def render(report):
    lines = [
        "# 统一三折折外对照结果", "",
        "四种既有参考方法沿用已核对的 M03 结果；其余方法在相同数据与划分上训练。",
        (f"新增 {report['new_training_runs']} 次折/种子训练，生成 "
         f"{report['new_output_runs']} 份折/种子预测。Set-NLL 的三种解码共用训练检查点。"),
        "每条记录在每个种子下取得一次折外预测。表中为完整 4200 条折外预测的三个种子均值±样本标准差。",
        "", "| 方法 | EM | 样本 F1 | Micro-F1 | Macro-F1 | 基数 MAE | 参数量范围 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = report["methods"][method]
        values = [f"{row['oof_mean'][measure]:.4f} ± {row['oof_sample_sd'][measure]:.4f}"
                  for measure in MEASURES]
        parameters = row["parameters"]
        parameter_range = (f"{min(parameters):,}" if min(parameters) == max(parameters)
                           else f"{min(parameters):,}–{max(parameters):,}")
        lines.append(f"| {method} | " + " | ".join(values) + f" | {parameter_range} |")
    lines.extend([
        "", "## 与 GRU 的病例配对比较", "",
        f"每个病例先对三个种子的指标差取平均，再按病例有放回抽样 {report['bootstrap_draws']:,} 次。",
        "正差值表示 GRU 占优；EM、样本 F1 取 GRU 减对照，基数 MAE 取对照减 GRU。",
        "", "| 对照 | EM 差值 [95% 区间] | 样本 F1 差值 [95% 区间] | 基数 MAE 差值 [95% 区间] |",
        "|---|---:|---:|---:|",
    ])
    for method, measures in report["paired_vs_gru"].items():
        cells = []
        for measure in ("exact", "sample_f1", "cardinality_mae"):
            row = measures[measure]
            low, high = row["ci95"]
            cells.append(f"{row['gru_advantage']:+.4f} [{low:+.4f}, {high:+.4f}]")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    pretraining = report["rl_vs_pretraining"]
    lines.extend([
        "", "## Set-RL 相对监督预训练起点", "",
        "同一次训练在第 50 轮以前选出的监督检查点，与第 51 至 100 轮选出的 RL 检查点使用相同折外病例和种子。",
        "", "| 阶段 | EM | 样本 F1 | Micro-F1 | Macro-F1 | 基数 MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, source in (("监督预训练起点", pretraining),
                         ("RL 训练后", report["methods"]["set_rl"])):
        cells = [f"{source['oof_mean'][measure]:.4f} ± {source['oof_sample_sd'][measure]:.4f}"
                 for measure in MEASURES]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    comparisons = pretraining["paired_final_minus_pretraining"]
    lines.extend([
        "", "按病例配对的 RL 训练后减监督预训练起点差值（基数 MAE 反向），正值表示 RL 训练后占优：",
        "", "| EM [95% 区间] | 样本 F1 [95% 区间] | 基数 MAE [95% 区间] |",
        "|---:|---:|---:|",
    ])
    cells = []
    for measure in ("exact", "sample_f1", "cardinality_mae"):
        item = comparisons[measure]
        lower, upper = item["ci95"]
        cells.append(f"{item['rl_advantage']:+.4f} [{lower:+.4f}, {upper:+.4f}]")
    lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "", "三折采用记录级划分。已有数据参与过模型开发，因此这组结果属于开发后的内部评价。",
        "区间描述这批记录与三个已训练实例下的配对差异；缺少患者标识，不能据此推断患者级泛化。",
        "参考模型与新模型来自不同训练脚本，报告的训练及推理时间不用于跨平台排名。", "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-results", type=Path,
                        default=ROOT / "reference_results")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    runs, rows = collect(args.reference_results, args.results)
    report = summarize(runs, rows, draws=args.bootstrap_draws)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "unified_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    path = args.results / "analysis_report.md"
    path.write_text(render(report), encoding="utf-8")
    print(f"VERIFIED {len(runs)}/{len(METHODS) * len(SEEDS) * len(FOLDS)} runs; wrote {path}")


if __name__ == "__main__":
    main()
