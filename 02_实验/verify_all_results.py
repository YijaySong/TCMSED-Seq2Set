"""Read-only integrity checks for the 4200-record experiment bundle.

This script does not train models and does not rewrite any result file.  It
checks that the copied data, split manifests, protocols, run artifacts and
out-of-fold predictions are mutually consistent.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FOLDS = (1, 2, 3)
SEEDS = (9, 17, 29)
LABEL_COUNT = 10
RECORD_COUNT = 4200
TEST_SIZE = 1400
EXPECTED_DATA_SHA256 = "1dabc3ff0edbbb931ceca244bc1e56ca87f6100a479a8bd43af0026feffb21b7"

UNIFIED_METHODS = (
    "mlp", "cnn", "symptom_transformer", "asl_mlp", "classifier_chain_legacy",
    "label_graph", "set_attention", "transformer_sequence_matched", "tcn_sequence",
    "patchtst_style_sequence", "mamba_style_v3", "set_nll_greedy",
    "set_nll_beam5", "set_nll_map", "set_rl",
)
UNIFIED_TRAIN_METHODS = (
    "mlp", "cnn", "symptom_transformer", "asl_mlp", "classifier_chain_legacy",
    "label_graph", "set_attention", "transformer_sequence_matched", "tcn_sequence",
    "patchtst_style_sequence", "mamba_style_v3", "set_nll", "set_rl",
)
REFERENCE_METHODS = ("sigmoid_small", "sigmoid_matched", "classifier_chain", "gru_stop")
MATCHED_METHODS = REFERENCE_METHODS
M04_METHODS = ("gru_stop", "gru_fixed_order", "gru_no_label_history", "gru_no_stop_cardinality")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_json(path: Path):
    if not path.is_file():
        raise AssertionError(f"missing file: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_dataset_and_splits() -> dict:
    data_files = [ROOT / "广安门心绞痛4200例.txt"]
    data_files.extend(
        ROOT / name / "广安门心绞痛4200例.txt"
        for name in ("unified_cv_comparison", "matched_encoder_comparison",
                     "gru_mechanism")
    )
    for path in data_files:
        assert path.is_file(), f"missing data file: {path.relative_to(ROOT)}"
        assert digest(path) == EXPECTED_DATA_SHA256, f"data hash differs: {path.relative_to(ROOT)}"

    protocol_path = ROOT / "evaluation_protocol" / "splits" / "protocol.json"
    split_protocol = load_json(protocol_path)
    assert split_protocol["record_count"] == RECORD_COUNT
    assert split_protocol["outer_folds"] == 3
    assert split_protocol["epochs"] == 100
    assert split_protocol["batch_size"] == 64
    assert split_protocol["data_sha256"] == EXPECTED_DATA_SHA256

    partitions = {}
    all_test_ids = []
    for fold in FOLDS:
        path = ROOT / "evaluation_protocol" / "splits" / f"fold_{fold}.csv"
        expected_hash = split_protocol["folds"][fold - 1]["sha256"]
        assert digest(path) == expected_hash, f"split hash differs: {path.relative_to(ROOT)}"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == RECORD_COUNT, f"wrong row count: {path.relative_to(ROOT)}"
        assert {row["split"] for row in rows} == {"train", "val", "test"}
        ids = [int(row["id"]) for row in rows]
        assert sorted(ids) == list(range(1, RECORD_COUNT + 1)), f"bad IDs: {path.relative_to(ROOT)}"
        partitions[fold] = {
            split: [int(row["id"]) for row in rows if row["split"] == split]
            for split in ("train", "val", "test")
        }
        assert len(partitions[fold]["test"]) == TEST_SIZE
        all_test_ids.extend(partitions[fold]["test"])
    assert sorted(all_test_ids) == list(range(1, RECORD_COUNT + 1)), "test folds do not partition all records"
    return {"split_protocol": split_protocol, "partitions": partitions}


def check_protocol(path: Path, expected_methods: tuple[str, ...], split_protocol_hash: str,
                   data_hash: str, expected_outputs: tuple[str, ...] | None = None) -> str:
    protocol = load_json(path)
    assert protocol.get("methods") == list(expected_methods), f"method list differs: {path.relative_to(ROOT)}"
    if expected_outputs is not None:
        assert protocol.get("output_methods") == list(expected_outputs), f"output method list differs: {path.relative_to(ROOT)}"
    assert protocol.get("folds") == list(FOLDS), f"fold list differs: {path.relative_to(ROOT)}"
    assert protocol.get("seeds") == list(SEEDS), f"seed list differs: {path.relative_to(ROOT)}"
    assert protocol.get("epochs") == 100, f"epoch setting differs: {path.relative_to(ROOT)}"
    assert protocol.get("batch_size") == 64, f"batch size differs: {path.relative_to(ROOT)}"
    if "split_protocol_sha256" in protocol:
        assert protocol["split_protocol_sha256"] == split_protocol_hash, f"split protocol differs: {path.relative_to(ROOT)}"
    if "data_sha256" in protocol:
        assert protocol["data_sha256"] == data_hash, f"data hash differs in protocol: {path.relative_to(ROOT)}"
    return digest(path)


def check_prediction(path: Path, expected_ids: list[int]) -> None:
    prediction = load_json(path)
    ids = prediction.get("ids")
    truth = prediction.get("truth")
    predicted = prediction.get("predicted")
    assert ids == expected_ids, f"test IDs differ: {path.relative_to(ROOT)}"
    assert isinstance(truth, list) and isinstance(predicted, list)
    assert len(truth) == len(predicted) == TEST_SIZE
    assert all(isinstance(row, list) and len(row) == LABEL_COUNT for row in truth)
    assert all(isinstance(row, list) and len(row) == LABEL_COUNT for row in predicted)
    assert all(value in (0, 1) for row in truth for value in row)
    assert all(value in (0, 1) for row in predicted for value in row)


def check_runs(base: Path, methods: tuple[str, ...], protocol_hash: str,
               partitions: dict, full_artifacts: bool) -> int:
    count = 0
    for method in methods:
        method_root = base / method
        assert method_root.is_dir(), f"missing method directory: {method_root.relative_to(ROOT)}"
        for fold in FOLDS:
            expected_ids = partitions[fold]["test"]
            for seed in SEEDS:
                folder = method_root / f"fold_{fold}" / f"seed_{seed}"
                summary_path = folder / "summary.json"
                prediction_path = folder / "predictions.json"
                summary = load_json(summary_path)
                assert summary.get("method") == method, f"method identity differs: {folder.relative_to(ROOT)}"
                assert summary.get("fold") == fold and summary.get("seed") == seed
                assert summary.get("protocol_sha256") == protocol_hash, f"protocol hash differs: {folder.relative_to(ROOT)}"
                assert summary.get("test_records") == TEST_SIZE
                check_prediction(prediction_path, expected_ids)
                if full_artifacts:
                    history = load_json(folder / "history.json")
                    assert isinstance(history, list) and len(history) == 100
                    assert [row.get("epoch") for row in history] == list(range(1, 101))
                    assert (folder / "best.pt").is_file(), f"missing checkpoint: {folder.relative_to(ROOT)}"
                count += 1
    return count


def check_forbidden_files() -> None:
    forbidden_suffixes = {".tex", ".cls", ".sty", ".bib", ".doc", ".docx", ".pdf", ".ppt", ".pptx"}
    forbidden = [path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in forbidden_suffixes]
    assert not forbidden, "paper or layout files found: " + ", ".join(str(p.relative_to(ROOT)) for p in forbidden)


def main() -> None:
    split_info = check_dataset_and_splits()
    split_protocol = split_info["split_protocol"]
    split_protocol_hash = digest(ROOT / "evaluation_protocol" / "splits" / "protocol.json")
    data_hash = EXPECTED_DATA_SHA256

    unified_protocol_hash = check_protocol(
        ROOT / "unified_cv_comparison" / "results" / "protocol.json",
        UNIFIED_TRAIN_METHODS, split_protocol_hash, data_hash,
        expected_outputs=UNIFIED_METHODS)
    reference_protocol_hash = check_protocol(
        ROOT / "unified_cv_comparison" / "reference_results" / "protocol.json",
        REFERENCE_METHODS, split_protocol_hash, data_hash)
    matched_protocol_hash = check_protocol(
        ROOT / "matched_encoder_comparison" / "results" / "protocol.json",
        MATCHED_METHODS, split_protocol_hash, data_hash)
    m04_protocol_hash = check_protocol(
        ROOT / "gru_mechanism" / "results" / "protocol.json",
        M04_METHODS, split_protocol_hash, data_hash)

    counts = {
        "unified": check_runs(ROOT / "unified_cv_comparison" / "results",
                               UNIFIED_METHODS, unified_protocol_hash,
                               split_info["partitions"], full_artifacts=True),
        "reference": check_runs(ROOT / "unified_cv_comparison" / "reference_results",
                                 REFERENCE_METHODS, reference_protocol_hash,
                                 split_info["partitions"], full_artifacts=False),
        "matched": check_runs(ROOT / "matched_encoder_comparison" / "results",
                               MATCHED_METHODS, matched_protocol_hash,
                               split_info["partitions"], full_artifacts=True),
        "m04": check_runs(ROOT / "gru_mechanism" / "results",
                           M04_METHODS, m04_protocol_hash,
                           split_info["partitions"], full_artifacts=True),
    }
    expected_analysis = (
        ROOT / "unified_cv_comparison" / "results" / "analysis_report.md",
        ROOT / "matched_encoder_comparison" / "results" / "analysis_report.md",
        ROOT / "gru_mechanism" / "results" / "analysis_report.md",
        ROOT / "cardinality_analysis" / "results" / "analysis_report.md",
    )
    for path in expected_analysis:
        assert path.is_file() and path.stat().st_size > 0, f"missing analysis report: {path.relative_to(ROOT)}"
    for path in (ROOT / "cardinality_analysis" / "results").glob("*"):
        assert path.is_file(), f"unexpected nested M05 result path: {path.relative_to(ROOT)}"
    check_forbidden_files()

    print("Data hash:", data_hash)
    print("Runs checked:", ", ".join(f"{name}={value}" for name, value in counts.items()))
    print("Out-of-fold records checked per method/seed: 4200")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
