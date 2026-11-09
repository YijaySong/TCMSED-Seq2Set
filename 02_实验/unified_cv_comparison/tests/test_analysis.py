import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import analyze_results as analysis


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class AnalysisTests(unittest.TestCase):
    def test_per_record_scores_and_bootstrap_pairing(self):
        scores = analysis.per_record([[1, 0, 1], [0, 1, 0]],
                                     [[1, 1, 0], [0, 1, 0]])
        np.testing.assert_allclose(scores["exact"], [0, 1])
        np.testing.assert_allclose(scores["sample_f1"], [0.5, 1])
        np.testing.assert_allclose(scores["cardinality_mae"], [0, 0])
        difference = np.array([1, 0, -1, 1], dtype=float)
        self.assertEqual(analysis.bootstrap_interval(difference, draws=300, seed=7),
                         analysis.bootstrap_interval(difference, draws=300, seed=7))

    def test_packaged_reference_protocol_matches_locked_split(self):
        root = Path(__file__).resolve().parents[1]
        split_path = root / "splits" / "protocol.json"
        split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
        data_hash = hashlib.sha256((root / "广安门心绞痛4200例.txt").read_bytes()).hexdigest()
        self.assertEqual(analysis.check_protocol(root / "reference_results",
                                                 analysis.REFERENCE_METHODS,
                                                 split_hash, data_hash),
                         hashlib.sha256((root / "reference_results" / "protocol.json").read_bytes()).hexdigest())

    def make_synthetic(self, root):
        split = {"record_count": 2, "data_file": "cases.txt"}
        data = root / "cases.txt"
        data.write_text("synthetic data", encoding="utf-8")
        split["data_sha256"] = analysis.sha256(data)
        split_path = root / "splits" / "protocol.json"
        save_json(split_path, split)
        split_hash = analysis.sha256(split_path)
        reference = root / "reference_results"
        results = root / "results"
        for base, methods in ((reference, ["gru_stop"]),
                              (results, ["symptom_transformer"])):
            protocol = {"methods": methods, "folds": [1], "seeds": [9],
                        "epochs": 100, "batch_size": 64,
                        "split_protocol_sha256": split_hash,
                        "data_sha256": split["data_sha256"]}
            if base == results:
                protocol["output_methods"] = methods
            save_json(base / "protocol.json", protocol)
            protocol_hash = analysis.sha256(base / "protocol.json")
            for method in methods:
                folder = base / method / "fold_1" / "seed_9"
                truth = np.array([[1, 0], [0, 1]])
                predicted = (truth if method == "gru_stop"
                             else np.array([[1, 0], [1, 0]]))
                save_json(folder / "predictions.json", {
                    "ids": [1, 2], "truth": truth.tolist(),
                    "predicted": predicted.tolist()})
                summary = {
                    "method": method, "fold": 1, "seed": 9,
                    "protocol_sha256": protocol_hash, "selection": {"epoch": 5},
                    "test": analysis.metrics(truth, predicted), "parameters": 100,
                    "training_seconds": 2.0, "test_inference_seconds": 0.1,
                    "test_records": 2}
                if base == results:
                    summary["training_method"] = method
                    history = [{"epoch": i, "stage": "supervised"}
                               for i in range(1, 101)]
                    summary["selection"] = history[4]
                    save_json(folder / "history.json", history)
                    (folder / "best.pt").write_bytes(b"checkpoint")
                    summary["artifact_sha256"] = {
                        name: analysis.sha256(folder / name)
                        for name in ("best.pt", "history.json", "predictions.json")}
                save_json(folder / "summary.json", summary)
        return reference, results

    def test_collect_checks_predictions_against_fold_truth_and_saved_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, results = self.make_synthetic(root)
            folds = {1: {"test": ([1, 2], None, np.array([[1, 0], [0, 1]]))}}
            overrides = (patch.object(analysis, "ROOT", root),
                         patch.object(analysis, "REFERENCE_METHODS", ("gru_stop",)),
                         patch.object(analysis, "NEW_TRAIN_METHODS", ("symptom_transformer",)),
                         patch.object(analysis, "NEW_METHODS", ("symptom_transformer",)),
                         patch.object(analysis, "METHODS", ("gru_stop", "symptom_transformer")),
                         patch.object(analysis, "FOLDS", (1,)),
                         patch.object(analysis, "SEEDS", (9,)),
                         patch.object(analysis, "load_data", return_value=({}, folds, 4, 2)))
            for override in overrides:
                override.start()
            try:
                runs, rows = analysis.collect(reference, results)
                self.assertEqual(len(runs), 2)
                self.assertAlmostEqual(rows["gru_stop", 9]["metrics"]["exact"], 1.0)
                self.assertAlmostEqual(rows["symptom_transformer", 9]["metrics"]["exact"], 0.5)

                path = results / "symptom_transformer" / "fold_1" / "seed_9" / "predictions.json"
                summary_path = results / "symptom_transformer" / "fold_1" / "seed_9" / "summary.json"
                prediction = analysis.read_json(path)
                prediction["ids"] = [2, 1]
                save_json(path, prediction)
                summary = analysis.read_json(summary_path)
                summary["artifact_sha256"]["predictions.json"] = analysis.sha256(path)
                save_json(summary_path, summary)
                with self.assertRaisesRegex(ValueError, "Test IDs or truth differ"):
                    analysis.collect(reference, results)
                prediction["ids"] = [1, 2]
                save_json(path, prediction)

                summary["artifact_sha256"]["predictions.json"] = analysis.sha256(path)
                summary["test"]["exact"] = 0.9
                save_json(summary_path, summary)
                with self.assertRaisesRegex(ValueError, "Saved exact differs"):
                    analysis.collect(reference, results)
            finally:
                for override in reversed(overrides):
                    override.stop()

    def test_set_nll_outputs_share_training_and_rl_uses_rl_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, results = self.make_synthetic(root)
            new_training = ("symptom_transformer", "set_nll", "set_rl")
            new_outputs = ("symptom_transformer", "set_nll_greedy",
                           "set_nll_beam5", "set_nll_map", "set_rl")
            protocol_path = results / "protocol.json"
            protocol = analysis.read_json(protocol_path)
            protocol["methods"] = list(new_training)
            protocol["output_methods"] = list(new_outputs)
            save_json(protocol_path, protocol)
            protocol_hash = analysis.sha256(protocol_path)
            existing_summary = results / "symptom_transformer" / "fold_1" / "seed_9" / "summary.json"
            summary = analysis.read_json(existing_summary)
            summary["protocol_sha256"] = protocol_hash
            save_json(existing_summary, summary)

            truth = np.array([[1, 0], [0, 1]])
            for method in new_outputs[1:]:
                folder = results / method / "fold_1" / "seed_9"
                predicted = truth.copy()
                save_json(folder / "predictions.json", {
                    "ids": [1, 2], "truth": truth.tolist(),
                    "predicted": predicted.tolist()})
                is_rl = method == "set_rl"
                history = [{"epoch": epoch,
                            "stage": ("rl" if is_rl and epoch > 50 else "nll")}
                           for epoch in range(1, 101)]
                save_json(folder / "history.json", history)
                (folder / "best.pt").write_bytes(b"rl checkpoint" if is_rl
                                                 else b"shared nll checkpoint")
                summary = {
                    "method": method,
                    "training_method": ("set_rl" if is_rl else "set_nll"),
                    "fold": 1, "seed": 9,
                    "selection": history[69] if is_rl else history[4],
                    "test": analysis.metrics(truth, predicted),
                    "parameters": 100, "training_seconds": 2.0,
                    "test_inference_seconds": 0.1, "test_records": 2,
                    "protocol_sha256": protocol_hash,
                }
                if is_rl:
                    summary["pretraining_selection"] = history[19]
                    summary["rl_stage_start_epoch"] = 51
                    pretraining_predicted = np.array([[1, 0], [1, 0]])
                    save_json(folder / "pretraining_predictions.json", {
                        "ids": [1, 2], "truth": truth.tolist(),
                        "predicted": pretraining_predicted.tolist()})
                    (folder / "pretraining_best.pt").write_bytes(b"pretraining checkpoint")
                    summary["pretraining_test"] = analysis.metrics(truth, pretraining_predicted)
                    summary["pretraining_inference_seconds"] = 0.1
                else:
                    summary["decoder"] = method.removeprefix("set_nll_")
                artifacts = ["best.pt", "history.json", "predictions.json"]
                if is_rl:
                    artifacts += ["pretraining_best.pt", "pretraining_predictions.json"]
                summary["artifact_sha256"] = {
                    name: analysis.sha256(folder / name)
                    for name in artifacts}
                save_json(folder / "summary.json", summary)

            folds = {1: {"test": ([1, 2], None, truth)}}
            overrides = (patch.object(analysis, "ROOT", root),
                         patch.object(analysis, "REFERENCE_METHODS", ("gru_stop",)),
                         patch.object(analysis, "NEW_TRAIN_METHODS", new_training),
                         patch.object(analysis, "NEW_METHODS", new_outputs),
                         patch.object(analysis, "METHODS", ("gru_stop",) + new_outputs),
                         patch.object(analysis, "FOLDS", (1,)),
                         patch.object(analysis, "SEEDS", (9,)),
                         patch.object(analysis, "load_data", return_value=({}, folds, 4, 2)))
            for override in overrides:
                override.start()
            try:
                runs, _ = analysis.collect(reference, results)
                self.assertEqual(len(runs), 6)
                _, rows = analysis.collect(reference, results)
                self.assertAlmostEqual(rows[analysis.RL_PRETRAINING, 9]["metrics"]["exact"],
                                       0.5)
                folder = results / "set_nll_map" / "fold_1" / "seed_9"
                checkpoint = folder / "best.pt"
                checkpoint.write_bytes(b"different nll checkpoint")
                summary_path = folder / "summary.json"
                summary = analysis.read_json(summary_path)
                summary["artifact_sha256"]["best.pt"] = analysis.sha256(checkpoint)
                save_json(summary_path, summary)
                with self.assertRaisesRegex(ValueError, "Set-NLL decoders do not share training"):
                    analysis.collect(reference, results)

                checkpoint.write_bytes(b"shared nll checkpoint")
                summary["artifact_sha256"]["best.pt"] = analysis.sha256(checkpoint)
                save_json(summary_path, summary)
                rl_summary_path = results / "set_rl" / "fold_1" / "seed_9" / "summary.json"
                rl_summary = analysis.read_json(rl_summary_path)
                rl_summary["pretraining_test"]["exact"] = 0.9
                save_json(rl_summary_path, rl_summary)
                with self.assertRaisesRegex(ValueError, "Saved exact differs"):
                    analysis.collect(reference, results)
                rl_summary["pretraining_test"]["exact"] = 0.5
                save_json(rl_summary_path, rl_summary)
                pretraining_checkpoint = rl_summary_path.parent / "pretraining_best.pt"
                pretraining_checkpoint.write_bytes(b"changed pretraining checkpoint")
                with self.assertRaisesRegex(ValueError, "Run artifact differs"):
                    analysis.collect(reference, results)
                pretraining_checkpoint.write_bytes(b"pretraining checkpoint")
                rl_summary["selection"] = {"epoch": 50, "stage": "nll"}
                save_json(rl_summary_path, rl_summary)
                with self.assertRaisesRegex(ValueError, "Set-RL stages or checkpoint differ"):
                    analysis.collect(reference, results)
            finally:
                for override in reversed(overrides):
                    override.stop()

    def test_summarize_uses_seed_sd_and_gru_favorable_difference(self):
        measures = analysis.MEASURES
        runs = {}
        rows = {}
        for method in ("gru_stop", "symptom_transformer", "set_rl"):
            for index, seed in enumerate((9, 17, 29)):
                runs[method, seed, 1] = {"parameters": 100,
                                         "training_seconds": 2.0,
                                         "test_inference_seconds": 0.1}
                mean = (0.8 + index * 0.1 if method == "gru_stop"
                        else 1.0 if method == "set_rl" else 0.5)
                rows[method, seed] = {
                    "metrics": {name: mean for name in measures},
                    "per_record": {
                        "exact": np.array([1., 1.]) if method in ("gru_stop", "set_rl")
                                 else np.array([1., 0.]),
                        "sample_f1": np.array([1., 1.]) if method in ("gru_stop", "set_rl")
                                     else np.array([1., 0.]),
                        "cardinality_mae": np.array([0., 0.]) if method in ("gru_stop", "set_rl")
                                           else np.array([0., 1.]),
                    },
                }
                if method == "set_rl":
                    rows[analysis.RL_PRETRAINING, seed] = {
                        "metrics": {name: 0.5 for name in measures},
                        "per_record": {
                            "exact": np.array([1., 0.]),
                            "sample_f1": np.array([1., 0.]),
                            "cardinality_mae": np.array([0., 1.]),
                        },
                    }
        with patch.object(analysis, "METHODS", ("gru_stop", "symptom_transformer", "set_rl")), \
             patch.object(analysis, "FOLDS", (1,)):
            report = analysis.summarize(runs, rows, draws=100)
            self.assertIn("Set-RL 相对监督预训练起点", analysis.render(report))
        self.assertAlmostEqual(report["methods"]["gru_stop"]["oof_mean"]["exact"], 0.9)
        self.assertAlmostEqual(report["methods"]["gru_stop"]["oof_sample_sd"]["exact"], 0.1)
        for measure in ("exact", "sample_f1", "cardinality_mae"):
            self.assertAlmostEqual(report["paired_vs_gru"]["symptom_transformer"]
                                   [measure]["gru_advantage"], 0.5)
            self.assertAlmostEqual(report["rl_vs_pretraining"]
                                   ["paired_final_minus_pretraining"]
                                   [measure]["rl_advantage"], 0.5)


if __name__ == "__main__":
    unittest.main()
