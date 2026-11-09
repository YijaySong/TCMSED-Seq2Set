import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from analyze_results import (bootstrap_interval, compare_m03_reference,
                             per_record, replay_checkpoint)
from run_all import load_data, model_for, test_predictions


class AnalysisTests(unittest.TestCase):
    def test_per_record_scores_keep_pairing(self):
        truth = [[1, 0, 1], [0, 1, 0]]
        predicted = [[1, 1, 0], [0, 1, 0]]
        result = per_record(truth, predicted)
        np.testing.assert_allclose(result["exact"], [0, 1])
        np.testing.assert_allclose(result["sample_f1"], [0.5, 1])
        np.testing.assert_allclose(result["cardinality_mae"], [0, 0])

    def test_bootstrap_interval_is_deterministic(self):
        differences = np.array([0, 0, 1, 1], dtype=float)
        self.assertEqual(bootstrap_interval(differences, draws=300, seed=7),
                         bootstrap_interval(differences, draws=300, seed=7))

    def test_locked_folds_cover_all_records_once(self):
        manifest, folds, _, labels = load_data()
        self.assertEqual(manifest["record_count"], 4200)
        self.assertEqual(labels, 10)
        self.assertEqual(sorted(record_id for fold in folds.values()
                                for record_id in fold["test"][0]),
                         list(range(1, 4201)))

    def test_m03_comparison_allows_a_different_action_order(self):
        old = {"ids": [1], "truth": [[1, 1]], "predicted": [[1, 1]],
               "traces": [[0, 1, 2]]}
        new = {**old, "traces": [[1, 0, 2]]}
        runs = {("gru_stop", 9, 1): ({}, new)}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gru_stop" / "fold_1" / "seed_9"
            path.mkdir(parents=True)
            (path / "predictions.json").write_text(json.dumps(old), encoding="utf-8")
            with patch("analyze_results.FOLDS", (1,)), patch("analyze_results.SEEDS", (9,)):
                result = compare_m03_reference(runs, Path(temporary))
        self.assertEqual(result, {"runs": 1, "trace_mismatches": 1})

    def test_checkpoint_replays_saved_sets(self):
        torch.manual_seed(9)
        model = model_for("gru_stop", 7, 3)
        tokens = [[2, 3], [3]]
        predicted, traces = test_predictions(model, "gru_stop", tokens)
        expected = {"predicted": predicted.tolist(), "traces": traces}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            torch.save({"model": model.state_dict(), "selection": {}},
                       folder / "best.pt")
            self.assertEqual(replay_checkpoint(
                folder, "gru_stop", 7, 3, tokens, expected), 0)


if __name__ == "__main__":
    unittest.main()
