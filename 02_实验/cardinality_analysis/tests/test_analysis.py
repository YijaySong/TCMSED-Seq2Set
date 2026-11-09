import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze import (ILLUSTRATIVE_SCREEN_PAIRS, bootstrap_interval, error_type,
                     cardinality_summary, fixed_baselines, has_screen_pair,
                     per_label_metrics, per_record, select_cases)


class M05AnalysisTests(unittest.TestCase):
    def test_fixed_baselines_use_only_training_records(self):
        records = {
            1: {"symptoms": ["a"], "truth": (0, 1)},
            2: {"symptoms": ["b"], "truth": (0, 1)},
            3: {"symptoms": ["c"], "truth": (2,)},
        }
        partitions = {1: {"train": [1, 2], "val": [], "test": [3]},
                      2: {"train": [1, 3], "val": [], "test": [2]},
                      3: {"train": [2, 3], "val": [], "test": [1]}}
        predictions, definitions = fixed_baselines(records, partitions, 3)
        self.assertEqual(predictions["most_common_full_set"][3], (0, 1))
        self.assertEqual(predictions["most_common_full_set"][2], (0, 1))
        self.assertEqual(predictions["top_six_marginal"][3], (0, 1, 2))
        self.assertEqual([row["train_support"] for row in definitions
                          if row["method"] == "most_common_full_set"], [2, 1, 1])

    def test_metrics_and_error_types(self):
        self.assertEqual(per_record((0, 1), (0, 2)), (0.0, 0.5, 0.0))
        self.assertEqual(error_type((0, 1), (0,)), "omission_only")
        self.assertEqual(error_type((0,), (0, 1)), "addition_only")
        self.assertEqual(error_type((0, 1), (0, 2)), "mixed")

    def test_per_label_metrics_compute_each_seed_before_averaging(self):
        records = {1: {"truth": (0,)}, 2: {"truth": (0, 1)},
                   3: {"truth": (1, 2)}}
        imperfect = {1: (0,), 2: (1,), 3: (0, 1)}
        perfect = {1: (0,), 2: (0, 1), 3: (1,)}
        predictions = {"gru_stop": {9: imperfect, 17: perfect, 29: imperfect}}
        seed_rows, summary = per_label_metrics(
            predictions, records, ["A", "B", "C"], methods=("gru_stop",))
        rows = [row for row in seed_rows if row["label"] == "A"]
        self.assertEqual([(row["support"], row["tp"], row["fp"], row["fn"])
                          for row in rows], [(2, 1, 1, 1), (2, 2, 0, 0), (2, 1, 1, 1)])
        label_a = next(row for row in summary if row["label"] == "A")
        self.assertEqual(label_a["support"], 2)
        self.assertAlmostEqual(label_a["precision_mean"], 2 / 3)
        self.assertAlmostEqual(label_a["recall_mean"], 2 / 3)
        self.assertAlmostEqual(label_a["f1_mean"], 2 / 3)
        self.assertAlmostEqual(label_a["f1_seed_sd"], 0.2886751345948129)
        self.assertAlmostEqual(label_a["fp_mean"], 2 / 3)
        label_c = next(row for row in summary if row["label"] == "C")
        self.assertEqual((label_c["support"], label_c["precision_mean"],
                          label_c["recall_mean"], label_c["f1_mean"]),
                         (1, 0.0, 0.0, 0.0))

    def test_cardinality_summary_separates_exact_under_and_over(self):
        records = {i: {"truth": tuple(range(i))} for i in range(1, 7)}
        gru = {i: tuple(range(n)) for i, n in
               enumerate((1, 1, 4, 4, 6, 6), start=1)}
        gru[1] = (1,)  # Same predicted count, wrong label set.
        no_stop = {i: tuple(range(6)) for i in records}
        predictions = {
            "gru_stop": {seed: gru for seed in (9, 17, 29)},
            "sigmoid_matched": {seed: gru for seed in (9, 17, 29)},
            "gru_no_stop_cardinality": {seed: no_stop for seed in (9, 17, 29)},
        }
        rows = cardinality_summary(predictions, records)
        gru_all = next(row for row in rows if row["method"] == "gru_stop" and
                       row["true_cardinality"] == "all")
        self.assertAlmostEqual(gru_all["cardinality_exact_mean"], 0.5)
        self.assertAlmostEqual(gru_all["underprediction_mean"], 1 / 6)
        self.assertAlmostEqual(gru_all["overprediction_mean"], 2 / 6)
        self.assertAlmostEqual(gru_all["cardinality_mae_mean"], 0.5)
        self.assertEqual(gru_all["cardinality_exact_seed_sd"], 0.0)
        self.assertEqual(per_record(records[1]["truth"], gru[1])[0], 0.0)
        gru_short = next(row for row in rows if row["method"] == "gru_stop" and
                         row["true_cardinality"] == "1-3")
        self.assertEqual(gru_short["n"], 3)
        self.assertAlmostEqual(gru_short["cardinality_exact_mean"], 1 / 3)
        self.assertAlmostEqual(gru_short["underprediction_mean"], 1 / 3)
        self.assertAlmostEqual(gru_short["overprediction_mean"], 1 / 3)
        self.assertTrue(any(row["method"] == "sigmoid_matched" for row in rows))
        no_stop_all = next(row for row in rows if
                           row["method"] == "gru_no_stop_cardinality" and
                           row["true_cardinality"] == "all")
        self.assertAlmostEqual(no_stop_all["cardinality_exact_mean"], 1 / 6)
        self.assertEqual(no_stop_all["underprediction_mean"], 0.0)
        self.assertAlmostEqual(no_stop_all["cardinality_mae_mean"], 2.5)

    def test_constant_paired_difference_has_point_interval(self):
        self.assertEqual(bootstrap_interval(__import__("numpy").ones(4), draws=100),
                         (1.0, 1.0))

    def test_illustrative_screen_checks_only_the_four_listed_pairs(self):
        self.assertEqual(len(ILLUSTRATIVE_SCREEN_PAIRS), 4)
        for first, second in ILLUSTRATIVE_SCREEN_PAIRS:
            self.assertTrue(has_screen_pair([first, second]))
            self.assertFalse(has_screen_pair([first]))
        self.assertFalse(has_screen_pair(["气短懒言", "动则汗出"]))

    def test_case_selection_requires_three_seed_agreement(self):
        records = {
            1: {"symptoms": ["食欲不振", "食欲旺盛"], "truth": (0,)},
            2: {"symptoms": ["b"], "truth": (0, 1)},
            3: {"symptoms": ["c"], "truth": (0,)},
            4: {"symptoms": ["d"], "truth": (0,)},
            5: {"symptoms": ["e"], "truth": (0,)},
        }
        predictions = {"gru_stop": {
            seed: {1: (0,), 2: (0,), 3: (0, 1), 4: (0,), 5: (0,)}
            for seed in (9, 17, 29)
        }}
        predictions["gru_stop"][29][4] = (0, 1)
        traces = {seed: {1: (0, 2), 2: (0, 2), 3: (0, 1, 2),
                         4: (0, 2), 5: (0, 2)}
                  for seed in (9, 17, 29)}
        partitions = {1: {"test": [1, 2, 3]}, 2: {"test": [4]},
                      3: {"test": [5]}}
        selected = select_cases(records, predictions, traces, partitions, ["A", "B"])
        self.assertEqual([row["record_id"] for row in selected["cases"]], [5, 2, 3])
        self.assertEqual(selected["eligible_count_before_screen"],
                         {"exact": 2, "omission_only": 1, "addition_only": 1})
        self.assertEqual(selected["eligible_count_after_screen"],
                         {"exact": 1, "omission_only": 1, "addition_only": 1})

    def test_case_selection_reports_empty_screened_category(self):
        records = {1: {"symptoms": ["苔厚腻", "苔薄白"], "truth": (0,)}}
        predictions = {"gru_stop": {seed: {1: (0,)} for seed in (9, 17, 29)}}
        traces = {seed: {1: (0, 1)} for seed in (9, 17, 29)}
        partitions = {1: {"test": [1]}, 2: {"test": []}, 3: {"test": []}}
        with self.assertRaisesRegex(ValueError, "No illustrative case remains"):
            select_cases(records, predictions, traces, partitions, ["A"])


if __name__ == "__main__":
    unittest.main()
