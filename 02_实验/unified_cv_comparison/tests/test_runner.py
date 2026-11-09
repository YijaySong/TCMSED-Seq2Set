import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import run_all as runner
from run_all import (FOLDS, METHODS, OUTPUT_METHODS, SEEDS, load_data,
                     batch_tokens, model_for, output_methods)


class UnifiedRunnerTests(unittest.TestCase):
    def test_locked_folds_cover_every_record_once(self):
        protocol, folds, vocab, labels = load_data()
        self.assertEqual(protocol["data_sha256"],
                         "1dabc3ff0edbbb931ceca244bc1e56ca87f6100a479a8bd43af0026feffb21b7")
        self.assertEqual((vocab, labels, FOLDS, SEEDS),
                         (42, 10, (1, 2, 3), (9, 17, 29)))
        tests = []
        for fold in FOLDS:
            self.assertEqual([len(folds[fold][role][0])
                              for role in ("train", "val", "test")],
                             [2380, 420, 1400])
            tests.extend(folds[fold]["test"][0])
        self.assertEqual(sorted(tests), list(range(1, 4201)))

    def test_models_match_intended_architectures(self):
        counts = {
            "mlp": 13578,
            "cnn": 157066,
            "symptom_transformer": 271882,
            "asl_mlp": 13578,
            "classifier_chain_legacy": 13678,
            "label_graph": 13578,
            "set_attention": 338058,
            "transformer_sequence_matched": 520715,
            "tcn_sequence": 1274699,
            "patchtst_style_sequence": 1695563,
            "mamba_style_v3": 682827,
            "set_nll": 48395,
            "set_rl": 48395,
        }
        self.assertEqual(set(METHODS), set(counts))
        train_y = np.array([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        for method, count in counts.items():
            model = model_for(method, 42, 10, train_y)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)

    def test_set_nll_decoder_rows_share_one_training_arm(self):
        self.assertEqual(output_methods("set_nll"),
                         ("set_nll_greedy", "set_nll_beam5", "set_nll_map"))
        self.assertEqual(len(OUTPUT_METHODS), len(METHODS) + 2)

    def test_symptom_permutation_does_not_change_encoder_outputs(self):
        x = torch.tensor([[2, 5, 8, 0], [8, 2, 5, 0]])
        for method in ("symptom_transformer", "transformer_sequence_matched",
                       "mamba_style_v3"):
            model = model_for(method, 42, 10)
            model.eval()
            with torch.no_grad():
                if method == "symptom_transformer":
                    output = model(x)
                elif method == "transformer_sequence_matched":
                    output = model.encode(x)
                else:
                    output = model.encoder(x)
            self.assertTrue(torch.allclose(output[0], output[1], atol=1e-5), method)

    def test_tcn_fixed_padding_removes_batch_neighbor_effect(self):
        model = model_for("tcn_sequence", 42, 10)
        model.eval()
        short = batch_tokens([[2, 3, 4], [5]], range(2), pad_to=40)
        long = batch_tokens([[2, 3, 4], list(range(5, 19))],
                            range(2), pad_to=40)
        with torch.no_grad():
            first = model.encoder(short)[0]
            second = model.encoder(long)[0]
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_set_training_saves_shared_decoders_and_rl_stage_checkpoint(self):
        rows = [[2, 3], [3, 4], [2, 5], [4, 5],
                [2, 4], [3, 5], [2, 3, 5], [3, 4, 5]]
        y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1],
                      [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=np.int64)
        folds = {1: {
            "train": (list(range(1, 9)), rows, y),
            "val": ([9, 10, 11, 12], rows[:4], y[:4]),
            "test": ([13, 14, 15, 16], rows[4:], y[4:]),
        }}
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(runner, "EPOCHS", 4), \
             patch.object(runner, "RL_PRETRAIN_EPOCHS", 2), \
             patch.object(runner, "BATCH", 4):
            results = Path(temporary)
            nll = runner.train_one("set_nll", 1, 9, folds, 8, 3,
                                   results, "synthetic-protocol")
            self.assertEqual({row["method"] for row in nll},
                             {"set_nll_greedy", "set_nll_beam5", "set_nll_map"})
            self.assertEqual(len({row["selection"]["epoch"] for row in nll}), 1)
            self.assertEqual(len({row["artifact_sha256"]["best.pt"] for row in nll}), 1)
            rl = runner.train_one("set_rl", 1, 9, folds, 8, 3,
                                  results, "synthetic-protocol")[0]
            self.assertGreaterEqual(rl["selection"]["epoch"], 3)
            self.assertLessEqual(rl["pretraining_selection"]["epoch"], 2)
            self.assertEqual(set(rl["artifact_sha256"]),
                             {"best.pt", "history.json", "predictions.json",
                              "pretraining_best.pt", "pretraining_predictions.json"})
            self.assertIn("exact", rl["pretraining_test"])

    def test_epoch_resume_matches_uninterrupted_training(self):
        rows = [[2, 3], [3, 4], [2, 5], [4, 5],
                [2, 4], [3, 5], [2, 3, 5], [3, 4, 5]]
        y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1],
                      [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=np.int64)
        folds = {1: {
            "train": (list(range(1, 9)), rows, y),
            "val": ([9, 10, 11, 12], rows[:4], y[:4]),
            "test": ([13, 14, 15, 16], rows[4:], y[4:]),
        }}
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(runner, "BATCH", 4), \
             patch.object(runner, "THRESHOLDS", np.array([0.5])):
            base = Path(temporary)
            with patch.object(runner, "EPOCHS", 4):
                runner.train_one("symptom_transformer", 1, 9, folds, 8, 3,
                                 base / "continuous", "synthetic-protocol")
            with patch.object(runner, "EPOCHS", 2):
                runner.train_one("symptom_transformer", 1, 9, folds, 8, 3,
                                 base / "resumed", "synthetic-protocol")
            with patch.object(runner, "EPOCHS", 4):
                runner.train_one("symptom_transformer", 1, 9, folds, 8, 3,
                                 base / "resumed", "synthetic-protocol")
            first = runner.run_folder(base / "continuous", "symptom_transformer", 1, 9)
            second = runner.run_folder(base / "resumed", "symptom_transformer", 1, 9)
            first_best = torch.load(first / "best.pt", map_location="cpu", weights_only=False)
            second_best = torch.load(second / "best.pt", map_location="cpu", weights_only=False)
            for name in first_best["model"]:
                self.assertTrue(torch.equal(first_best["model"][name],
                                            second_best["model"][name]), name)
            self.assertEqual((first / "predictions.json").read_bytes(),
                             (second / "predictions.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
