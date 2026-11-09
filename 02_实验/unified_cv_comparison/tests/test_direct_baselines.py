import unittest

import numpy as np
import torch

from direct_baselines import METHODS, build, loss, predict_thresholds


class DirectBaselineTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([[2, 4, 0], [3, 5, 6]], dtype=torch.long)
        self.y = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)

    def test_legacy_model_parameter_counts(self):
        expected = {
            "mlp": 13578,
            "cnn": 157066,
            "asl_mlp": 13578,
            "classifier_chain_legacy": 13678,
            "label_graph": 13578,
            "set_attention": 338058,
        }
        self.assertEqual(set(METHODS), set(expected))
        train_y = np.zeros((3, 10), dtype=np.float32)
        for method, count in expected.items():
            model = build(method, 42, 10, train_y)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count,
                             method)

    def test_all_losses_backpropagate(self):
        torch.set_num_threads(1)
        for method in METHODS:
            model = build(method, 8, 3, self.y.numpy())
            with self.subTest(method=method):
                value = loss(model, method, self.x, self.y)
                self.assertTrue(torch.isfinite(value))
                value.backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_graph_uses_training_cooccurrence(self):
        model = build("label_graph", 8, 3, self.y.numpy())
        expected = torch.tensor([[0.5, 0.0, 0.5],
                                 [0.0, 1.0, 0.0],
                                 [0.5, 0.0, 0.5]])
        torch.testing.assert_close(model.adjacency, expected)

    def test_classifier_chain_threshold_changes_later_decisions(self):
        model = build("classifier_chain_legacy", 4, 2,
                      np.zeros((1, 2), dtype=np.float32))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.head.bias[:] = torch.tensor([0.4, -0.4])
            model.head.weight[1, model.encoder[0].out_features] = 2.0
        model.eval()
        x = torch.tensor([[2]], dtype=torch.long)
        predictions = predict_thresholds(model, "classifier_chain_legacy", x,
                                         [0.5, 0.7])
        self.assertEqual(predictions.shape, (2, 1, 2))
        self.assertEqual(predictions[0, 0].tolist(), [1, 1])
        self.assertEqual(predictions[1, 0].tolist(), [1, 0])
        self.assertEqual(predictions[0, 0].tolist(), model.predict(x)[0].int().tolist())

    def test_thresholds_guarantee_nonempty_direct_sets(self):
        model = build("mlp", 8, 3, self.y.numpy())
        model.eval()
        predicted = predict_thresholds(model, "mlp", self.x, [0.01, 0.99])
        self.assertEqual(predicted.shape, (2, 2, 3))
        self.assertTrue(torch.all(predicted.sum(2) >= 1))


if __name__ == "__main__":
    unittest.main()
