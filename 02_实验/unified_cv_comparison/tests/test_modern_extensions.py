import unittest

import torch

from modern_extensions import build
from run_modern_encoders import PatchTSTStyleEncoder, SequenceSetModel, TCNEncoder


class ModernExtensionTests(unittest.TestCase):
    def test_uses_original_models_and_parameter_counts(self):
        expected = {
            "tcn_sequence": (TCNEncoder, 1274699),
            "patchtst_style_sequence": (PatchTSTStyleEncoder, 1695563),
        }
        for method, (encoder_class, parameter_count) in expected.items():
            with self.subTest(method=method):
                model = build(method, 42, 10)
                self.assertIsInstance(model, SequenceSetModel)
                self.assertIsInstance(model.encoder, encoder_class)
                self.assertEqual(sum(p.numel() for p in model.parameters()),
                                 parameter_count)
        with self.assertRaises(ValueError):
            build("unknown", 42, 10)

    def test_generation_uses_unique_labels_and_stops(self):
        inputs = torch.tensor([[2, 3, 4, 5, 0], [6, 7, 8, 9, 10]])
        for method in ("tcn_sequence", "patchtst_style_sequence"):
            with self.subTest(method=method):
                torch.manual_seed(9)
                model = build(method, 42, 10).eval()
                with torch.no_grad():
                    predicted, traces = model.generate(inputs)
                self.assertEqual(tuple(predicted.shape), (2, 10))
                self.assertEqual(len(traces), 2)
                for row, trace in zip(predicted.tolist(), traces):
                    self.assertGreaterEqual(sum(row), 1)
                    self.assertLessEqual(sum(row), 10)
                    self.assertEqual(trace[-1], 10)
                    self.assertEqual(len(trace), len(set(trace)))
                    self.assertEqual({i for i, selected in enumerate(row) if selected},
                                     set(trace[:-1]))

    def test_encoders_depend_on_symptom_input_order(self):
        inputs = torch.tensor([
            [2, 3, 4, 5, 6, 7, 8, 9],
            [9, 8, 7, 6, 5, 4, 3, 2],
        ])
        for method in ("tcn_sequence", "patchtst_style_sequence"):
            with self.subTest(method=method):
                torch.manual_seed(9)
                encoder = build(method, 42, 10).encoder.eval()
                with torch.no_grad():
                    context = encoder(inputs)
                self.assertFalse(torch.allclose(context[0], context[1], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
