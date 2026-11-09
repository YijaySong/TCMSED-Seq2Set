import unittest

import torch

from gru_sequence import SymptomGRU, random_targets
from models import GRUNoLabelHistory, NoStopCardinalityGRU, fixed_targets
from run_all import METHODS, model_for


class MechanismTests(unittest.TestCase):
    def test_fixed_targets_use_vocabulary_order_and_stop(self):
        y = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.float32)
        self.assertEqual(fixed_targets(y).tolist(), [[0, 2, 4], [1, 4, -100]])

    def test_all_variants_preserve_the_mean_symptom_encoder(self):
        parameters = []
        for method in METHODS:
            torch.manual_seed(17)
            model = model_for(method, 42, 10)
            parameters.append({key: value.detach().clone() for key, value in
                               model.named_parameters() if key.startswith(("symptoms.", "context."))})
        for candidate in parameters[1:]:
            for key in parameters[0]:
                self.assertTrue(torch.equal(parameters[0][key], candidate[key]))

    def test_no_history_hides_previous_identity_but_keeps_mask(self):
        model = GRUNoLabelHistory(7, 3, embedding=8, hidden=12)
        model.eval()
        hidden = torch.zeros(1, 12)
        context = torch.zeros(1, 12)
        chosen = torch.tensor([[True, False, False]])
        left, left_hidden = model.step(torch.tensor([0]), hidden, context, chosen)
        right, right_hidden = model.step(torch.tensor([1]), hidden, context, chosen)
        self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.equal(left_hidden, right_hidden))
        self.assertTrue(torch.isneginf(left[0, 0]))
        self.assertTrue(torch.isfinite(left[0, 3]))

    def test_no_stop_loss_and_cardinality_decoding(self):
        model = NoStopCardinalityGRU(7, 3, embedding=8, hidden=12)
        x = torch.tensor([[2, 3], [3, 0]])
        y = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)
        targets = random_targets(y, torch.Generator().manual_seed(9))
        self.assertTrue(torch.isfinite(model.loss(x, targets)))
        with torch.no_grad():
            model.cardinality_head.weight.zero_()
            model.cardinality_head.bias.copy_(torch.tensor([0., 0., 5.]))
        model.eval()
        predicted, traces = model.generate(x)
        self.assertEqual(predicted.sum(1).tolist(), [3, 3])
        self.assertTrue(all(len(set(trace)) == 3 for trace in traces))
        self.assertTrue(all(max(trace) < 3 for trace in traces))

    def test_full_and_fixed_order_use_exact_main_gru_class(self):
        for method in ("gru_stop", "gru_fixed_order"):
            self.assertIs(type(model_for(method, 42, 10)), SymptomGRU)


if __name__ == "__main__":
    unittest.main()
