import unittest

import torch

from gru_sequence import SymptomGRU
from models import ClassifierChain, SigmoidModel


class MatchedEncoderTests(unittest.TestCase):
    def test_seed_gives_identical_encoder_initialization(self):
        torch.manual_seed(17)
        direct = SigmoidModel(42, 10, matched=True)
        torch.manual_seed(17)
        chain = ClassifierChain(42, 10)
        torch.manual_seed(17)
        gru = SymptomGRU(42, 10)
        for candidate in (chain, gru):
            for name in ("symptoms", "context"):
                left = dict(getattr(direct, name).named_parameters())
                right = dict(getattr(candidate, name).named_parameters())
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    self.assertTrue(torch.equal(left[key], right[key]))

    def test_chain_threshold_changes_later_conditionals(self):
        model = ClassifierChain(4, 2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.head.bias[1] = -1.3862944
            model.head.weight[1, 256] = 2.7725888
        model.eval()
        predicted = model.predict_thresholds(torch.tensor([[2]]), [0.4, 0.6])
        self.assertEqual(predicted.tolist(), [[[1, 1]], [[1, 0]]])


if __name__ == "__main__":
    unittest.main()
