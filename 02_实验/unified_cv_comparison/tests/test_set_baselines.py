import unittest

import numpy as np
import torch

from set_baselines import build, decode, train_step


def batch_tokens(rows, indices):
    selected = [rows[int(i)] for i in indices]
    batch = torch.zeros((len(selected), max(map(len, selected))), dtype=torch.long)
    for row, values in enumerate(selected):
        batch[row, :len(values)] = torch.tensor(values)
    return batch


class SetBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.model = build(vocab=8, k=3)
        self.rows = [[2, 3, 5], [3, 4], [2, 5, 6]]
        self.x = batch_tokens(self.rows, range(len(self.rows)))
        self.y = torch.tensor([[1., 0., 1.], [0., 1., 0.], [1., 1., 0.]])

    def test_exact_nll_has_finite_gradients(self):
        self.model.zero_grad(set_to_none=True)
        value = train_step(self.model, self.x, self.y, "nll")
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(np.isfinite(value))
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
        self.assertTrue(any(torch.count_nonzero(grad) for grad in gradients))

    def test_rl_stage_updates_supervised_initialization(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        train_step(self.model, self.x, self.y, "nll")
        optimizer.step()
        supervised = {name: value.detach().clone()
                      for name, value in self.model.state_dict().items()}
        optimizer.zero_grad(set_to_none=True)
        value = train_step(self.model, self.x, self.y, "rl",
                           torch.Generator().manual_seed(17), samples=8)
        self.assertTrue(np.isfinite(value))
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad)
                            for p in self.model.parameters()))
        optimizer.step()
        self.assertTrue(any(not torch.equal(value, supervised[name])
                            for name, value in self.model.state_dict().items()))

    def test_all_decoders_return_legal_nonempty_sets(self):
        for mode in ("greedy", "beam5", "map"):
            with self.subTest(mode=mode):
                predicted, traces = decode(self.model, self.rows, mode,
                                           batch_tokens, batch_size=2)
                self.assertEqual(predicted.shape, (len(self.rows), 3))
                self.assertTrue(np.isin(predicted, [0, 1]).all())
                self.assertTrue((predicted.sum(axis=1) >= 1).all())
                if mode == "greedy":
                    self.assertEqual(len(traces), len(self.rows))
                else:
                    self.assertIsNone(traces)

    def test_exact_map_covers_ten_label_vocabulary(self):
        model = build(vocab=42, k=10)
        predicted, traces = decode(model, [[2, 4, 7], [3, 8]], "map",
                                   batch_tokens, batch_size=8)
        self.assertEqual(predicted.shape, (2, 10))
        self.assertTrue((predicted.sum(axis=1) >= 1).all())
        self.assertIsNone(traces)


if __name__ == "__main__":
    unittest.main()
