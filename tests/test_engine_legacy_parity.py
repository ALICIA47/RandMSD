"""Optional numeric audit against an untouched, independently trusted checkout.

Set RANDMSD_LEGACY_ROOT to request the audit. Only the named pure functions are
compiled from that local source; legacy dataset/native-extension imports are
not needed. This test does not bundle or modify the historical checkout.
"""

import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from randmsd.engine import learning_rate_at_epoch
from randmsd.losses import class_weights, weighted_cross_entropy


def source_functions(path, names, namespace):
    parsed = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError("Expected historical functions are absent")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@unittest.skipUnless(os.environ.get("RANDMSD_LEGACY_ROOT"), "historical checkout not requested")
class HistoricalRecipeParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        root = Path(os.environ["RANDMSD_LEGACY_ROOT"])
        cls.schedule = staticmethod(source_functions(root / "training_modernization.py", ["epoch_learning_rate"],
                                                       {"math": math})["epoch_learning_rate"])
        cls.losses = source_functions(root / "RandLANet.py", ["get_loss", "compute_loss"],
                                      {"torch": torch, "np": np, "nn": torch.nn,
                                       "F": torch.nn.functional})

    def test_all_historical_learning_rates(self):
        for schedule, epochs, warmup, floor in (("exponential", 74, 0, 0), ("cosine", 100, 5, 1e-5),
                                                 ("cosine", 1, 0, 1e-5)):
            recipe = {"learning_rate": .01, "epochs": epochs, "warmup_epochs": warmup,
                      "min_learning_rate": floor, "schedule": schedule, "lr_decay": .95}
            for epoch in range(epochs):
                historical = self.schedule(.01, epoch, epochs, schedule=schedule,
                                            warmup_epochs=warmup, min_lr=floor, decay=.95)
                self.assertEqual(learning_rate_at_epoch(recipe, epoch), historical)

    def test_unsmoothed_and_smoothed_loss_values_and_gradients(self):
        for classes, ignored in ((13, False), (19, True)):
            for smoothing in (0.0, .2):
                with self.subTest(classes=classes, smoothing=smoothing):
                    generator = torch.Generator().manual_seed(20260620)
                    values = torch.randn(2, classes, 17, generator=generator)
                    clean_logits, historical_logits = values.clone().requires_grad_(), values.clone().requires_grad_()
                    labels = torch.randint(classes, (2, 17), generator=generator)
                    historical_labels = labels + 1 if ignored else labels.clone()
                    if ignored:
                        historical_labels[0, :3] = 0
                        labels[0, :3] = -1
                    weights = class_weights(np.arange(1, classes + 1))
                    cfg = SimpleNamespace(num_classes=classes, ignored_label_inds=[0] if ignored else [],
                                          class_weights=weights.numpy()[None], seg_loss_mode="legacy_wce",
                                          label_smoothing=smoothing)
                    legacy, _ = self.losses["compute_loss"](
                        {"logits": historical_logits, "labels": historical_labels}, cfg)
                    clean = weighted_cross_entropy(clean_logits, labels, weights, smoothing)
                    self.assertTrue(torch.equal(clean, legacy))
                    clean.backward()
                    legacy.backward()
                    self.assertTrue(torch.equal(clean_logits.grad, historical_logits.grad))


if __name__ == "__main__":
    unittest.main()
