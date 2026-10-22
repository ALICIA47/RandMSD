"""Optional release audit against an untouched historical checkout.

Set RANDMSD_LEGACY_ROOT to the checkout containing RSNativeNet.py. This test is
skipped in a normal standalone installation; no research checkout is bundled.
"""

import copy
import importlib.util
import os
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from randmsd import ModelConfig, RandMSD, ReferenceRandMSD
from _synthetic import build_hierarchy


@unittest.skipUnless(os.environ.get("RANDMSD_LEGACY_ROOT"), "historical checkout not requested")
class LegacyParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        root = Path(os.environ["RANDMSD_LEGACY_ROOT"]).resolve()
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("randmsd_historical_model", root / "RSNativeNet.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.legacy_class = module.RSNativeNetwork

    def assert_same_state(self, left, right):
        expected, actual = left.state_dict(), right.state_dict()
        self.assertEqual(expected.keys(), actual.keys())
        for key, value in expected.items():
            self.assertTrue(torch.equal(value, actual[key]), key)

    def test_seeded_parameters_eval_and_training(self):
        for dataset in ("S3DIS", "SemanticKITTI", "SensatUrban"):
            config = ModelConfig.for_dataset(dataset)
            values = asdict(config)
            # Historical SensatUrban invokes the shared S3DIS training engine
            # with its 13-class XYZ+RGB configuration (same full backbone).
            if dataset == "SensatUrban":
                values["name"] = "S3DIS"
            values.update(rs_native_core_lite=True, rs_native_analysis_enabled=False,
                          rs_native_synthesis_enabled=False, rs_native_disable_global_context=True)
            for optimized in (False, True):
                with self.subTest(dataset=dataset, optimized=optimized):
                    options = dict(values)
                    if optimized:
                        options.update(rs_native_eval_chunk_size=37,
                                       rs_native_activation_checkpointing=True,
                                       rs_native_activation_checkpoint_layers=1)
                    torch.manual_seed(20260620)
                    historical = self.legacy_class(SimpleNamespace(**options))
                    torch.manual_seed(20260620)
                    clean = (RandMSD(config, eval_chunk_size=37) if optimized
                             else ReferenceRandMSD(config))
                    self.assert_same_state(historical, clean)
                    batch = build_hierarchy(config, batch_size=2)
                    historical.eval(), clean.eval()
                    with torch.no_grad():
                        left = historical(copy.deepcopy(batch))["logits"]
                        right = clean(copy.deepcopy(batch))["logits"]
                    self.assertTrue(torch.equal(left, right))
                    historical.train(), clean.train()
                    torch.manual_seed(41000)
                    left = historical(copy.deepcopy(batch))["logits"]
                    left.square().mean().backward()
                    torch.manual_seed(41000)
                    right = clean(copy.deepcopy(batch))["logits"]
                    right.square().mean().backward()
                    self.assertTrue(torch.equal(left, right))
                    for (name_l, par_l), (name_r, par_r) in zip(
                            historical.named_parameters(), clean.named_parameters()):
                        self.assertEqual(name_l, name_r)
                        self.assertTrue(torch.equal(par_l.grad, par_r.grad), name_l)
                    self.assert_same_state(historical, clean)


if __name__ == "__main__":
    unittest.main()
