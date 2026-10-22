"""Strict, restricted checkpoint loading contracts."""

import copy
import hashlib
import pickle
import tempfile
import unittest
from pathlib import Path

import torch

from randmsd import ModelConfig, ReferenceRandMSD, load_checkpoint


class CheckpointContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_raw_and_historical_archive(self):
        config = ModelConfig.for_dataset("SemanticKITTI")
        source, target = ReferenceRandMSD(config), ReferenceRandMSD(config)
        for archive in (source.state_dict(), {"epoch": 12, "model_state_dict": source.state_dict()}):
            with self.subTest(archive="model_state_dict" in archive), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "model.pt"
                torch.save(archive, path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                info = load_checkpoint(target, path, expected_sha256=digest)
                self.assertEqual(info["sha256"], digest)
                for key, tensor in source.state_dict().items():
                    self.assertTrue(torch.equal(tensor, target.state_dict()[key]))

    def test_bad_keys_and_shapes_leave_model_unchanged(self):
        model = ReferenceRandMSD(ModelConfig.for_dataset("SemanticKITTI"))
        before = copy.deepcopy(model.state_dict())
        wrong_keys = {"module." + key: value for key, value in before.items()}
        wrong_shapes = copy.deepcopy(before)
        wrong_shapes["classifier.weight"] = torch.zeros(1)
        for state in (wrong_keys, wrong_shapes):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "bad.pt"
                torch.save(state, path)
                with self.assertRaises(RuntimeError):
                    load_checkpoint(model, path)
                for key, tensor in before.items():
                    self.assertTrue(torch.equal(tensor, model.state_dict()[key]), key)

    def test_bad_hash_and_unsafe_payload_rejected(self):
        model = ReferenceRandMSD(ModelConfig.for_dataset("SemanticKITTI"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "unsafe.pt"
            torch.save({"model_state_dict": model.state_dict(), "config": Path("example")}, path)
            with self.assertRaises(ValueError):
                load_checkpoint(model, path, expected_sha256="0" * 64)
            # Path objects need arbitrary-pickle permission on modern PyTorch;
            # the loader must not retry unrestricted deserialization silently.
            with self.assertRaises(pickle.UnpicklingError):
                load_checkpoint(model, path)


if __name__ == "__main__":
    unittest.main()
