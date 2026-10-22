"""Architecture and CPU equivalence tests; no training-data/GPU claims."""

import copy
import unittest

import torch

from randmsd import ModelConfig, RandMSD, ReferenceRandMSD
from rsnative_corelite.optimized import eval_mixer_forward
from rsnative_corelite.reference import NormalizedLocalMixer
from _synthetic import build_hierarchy


EXPECTED_PARAMETERS = {"S3DIS": 1_976_381, "SemanticKITTI": 754_307,
                       "SensatUrban": 1_976_381}


class ModelContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_same_state(self, expected, actual):
        left, right = expected.state_dict(), actual.state_dict()
        self.assertEqual(left.keys(), right.keys())
        for key, value in left.items():
            self.assertTrue(torch.equal(value, right[key]), key)

    def test_dataset_contracts_and_chunked_inference(self):
        for dataset, count in EXPECTED_PARAMETERS.items():
            with self.subTest(dataset=dataset):
                config = ModelConfig.for_dataset(dataset)
                torch.manual_seed(20260620)
                reference = ReferenceRandMSD(config).eval()
                torch.manual_seed(20260620)
                optimized = RandMSD(config, eval_chunk_size=37).eval()
                self.assertEqual(sum(p.numel() for p in reference.parameters()), count)
                self.assert_same_state(reference, optimized)
                batch = build_hierarchy(config)
                with torch.no_grad():
                    left = reference(copy.deepcopy(batch))
                    right = optimized(copy.deepcopy(batch))
                self.assertEqual(tuple(right["logits"].shape), (1, config.num_classes, 256))
                self.assertTrue(torch.equal(left["logits"], right["logits"]))
                self.assertTrue(torch.equal(left["point_features"], right["point_features"]))

    def test_training_gradients_and_batch_norm_buffers(self):
        for dataset in EXPECTED_PARAMETERS:
            with self.subTest(dataset=dataset):
                config = ModelConfig.for_dataset(dataset)
                torch.manual_seed(20260620)
                reference = ReferenceRandMSD(config).train()
                torch.manual_seed(20260620)
                optimized = RandMSD(config, activation_checkpoint_layers=1).train()
                batch = build_hierarchy(config, batch_size=2)
                torch.manual_seed(41000)
                left = reference(copy.deepcopy(batch))["logits"]
                left.square().mean().backward()
                torch.manual_seed(41000)
                right = optimized(copy.deepcopy(batch))["logits"]
                right.square().mean().backward()
                self.assertTrue(torch.equal(left, right))
                for (name_l, par_l), (name_r, par_r) in zip(
                        reference.named_parameters(), optimized.named_parameters()):
                    self.assertEqual(name_l, name_r)
                    self.assertIsNotNone(par_l.grad)
                    self.assertTrue(torch.equal(par_l.grad, par_r.grad), name_l)
                self.assert_same_state(reference, optimized)
                # An actual update also preserves parameter equality.
                torch.optim.SGD(reference.parameters(), lr=0.01).step()
                torch.optim.SGD(optimized.parameters(), lr=0.01).step()
                self.assert_same_state(reference, optimized)

    def test_default_8192_chunk_boundary(self):
        torch.manual_seed(20260620)
        mixer = NormalizedLocalMixer(8, 8).eval()
        feature = torch.randn(1, 8, 8201)
        xyz = torch.randn(1, 8201, 3)
        neighbors = torch.randint(8201, (1, 8201, 16))
        with torch.no_grad():
            expected = mixer(feature, xyz, neighbors)
            actual = eval_mixer_forward(mixer, feature, xyz, neighbors)
        self.assertTrue(torch.equal(expected, actual))

    def test_rejects_unknown_config_and_invalid_optimization(self):
        with self.assertRaises(ValueError):
            ModelConfig.for_dataset("Toronto3D")
        config = ModelConfig.for_dataset("S3DIS")
        for settings in ({"eval_chunk_size": -1}, {"activation_checkpoint_layers": 6}):
            with self.assertRaises(ValueError):
                RandMSD(config, **settings)


if __name__ == "__main__":
    unittest.main()
