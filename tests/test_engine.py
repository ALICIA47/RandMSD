"""CPU-only end-to-end workflow tests; synthetic clouds are not accuracy tests."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from randmsd import ModelConfig
from randmsd.data import PreparedCloudDataset
from randmsd.engine import (epoch_pass, learning_rate_at_epoch, run_training,
                           run_voting, seed_everything, make_dataset, model_for_config,
                           atomic_checkpoint, sha256)
from randmsd import load_checkpoint
from randmsd.losses import class_weights, weighted_cross_entropy
from randmsd.metrics import confusion_matrix, segmentation_metrics


class TinyModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = torch.nn.Linear(6, 13)
        self.dropout = torch.nn.Dropout(.2)

    def forward(self, endpoints):
        return {"logits": self.linear(self.dropout(endpoints["features"])).transpose(1, 2)}


def prepare_tiny(root):
    rng = np.random.RandomState(42)
    records = []
    for split in ("training", "validation", "test"):
        name = split + "_cloud"
        xyz = rng.normal(size=(96, 3)).astype(np.float32)
        labels = (np.arange(96) % 13).astype(np.int64)
        features = rng.uniform(size=(96, 3)).astype(np.float32)
        arrays = {"xyz": xyz, "features": features, "projection": np.repeat(np.arange(96), 2)}
        if split != "test":
            arrays.update(labels=labels, raw_labels=np.repeat(labels, 2))
        np.savez(root / (name + ".npz"), **arrays)
        records.append({"name": name, "path": name + ".npz", "split": split, "num_points": 96})
    (root / "manifest.json").write_text(json.dumps({"dataset": "S3DIS", "format_version": 1,
                                                   "clouds": records}), encoding="utf-8")


class EngineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        prepare_tiny(self.root)
        self.config = {"dataset": "S3DIS", "seed": 42, "num_points": 32,
                       "class_counts": [10] * 13,
                       "training": {"epochs": 2, "batch_size": 2, "validation_batch_size": 2,
                                    "steps_per_epoch": 1, "validation_steps": 1,
                                    "optimizer": "adamw", "learning_rate": .001,
                                    "weight_decay": .0001, "schedule": "cosine", "noise_std": 0}}

    def tearDown(self):
        self.directory.cleanup()

    def test_losses_and_metrics(self):
        logits = torch.tensor([[[1., 2., 3.], [0., 1., 4.]]], requires_grad=True)
        labels = torch.tensor([[0, 1, -1]])
        weights = torch.tensor([2., 4.])
        expected = torch.nn.functional.cross_entropy(logits[:, :, :2], labels[:, :2],
                                                      weight=weights, reduction="none").mean()
        self.assertTrue(torch.equal(weighted_cross_entropy(logits, labels, weights), expected))
        weighted_cross_entropy(logits, torch.full_like(labels, -1), weights).backward()
        self.assertEqual(float(logits.grad.abs().sum()), 0)
        np.testing.assert_array_equal(confusion_matrix(np.array([0, 1, -1]), np.array([0, 0, 0]), 2),
                                      np.array([[1, 0], [1, 0]]))
        self.assertEqual(segmentation_metrics(np.eye(2, dtype=np.int64))["miou"], 100)
        with self.assertRaises(ValueError):
            class_weights([1, 0])

    def test_schedules(self):
        recipe = {"learning_rate": .01, "warmup_epochs": 5, "epochs": 100,
                  "schedule": "cosine", "min_learning_rate": 1e-5}
        self.assertAlmostEqual(learning_rate_at_epoch(recipe, 0), .002)
        self.assertAlmostEqual(learning_rate_at_epoch(recipe, 99), 1e-5)
        recipe.update(schedule="exponential", warmup_epochs=0, lr_decay=.95)
        self.assertAlmostEqual(learning_rate_at_epoch(recipe, 0), .0095)
        self.assertAlmostEqual(learning_rate_at_epoch(recipe, 2), .01 * .95 ** 3)
        recipe["exponential_offset"] = 0
        self.assertAlmostEqual(learning_rate_at_epoch(recipe, 0), .01)
        self.assertAlmostEqual(learning_rate_at_epoch(recipe, 2), .01 * .95 ** 2)
        recipe.update(schedule="cosine", epochs=1)
        self.assertAlmostEqual(learning_rate_at_epoch(recipe, 0), 1e-5)

    def test_smoothed_loss_weights_every_class_in_uniform_term(self):
        logits = torch.tensor([[[1., -.5, 2.], [.1, 1.5, -.8], [-.9, .7, .4]]],
                              requires_grad=True)
        labels = torch.tensor([[0, 2, -1]])
        weights = torch.tensor([1.2, 3.5, 6.1])
        epsilon = .2
        flat = logits.transpose(1, 2).reshape(-1, 3)[:2]
        log_probability = torch.log_softmax(flat, dim=1)
        expected = (-(1-epsilon) * weights[labels[0, :2]] * log_probability[
                    torch.arange(2), labels[0, :2]]
                    - epsilon / 3 * (weights * log_probability).sum(1)).mean()
        actual = weighted_cross_entropy(logits, labels, weights, epsilon)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        expected_gradient = torch.autograd.grad(expected, logits, retain_graph=True)[0]
        actual_gradient = torch.autograd.grad(actual, logits)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-6)
        incorrect = (torch.nn.functional.cross_entropy(flat, labels[0, :2],
                     reduction="none", label_smoothing=epsilon) * weights[labels[0, :2]]).mean()
        self.assertGreater(float((actual-incorrect).abs().detach()), .01)

    def test_training_resume_restores_sampler_optimizer_and_rng(self):
        full = self.root / "full"
        interrupted = self.root / "interrupted"
        run_training(self.config, self.root, full, device="cpu", model_factory=TinyModel)
        calls = []

        def interrupt_after_first_epoch(*args, **kwargs):
            if len(calls) == 2:
                raise RuntimeError("simulated interruption after checkpoint")
            calls.append(1)
            return epoch_pass(*args, **kwargs)

        with patch("randmsd.engine.epoch_pass", interrupt_after_first_epoch):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                run_training(self.config, self.root, interrupted, device="cpu", model_factory=TinyModel)
        run_training(self.config, self.root, interrupted, device="cpu", resume=True, model_factory=TinyModel)
        complete = torch.load(full / "last.pt", weights_only=True)
        resumed = torch.load(interrupted / "last.pt", weights_only=True)
        for key, value in complete["model_state_dict"].items():
            self.assertTrue(torch.equal(value, resumed["model_state_dict"][key]), key)
        self.assertEqual(complete["epoch"], 2)
        self.assertEqual(complete["best_epoch"], resumed["best_epoch"])
        with self.assertRaises(FileExistsError):
            run_training(self.config, self.root, full, device="cpu", model_factory=TinyModel)
        changed = copy.deepcopy(self.config)
        changed["seed"] = 43
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            run_training(changed, self.root, full, device="cpu", resume=True, model_factory=TinyModel)

    def test_best_first_transaction_recovers_after_each_interrupted_last_write(self):
        for failed_epoch in (1, 2):
            output = self.root / ("transaction_" + str(failed_epoch))
            validation_calls = []

            def always_improve(*args, **kwargs):
                result = epoch_pass(*args, **kwargs)
                if kwargs.get("optimizer") is None:
                    validation_calls.append(1)
                    result["miou"] = float(10 * len(validation_calls))
                return result

            def interrupt_last(path, payload):
                if Path(path).name == "last.pt" and payload["epoch"] == failed_epoch:
                    raise OSError("simulated power loss between best and last")
                return atomic_checkpoint(path, payload)

            with patch("randmsd.engine.atomic_checkpoint", interrupt_last), patch(
                    "randmsd.engine.epoch_pass", always_improve):
                with self.assertRaisesRegex(OSError, "simulated power loss"):
                    run_training(self.config, self.root, output, device="cpu", model_factory=TinyModel)
            durable_best = torch.load(output / "best.pt", weights_only=True)
            self.assertEqual(durable_best["epoch"], failed_epoch)
            result = run_training(self.config, self.root, output, device="cpu", resume=True, model_factory=TinyModel)
            self.assertEqual(result["status"], "COMPLETE")
            last = torch.load(output / "last.pt", weights_only=True)
            best = torch.load(output / "best.pt", weights_only=True)
            self.assertEqual(last["best_epoch"], best["epoch"])
            self.assertEqual(last["best_checkpoint_sha256"], sha256(output / "best.pt"))
            self.assertEqual(len(json.loads((output / "history.json").read_text())), 2)

    def test_completed_resume_rebuilds_current_best_and_history(self):
        config = copy.deepcopy(self.config)
        config["training"]["epochs"] = 1
        output = self.root / "repair_current"
        run_training(config, self.root, output, device="cpu", model_factory=TinyModel)
        saved = torch.load(output / "last.pt", weights_only=True)
        (output / "best.pt").unlink()
        (output / "history.json").write_text("[]", encoding="utf-8")
        result = run_training(config, self.root, output, device="cpu", resume=True, model_factory=TinyModel)
        self.assertEqual(result["status"], "COMPLETE")
        restored = torch.load(output / "best.pt", weights_only=True)
        for key, value in saved["model_state_dict"].items():
            self.assertTrue(torch.equal(value, restored["model_state_dict"][key]))
        self.assertEqual(json.loads((output / "history.json").read_text()), saved["history"])

    def test_missing_or_modified_older_best_is_refused(self):
        for modification in ("missing", "wrong_metadata", "wrong_weights"):
            output = self.root / ("old_best_" + modification)
            with patch("randmsd.engine.epoch_pass", return_value={"miou": 50.0}):
                run_training(self.config, self.root, output, device="cpu", model_factory=TinyModel)
            saved = torch.load(output / "last.pt", weights_only=True)
            self.assertEqual(saved["best_epoch"], 1)
            self.assertEqual(saved["epoch"], 2)
            if modification == "missing":
                (output / "best.pt").unlink()
            else:
                best = torch.load(output / "best.pt", weights_only=True)
                if modification == "wrong_metadata":
                    best["best_miou"] = 49.0
                    best["history"][0]["sampled_validation"]["miou"] = 49.0
                else:
                    best["model_state_dict"]["linear.bias"][0] += 1
                atomic_checkpoint(output / "best.pt", best)
            with self.assertRaisesRegex(ValueError, "Older best.pt"):
                run_training(self.config, self.root, output, device="cpu", resume=True, model_factory=TinyModel)

    def test_full_cloud_voting_and_projection(self):
        seed_everything(42)
        dataset = PreparedCloudDataset(self.root, "S3DIS", "validation", num_points=32,
                                       seed=42, noise_std=0)
        result = run_voting(TinyModel(self.config), dataset, self.root / "vote", device="cpu",
                            num_classes=13, batch_size=2, first_coverage=.05,
                            max_patches=200, save_predictions=True)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["metric_space"], "original_points")
        self.assertEqual(result["metrics"]["point_count"], 192)
        self.assertGreaterEqual(result["clouds"][0]["min_visits"], 1)
        with np.load(self.root / "vote" / "confusion.npz") as archive:
            self.assertEqual(int(archive["original_confusion"].sum()), 192)
        with np.load(self.root / "vote" / "validation_cloud.npz") as archive:
            self.assertEqual(archive["original_labels"].shape, (192,))

    def test_frozen_backbone_training_checkpoint_and_voting(self):
        config = copy.deepcopy(self.config)
        config["training"]["epochs"] = 1
        result = run_training(config, self.root, self.root / "real_model", device="cpu")
        self.assertEqual(result["status"], "COMPLETE")
        model = model_for_config(config)
        self.assertEqual(sum(value.numel() for value in model.parameters()), 1976381)
        load_checkpoint(model, self.root / "real_model" / "best.pt")
        data = make_dataset(self.root, config, "validation", steps=1)
        voted = run_voting(model, data, self.root / "real_vote", device="cpu",
                           num_classes=13, batch_size=2, first_coverage=.05, max_patches=200)
        self.assertEqual(voted["metrics"]["point_count"], 192)

    def test_incomplete_coverage_has_no_accepted_metrics(self):
        dataset = PreparedCloudDataset(self.root, "S3DIS", "validation", num_points=4,
                                       seed=42, noise_std=0)
        with self.assertRaisesRegex(RuntimeError, "max_patches"):
            run_voting(TinyModel(self.config), dataset, self.root / "incomplete", device="cpu",
                       num_classes=13, first_coverage=.5, max_patches=1)
        self.assertFalse((self.root / "incomplete" / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
