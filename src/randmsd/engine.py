"""Portable serial training and coverage-based evaluation for prepared clouds.

These entrypoints preserve the frozen backbone, weighted-loss convention and
explicit recipe controls. The clean NumPy/SciPy sampler is not a claim
of bitwise reproduction of the historical training executables.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import ModelConfig, RandMSD
from .data import PreparedCloudDataset
from .losses import LEGACY_CLASS_COUNTS, class_weights, weighted_cross_entropy
from .metrics import confusion_matrix, segmentation_metrics


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    source_root = Path(__file__).resolve().parents[1]
    return {path.relative_to(source_root).as_posix(): sha256(path)
            for folder in ("randmsd", "rsnative_corelite")
            for path in sorted((source_root / folder).rglob("*.py"))}


def write_json(path, value):
    """Replace one JSON artifact atomically, without truncating the old file."""
    target = Path(path)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                        allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_checkpoint(path, payload):
    target = Path(path)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    np_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    values = state["numpy"]
    np.random.set_state((values[0], np.asarray(values[1], np.uint32), *values[2:]))
    if torch.cuda.is_available() and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA device count differs from the checkpoint")
        torch.cuda.set_rng_state_all(state["cuda"])


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def load_config(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Configuration must be a JSON object")
    ModelConfig.for_dataset(result["dataset"])
    if "seed" not in result or "num_points" not in result:
        raise ValueError("Configuration must state seed and num_points explicitly")
    if int(result["seed"]) < 0 or int(result["num_points"]) < 1:
        raise ValueError("seed must be nonnegative; num_points must be positive")
    return result


def make_dataset(root, config, split, *, steps, seed=None):
    recipe = config.get("training", {})
    return PreparedCloudDataset(
        root, config["dataset"], split=split, num_points=int(config["num_points"]),
        seed=int(config["seed"] if seed is None else seed), steps=int(steps),
        cache_size=int(config.get("cache_size", 2)),
        noise_std=float(recipe.get("noise_std", .35)),
        color_drop_prob=float(recipe.get("color_drop_prob", 0)),
        auto_contrast_prob=float(recipe.get("auto_contrast_prob", 0)),
        augment_xyz=bool(recipe.get("augment_xyz", False)),
        exposure_prob=float(recipe.get("exposure_prob", 0)),
        exposure_quantile=float(recipe.get("exposure_quantile", .25)),
        model_config=ModelConfig.for_dataset(config["dataset"]))


def model_for_config(config):
    implementation = config.get("implementation", {})
    return RandMSD(ModelConfig.for_dataset(config["dataset"]),
                   eval_chunk_size=int(implementation.get("eval_chunk_size", 8192)),
                   activation_checkpoint_layers=int(implementation.get("activation_checkpoint_layers", 1)))


def learning_rate_at_epoch(recipe, epoch):
    """Historical epoch-indexed schedules; epoch 0 is the first epoch.

    Exponential decay defaults to offset 1: a base rate of .01 and decay .95
    start at .0095, as in the historical SemanticKITTI executable. Offset 0
    explicitly selects the TensorFlow-style first-epoch base rate. Linear
    warmup applies only to the cosine schedule.
    """
    base = float(recipe["learning_rate"])
    warmup = int(recipe.get("warmup_epochs", 0))
    schedule = recipe.get("schedule", "cosine")
    epochs = int(recipe["epochs"])
    if epochs < 1 or not 0 <= epoch < epochs:
        raise ValueError("epoch must be in [0, epochs)")
    if schedule == "exponential":
        offset = int(recipe.get("exponential_offset", 1))
        if offset not in (0, 1):
            raise ValueError("exponential_offset must be 0 or 1")
        return base * float(recipe.get("lr_decay", .95)) ** (epoch + offset)
    if schedule == "cosine":
        if warmup and epoch < warmup:
            return base * (epoch + 1) / warmup
        floor = float(recipe.get("min_learning_rate", 1e-5))
        if epochs - warmup <= 1:
            return floor
        denominator = epochs - warmup - 1
        progress = min(1.0, max(0.0, (epoch - warmup) / denominator))
        factor = .5 * (1 + math.cos(math.pi * progress))
        return floor + (base - floor) * factor
    raise ValueError("schedule must be cosine or exponential")


def epoch_pass(model, loader, weights, device, *, optimizer=None, label_smoothing=0.0):
    training = optimizer is not None
    model.train(training)
    matrix = np.zeros((len(weights), len(weights)), dtype=np.int64)
    total_loss, batches = 0.0, 0
    started = time.monotonic()
    for batch in loader:
        labels = batch["labels"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(move_to_device(batch, device))["logits"]
            loss = weighted_cross_entropy(logits, labels, weights, label_smoothing)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite segmentation loss")
            if training:
                loss.backward()
                optimizer.step()
        prediction = logits.detach().argmax(1).cpu().numpy()
        matrix += confusion_matrix(labels.detach().cpu().numpy(), prediction, len(weights))
        total_loss += float(loss.detach().cpu())
        batches += 1
        if batches % 50 == 0:
            print("{} step={:05d} loss={:.6f} mean_ms_per_batch={:.2f}".format(
                "TRAIN" if training else "VALIDATION", batches, total_loss / batches,
                1000 * (time.monotonic() - started) / batches), flush=True)
    if not batches or not matrix.sum():
        raise ValueError("Epoch has no batches or no labeled points")
    return {**segmentation_metrics(matrix), "loss": total_loss / batches,
            "seconds": time.monotonic() - started, "batches": batches}


def dataset_class_counts(dataset, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for index in range(len(dataset.records)):
        labels = dataset.load_cloud(index).labels
        counts += np.bincount(labels[labels >= 0], minlength=num_classes)
    return counts.tolist()


def _validate_training_checkpoint(payload, config, hashes, manifest_hash):
    if payload.get("format_version") != 1 or payload.get("config") != config:
        raise ValueError("Resume configuration differs from the checkpoint")
    if payload.get("source_hashes") != hashes or payload.get("manifest_sha256") != manifest_hash:
        raise ValueError("Resume source code or dataset manifest differs")
    epoch = int(payload["epoch"])
    history = payload["history"]
    if not 1 <= epoch <= int(config["training"]["epochs"]) or len(history) != epoch:
        raise ValueError("Checkpoint epoch/history is inconsistent")
    if [record["epoch"] for record in history] != list(range(1, epoch + 1)):
        raise ValueError("Checkpoint history is not a contiguous completed-epoch record")
    best_epoch, best_miou = int(payload["best_epoch"]), float(payload["best_miou"])
    if (not 1 <= best_epoch <= epoch or not math.isfinite(best_miou)
            or history[best_epoch - 1]["sampled_validation"]["miou"] != best_miou):
        raise ValueError("Checkpoint best epoch/metric is inconsistent with its history")
    for key in ("model_state_dict", "optimizer_state_dict", "train_sampler_state",
                "validation_sampler_state", "rng_state"):
        if key not in payload:
            raise ValueError("Training checkpoint is missing " + key)


def recover_training_checkpoint(output, config, hashes, manifest_hash):
    """Reconcile the best-first checkpoint transaction before resuming.

    ``best.pt`` and ``last.pt`` are each atomic, but cannot be replaced as one
    filesystem operation. A new best therefore carries a complete resumable
    epoch. If a crash interrupts the following ``last.pt`` replacement, its
    validated history prefix lets us promote that durable best to last. An
    absent/stale older best cannot be reconstructed from later non-best model
    weights and is refused rather than silently reporting completion.
    """
    output = Path(output)
    last_path, best_path = output / "last.pt", output / "best.pt"
    saved = torch.load(last_path, map_location="cpu", weights_only=True) if last_path.is_file() else None
    if saved is not None:
        _validate_training_checkpoint(saved, config, hashes, manifest_hash)
    best, best_error = None, None
    if best_path.is_file():
        try:
            best = torch.load(best_path, map_location="cpu", weights_only=True)
            _validate_training_checkpoint(best, config, hashes, manifest_hash)
            if int(best["best_epoch"]) != int(best["epoch"]):
                raise ValueError("best.pt does not hold its own best epoch")
        except Exception as error:
            best, best_error = None, error
    repaired = False
    if saved is None:
        if best is None:
            raise ValueError("No consistent resumable last.pt or best.pt") from best_error
        saved, repaired = dict(best), True
    elif best is not None and int(best["epoch"]) > int(saved["epoch"]):
        if best["history"][:int(saved["epoch"])] != saved["history"]:
            raise ValueError("Ahead best.pt history does not extend last.pt; refusing ambiguous recovery")
        saved, repaired = dict(best), True
    same_best = (best is not None and int(best["epoch"]) == int(saved["best_epoch"])
                 and best["best_miou"] == saved["best_miou"])
    actual_hash = sha256(best_path) if best is not None else None
    expected_hash = saved.get("best_checkpoint_sha256")
    if not same_best or (expected_hash is not None and expected_hash != actual_hash):
        if int(saved["best_epoch"]) != int(saved["epoch"]):
            raise ValueError("Older best.pt is missing, inconsistent, or modified; correct best weights are required") from best_error
        best = dict(saved)
        best.pop("best_checkpoint_sha256", None)
        atomic_checkpoint(best_path, best)
        actual_hash, repaired = sha256(best_path), True
    if repaired or saved.get("best_checkpoint_sha256") != actual_hash:
        saved = dict(saved)
        saved["best_checkpoint_sha256"] = actual_hash
        atomic_checkpoint(last_path, saved)
    write_json(output / "history.json", saved["history"])
    return saved


def run_training(config, data_root, output, *, device="cuda:0", resume=False,
                 model_factory=None):
    """Train with sampled validation; best selection never uses official test labels."""
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Nonempty output directory; use --resume explicitly")
    if resume and not any((output / name).is_file() for name in ("last.pt", "best.pt")):
        raise FileNotFoundError("--resume requires a resumable output/last.pt or output/best.pt")
    output.mkdir(parents=True, exist_ok=True)
    recipe = config["training"]
    epochs = int(recipe["epochs"])
    batch_size = int(recipe["batch_size"])
    val_batch_size = int(recipe.get("validation_batch_size", batch_size))
    train_steps, val_steps = int(recipe["steps_per_epoch"]), int(recipe["validation_steps"])
    if min(epochs, batch_size, val_batch_size, train_steps, val_steps) < 1:
        raise ValueError("Epoch, batch and step counts must be positive")
    seed_everything(int(config["seed"]))
    train_data = make_dataset(data_root, config, "training", steps=train_steps * batch_size)
    val_data = make_dataset(data_root, config, "validation", steps=val_steps * val_batch_size,
                            seed=int(config["seed"]) + 1)
    loaders = [DataLoader(data, batch_size=size, shuffle=False, num_workers=0,
                          collate_fn=data.collate_fn)
               for data, size in ((train_data, batch_size), (val_data, val_batch_size))]
    model = (model_factory or model_for_config)(config).to(device)
    classes = ModelConfig.for_dataset(config["dataset"]).num_classes
    counts = config.get("class_counts", LEGACY_CLASS_COUNTS.get(config["dataset"]))
    if counts is None:
        counts = dataset_class_counts(train_data, classes)
    mode = recipe.get("class_weight_mode", "inverse_sqrt" if config["dataset"] == "SensatUrban" else "legacy")
    weights = class_weights(counts, mode).to(device)
    if len(weights) != classes:
        raise ValueError("class_counts length does not match model classes")
    optimizer_name = recipe.get("optimizer", "adamw")
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(optimizer_name)
    if optimizer_class is None:
        raise ValueError("optimizer must be adam or adamw")
    optimizer = optimizer_class(model.parameters(), lr=float(recipe["learning_rate"]),
                                weight_decay=float(recipe.get("weight_decay", 0)))
    hashes = source_hashes()
    manifest_hash = sha256(Path(data_root) / "manifest.json")
    runtime = {"python": sys.version, "torch": str(torch.__version__),
               "cuda": torch.version.cuda, "device": str(device),
               "cpu_threads": torch.get_num_threads(),
               "seed_does_not_guarantee_cross_platform_bitwise_identity": True}
    start, best, best_epoch, history, best_hash = 0, -1.0, -1, [], None
    if resume:
        saved = recover_training_checkpoint(output, config, hashes, manifest_hash)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        train_data.load_state_dict(saved["train_sampler_state"])
        val_data.load_state_dict(saved["validation_sampler_state"])
        restore_rng(saved["rng_state"])
        start, best = int(saved["epoch"]), float(saved["best_miou"])
        best_epoch, history = int(saved["best_epoch"]), saved["history"]
        best_hash = saved["best_checkpoint_sha256"]
    write_json(output / "config.json", config)
    print("RandMSD parameters:", sum(parameter.numel() for parameter in model.parameters()), flush=True)
    for epoch in range(start, epochs):
        rate = learning_rate_at_epoch(recipe, epoch)
        for group in optimizer.param_groups:
            group["lr"] = rate
        train_result = epoch_pass(model, loaders[0], weights, device, optimizer=optimizer,
                                  label_smoothing=float(recipe.get("label_smoothing", 0)))
        validation = epoch_pass(model, loaders[1], weights, device,
                                label_smoothing=float(recipe.get("label_smoothing", 0)))
        record = {"epoch": epoch + 1, "learning_rate": rate,
                  "training": train_result, "sampled_validation": validation}
        history.append(record)
        improved = validation["miou"] > best
        if improved:
            best, best_epoch = validation["miou"], epoch + 1
        payload = {"format_version": 1, "config": config, "epoch": epoch + 1,
                   "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                   "best_miou": best, "best_epoch": best_epoch, "history": history,
                   "class_counts": counts, "class_weight_mode": mode,
                   "train_sampler_state": train_data.state_dict(),
                   "validation_sampler_state": val_data.state_dict(),
                   "rng_state": rng_state(), "source_hashes": hashes,
                   "manifest_sha256": manifest_hash,
                   "runtime": runtime,
                   "selection_protocol": "sampled_validation_not_full_cloud_or_official_test"}
        if improved:
            atomic_checkpoint(output / "best.pt", payload)
            best_hash = sha256(output / "best.pt")
        payload["best_checkpoint_sha256"] = best_hash
        atomic_checkpoint(output / "last.pt", payload)
        write_json(output / "history.json", history)
        print(json.dumps(record, allow_nan=False), flush=True)
    result = {"status": "COMPLETE", "epochs": epochs, "best_epoch": best_epoch,
              "best_sampled_validation_miou": best,
              "next_step": "Evaluate best.pt using coverage-based full-cloud voting."}
    write_json(output / "training_result.json", result)
    return result


def run_voting(model, dataset, output, *, device="cuda:0", num_classes,
               batch_size=1, votes=1, first_coverage=.5, coverage_step=1.0,
               smooth=.95, max_patches=1000000, save_predictions=False):
    """Probability EMA until the global possibility threshold is reached.

    A coverage vote is not one forward pass per point. Every duplicate index
    in a padded patch is fused sequentially, so repeated-index updates are
    well-defined instead of relying on NumPy advanced-assignment behavior.
    """
    if min(int(batch_size), int(votes), int(max_patches)) < 1:
        raise ValueError("Batch size, votes and max_patches must be positive")
    if first_coverage <= .001 or coverage_step <= 0 or not 0 <= smooth < 1:
        raise ValueError("Require first_coverage > .001, coverage_step > 0, smooth in [0,1)")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    probabilities = [np.zeros((int(row["num_points"]), num_classes), dtype=np.float32)
                     for row in dataset.records]
    visits = [np.zeros(len(values), dtype=np.uint32) for values in probabilities]
    target = float(first_coverage + (votes - 1) * coverage_step)
    patches, batches = 0, 0
    model.eval()
    started = time.monotonic()
    with torch.inference_mode():
        while min(dataset.min_possibility) < target or any(np.any(values == 0) for values in visits):
            count = min(int(batch_size), int(max_patches) - patches)
            if count < 1:
                raise RuntimeError("max_patches reached before complete possibility coverage; no final score accepted")
            samples = [dataset[patches + index] for index in range(count)]
            batch = dataset.collate_fn(samples)
            logits = model(move_to_device(batch, device))["logits"]
            predicted = torch.softmax(logits, 1).transpose(1, 2).cpu().numpy()
            if not np.isfinite(predicted).all():
                raise FloatingPointError("Non-finite predicted probabilities")
            for sample, values in zip(samples, predicted):
                cloud = int(sample["cloud_inds"])
                indices = np.asarray(sample["input_inds"], dtype=np.int64)
                unique, counts = np.unique(indices, return_counts=True)
                if counts.max() == 1:
                    probabilities[cloud][indices] = (smooth * probabilities[cloud][indices]
                                                     + (1-smooth) * values)
                else:
                    for index, value in zip(indices, values):
                        probabilities[cloud][index] = smooth * probabilities[cloud][index] + (1-smooth) * value
                visits[cloud][unique] += 1
            patches += count
            batches += 1
            if batches % 50 == 0:
                print("VOTING patches={} min_coverage={:.4f} target={:.4f}".format(
                    patches, min(dataset.min_possibility), target), flush=True)
    grid_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    raw_matrix = np.zeros_like(grid_matrix)
    raw_complete, all_grid_labeled, cloud_results = True, True, []
    for index, values in enumerate(probabilities):
        cloud = dataset.load_cloud(index)
        prediction = values.argmax(1).astype(np.uint8)
        grid_matrix += confusion_matrix(cloud.labels, prediction, num_classes)
        all_grid_labeled &= bool(np.any(cloud.labels >= 0))
        raw_complete &= cloud.projection is not None and cloud.raw_labels is not None
        if cloud.raw_labels is not None:
            for offset in range(0, len(cloud.projection), 1000000):
                selection = slice(offset, offset + 1000000)
                raw_matrix += confusion_matrix(cloud.raw_labels[selection],
                    prediction[cloud.projection[selection]], num_classes)
        if save_predictions:
            name = str(dataset.cloud_names[index])
            if Path(name).name != name or name in (".", "..") or "\\" in name:
                raise ValueError("Cloud name must be a safe file basename")
            arrays = {"grid_labels": prediction}
            if cloud.projection is not None:
                arrays["original_labels"] = prediction[cloud.projection]
            np.savez_compressed(output / (name + ".npz"), **arrays)
        cloud_results.append({"name": dataset.cloud_names[index], "grid_points": len(prediction),
                              "min_visits": int(visits[index].min()), "max_visits": int(visits[index].max()),
                              "min_possibility": float(dataset.min_possibility[index]),
                              "original_points": len(cloud.projection) if cloud.projection is not None else None})
    np.savez_compressed(output / "confusion.npz", grid_confusion=grid_matrix, original_confusion=raw_matrix)
    metric_space = "original_points" if raw_complete and raw_matrix.sum() else (
        "prepared_grid_points" if all_grid_labeled and grid_matrix.sum() else "unlabeled_no_score")
    headline = raw_matrix if metric_space == "original_points" else grid_matrix
    result = {"status": "COMPLETE", "protocol": "full_cloud_possibility_coverage_probability_ema",
              "dataset": dataset.dataset, "split": dataset.split, "metric_space": metric_space,
              "not_an_official_test_server_score": True, "patch_forwards": patches, "batches": batches,
              "coverage_votes": int(votes), "target_possibility": target, "smooth": float(smooth),
              "seconds_including_sampling_model_fusion_and_metrics": time.monotonic() - started,
              "clouds": cloud_results, "metrics": segmentation_metrics(headline) if metric_space != "unlabeled_no_score" else None,
              "grid_metrics": segmentation_metrics(grid_matrix) if grid_matrix.sum() else None}
    write_json(output / "metrics.json", result)
    return result
