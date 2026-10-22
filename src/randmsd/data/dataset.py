"""Stateful spatial-coverage patches and ordinary random-prefix hierarchies.

The reference SciPy KNN backend avoids custom compilation. It is not the
compiled benchmark backend; exact distance ties may select different neighbors.
"""
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset, get_worker_info

from rsnative_corelite.config import CoreLiteConfig
from .io import DEFAULT_POINTS, canonical_dataset, file_sha256, load_cloud


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _knn(support, query, k):
    count = min(k, len(support))
    indices = cKDTree(support, leafsize=40).query(query, k=count, workers=1)[1]
    if count == 1:
        indices = indices[:, None]
    if count < k:
        indices = np.concatenate([indices, np.repeat(indices[:, -1:], k-count, axis=1)], axis=1)
    return indices.astype(np.int64)


def augment_rgb(colors, rng, color_drop_prob=0.0, auto_contrast_prob=0.0):
    """Auto-contrast then whole-patch RGB dropout, using a caller-owned RNG."""
    result = np.asarray(colors).copy()
    if auto_contrast_prob > 0 and rng.rand() < auto_contrast_prob:
        low, high = result.min(0, keepdims=True), result.max(0, keepdims=True)
        contrast = (result-low) / np.maximum(high-low, 1e-6)
        blend = float(rng.rand())
        result[:] = (1-blend)*result + blend*contrast
    if color_drop_prob > 0 and rng.rand() < color_drop_prob:
        result[:] = 0
    np.clip(result, 0, 1, out=result)
    return result


def build_hierarchy(xyz, features, config, seed=None):
    """Build model inputs from [N,C] or [B,N,C] arrays.

    Pass a seed to shuffle points/features together once before prefix sampling.
    Already shuffled dataset patches use seed=None; each level retains the
    first floor(N/ratio) centers, KNN supports, and nearest-center interpolation.
    """
    xyz, features = _array(xyz).astype(np.float32), _array(features).astype(np.float32)
    if xyz.ndim == 2:
        xyz, features = xyz[None], features[None]
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or features.shape[:2] != xyz.shape[:2]:
        raise ValueError("Expected xyz [B,N,3] and aligned features [B,N,C]")
    if features.shape[-1] != config.input_channels or xyz.shape[1] == 0:
        raise ValueError("Incorrect model feature width or empty patch")
    if seed is not None:
        rng = np.random.RandomState(seed)
        permutations = np.stack([rng.permutation(xyz.shape[1]) for _ in xyz])
        batch = np.arange(len(xyz))[:, None]
        xyz, features = xyz[batch, permutations], features[batch, permutations]
    result = {key: [] for key in ("xyz", "neigh_idx", "sub_idx", "interp_idx")}
    current = xyz
    for ratio in config.sub_sampling_ratio:
        count = max(1, current.shape[1] // int(ratio))
        neighbors = np.stack([_knn(points, points, config.k_n) for points in current])
        sub = current[:, :count]
        interpolation = np.stack([_knn(support, query, 1) for support, query in zip(sub, current)])
        result["xyz"].append(torch.from_numpy(np.ascontiguousarray(current)))
        result["neigh_idx"].append(torch.from_numpy(neighbors))
        result["sub_idx"].append(torch.from_numpy(np.ascontiguousarray(neighbors[:, :count])))
        result["interp_idx"].append(torch.from_numpy(interpolation))
        current = sub
    result["features"] = torch.from_numpy(np.ascontiguousarray(features))
    return result


class PreparedCloudDataset(Dataset):
    """Safe-NPZ data adapter with bounded cloud cache and serial sampler state.

    Dataset labels are contiguous 0..K-1, with -1 ignored. Evaluation uses no
    labels for selecting patches. Use num_workers=0 to maintain global coverage.
    This clean workflow does not claim bitwise historical training reproduction.
    """
    def __init__(self, root, dataset, split="training", num_points=None, seed=20260620,
                 steps=None, cache_size=2, noise_std=0.35, model_config=None,
                 color_drop_prob=0.0, auto_contrast_prob=0.0, augment_xyz=False,
                 exposure_prob=0.0, exposure_quantile=0.25):
        self.root = Path(root).resolve()
        self.dataset = canonical_dataset(dataset)
        self.config = model_config or CoreLiteConfig.for_dataset(self.dataset)
        if split not in ("training", "validation", "test"):
            raise ValueError("Unknown split")
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("dataset") != self.dataset or manifest.get("format_version") != 1:
            raise ValueError("Dataset/manifest format mismatch")
        self.records = [row for row in manifest["clouds"] if row["split"] == split]
        if not self.records:
            raise ValueError("No clouds in requested split: " + split)
        self.cloud_names = [row["name"] for row in self.records]
        if len(set(self.cloud_names)) != len(self.cloud_names):
            raise ValueError("Duplicate cloud names")
        for row in self.records:
            path = (self.root / row["path"]).resolve()
            if self.root not in path.parents or not path.is_file() or int(row["num_points"]) < 1:
                raise ValueError("Invalid cloud manifest path/count")
        self.split, self.num_points = split, int(num_points or DEFAULT_POINTS[self.dataset])
        self.coverage_enabled = not (self.dataset == "SemanticKITTI" and split == "training")
        self.steps = int(steps) if steps is not None else len(self.records)
        if self.num_points < 1 or self.steps < 1 or int(cache_size) < 1:
            raise ValueError("Point, step and cache counts must be positive")
        self.cache_size, self._cache = int(cache_size), OrderedDict()
        self._verified_hashes = set()
        self.noise_std = 0.0 if self.dataset == "SemanticKITTI" else float(noise_std)
        self.color_drop_prob, self.auto_contrast_prob = float(color_drop_prob), float(auto_contrast_prob)
        self.augment_xyz = bool(augment_xyz)
        self.exposure_prob = float(exposure_prob) if split == "training" else 0.0
        self.exposure_quantile = float(exposure_quantile)
        for probability in (self.color_drop_prob, self.auto_contrast_prob,
                            self.exposure_prob, self.exposure_quantile):
            if not 0 <= probability <= 1:
                raise ValueError("Augmentation probabilities must lie in [0,1]")
        self.reset_possibility(seed)
        self._rare_pools = {}
        if self.exposure_prob:
            if self.dataset == "SemanticKITTI":
                raise ValueError("Exposure sampling is implemented for RGB datasets only")
            self._init_exposure()

    def reset_possibility(self, seed):
        self.rng = np.random.RandomState(int(seed))
        self.possibility = ([(self.rng.rand(int(row["num_points"])) * 1e-3).astype(np.float32)
                             for row in self.records] if self.coverage_enabled else [])
        self.min_possibility = [float(values.min()) for values in self.possibility]

    def state_dict(self):
        """Tensor/list-only state suitable for torch.load(weights_only=True)."""
        state = self.rng.get_state()
        return {"cloud_names": self.cloud_names, "split": self.split,
                "possibility": [torch.from_numpy(v.copy()) for v in self.possibility],
                "rng": [state[0], torch.from_numpy(state[1].astype(np.int64)),
                        int(state[2]), int(state[3]), float(state[4])]}

    def load_state_dict(self, state):
        if state["cloud_names"] != self.cloud_names or state["split"] != self.split:
            raise ValueError("Sampler checkpoint belongs to a different split")
        values = [_array(v).astype(np.float32).copy() for v in state["possibility"]]
        expected_records = self.records if self.coverage_enabled else []
        if len(values) != len(expected_records) or any(v.shape != (int(r["num_points"]),)
                                                     for v, r in zip(values, expected_records)):
            raise ValueError("Sampler checkpoint point counts differ")
        self.possibility = values
        self.min_possibility = [float(v.min()) for v in values]
        rng = state["rng"]
        self.rng.set_state((rng[0], _array(rng[1]).astype(np.uint32), *rng[2:]))

    def _init_exposure(self):
        counts = np.zeros(self.config.num_classes, dtype=np.int64)
        pools = {}
        for index in range(len(self.records)):
            labels = self.load_cloud(index).labels
            counts += np.bincount(labels[labels >= 0], minlength=len(counts))
            for label in np.unique(labels[labels >= 0]):
                pools.setdefault(int(label), []).append((index, np.flatnonzero(labels == label)))
        present = np.flatnonzero(counts)
        if not len(present):
            raise ValueError("Exposure sampling requires labeled training clouds")
        rare = present[counts[present] <= np.quantile(counts[present], self.exposure_quantile)]
        self._rare_classes = rare
        weights = 1 / np.sqrt(counts[rare])
        self._rare_probs = weights / weights.sum()
        self._rare_pools = {int(label): pools[int(label)] for label in rare}

    def load_cloud(self, index):
        index = int(index)
        if index not in self._cache:
            path = self.root / self.records[index]["path"]
            digest = self.records[index].get("npz_sha256")
            if digest and index not in self._verified_hashes:
                if file_sha256(path) != digest:
                    raise ValueError("Cloud checksum differs from manifest: " + self.cloud_names[index])
                self._verified_hashes.add(index)
            cloud = load_cloud(path, self.dataset)
            if len(cloud.xyz) != int(self.records[index]["num_points"]):
                raise ValueError("Manifest point count differs from NPZ")
            self._cache[index] = cloud
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(index)
        return self._cache[index]

    def __len__(self):
        return self.steps

    def __getitem__(self, item):
        if get_worker_info() is not None:
            raise RuntimeError("Stateful coverage sampler requires num_workers=0")
        if self._rare_pools and self.rng.rand() < self.exposure_prob:
            label = int(self.rng.choice(self._rare_classes, p=self._rare_probs))
            pools = self._rare_pools[label]
            cloud_index, candidates = pools[self.rng.randint(len(pools))]
            center_index = candidates[np.argmin(self.possibility[cloud_index][candidates])]
        elif self.dataset == "SemanticKITTI" and self.split == "training":
            # Fixed-step clean runner samples scans uniformly with replacement;
            # the historical frame DataLoader used a shuffled scan traversal.
            cloud_index = self.rng.randint(len(self.records))
            center_index = self.rng.randint(int(self.records[cloud_index]["num_points"]))
        else:
            cloud_index = int(np.argmin(self.min_possibility))
            center_index = int(np.argmin(self.possibility[cloud_index]))
        cloud = self.load_cloud(cloud_index)
        center = cloud.xyz[center_index:center_index+1].copy()
        if self.noise_std:
            center += self.rng.normal(scale=self.noise_std, size=(1, 3)).astype(np.float32)
        indices = cloud.tree.query(center, k=min(len(cloud.xyz), self.num_points), workers=1)[1].reshape(-1)
        indices = self.rng.permutation(indices)
        distances = np.sum((cloud.xyz[indices] - center) ** 2, axis=1)
        maximum = float(distances.max())
        delta = (np.ones_like(distances) if maximum == 0 else (1-distances/maximum)**2)
        if self.coverage_enabled:
            self.possibility[cloud_index][indices] += delta
            self.min_possibility[cloud_index] = float(self.possibility[cloud_index].min())
        if len(indices) < self.num_points:
            indices = np.concatenate([indices, self.rng.choice(indices, self.num_points-len(indices))])
        xyz = cloud.xyz[indices].copy()
        if self.dataset != "SemanticKITTI":
            xyz -= center
        features = cloud.features[indices].copy()
        if self.split == "training" and self.dataset != "SemanticKITTI":
            features = augment_rgb(features, self.rng, self.color_drop_prob, self.auto_contrast_prob)
        feature_xyz = xyz.copy()
        if self.split == "training" and self.augment_xyz:
            angle = self.rng.uniform(0, 2*np.pi)
            c, s = np.cos(angle), np.sin(angle)
            original_x, original_y = feature_xyz[:, 0].copy(), feature_xyz[:, 1].copy()
            feature_xyz[:, 0] = c*original_x + s*original_y
            feature_xyz[:, 1] = -s*original_x + c*original_y
            feature_xyz *= self.rng.uniform(.8, 1.2, (1, 3)).astype(np.float32)
            feature_xyz[:, 0] *= self.rng.choice([-1, 1])
            feature_xyz += self.rng.normal(scale=.001, size=xyz.shape).astype(np.float32)
        return {"xyz": xyz.astype(np.float32),
                "features": np.concatenate([feature_xyz, features], axis=1).astype(np.float32),
                "labels": cloud.labels[indices].copy(), "input_inds": indices.astype(np.int64),
                "cloud_inds": np.int64(cloud_index)}

    def collate_fn(self, batch):
        stacked = {key: np.stack([row[key] for row in batch]) for key in batch[0]}
        endpoints = build_hierarchy(stacked.pop("xyz"), stacked.pop("features"), self.config)
        endpoints.update({key: torch.from_numpy(value) for key, value in stacked.items()})
        return endpoints
