"""CPU contract tests for all three prepared-data adapters and hierarchy."""
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from randmsd.data import PreparedCloudDataset, build_hierarchy, load_cloud, map_kitti_labels
from randmsd.data.io import read_ply
from randmsd.data.dataset import augment_rgb
from randmsd.preprocess import convert_prepared
from rsnative_corelite.config import CoreLiteConfig


def write_ply(path, xyz, labels, rgb=(255, 128, 0)):
    header = ("ply\nformat ascii 1.0\nelement vertex " + str(len(xyz)) +
              "\nproperty float x\nproperty float y\nproperty float z\n" +
              "property uchar red\nproperty uchar green\nproperty uchar blue\n" +
              "property uchar class\nend_header\n")
    rows = [" ".join(map(str, [*point, *rgb, int(label)]))
            for point, label in zip(xyz, labels)]
    path.write_text(header + "\n".join(rows) + "\n", encoding="ascii")


class DataTests(unittest.TestCase):
    def test_hierarchy_prefix_and_alignment(self):
        xyz = np.random.RandomState(3).randn(100, 3).astype(np.float32)
        features = np.concatenate([xyz, xyz], axis=1)
        cfg = CoreLiteConfig.for_dataset("S3DIS")
        result = build_hierarchy(xyz, features, cfg, seed=4)
        self.assertEqual([v.shape[1] for v in result["xyz"]], [100, 25, 6, 1, 1])
        np.testing.assert_array_equal(result["features"][0, :, :3], result["xyz"][0][0])
        for level in range(4):
            count = result["xyz"][level+1].shape[1]
            torch.testing.assert_close(result["xyz"][level+1], result["xyz"][level][:, :count])
            self.assertTrue(torch.all(result["interp_idx"][level] < count))
        self.assertEqual(result["neigh_idx"][4].shape[-1], 16)

    def test_all_adapters_and_sampler_restore(self):
        rng = np.random.RandomState(4)
        xyz = rng.rand(80, 3).astype(np.float32)
        for dataset in ("S3DIS", "SemanticKITTI", "SensatUrban"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source, target = root / "source", root / "target"
                source.mkdir()
                if dataset == "SemanticKITTI":
                    sequence = source / "08"
                    for name in ("velodyne", "remissions", "labels"):
                        (sequence / name).mkdir(parents=True)
                    np.save(sequence / "velodyne/000000.npy", xyz)
                    np.save(sequence / "remissions/000000.npy", np.ones(len(xyz)) * .7)
                    np.save(sequence / "labels/000000.npy", np.arange(len(xyz)) % 20)
                else:
                    name = "Area_5_office_1" if dataset == "S3DIS" else "birmingham_block_1"
                    write_ply(source / (name + ".ply"), xyz, np.arange(len(xyz)) % 13)
                manifest = convert_prepared(dataset, source, target)
                self.assertEqual(manifest["clouds"][0]["split"], "validation")
                ds = PreparedCloudDataset(target, dataset, "validation", num_points=128, steps=2)
                state = ds.state_dict()
                first = ds[0]
                stream = io.BytesIO()
                torch.save(state, stream)
                stream.seek(0)
                ds.load_state_dict(torch.load(stream, weights_only=True))
                repeated = ds[0]
                for key in first:
                    np.testing.assert_array_equal(first[key], repeated[key])
                batch = ds.collate_fn([first, ds[1]])
                self.assertEqual(batch["labels"].shape, (2, 128))
                self.assertEqual(batch["features"].shape[-1], 4 if dataset == "SemanticKITTI" else 6)
                self.assertEqual(batch["cloud_inds"].shape, (2,))
                self.assertTrue(ds.min_possibility[0] > state["possibility"][0].min())
                if dataset == "SemanticKITTI":
                    np.testing.assert_array_equal(first["features"][:, :3], first["xyz"])

    def test_label_mapping(self):
        labels = map_kitti_labels([0, 10, 252, (53 << 16) + 81, 999], raw=True)
        np.testing.assert_array_equal(labels, [-1, 0, 0, 18, -1])
        with self.assertRaises(ValueError):
            map_kitti_labels([20])

    def test_rgb_disabled_augmentation_preserves_rng(self):
        colors = np.array([[.1, .3, .5], [.9, .8, .7]], np.float32)
        original = colors.copy()
        rng, control = np.random.RandomState(7), np.random.RandomState(7)
        result = augment_rgb(colors, rng)
        np.testing.assert_array_equal(result, colors)
        self.assertEqual(rng.rand(), control.rand())
        result = augment_rgb(colors, rng, color_drop_prob=1, auto_contrast_prob=1)
        np.testing.assert_array_equal(result, 0)
        np.testing.assert_array_equal(colors, original)

    def test_rgb_dtype_normalization_not_double_scaled(self):
        for integer in (False, True):
            with self.subTest(integer=integer), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source = root / "source"
                source.mkdir()
                path = source / "Area_5_room_1.ply"
                if integer:
                    write_ply(path, np.zeros((2, 3)), [0, 1], rgb=(1, 0, 1))
                    expected = np.array([1/255, 0, 1/255], dtype=np.float32)
                else:
                    records = np.zeros(2, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                        ("red", "<f8"), ("green", "<f8"), ("blue", "<f8"), ("class", "u1")])
                    expected = np.array([.1, .3, .7], np.float32)
                    for key, value in zip(("red", "green", "blue"), expected):
                        records[key] = value
                    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 2\n" +
                        "property float x\nproperty float y\nproperty float z\n" +
                        "property double red\nproperty double green\nproperty double blue\n" +
                        "property uchar class\nend_header\n")
                    with path.open("wb") as stream:
                        stream.write(header.encode("ascii"))
                        records.tofile(stream)
                manifest = convert_prepared("S3DIS", source, root / "ready")
                cloud = load_cloud(root / "ready" / manifest["clouds"][0]["path"], "S3DIS")
                np.testing.assert_array_equal(cloud.features[0], expected)

    def test_checksum_and_training_exposure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            xyz = np.arange(60).reshape(20, 3).astype(np.float32)
            write_ply(source / "Area_1_room_1.ply", xyz, np.arange(20) % 13)
            manifest = convert_prepared("S3DIS", source, root / "ready")
            ds = PreparedCloudDataset(root / "ready", "S3DIS", "training", num_points=32,
                                      exposure_prob=1, exposure_quantile=.25,
                                      color_drop_prob=1, augment_xyz=True)
            first = ds[0]
            self.assertEqual(first["features"].shape, (32, 6))
            np.testing.assert_array_equal(first["features"][:, 3:], 0)
            file = root / "ready" / manifest["clouds"][0]["path"]
            with file.open("ab") as stream:
                stream.write(b"modified")
            ds = PreparedCloudDataset(root / "ready", "S3DIS", "training")
            with self.assertRaisesRegex(ValueError, "checksum"):
                ds.load_cloud(0)

    def test_pickle_trust_boundary_and_projection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            xyz = np.arange(15).reshape(5, 3).astype(np.float32)
            write_ply(source / "Area_5_room_1.ply", xyz, np.arange(5))
            with (source / "Area_5_room_1_proj.pkl").open("wb") as stream:
                pickle.dump([np.array([0, 1, 1, 4]), np.array([0, 1, 1, 4])], stream)
            with self.assertRaisesRegex(ValueError, "trust-pickle"):
                convert_prepared("S3DIS", source, root / "blocked", include_projections=True)
            manifest = convert_prepared("S3DIS", source, root / "ready", include_projections=True,
                                        trust_pickle=True)
            cloud = load_cloud(root / "ready" / manifest["clouds"][0]["path"], "S3DIS")
            np.testing.assert_array_equal(cloud.projection, [0, 1, 1, 4])
            np.testing.assert_array_equal(cloud.raw_labels, [0, 1, 1, 4])

    def test_manifest_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "manifest.json").write_text(json.dumps(dict(dataset="S3DIS", format_version=1,
                clouds=[dict(name="bad", split="validation", path="../outside.npz", num_points=4)])))
            with self.assertRaisesRegex(ValueError, "manifest path"):
                PreparedCloudDataset(root, "S3DIS", "validation")

    def test_truncated_ply_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "bad.ply"
            file.write_text("ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
                            "property float x\nend_header\n")
            with self.assertRaisesRegex(ValueError, "Truncated PLY body"):
                read_ply(file)


if __name__ == "__main__":
    unittest.main()
