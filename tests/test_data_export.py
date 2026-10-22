"""Local benchmark-format checks; no external submission/network operation."""
import tempfile
import unittest
from pathlib import Path
import os
import zipfile

import numpy as np

from randmsd.data.export import export_evaluation, kitti_original_ids, write_kitti_prediction
from randmsd.data.io import SENSAT_TEST


class ExportTests(unittest.TestCase):
    def test_kitti_all_ids_instance_bits_and_overwrite(self):
        expected = [10, 11, 15, 18, 20, 30, 31, 32, 40, 44, 48, 49, 50, 51, 70, 71, 72, 80, 81]
        values = kitti_original_ids(np.arange(19), 19)
        np.testing.assert_array_equal(values, expected)
        self.assertTrue(np.all(values >> 16 == 0))
        with tempfile.TemporaryDirectory() as folder:
            target = write_kitti_prediction(np.arange(19), folder, "11", "000000", 19)
            np.testing.assert_array_equal(np.fromfile(target, dtype="<u4"), expected)
            with self.assertRaises(FileExistsError):
                write_kitti_prediction(np.arange(19), folder, "11", "000000", 19)
            for sequence, frame in [("../11", "000000"), ("11", "../000000"), ("08", "000000")]:
                with self.assertRaises(ValueError):
                    write_kitti_prediction(np.arange(19), folder, sequence, frame, 19)

    def test_invalid_labels_and_original_point_count(self):
        for labels in [np.array([-1]), np.array([19]), np.array([1.2]), np.array([[1]])]:
            with self.assertRaises(ValueError):
                kitti_original_ids(labels, 1)
        with self.assertRaises(ValueError):
            kitti_original_ids(np.array([1]), 2)

    def test_partial_kitti_export_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            predictions = root / "predictions"
            raw = root / "raw/11/velodyne"
            predictions.mkdir()
            raw.mkdir(parents=True)
            np.zeros((3, 4), dtype=np.float32).tofile(raw / "000000.bin")
            np.savez(predictions / "11_000000.npz", original_labels=np.array([0, 1, 18], np.uint8))
            with self.assertRaisesRegex(ValueError, "every sequence"):
                export_evaluation("SemanticKITTI", predictions, root / "raw", root / "complete")
            self.assertFalse((root / "complete").exists())
            relative_output = os.path.relpath(root / "partial", Path.cwd())
            result = export_evaluation("SemanticKITTI", predictions, root / "raw", relative_output,
                                       require_all_kitti=False)
            self.assertEqual(result["clouds"][0]["points"], 3)
            with self.assertRaises(FileExistsError):
                export_evaluation("SemanticKITTI", predictions, root / "raw", root / "partial",
                                   require_all_kitti=False)

    def test_sensat_six_root_label_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            predictions, raw = root / "predictions", root / "raw"
            predictions.mkdir()
            raw.mkdir()
            for name in SENSAT_TEST:
                (raw / (name + ".ply")).write_text("ply\nformat ascii 1.0\nelement vertex 3\n"
                    "property float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n1 1 1\n2 2 2\n")
                np.savez(predictions / (name + ".npz"), original_labels=np.array([0, 6, 12], np.uint8))
            result = export_evaluation("SensatUrban", predictions, raw, root / "submission")
            self.assertEqual(len(result["clouds"]), 6)
            with zipfile.ZipFile(root / "submission/submission.zip") as archive:
                self.assertEqual(set(archive.namelist()), {name + ".label" for name in SENSAT_TEST})
                for name in archive.namelist():
                    np.testing.assert_array_equal(np.frombuffer(archive.read(name), dtype=np.uint8), [0, 6, 12])


if __name__ == "__main__":
    unittest.main()
