import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("aggregate_folds", ROOT/"tools/aggregate_folds.py")
folds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(folds)


class ReleaseToolsTests(unittest.TestCase):
    def test_recipe_shapes_and_frozen_settings(self):
        from randmsd import ModelConfig
        for name in ("s3dis", "semantickitti", "sensaturban"):
            config = json.loads((ROOT/"configs"/(name+".json")).read_text())
            frozen = ModelConfig.for_dataset(config["dataset"])
            self.assertEqual(frozen.k_n, 16)
            self.assertGreater(config["num_points"], 1000)
            self.assertFalse(config["training"]["augment_xyz"])
        kitti = json.loads((ROOT/"configs/semantickitti.json").read_text())
        from randmsd.engine import learning_rate_at_epoch
        self.assertEqual(learning_rate_at_epoch(kitti["training"], 0), .0095)

    def test_pool_confusion_not_per_fold_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            total = np.zeros((13, 13), dtype=np.int64)
            for index in range(6):
                path = Path(folder)/str(index)/"confusion.npz"
                path.parent.mkdir()
                matrix = np.eye(13, dtype=np.int64) * (index+1)
                matrix[0, 1] = 3*index
                np.savez(path, original_confusion=matrix, grid_confusion=matrix)
                paths.append(path); total += matrix
            result = folds.aggregate(paths, "original")
            np.testing.assert_array_equal(result["confusion"], total)
            self.assertEqual(result["point_count"], int(total.sum()))
            with self.assertRaises(ValueError): folds.aggregate([paths[0]]*6, "original")


if __name__ == "__main__":
    unittest.main()
