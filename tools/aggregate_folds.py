"""Pool six disjoint S3DIS confusion matrices before computing metrics."""
import argparse
import json
from pathlib import Path

import numpy as np

from randmsd.metrics import segmentation_metrics


def aggregate(paths, space):
    paths = [Path(path).resolve() for path in paths]
    if len(paths) != 6 or len(set(paths)) != 6:
        raise ValueError("Six different held-out-fold confusion files are required")
    key = {"original": "original_confusion", "grid": "grid_confusion"}[space]
    total = np.zeros((13, 13), dtype=np.int64)
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            matrix = arrays[key]
        if matrix.shape != total.shape or matrix.dtype.kind not in "iu" or np.any(matrix < 0) or not matrix.sum():
            raise ValueError("Missing, empty, or invalid 13-class confusion: " + str(path))
        receipt = path.with_name("metrics.json")
        if receipt.exists():
            info = json.loads(receipt.read_text(encoding="utf-8"))
            if info.get("dataset") != "S3DIS" or info.get("status") != "COMPLETE":
                raise ValueError("Not a completed S3DIS evaluation: " + str(path))
            if space == "original" and info.get("metric_space") != "original_points":
                raise ValueError("Original-point coverage was not verified: " + str(path))
        total += matrix
    return {"aggregation": "pooled_confusion_before_metrics", "space": space,
            "folds": 6, "point_count": int(total.sum()),
            "metrics": segmentation_metrics(total), "confusion": total.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confusions", nargs=6, required=True)
    parser.add_argument("--space", choices=("original", "grid"), default="original")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.confusions, args.space)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()

