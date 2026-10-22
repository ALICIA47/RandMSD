"""Convert existing RandLA prepared trees to safe, portable NPZ clouds.

This converter does NOT recompute the paper's voxel subsampling. Supply existing
grid PLY files or SemanticKITTI sequences_0.06 arrays. No KDTree pickle is read;
trees are rebuilt at runtime. Projection pickles are read only on explicit opt-in.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from .data.io import (KITTI_TRAIN, NUM_CLASSES, SENSAT_CLOUDS, SENSAT_TEST, SENSAT_VALIDATION,
                      canonical_dataset, file_sha256, load_cloud, map_kitti_labels, read_ply)


def _trusted_projection(path, trust_pickle):
    if not path.is_file():
        return None, None
    if not trust_pickle:
        raise ValueError("Projection pickle found; pass --trust-pickle only for trusted local artifacts: " + str(path))
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if isinstance(value, (tuple, list)):
        return np.asarray(value[0]).reshape(-1), (np.asarray(value[1]).reshape(-1)
                                                if len(value) > 1 else None)
    return np.asarray(value).reshape(-1), None


def convert_prepared(dataset, source, output, test_area=5, trust_pickle=False,
                     raw_kitti=None, include_projections=False, exclude=()):
    """Import every available grid cloud, preserving point order and coordinates.

    Missing SensatUrban tiles are recorded, never downloaded/recovered silently.
    KITTI prepared labels are learning IDs (0 ignored, 1..19 classes).
    """
    dataset, source, output = canonical_dataset(dataset), Path(source), Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite existing prepared manifest")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if not 1 <= int(test_area) <= 6:
        raise ValueError("S3DIS held-out area must be 1..6")
    output.mkdir(parents=True, exist_ok=True)
    (output / "clouds").mkdir(exist_ok=True)
    records, discovered = [], []
    if dataset == "SemanticKITTI":
        paths = sorted(source.glob("[0-9][0-9]/velodyne/*.npy"))
    else:
        paths = sorted(source.glob("*.ply"))
    if not paths:
        raise ValueError("No prepared grid files found; point --source to the grid directory")
    for path in paths:
        sequence = path.parent.parent.name if dataset == "SemanticKITTI" else None
        name = sequence + "_" + path.stem if sequence else path.stem
        discovered.append(name)
        if name in exclude:
            continue
        if dataset == "SemanticKITTI":
            if not 0 <= int(sequence) <= 21:
                raise ValueError("SemanticKITTI sequence must be 00..21")
            split = ("training" if sequence in KITTI_TRAIN else
                     "validation" if sequence == "08" else "test")
            xyz = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
            remission = path.parent.parent / "remissions" / path.name
            if not remission.is_file():
                raise FileNotFoundError("Required voxel-aligned remission sidecar: " + str(remission))
            features = np.asarray(np.load(remission, allow_pickle=False), np.float32).reshape(-1, 1)
            label_file = path.parent.parent / "labels" / path.name
            labels = (map_kitti_labels(np.load(label_file, allow_pickle=False))
                      if label_file.is_file() else np.full(len(xyz), -1, np.int64))
            projection_file = path.parent.parent / "proj" / (path.stem + "_proj.pkl")
        else:
            data = read_ply(path)
            fields = set(data.dtype.names)
            if not {"x", "y", "z", "red", "green", "blue"} <= fields:
                raise ValueError("Missing XYZ/RGB fields: " + str(path))
            xyz = np.column_stack([data[c] for c in ("x", "y", "z")]).astype(np.float32)
            features = np.column_stack([data[c] for c in ("red", "green", "blue")]).astype(np.float32)
            integer_rgb = all(np.issubdtype(data[c].dtype, np.integer) for c in ("red", "green", "blue"))
            if integer_rgb or features.max() > 1.5:
                features /= 255
            label_field = next((key for key in ("class", "label", "semantic_class") if key in fields), None)
            labels = (np.asarray(data[label_field], np.int64) if label_field
                      else np.full(len(xyz), -1, np.int64))
            if dataset == "S3DIS":
                if not name.startswith("Area_") or len(name.split("_")) < 3:
                    raise ValueError("S3DIS cloud names must start Area_<area>_...")
                if not 1 <= int(name.split("_")[1]) <= 6:
                    raise ValueError("S3DIS area must be 1..6")
                split = "validation" if int(name.split("_")[1]) == int(test_area) else "training"
            else:
                if name not in SENSAT_CLOUDS:
                    raise ValueError("Unknown SensatUrban cloud name: " + name)
                split = ("validation" if name in SENSAT_VALIDATION else
                         "test" if name in SENSAT_TEST else "training")
            projection_file = source / (name + "_proj.pkl")
        payload = dict(xyz=xyz, features=features, labels=labels)
        if include_projections:
            projection, raw_labels = _trusted_projection(projection_file, trust_pickle)
            if projection is not None:
                payload["projection"] = projection.astype(np.int64)
                if dataset == "SemanticKITTI":
                    raw_file = (Path(raw_kitti) / sequence / "labels" / (path.stem + ".label")
                                if raw_kitti else None)
                    if raw_file is not None and raw_file.is_file():
                        raw_labels = map_kitti_labels(np.fromfile(raw_file, dtype=np.uint32), raw=True)
                    else:
                        raw_labels = None  # evaluation can still export raw-point predictions
                if raw_labels is not None:
                    payload["raw_labels"] = np.asarray(raw_labels, dtype=np.int64)
        target = output / "clouds" / (name + ".npz")
        if target.exists():
            raise FileExistsError("Refusing to overwrite cloud: " + str(target))
        np.savez_compressed(target, **payload)
        validated = load_cloud(target, dataset)
        counts = np.bincount(validated.labels[validated.labels >= 0], minlength=NUM_CLASSES[dataset])
        records.append(dict(name=name, split=split, path=target.relative_to(output).as_posix(),
                            num_points=len(xyz), class_counts=counts.tolist(),
                            npz_sha256=file_sha256(target),
                            has_projection="projection" in payload, has_raw_labels="raw_labels" in payload))
        print("Imported", name, split, len(xyz), flush=True)
        del validated
    manifest = dict(format_version=1, dataset=dataset, clouds=records,
                    preprocessing="Imported existing grid; voxelization not recomputed",
                    imported_from=str(source.resolve()), excluded=list(exclude),
                    s3dis_validation_area=int(test_area) if dataset == "S3DIS" else None,
                    available_sensat_validation=sorted(set(discovered) & SENSAT_VALIDATION)
                    if dataset == "SensatUrban" else None)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("S3DIS", "SemanticKITTI", "SensatUrban"))
    parser.add_argument("--source", type=Path, required=True,
                        help="input_0.040, sequences_0.06, or grid_0.200[_author_recovered]")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-area", type=int, default=5)
    parser.add_argument("--include-projections", action="store_true")
    parser.add_argument("--trust-pickle", action="store_true",
                        help="Permit existing local projection pickle loading (arbitrary-code boundary)")
    parser.add_argument("--raw-kitti", type=Path, help="Raw sequences directory for original-point labels")
    parser.add_argument("--exclude", nargs="*", default=[], help="Explicit cloud names to omit")
    args = parser.parse_args()
    convert_prepared(args.dataset, args.source, args.output, args.test_area, args.trust_pickle,
                     args.raw_kitti, args.include_projections, args.exclude)


if __name__ == "__main__":
    main()
