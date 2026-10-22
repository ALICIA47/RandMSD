"""Export original-point evaluation predictions to verified benchmark formats.

SemanticKITTI: original semantic IDs in little-endian uint32, instance bits 0.
SensatUrban: six root-level uint8 label files in submission.zip, IDs 0..12.
Formats follow the research archive's verified official exporters; this utility
does not submit anything or claim an official benchmark score.
"""
import argparse
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from .io import SENSAT_TEST, canonical_dataset, file_sha256


# utils/semantic-kitti.yaml learning_map_inv[1..19]. Clean model IDs are 0..18.
KITTI_ORIGINAL_IDS = np.array([10, 11, 15, 18, 20, 30, 31, 32, 40, 44,
                               48, 49, 50, 51, 70, 71, 72, 80, 81], dtype="<u4")
KITTI_TEST_SCAN_COUNTS = {"11": 921, "12": 1061, "13": 3281, "14": 631,
    "15": 1901, "16": 1731, "17": 491, "18": 1801, "19": 4981,
    "20": 831, "21": 2721}


def _labels(values, classes, expected_points):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("Predictions must be a one-dimensional integer array")
    if len(values) != int(expected_points) or len(values) < 1:
        raise ValueError("Prediction count differs from original-point count")
    if np.any(values < 0) or np.any(values >= classes):
        raise ValueError("Prediction ID outside valid train-class range")
    return values


def kitti_original_ids(predictions, expected_points):
    values = _labels(predictions, 19, expected_points)
    return KITTI_ORIGINAL_IDS[values.astype(np.int64)]


def write_kitti_prediction(predictions, output_root, sequence, frame, expected_points):
    sequence, frame = str(sequence), str(frame)
    if not re.fullmatch(r"(?:1[1-9]|2[01])", sequence):
        raise ValueError("Official SemanticKITTI test sequence must be 11..21")
    if not re.fullmatch(r"[0-9]{6}", frame):
        raise ValueError("Frame must be a six-digit basename")
    values = kitti_original_ids(predictions, expected_points)
    root = Path(output_root).resolve()
    target = root / "sequences" / sequence / "predictions" / (frame + ".label")
    if root not in target.resolve().parents:
        raise ValueError("Output escapes submission directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        values.tofile(stream)
    return target


def _original_predictions(path):
    with np.load(path, allow_pickle=False) as data:
        if "original_labels" not in data:
            raise ValueError("Missing original_labels; evaluate data with original-to-grid projection: " + str(path))
        return np.asarray(data["original_labels"])


def _ply_vertex_count(path):
    """Read only the vertex header, avoiding multi-GB raw cloud allocation."""
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Raw reference is not a PLY")
        count = None
        for _ in range(10000):
            line = stream.readline()
            if not line:
                break
            words = line.decode("ascii").split()
            if words[:2] == ["element", "vertex"]:
                count = int(words[2])
            if words == ["end_header"]:
                if count is not None and count > 0:
                    return count
                break
    raise ValueError("Invalid raw PLY vertex count")


def export_evaluation(dataset, predictions, raw_root, output, *, require_all_kitti=True):
    """Export an evaluation folder; reject missing original points/overwrites.

    For KITTI, raw_root is the raw sequences directory. For SensatUrban, it is
    the directory containing the six original test PLY files. By default KITTI
    requires sequences 11..21; partial export is an explicit format-check option.
    """
    dataset = canonical_dataset(dataset)
    if dataset == "S3DIS":
        raise ValueError("S3DIS has no official held-out submission format here")
    predictions, raw_root, output = Path(predictions).resolve(), Path(raw_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing submission output")
    files = sorted(path for path in predictions.glob("*.npz") if path.stem != "confusion")
    if not files:
        raise ValueError("No per-cloud prediction NPZ files")
    records = []
    if dataset == "SemanticKITTI":
        predicted_frames = {}
        for path in files:
            match = re.fullmatch(r"(1[1-9]|2[01])_([0-9]{6})", path.stem)
            if not match:
                raise ValueError("Expected official test cloud name XX_NNNNNN: " + path.stem)
            sequence, frame = match.groups()
            raw = raw_root / sequence / "velodyne" / (frame + ".bin")
            size = raw.stat().st_size
            if size < 16 or size % 16:
                raise ValueError("Invalid raw KITTI scan byte count")
            count = size // 16
            kitti_original_ids(_original_predictions(path), count)
            predicted_frames.setdefault(sequence, set()).add(frame)
            records.append(dict(name=path.stem, source=path, points=count,
                                sequence=sequence, frame=frame))
        if require_all_kitti and set(predicted_frames) != {str(i) for i in range(11, 22)}:
            raise ValueError("Complete official export requires every sequence 11..21")
        for sequence, frames in predicted_frames.items():
            raw_frames = {path.stem for path in (raw_root / sequence / "velodyne").glob("*.bin")}
            if frames != raw_frames:
                raise ValueError("Raw/predicted frame set differs in sequence " + sequence)
            if require_all_kitti and len(frames) != KITTI_TEST_SCAN_COUNTS[sequence]:
                raise ValueError("Official test scan count differs in sequence " + sequence)
    else:
        if {path.stem for path in files} != SENSAT_TEST:
            raise ValueError("SensatUrban official export requires exactly the six test clouds")
        for path in files:
            count = _ply_vertex_count(raw_root / (path.stem + ".ply"))
            _labels(_original_predictions(path), 13, count)
            records.append(dict(name=path.stem, source=path, points=count))
    output.mkdir(parents=True)
    result = []
    for record in records:
        values = _original_predictions(record["source"])
        if dataset == "SemanticKITTI":
            target = write_kitti_prediction(values, output, record["sequence"], record["frame"], record["points"])
        else:
            target = output / (record["name"] + ".label")
            with target.open("xb") as stream:
                _labels(values, 13, record["points"]).astype(np.uint8).tofile(stream)
        result.append(dict(name=record["name"], points=record["points"],
                           path=target.relative_to(output).as_posix(), sha256=file_sha256(target)))
    if dataset == "SensatUrban":
        with zipfile.ZipFile(output / "submission.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for record in result:
                archive.write(output / record["path"], arcname=record["path"])
        with zipfile.ZipFile(output / "submission.zip") as archive:
            if set(archive.namelist()) != {name + ".label" for name in SENSAT_TEST}:
                raise RuntimeError("Unexpected ZIP member paths")
            for record in result:
                if archive.getinfo(record["path"]).file_size != record["points"]:
                    raise RuntimeError("SensatUrban ZIP point count mismatch")
    manifest = dict(dataset=dataset, format_only_not_submitted=True, clouds=result,
                    complete_kitti_sequences=(require_all_kitti if dataset == "SemanticKITTI" else None))
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("SemanticKITTI", "SensatUrban"))
    parser.add_argument("--predictions", type=Path, required=True, help="evaluate --save-predictions output")
    parser.add_argument("--raw-root", type=Path, required=True, help="Raw sequences or original test PLY directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial-kitti", action="store_true",
                        help="Format check only: omit the full 11..21 sequence-set requirement")
    args = parser.parse_args()
    result = export_evaluation(args.dataset, args.predictions, args.raw_root, args.output,
                               require_all_kitti=not args.allow_partial_kitti)
    print("Exported", len(result["clouds"]), "clouds to", args.output)


if __name__ == "__main__":
    main()
