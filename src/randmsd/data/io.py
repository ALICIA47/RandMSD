"""Portable prepared-cloud I/O. Runtime inputs never unpickle Python objects."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


DATASETS = ("S3DIS", "SemanticKITTI", "SensatUrban")
DEFAULT_POINTS = dict(S3DIS=40960, SemanticKITTI=45056, SensatUrban=65536)
NUM_CLASSES = dict(S3DIS=13, SemanticKITTI=19, SensatUrban=13)
SENSAT_VALIDATION = {"birmingham_block_1", "birmingham_block_5",
                     "cambridge_block_10", "cambridge_block_7"}
SENSAT_TEST = {"birmingham_block_2", "birmingham_block_8", "cambridge_block_15",
               "cambridge_block_22", "cambridge_block_16", "cambridge_block_27"}
SENSAT_CLOUDS = ({"birmingham_block_" + str(i) for i in range(14)} |
                {"cambridge_block_" + str(i) for i in
                 (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18,
                  19, 20, 21, 22, 23, 25, 26, 27, 28, 32, 33, 34)})
KITTI_TRAIN = {"00", "01", "02", "03", "04", "05", "06", "07", "09", "10"}
KITTI_LEARNING_MAP = {0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5,
    18: 4, 20: 5, 30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11,
    49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17,
    80: 18, 81: 19, 99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5,
    257: 5, 258: 4, 259: 5}


def canonical_dataset(name):
    for candidate in DATASETS:
        if candidate.lower() == str(name).lower():
            return candidate
    raise ValueError("dataset must be one of " + ", ".join(DATASETS))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def map_kitti_labels(labels, raw=False):
    """Return 0..18 classes and -1 ignored; raw input may include instance bits."""
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if raw:
        values = values & 0xFFFF
        lut = np.zeros(65536, dtype=np.int64)
        for key, value in KITTI_LEARNING_MAP.items():
            lut[key] = value
        values = lut[values]
    if np.any((values < 0) | (values > 19)):
        raise ValueError("KITTI prepared labels must be learning IDs 0..19")
    return values - 1


def read_ply(path):
    """Read scalar vertex-only PLY (ASCII or binary); no executable content.

    Lists/meshes are deliberately rejected rather than silently misparsed.
    """
    types = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
             "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
             "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
             "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Not a PLY file: " + str(path))
        fmt, count, fields, element = None, None, [], None
        for _ in range(10000):
            line = stream.readline()
            if not line:
                raise ValueError("Truncated PLY header")
            words = line.decode("ascii").strip().split()
            if not words or words[0] in ("comment", "obj_info"):
                continue
            if words[0] == "end_header":
                break
            if words[0] == "format":
                fmt = words[1]
            elif words[0] == "element":
                element = words[1]
                if element != "vertex" and int(words[2]):
                    raise ValueError("Only vertex-only PLY is supported")
                if element == "vertex":
                    count = int(words[2])
            elif words[0] == "property" and element == "vertex":
                if words[1] not in types or len(words) != 3:
                    raise ValueError("Only scalar PLY properties are supported")
                fields.append((words[2], types[words[1]]))
        if count is None or not fields or count < 1:
            raise ValueError("Missing/nonempty PLY vertices")
        if fmt == "ascii":
            matrix = np.loadtxt(stream, max_rows=count, ndmin=2)
            if matrix.shape != (count, len(fields)):
                raise ValueError("PLY vertex count/property mismatch")
            data = np.empty(count, dtype=fields)
            for index, (name, _) in enumerate(fields):
                data[name] = matrix[:, index]
        elif fmt in ("binary_little_endian", "binary_big_endian"):
            order = "<" if fmt == "binary_little_endian" else ">"
            data = np.fromfile(stream, dtype=[(n, order + t) for n, t in fields], count=count)
            if len(data) != count:
                raise ValueError("Truncated PLY body")
        else:
            raise ValueError("Unsupported PLY format")
    return data


@dataclass
class Cloud:
    xyz: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    projection: np.ndarray | None = None
    raw_labels: np.ndarray | None = None
    tree: object = None


def load_cloud(path, dataset):
    dataset = canonical_dataset(dataset)
    with np.load(path, allow_pickle=False) as data:
        xyz = np.asarray(data["xyz"], dtype=np.float32)
        features = np.asarray(data["features"], dtype=np.float32)
        labels = (np.asarray(data["labels"], dtype=np.int64).reshape(-1)
                  if "labels" in data else np.full(len(xyz), -1, dtype=np.int64))
        projection = (np.asarray(data["projection"], dtype=np.int64).reshape(-1)
                      if "projection" in data else None)
        raw_labels = (np.asarray(data["raw_labels"], dtype=np.int64).reshape(-1)
                      if "raw_labels" in data else None)
    channels = 1 if dataset == "SemanticKITTI" else 3
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("xyz must be a nonempty [N,3] array")
    if features.shape != (len(xyz), channels) or labels.shape != (len(xyz),):
        raise ValueError("Feature/label shape mismatch")
    if not np.isfinite(xyz).all() or not np.isfinite(features).all():
        raise ValueError("Non-finite point cloud")
    if dataset != "SemanticKITTI" and (features.min() < 0 or features.max() > 1):
        raise ValueError("Prepared RGB must be in [0,1]")
    for label_array in (labels, raw_labels):
        if label_array is not None and np.any((label_array < -1) | (label_array >= NUM_CLASSES[dataset])):
            raise ValueError("Labels must be train IDs or -1 ignored")
    if projection is not None:
        if np.any((projection < 0) | (projection >= len(xyz))):
            raise ValueError("Projection is outside grid point range")
        if raw_labels is not None and len(raw_labels) != len(projection):
            raise ValueError("Raw labels and projection differ in length")
    elif raw_labels is not None:
        raise ValueError("raw_labels require an original-to-grid projection")
    return Cloud(xyz, features, labels, projection, raw_labels, cKDTree(xyz, leafsize=40))
