"""CPU model smoke test with the public hierarchy builder (not an accuracy run)."""

import argparse

import numpy as np
import torch

from randmsd import ModelConfig, RandMSD
from randmsd.data import build_hierarchy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("S3DIS", "SemanticKITTI", "SensatUrban"), default="S3DIS")
    parser.add_argument("--points", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20260620)
    rng = np.random.default_rng(20260620)
    config = ModelConfig.for_dataset(args.dataset)
    xyz = rng.normal(size=(args.points, 3)).astype(np.float32)
    attributes = rng.random((args.points, config.input_channels - 3), dtype=np.float32)
    features = np.concatenate((xyz, attributes), axis=1)
    # Random-prefix centers and KNN supports are built outside the backbone.
    batch = build_hierarchy(xyz, features, config, seed=20260620)
    model = RandMSD(config).eval()
    with torch.no_grad():
        result = model(batch)
    print("dataset={} parameters={} logits={} finite={}".format(
        args.dataset, sum(p.numel() for p in model.parameters()),
        tuple(result["logits"].shape), bool(torch.isfinite(result["logits"]).all())))


if __name__ == "__main__":
    main()
