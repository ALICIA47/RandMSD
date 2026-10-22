"""FP32 model-only forward timing; hierarchy construction is outside timing."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import torch

from randmsd import ModelConfig, RandMSD, ReferenceRandMSD
from randmsd.data import build_hierarchy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("S3DIS", "SemanticKITTI", "SensatUrban"), default="S3DIS")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--points", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true",
                        help="Explicitly allow an occupied GPU; results are not exclusive-device timings")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite benchmark output")
    if min(args.batch_size, args.warmup, args.iterations, args.threads) < 1:
        raise ValueError("Batch, warmup, iteration and thread counts must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not args.allow_shared_gpu:
        try:
            query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                   check=True, capture_output=True, text=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise RuntimeError("Cannot verify an idle GPU; inspect occupancy or explicitly allow sharing") from error
        if query.stdout.strip():
            raise RuntimeError("GPU compute process detected; no process was stopped")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    cfg = ModelConfig.for_dataset(args.dataset)
    points = args.points or {"S3DIS": 40960, "SemanticKITTI": 45056, "SensatUrban": 65536}[args.dataset]
    if points < 1:
        raise ValueError("Point count must be positive")
    rng = np.random.default_rng(args.seed)
    xyz = rng.normal(size=(args.batch_size, points, 3)).astype(np.float32)
    attributes = rng.random((args.batch_size, points, cfg.input_channels-3), dtype=np.float32)
    prepared = build_hierarchy(xyz, np.concatenate([xyz, attributes], axis=-1), cfg, seed=args.seed)
    prepared = {key: [part.to(device) for part in value] if isinstance(value, list) else value.to(device)
                for key, value in prepared.items()}
    model = (ReferenceRandMSD if args.reference else RandMSD)(cfg).to(device).eval()
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(prepared)
        sync()
        # The network stores logits in its endpoint dictionary: drop the last
        # warmup output so allocator baselines exclude a retained output tensor.
        prepared.pop("logits", None)
        if device.type == "cuda":
            sync()
            initial = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            initial = None
        elapsed = []
        for _ in range(args.iterations):
            prepared.pop("logits", None)
            sync(); begin = time.perf_counter()
            logits = model(prepared)["logits"]
            sync(); elapsed.append(1000 * (time.perf_counter()-begin))
            del logits
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        finite = bool(torch.isfinite(prepared["logits"]).all().item())
    result = {
        "status": "COMPLETE" if finite else "FAILED_NONFINITE",
        "scope": "model_forward_on_preconstructed_synthetic_hierarchy",
        "excluded": ["sampling", "knn", "hierarchy_construction", "h2d", "warmup", "scene_fusion", "projection", "io"],
        "dataset": args.dataset, "points": points, "batch_size": args.batch_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "implementation": "reference" if args.reference else "optimized",
        "precision": "FP32; TF32 off", "seed": args.seed,
        "warmup": args.warmup, "iterations": args.iterations,
        "milliseconds": elapsed, "mean_ms": statistics.mean(elapsed),
        "median_ms": statistics.median(elapsed),
        "peak_allocated_mib": peak/2**20 if peak is not None else None,
        "incremental_peak_mib": (peak-initial)/2**20 if peak is not None else None,
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "threads": args.threads,
        "gpu_sharing_explicitly_allowed": args.allow_shared_gpu,
        "random_initialization": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps(result, indent=2))
    if not finite:
        raise RuntimeError("Non-finite output; benchmark rejected")


if __name__ == "__main__":
    main()

