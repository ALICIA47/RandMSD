"""Full-cloud voting on a prepared split; local scores are not official test scores."""

import argparse
import json
from pathlib import Path

from . import load_checkpoint
from .engine import (load_config, make_dataset, model_for_config, run_voting,
                     seed_everything, sha256, write_json)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trusted-checkpoint", action="store_true", help="Allow unsafe legacy pickle loading only for an independently trusted archive")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--save-predictions", action="store_true", help="Save uint8 train IDs; NOT ready-made benchmark submission IDs")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation output must be new or empty")
    seed_everything(int(config["seed"]))
    model = model_for_config(config)
    checkpoint = load_checkpoint(model, args.checkpoint, trusted=args.trusted_checkpoint,
                                  expected_sha256=args.checkpoint_sha256)
    checkpoint_config = checkpoint["metadata"].get("config", {})
    if checkpoint_config and checkpoint_config.get("dataset") != config["dataset"]:
        raise ValueError("Checkpoint and evaluation dataset differ")
    recipe = config.get("evaluation", {})
    dataset = make_dataset(args.data_root, config, args.split, steps=1)
    model.to(args.device)
    result = run_voting(model, dataset, output, device=args.device,
                        num_classes=model.classifier.out_channels,
                        batch_size=int(recipe.get("batch_size", 1)),
                        votes=int(recipe.get("votes", 1)),
                        first_coverage=float(recipe.get("first_coverage", recipe.get("coverage", .5))),
                        coverage_step=float(recipe.get("coverage_step", 1.0)),
                        smooth=float(recipe.get("smooth", .95)),
                        max_patches=int(recipe.get("max_patches", 1000000)),
                        save_predictions=args.save_predictions or args.split == "test")
    result.update({"seed": int(config["seed"]), "checkpoint_sha256": checkpoint["sha256"],
                   "manifest_sha256": sha256(Path(args.data_root) / "manifest.json"),
                   "config": config})
    write_json(output / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
