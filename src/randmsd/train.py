"""Train the frozen RandMSD model on a prepared dataset (see configs/)."""

import argparse
import json

from .engine import load_config, run_training


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true", help="Strictly resume last.pt; recover an interrupted best-first checkpoint transaction when consistent")
    args = parser.parse_args(argv)
    result = run_training(load_config(args.config), args.data_root, args.output,
                          device=args.device, resume=args.resume)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
