import argparse
import json
from pathlib import Path

from config import config
from experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Run conditional backdoor ICL experiment on SST-2."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Path to save results JSON.",
    )
    parser.add_argument(
        "--max_eval_examples",
        type=int,
        default=None,
        help="Override max_eval_examples (optional).",
    )
    args = parser.parse_args()

    cfg = config()
    if args.max_eval_examples is not None:
        cfg["max_eval_examples"] = args.max_eval_examples

    results = run_experiment(cfg)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
