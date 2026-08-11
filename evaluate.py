"""CLI entry point for evaluating a saved experiment."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a saved model checkpoint.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing the saved experiment artifacts.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Dataset split to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the evaluation command."""
    args = parse_args()
    # TODO: Restore preprocessing and the model, then compute and save metrics.
    _ = args


if __name__ == "__main__":
    main()

