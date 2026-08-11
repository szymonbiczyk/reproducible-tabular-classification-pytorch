"""CLI entry point for predicting a single tabular record."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for prediction."""
    parser = argparse.ArgumentParser(description="Predict from a saved model checkpoint.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing the saved experiment artifacts.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a JSON record to classify.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the prediction command."""
    args = parse_args()
    # TODO: Load the record, preprocessing, and model, then emit a prediction.
    _ = args


if __name__ == "__main__":
    main()

