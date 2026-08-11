"""CLI entry point for training a configured experiment."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train a configured PyTorch model.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to an experiment YAML configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the training command."""
    args = parse_args()
    # TODO: Load the config, prepare data, train the model, and save run artifacts.
    _ = args


if __name__ == "__main__":
    main()

