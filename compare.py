"""CLI entry point for comparing completed experiment runs."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for experiment comparison."""
    parser = argparse.ArgumentParser(description="Compare metrics across experiment runs.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/runs"),
        help="Directory containing experiment run folders.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the experiment comparison command."""
    args = parse_args()
    # TODO: Discover completed runs and present their saved metrics side by side.
    _ = args


if __name__ == "__main__":
    main()

