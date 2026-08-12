"""Load and clean Adult Income data and create reproducible splits."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import train_test_split


ADULT_COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

TARGET_COLUMN = "income"
TARGET_MAPPING = {
    "<=50K": 0,
    ">50K": 1,
    "<=50K.": 0,
    ">50K.": 1,
}

ADULT_RAW_URLS = {
    "adult.data": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    ),
    "adult.test": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    ),
}


@dataclass(frozen=True)
class AdultDataSplits:
    """Clean train, validation, and official test dataframes."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def ensure_raw_adult_data(raw_dir: Path) -> None:
    """Download missing Adult files from the official UCI URLs."""

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in ADULT_RAW_URLS.items():
        destination = raw_dir / filename
        if destination.is_file() and destination.stat().st_size > 0:
            continue

        temporary_path = raw_dir / f".{filename}.download"
        try:
            with urlopen(url, timeout=60) as response, temporary_path.open("wb") as output:
                shutil.copyfileobj(response, output)

            if temporary_path.stat().st_size == 0:
                raise OSError(f"Downloaded an empty file from {url}")
            temporary_path.replace(destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise


def load_adult_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the official Adult train and test files without cleaning them."""

    raw_dir = Path(raw_dir)
    train_path = raw_dir / "adult.data"
    test_path = raw_dir / "adult.test"
    missing_paths = [path for path in (train_path, test_path) if not path.is_file()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing raw Adult data file(s): {missing}")

    read_options = {
        "header": None,
        "names": ADULT_COLUMN_NAMES,
        "skipinitialspace": True,
        "na_values": ["?"],
        "comment": "|",
        "skip_blank_lines": True,
    }
    train_df = pd.read_csv(train_path, **read_options)
    test_df = pd.read_csv(test_path, **read_options)
    return train_df, test_df


def clean_adult_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the Adult schema, missing-value, and target-label contract."""

    missing_columns = set(ADULT_COLUMN_NAMES) - set(df.columns)
    extra_columns = set(df.columns) - set(ADULT_COLUMN_NAMES)
    if missing_columns or extra_columns:
        raise ValueError(
            "Adult dataframe columns do not match the contract. "
            f"Missing: {sorted(missing_columns)}; extra: {sorted(extra_columns)}"
        )

    cleaned = df.loc[:, ADULT_COLUMN_NAMES].copy()
    string_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in string_columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()
        cleaned[column] = cleaned[column].replace("?", pd.NA)

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")

    unknown_targets = set(cleaned[TARGET_COLUMN].dropna().unique()) - set(TARGET_MAPPING)
    if unknown_targets:
        raise ValueError(f"Unknown Adult target labels: {sorted(unknown_targets)}")

    mapped_target = cleaned[TARGET_COLUMN].map(TARGET_MAPPING)
    if mapped_target.isna().any():
        raise ValueError("Adult target column contains missing labels")
    cleaned[TARGET_COLUMN] = mapped_target.astype("int8")

    return cleaned


def split_adult_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> AdultDataSplits:
    """Split only the original train source and preserve the official test set."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_df.empty or test_df.empty:
        raise ValueError("Adult train and test dataframes must not be empty")
    if TARGET_COLUMN not in train_df or TARGET_COLUMN not in test_df:
        raise ValueError(f"Both dataframes must contain the {TARGET_COLUMN!r} column")

    train_split, validation_split = train_test_split(
        train_df,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=train_df[TARGET_COLUMN],
    )

    return AdultDataSplits(
        train=train_split.reset_index(drop=True),
        validation=validation_split.reset_index(drop=True),
        test=test_df.reset_index(drop=True).copy(),
    )


def load_adult_splits(
    raw_dir: Path,
    validation_fraction: float,
    seed: int,
    download: bool = True,
) -> AdultDataSplits:
    """Load, clean, and split Adult data according to the project contract."""

    raw_dir = Path(raw_dir)
    if download:
        ensure_raw_adult_data(raw_dir)

    raw_train, raw_test = load_adult_raw(raw_dir)
    clean_train = clean_adult_dataframe(raw_train)
    clean_test = clean_adult_dataframe(raw_test)
    return split_adult_data(clean_train, clean_test, validation_fraction, seed)


def summarize_splits(splits: AdultDataSplits) -> dict[str, object]:
    """Return row counts, column count, and target counts for each split."""

    frames = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }

    return {
        "rows": {name: len(frame) for name, frame in frames.items()},
        "columns": len(ADULT_COLUMN_NAMES),
        "target_distribution": {
            name: {
                int(label): int(count)
                for label, count in frame[TARGET_COLUMN]
                .value_counts()
                .sort_index()
                .items()
            }
            for name, frame in frames.items()
        },
    }

