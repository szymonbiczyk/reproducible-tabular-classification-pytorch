"""Tests for the Adult Income data contract."""

import pandas as pd
from pandas.testing import assert_frame_equal

from tabular_classifier.data import (
    ADULT_COLUMN_NAMES,
    clean_adult_dataframe,
    split_adult_data,
    summarize_splits,
)


def _raw_frame(targets: list[str]) -> pd.DataFrame:
    row_count = len(targets)
    return pd.DataFrame(
        {
            "age": range(20, 20 + row_count),
            "workclass": [" Private "] * row_count,
            "fnlwgt": range(100_000, 100_000 + row_count),
            "education": [" Bachelors "] * row_count,
            "education-num": [13] * row_count,
            "marital-status": [" Never-married "] * row_count,
            "occupation": [" Prof-specialty "] * row_count,
            "relationship": [" Not-in-family "] * row_count,
            "race": [" White "] * row_count,
            "sex": [" Male "] * row_count,
            "capital-gain": [0] * row_count,
            "capital-loss": [0] * row_count,
            "hours-per-week": [40] * row_count,
            "native-country": [" United-States "] * row_count,
            "income": targets,
        },
        columns=ADULT_COLUMN_NAMES,
    )


def test_cleaning_maps_all_targets_and_normalizes_strings() -> None:
    raw = _raw_frame([" <=50K ", " >50K ", " <=50K. ", " >50K. "])
    raw.loc[1, "workclass"] = " ? "

    cleaned = clean_adult_dataframe(raw)

    assert cleaned["income"].tolist() == [0, 1, 0, 1]
    assert cleaned.loc[0, "workclass"] == "Private"
    assert pd.isna(cleaned.loc[1, "workclass"])
    assert cleaned.loc[0, "education"] == "Bachelors"


def test_split_is_deterministic_and_has_target_distributions() -> None:
    train = clean_adult_dataframe(_raw_frame(["<=50K", ">50K"] * 10))
    test = clean_adult_dataframe(_raw_frame(["<=50K.", ">50K."] * 2))

    first = split_adult_data(train, test, validation_fraction=0.2, seed=42)
    second = split_adult_data(train, test, validation_fraction=0.2, seed=42)

    assert_frame_equal(first.train, second.train)
    assert_frame_equal(first.validation, second.validation)
    assert_frame_equal(first.test, test.reset_index(drop=True))

    summary = summarize_splits(first)
    assert summary["rows"] == {"train": 16, "validation": 4, "test": 4}
    assert summary["columns"] == 15
    assert all(summary["target_distribution"][name] for name in summary["rows"])
