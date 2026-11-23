from datetime import datetime, timezone
import pandas as pd
import pytest
from canonical_pipeline.schema import parse_datetime, standardize


@pytest.mark.parametrize("value", [1566432000, 1566432000000, 1566432000000000,
                                   1566432000000000000, "1566432000", "1566432000000",
                                   "2019-08-22", datetime(2019, 8, 22, tzinfo=timezone.utc)])
def test_epoch_units_and_iso_dates_agree(value):
    assert parse_datetime(value) == pd.Timestamp("2019-08-22", tz="UTC")


def test_mixed_seconds_and_milliseconds_are_parsed_per_record():
    frame = pd.DataFrame({"id": [1, 2, 3], "source_type": ["legal"] * 3, "content": ["text"] * 3,
                          "date": pd.Series([1566432000, 1566432000000, "2013-12-24"], dtype=object),
                          "modified": pd.Series([1791551996017, None, "2026-01-01"], dtype=object)})
    result = standardize(frame)
    assert result["date"].dt.year.tolist() == [2019, 2019, 2013]
    assert result["modified"].iloc[0] == pd.Timestamp("2026-10-09T13:19:56.017Z")


@pytest.mark.parametrize("value", [None, "invalid", float("nan"), float("inf"), True])
def test_invalid_dates_are_missing(value):
    assert pd.isna(parse_datetime(value))


def test_typed_1970_date_is_not_guessed_to_be_corrupt():
    value = pd.Timestamp("1970-01-19T03:07:12Z")
    assert parse_datetime(value) == value
    assert parse_datetime(1791551996017123456) == pd.Timestamp("2026-10-09T13:19:56.017123456Z")
