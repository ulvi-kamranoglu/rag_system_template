"""Baseline comparison is stable even when a current snapshot advances."""
from wp_tracker.diff import compare
from wp_tracker.records import FIELDS, dataset


def test_outputs_have_only_common_columns(record):
    assert list(dataset([record()], "new").columns) == FIELDS
    assert list(dataset([record()], "deleted").columns) == FIELDS


def test_id_deltas_against_fixed_baseline_remain_stable(record):
    baseline = [record(number=1, title="Original")]
    latest = [record(number=1, title="Changed"), record(number=2)]
    assert [x["id"] for x in compare(baseline, latest).new] == ["page:2"]
    assert [x["id"] for x in compare(baseline, latest).new] == ["page:2"]
    assert compare(baseline, latest).deleted == []


def test_reject_unsupported_event_type(settings):
    import pytest
    with pytest.raises(ValueError, match="Unsupported"):
        settings.output_file("unsupported_event")
    with pytest.raises(ValueError, match="Unsupported"):
        dataset([], "unsupported_event")
