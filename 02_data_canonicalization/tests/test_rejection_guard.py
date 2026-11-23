"""Rejected records must stop the stage instead of disappearing silently."""
import json

import pandas as pd
import pytest

from canonical_pipeline.pipeline import process, quality_options, rejection_guard

BODY = "Useful university information about admissions and study programmes."


@pytest.fixture(autouse=True)
def fixed_language(monkeypatch):
    monkeypatch.setattr("canonical_pipeline.pipeline.detector", lambda: object())
    monkeypatch.setattr("canonical_pipeline.pipeline.detect_language", lambda *args: {
        "language": "az", "confidence": 1.0, "uncertain": False})


def legal_row(identifier, clause, number):
    return dict(id=identifier, source_type="legal", source="e-qanun.az", title="Nizamnamə",
                content=f"{BODY} {number}", url="https://e-qanun.az/framework/43163",
                language="az", metadata=json.dumps({"clause_id": clause}))


def write(tmp_path, rows):
    source = tmp_path / "batch.parquet"
    pd.DataFrame(rows).to_parquet(source)
    return source


def test_identity_rejection_stops_before_archive(tmp_path):
    rows = [legal_row("legal:43163:1.1", "1.1", 1), legal_row("legal:43163:1.2", "bad", 2)]
    with pytest.raises(ValueError, match="Rejection guard.*invalid identity.*legal:43163:1.2"):
        process([write(tmp_path, rows)], tmp_path / "archive", tmp_path / "shared")
    archive = tmp_path / "archive"
    assert not list(archive.glob("run_*")) and not list(archive.glob("base_data/*.parquet"))
    assert not list((tmp_path / "shared").glob("*.parquet"))


def test_dry_run_reports_guard_without_raising(tmp_path):
    rows = [legal_row("legal:43163:1.1", "1.1", 1), legal_row("legal:43163:1.2", "bad", 2)]
    result = process([write(tmp_path, rows)], tmp_path / "archive", tmp_path / "shared", True)
    guard = result["rejection_guard"]
    assert guard["identity_rejections"] == 1 and guard["content_rejections"] == 0
    assert guard["violations"]
    assert result["rejected_samples"][0]["category"] == "identity"


def test_identity_tolerance_is_configurable(tmp_path):
    rows = [legal_row("legal:43163:1.1", "1.1", 1), legal_row("legal:43163:1.2", "bad", 2)]
    result = process([write(tmp_path, rows)], tmp_path / "archive", tmp_path / "shared",
                     max_identity_rejections=1)
    assert result["output_rows"] == 1 and result["rejection_guard"]["violations"] == []


def test_small_number_of_short_posts_is_tolerated(tmp_path):
    rows = [dict(id=f"news:{i}", source_type="news", title="", content=BODY + str(i), language="az",
                 metadata="{}", url="") for i in range(1, 4)]
    rows.append(dict(id="news:9", source_type="news", title="", content="Hi", language="az", metadata="{}", url=""))
    result = process([write(tmp_path, rows)], tmp_path / "archive", tmp_path / "shared")
    assert result["output_rows"] == 3
    assert result["rejection_guard"]["content_rejections"] == 1


def test_content_ratio_limit():
    content = [{"category": "content"}] * 6
    assert rejection_guard(content, 100)["violations"]            # limit max(5, 5) = 5
    assert not rejection_guard(content, 200)["violations"]        # limit max(5, 10) = 10
    assert not rejection_guard(content, 100, content_rejection_tolerance=6)["violations"]


def test_quality_options_defaults_and_overrides():
    assert quality_options({}) == {"min_content_chars": 6, "max_identity_rejections": 0,
                                   "max_content_rejection_ratio": 0.05, "content_rejection_tolerance": 5}
    assert quality_options({"quality": {"max_identity_rejections": 3}})["max_identity_rejections"] == 3


@pytest.mark.parametrize("options", [dict(max_identity_rejections=-1), dict(max_content_rejection_ratio=2),
                                     dict(content_rejection_tolerance=True)])
def test_invalid_guard_settings_rejected(tmp_path, options):
    with pytest.raises(ValueError, match="must be"):
        process([write(tmp_path, [legal_row("legal:43163:1.1", "1.1", 1)])],
                tmp_path / "archive", tmp_path / "shared", True, **options)
