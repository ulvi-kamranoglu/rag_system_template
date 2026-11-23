"""Regression cases for the expanded canonicalization rules; run manually."""
from pathlib import Path

import pandas as pd
import pytest

from canonical_pipeline.cleaning import clean_text, content_rejection
from canonical_pipeline.language import detect_language
from canonical_pipeline.pipeline import process


@pytest.mark.parametrize("text,expected", [
    ("Azərbaycan Dövlət İqtisad Universitetində tələbələr üçün təhsil imkanları haqqında məlumat verilir.", "az"),
    ("The university provides students with information about education and admission requirements.", "en"),
    ("Университет предоставляет студентам информацию об обучении и правилах поступления.", "ru"),
])
def test_text_language(text, expected):
    assert detect_language("", text)["language"] == expected


def test_encoded_html_scripts_and_invisible_unicode():
    value = "&lt;p&gt;Hello\u200bworld&lt;/p&gt;&lt;script&gt;bad()&lt;/script&gt;"
    assert clean_text(value) == "Hello world"
    assert clean_text("&amp;lt;p&amp;gt;Hello&amp;lt;/p&amp;gt;") == "Hello"
    assert clean_text("Cafe\u0301\u202e\x00") == "Café"
    assert clean_text("hello\u2009world") == "hello world"
    assert clean_text("FranÃ§ais") == "Français"


@pytest.mark.parametrize("text", ["", "www.example.com", "https://a.org https://b.org", "Hi", "!!!"])
def test_reject_unusable_content(text):
    assert content_rejection(text) is not None


def test_quality_then_dedup_keeps_identity_and_reports_source_rows(tmp_path):
    source = tmp_path / "chatbot_dataset_v1.3.parquet"
    body = "The university offers educational opportunities for international students."
    rows = [
        {"id": "1", "source_type": "page", "title": "Empty", "content": "<p></p>"},
        {"id": "2", "source_type": "page", "title": "Education", "content": f"<p>{body}</p>"},
        {"id": "3", "source_type": "post", "title": "Other", "content": body.upper()},
        {"id": "4", "source_type": "page", "title": "Education", "content": body},
        {"id": "5", "source_type": "page", "title": "", "content": "A different useful article about university admissions."},
        {"id": "6", "source_type": "form", "title": "Contact", "content": "www.example.com"},
        {"id": "7", "source_type": "form", "title": "Answer", "content": "Yes"},
    ]
    for row in rows:
        row.update(language="ru", metadata="{}", url="https://UNEC.edu.az/?utm_source=test")
    pd.DataFrame(rows).to_parquet(source)
    result = process([source], tmp_path / "archive", tmp_path / "shared")
    output = pd.read_parquet(result["output"])
    assert output["id"].tolist() == ["page:2", "post:3", "page:5"]
    assert set(output["language"]) == {"en"}
    assert output["search_text"].tolist() == output["content"].tolist()
    assert output["url"].tolist() == ["https://unec.edu.az/"] * 3
    assert result["input_rows"] == result["output_rows"] + result["rejected_rows"] + result["duplicate_removed_rows"]
    assert result["duplicate_removed_rows"] == 1
    removed = result["duplicates"]["removed_records"]
    assert {row["row_index"] for row in removed} == {3}
    assert all(row["kept_id"] == "page:2" and row["kept_row_index"] == 1 for row in removed)
    assert (Path(result["output"]).parent / "reports/text_quality_report.json").is_file()


def test_short_threshold_is_configurable_and_dry_run_is_read_only(tmp_path):
    source = tmp_path / "faq.parquet"
    pd.DataFrame([{"id": "1", "source_type": "form", "title": "Agreement", "content": "Yes"}]).to_parquet(source)
    before = source.read_bytes()
    result = process([source], tmp_path / "archive", tmp_path / "shared", True, min_content_chars=3)
    assert result["output_rows"] == 1
    assert source.read_bytes() == before
    assert not (tmp_path / "archive").exists()
    assert not (tmp_path / "shared").exists()


def test_duplicate_source_positions_after_rejections_and_custom_index(tmp_path, monkeypatch):
    monkeypatch.setattr("canonical_pipeline.pipeline.detector", lambda: object())
    monkeypatch.setattr("canonical_pipeline.pipeline.detect_language", lambda *args: {
        "language": "en", "confidence": 1.0, "uncertain": False})
    source = tmp_path / "indexed.parquet"
    bodies = ["", "Useful university information.", "www.example.com",
              "Useful university information.", "Different university information."]
    pd.DataFrame([{"id": str(i), "source_type": "form", "title": "Title",
                   "content": body} for i, body in enumerate(bodies)],
                 index=[100, 200, 300, 400, 500]).to_parquet(source)
    result = process([source], tmp_path / "archive", tmp_path / "shared", dry_run=True)
    assert result["output_rows"] == 2
    assert result["rejected_rows"] == 2
    removed = result["duplicates"]["removed_records"]
    assert len(removed) == 1
    assert removed[0]["row_index"] == 3
    assert removed[0]["kept_row_index"] == 1
    assert removed[0]["id"] == "form:3"
    assert removed[0]["kept_id"] == "form:1"
    assert not (tmp_path / "shared").exists()
