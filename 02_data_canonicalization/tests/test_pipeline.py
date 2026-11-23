from pathlib import Path
import pandas as pd
import pytest

pytest.importorskip("pyarrow", reason="Parquet integration tests require pyarrow")
from canonical_pipeline.pipeline import process


def sample(row_id: str) -> pd.DataFrame:
    return pd.DataFrame([{"id": row_id, "source_type": "page", "title": "Title", "content": "<p>First</p><p>Second</p>", "metadata": "{}", "language": "az"}])


def test_single_and_dry_run(tmp_path: Path):
    path = tmp_path / "a.parquet"
    sample("1").to_parquet(path)
    output = tmp_path / "shared/a_canonical.parquet"
    report = process([path], tmp_path / "archive", output.parent, dry_run=True)
    assert report["mode"] == "single"
    assert not output.exists()
    report = process([path], tmp_path / "archive", output.parent)
    assert report["output_rows"] == 1
    df = pd.read_parquet(output)
    assert df.iloc[0]["search_text"] == "First\n\nSecond"


def test_multiple_inputs_merge(tmp_path: Path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    sample("1").to_parquet(a)
    sample("2").to_parquet(b)
    out = tmp_path / "shared/a__b_canonical.parquet"
    report = process([a, b], tmp_path / "archive", out.parent)
    assert report["mode"] == "merge"
    assert len(pd.read_parquet(out)) == 1
    assert report["duplicate_removed_rows"] == 1
