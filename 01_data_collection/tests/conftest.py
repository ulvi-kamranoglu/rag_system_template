from dataclasses import replace
from pathlib import Path

import pytest

from wp_tracker.config import load_settings


@pytest.fixture
def settings(tmp_path: Path):
    source = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    original = load_settings(source)
    return replace(
        original,
        project_root=tmp_path,
        import_dir=tmp_path / "data/import",
        output_dir=tmp_path / "archive_data/01_raw",
        state_dir=tmp_path / "data/state",
        history_dir=tmp_path / "data/history",
        reports_dir=tmp_path / "data/state",
        shared_raw_dir=tmp_path / "shared_data/01_raw",
        delay_seconds=0,
    )


@pytest.fixture
def record():
    def create(kind="page", number=10, title="Old title", content="Old body", **kwargs):
        import json
        row = {
            "id": f"{kind}:{number}", "source_type": kind,
            "title": title, "content": content, "url": f"https://unec.edu.az/{number}",
            "date": None, "modified": None, "source": "unec.edu.az",
            "language": "az", "search_text": content,
            "metadata": json.dumps({"wp_id": str(number)}),
        }
        row.update(kwargs)
        return row
    return create
