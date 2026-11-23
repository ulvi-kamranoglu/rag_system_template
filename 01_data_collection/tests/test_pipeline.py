from dataclasses import replace

import pytest

from wp_tracker.pipeline import _guard_deletions


def test_deletion_guard_blocks_abnormal_shrink(settings, record):
    old = [record(number=i) for i in range(1, 101)]
    deleted = old[:10]
    with pytest.raises(RuntimeError, match="Safety stop"):
        _guard_deletions(old, deleted, settings, False)


def test_deletion_guard_allows_small_shrink(settings, record):
    old = [record(number=i) for i in range(1, 101)]
    _guard_deletions(old, old[:3], settings, False)


def test_deletion_guard_can_be_overridden(settings, record):
    old = [record(number=1)]
    _guard_deletions(old, old, settings, True)


def test_settings_reject_bad_fraction(settings):
    from wp_tracker.config import load_settings
    from pathlib import Path
    config = settings.project_root / "config" / "settings.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("site:\n  base_url: https://unec.edu.az\nsafety:\n  max_deleted_fraction: 1.5\n")
    with pytest.raises(ValueError, match="fraction"):
        load_settings(config)
