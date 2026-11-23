"""News-only storage ownership and collection settings."""
from dataclasses import dataclass
from pathlib import Path
import math
import yaml

SITEMAP_URL = "https://news.unec.edu.az/sitemap.xml"


@dataclass(frozen=True)
class NewsSettings:
    root: Path
    archive_root: Path | None = None
    output_root: Path | None = None
    state_root: Path | None = None
    import_dir: Path | None = None
    delay_seconds: float = 1.0
    max_deleted_fraction: float = 0.05

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).resolve())
        if not math.isfinite(self.delay_seconds) or self.delay_seconds < 0:
            raise ValueError("News delay must be finite and nonnegative")
        if not 0 <= self.max_deleted_fraction <= 1:
            raise ValueError("News max_deleted_fraction must be between 0 and 1")

    @property
    def state_dir(self):
        return (self.state_root or self.root / "data/state/news") / "news.unec.edu.az"

    @property
    def baseline_file(self):
        return self.state_dir / "baseline.parquet"

    @property
    def snapshot_file(self):
        return self.state_dir / "snapshot.parquet"

    @property
    def report_file(self):
        return self.state_dir / "latest.json"

    @property
    def lock_file(self):
        return self.state_dir / "collector.lock"

    @property
    def archive_dir(self):
        return (self.archive_root or self.root.parent / "archive_data/01_raw/sources/news") / "news.unec.edu.az"

    @property
    def output_dir(self):
        # storage.recover() uses output_dir to tidy an empty run folder.
        return self.archive_dir

    @property
    def shared_dir(self):
        return (self.output_root or self.root / "data/output/news") / "news.unec.edu.az"

    @property
    def seed_file(self):
        return (self.import_dir or self.root / "data/import") / "news.parquet"


def load_settings(config_path):
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    news = config.get("news", {})
    if not isinstance(news, dict):
        raise ValueError("news must be a configuration mapping")
    news = dict(news)
    news.pop('enabled', None)
    if set(news) - {"delay_seconds"}:
        raise ValueError("Unknown news settings; only enabled and delay_seconds are configurable")
    from ..config import load_settings as load_parent
    parent = load_parent(path)
    return NewsSettings(root=parent.project_root, archive_root=parent.output_dir / 'sources/news',
                        output_root=parent.project_root / 'data/output/news',
                        state_root=parent.state_dir / 'news', import_dir=parent.import_dir,
                        max_deleted_fraction=parent.max_deleted_fraction, **news)
