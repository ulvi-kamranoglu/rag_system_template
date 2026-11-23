"""Validated configuration with project-relative directories."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import yaml
from pipeline_storage import default_config, run_path, storage_paths, configuration_root


@dataclass(frozen=True)
class Settings:
    project_root: Path
    base_url: str
    default_language: str
    timezone: str
    user_agent: str
    per_page: int
    timeout_seconds: int
    retries: int
    delay_seconds: float
    max_pages_per_source: int
    max_deleted_fraction: float
    import_dir: Path
    output_dir: Path
    state_dir: Path
    history_dir: Path
    reports_dir: Path
    shared_raw_dir: Path

    @property
    def hostname(self) -> str:
        return urlparse(self.base_url).hostname or ""

    @property
    def baseline_file(self) -> Path:
        return self.state_dir / "baseline.parquet"

    @property
    def snapshot_file(self) -> Path:
        return self.state_dir / "snapshot.parquet"

    @property
    def lock_file(self) -> Path:
        return self.state_dir / "tracker.lock"

    def new_run_dir(self, kind: str = "run") -> Path:
        """Allocate a unique path; no directory is created until commit."""
        if kind not in {"run", "bootstrap"}:
            raise ValueError(f"Unsupported run kind: {kind}")
        return run_path(self.output_dir)

    def output_file(self, event: str, run_dir: Path | None = None) -> Path:
        """Resolve an explicit run output or the latest successfully committed one.

        Without a new commit pointer, look for the centralized shared filename.
        """
        if event not in {"new", "deleted"}:
            raise ValueError(f"Unsupported event type: {event}")
        filename = f"chatbot_dataset_{event}.parquet"
        if run_dir is not None:
            return Path(run_dir) / filename
        report_file = self.reports_dir / "latest.json"
        if report_file.is_file():
            report = json.loads(report_file.read_text(encoding="utf-8"))
            saved_path = report.get("outputs", {}).get(event)
            if saved_path:
                candidate = Path(saved_path)
                if not candidate.is_absolute():
                    candidate = self.project_root / candidate
                candidate = candidate.resolve()
                if not candidate.is_relative_to(self.output_dir.resolve()) or candidate.name != filename:
                    raise ValueError(f"Invalid output path in latest report: {candidate}")
                return candidate
        return self.shared_raw_dir / filename


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path).resolve() if path else default_config(__file__, "config/settings.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")
    site = cfg.get("site", {})
    net = cfg.get("network", {})
    safety = cfg.get("safety", {})
    storage = cfg.get("storage", {})
    root = configuration_root(config_path, __file__, "01_data_collection")
    if cfg.get("source_project_root"):
        root = Path(cfg["source_project_root"]).resolve()
    archive_dir, shared_dir = storage_paths(root, cfg, "01_raw")

    def directory(key: str, default: str) -> Path:
        path_value = Path(storage.get(key, default))
        return path_value if path_value.is_absolute() else root / path_value

    url = str(site.get("base_url", "https://unec.edu.az")).rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("site.base_url must be a valid HTTPS URL without credentials")
    tz = str(site.get("timezone", "Asia/Baku"))
    ZoneInfo(tz)
    settings = Settings(
        project_root=root,
        base_url=url,
        default_language=str(site.get("default_language", "az")),
        timezone=tz,
        user_agent=str(site.get("user_agent", "UNECWordPressChangeTracker/1.0")),
        per_page=int(net.get("per_page", 100)),
        timeout_seconds=int(net.get("timeout_seconds", 30)),
        retries=int(net.get("retries", 3)),
        delay_seconds=float(net.get("delay_seconds", 0.25)),
        max_pages_per_source=int(net.get("max_pages_per_source", 500)),
        max_deleted_fraction=float(safety.get("max_deleted_fraction", 0.05)),
        import_dir=directory("import_dir", "data/import"),
        output_dir=archive_dir,
        state_dir=directory("state_dir", "data/state"),
        history_dir=directory("history_dir", "data/history"),
        reports_dir=directory("state_dir", "data/state"),
        shared_raw_dir=shared_dir,
    )
    if not 1 <= settings.per_page <= 100:
        raise ValueError("network.per_page must be between 1 and 100")
    if settings.timeout_seconds <= 0 or settings.retries < 0 or settings.delay_seconds < 0:
        raise ValueError("Invalid network settings")
    if settings.max_pages_per_source <= 0:
        raise ValueError("network.max_pages_per_source must be positive")
    if not 0 <= settings.max_deleted_fraction <= 1:
        raise ValueError("safety.max_deleted_fraction must be between 0 and 1")
    return settings
