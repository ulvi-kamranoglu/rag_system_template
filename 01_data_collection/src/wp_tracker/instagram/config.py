"""Separate Instagram state/archives/shared directories to avoid WordPress overwrite."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

from .schema import profile_name


DEFAULT_PROFILES = ["uneceduaz", "adalat.muradov"]


@dataclass(frozen=True)
class InstagramSettings:
    root: Path
    username: str
    archive_dir: Path
    shared_dir: Path
    state_dir: Path
    auth_file: Path
    max_scrolls: int = 40
    # Consecutive non-pinned posts not newer than the cutoff that end a scan.
    # Must exceed Instagram's limit of three pinned posts.
    boundary_old_posts: int = 4
    boundary_quiet_scrolls: int = 3

    @property
    def snapshot_file(self):
        return self.state_dir / "snapshot.parquet"

    @property
    def baseline_file(self):
        return self.state_dir / "baseline.parquet"

    @property
    def cutoff_file(self):
        return self.state_dir / "cutoff.json"

    @property
    def pending_file(self):
        return self.state_dir / "pending.jsonl"

    @property
    def skipped_file(self):
        return self.state_dir / "skipped.jsonl"

    @property
    def report_file(self):
        return self.state_dir / "latest.json"

    @property
    def lock_file(self):
        return self.state_dir / "collector.lock"

    @property
    def output_dir(self):
        return self.archive_dir


def _instagram_mapping(config_path: Path) -> dict:
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    ig = cfg.get("instagram", {})
    if not isinstance(ig, dict):
        raise ValueError("instagram config must be a mapping")
    return ig


def listed_profiles(config_path: Path) -> list[str]:
    ig = _instagram_mapping(config_path)
    profiles = ig.get("profiles", DEFAULT_PROFILES)
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("instagram.profiles must be a nonempty list")
    return [profile_name(name) for name in profiles]


def project_root(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    if config.get('source_project_root'):
        return Path(config['source_project_root']).resolve()
    config_path = config_path.resolve()
    root = config_path.parent.parent
    if root.name == "wp_tracker" and root.parent.name == "src":
        root = root.parent.parent.parent
    if config_path.parent.name == "config" and config_path.parent.parent.name == "wp_tracker":
        root = config_path.parents[3]
    return root


def load_ig_settings(config_path: Path, username: str) -> InstagramSettings:
    config_path = config_path.resolve()
    ig = _instagram_mapping(config_path)
    profiles = listed_profiles(config_path)
    username = profile_name(username)
    if username not in profiles:
        raise ValueError(f"Instagram profile {username} not allowed by config: {profiles}")
    root = project_root(config_path)

    def resolve(key, default):
        given = Path(ig.get(key, default))
        return given.resolve() if given.is_absolute() else (root / given).resolve()

    from ..config import load_settings
    parent = load_settings(config_path)
    state = resolve("state_dir", str(parent.state_dir / "instagram")) / username
    auth_root = resolve("auth_dir", "data/instagram_auth")
    boundary = int(ig.get("boundary_old_posts", 4))
    if boundary < 4:
        raise ValueError("instagram.boundary_old_posts must be at least 4 (Instagram allows up to 3 pinned posts)")
    return InstagramSettings(
        root=root, username=username,
        archive_dir=parent.output_dir / "sources/instagram" / username,
        shared_dir=parent.project_root / "data/output/instagram" / username,
        state_dir=state,
        auth_file=auth_root / f"{username}.json",
        max_scrolls=int(ig.get("max_scrolls", 40)),
        boundary_old_posts=boundary,
        boundary_quiet_scrolls=int(ig.get("boundary_quiet_scrolls", 3)),
    )
