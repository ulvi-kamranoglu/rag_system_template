"""Batch orchestration, quality reporting and atomic output publishing."""
import logging
from collections import Counter
from pathlib import Path
import pandas as pd
from pipeline_storage import archive_run, publish, recover_publication, run_path, write_json, validate_json, batch_stem
import json
import re
import os
import shutil
from tempfile import TemporaryDirectory
from pipeline_storage.documents import deleted_ids, document_id
from .schema import COLUMNS, standardize
from .cleaning import clean_text, content_rejection
from .identity import canonicalize_row, canonical_identity
from .duplicates import duplicate_report, remove_duplicates
from .language import detect_language, detector
from filelock import FileLock
from pipeline_storage.versioning import latest, next_name, version, commit_base

LOG = logging.getLogger(__name__)


def identity_alias(row):
    """Compare a legacy baseline ID to a newly normalized upload safely."""
    try:
        kind, _, identifier = canonical_identity(row)
        return kind, identifier
    except (ValueError, TypeError):
        return row['source_type'], row['id']


QUALITY_OPTIONS = {"min_content_chars": 6, "max_identity_rejections": 0,
                   "max_content_rejection_ratio": 0.05, "content_rejection_tolerance": 5}


def quality_options(config: dict) -> dict:
    """process() keyword arguments from the YAML quality section."""
    quality = (config or {}).get("quality") or {}
    return {key: quality.get(key, default) for key, default in QUALITY_OPTIONS.items()}


def rejection_guard(rejected, input_rows, *, max_identity_rejections=0,
                    max_content_rejection_ratio=0.05, content_rejection_tolerance=5):
    """Fail loudly instead of silently dropping records.

    Identity/metadata errors are structural, so the default allows none. Low-value
    content is normal (historically under 0.3%); a small tolerance keeps short
    batches from stopping on one or two empty posts.
    """
    identity = sum(entry.get("category") == "identity" for entry in rejected)
    content = len(rejected) - identity
    content_limit = max(content_rejection_tolerance, int(max_content_rejection_ratio * input_rows))
    violations = []
    if identity > max_identity_rejections:
        violations.append(f"{identity} record(s) have invalid identity/metadata (allowed {max_identity_rejections}).")
    if content > content_limit:
        violations.append(f"{content} of {input_rows} record(s) rejected for content (allowed {content_limit}).")
    return {"identity_rejections": identity, "content_rejections": content,
            "max_identity_rejections": max_identity_rejections, "content_rejection_limit": content_limit,
            "violations": violations}


def validate_run(folder: Path) -> None:
    report = json.loads((folder / "run_report.json").read_text(encoding="utf-8"))
    filename = report.get("output_file", "canonical_dataset.parquet")
    if Path(filename).name != filename:
        raise ValueError("Invalid canonical output filename")
    frame = pd.read_parquet(folder / filename)
    if list(frame.columns) != COLUMNS:
        raise ValueError("Invalid canonical output schema")
    if report.get("base_source_dir") == "base_snapshot":
        base = pd.read_parquet(folder / "base_snapshot" / report["base_output_file"])
        if list(base.columns) != COLUMNS or len(base) != report["base_output_rows"]:
            raise ValueError("Invalid cumulative canonical snapshot")
    validate_json(folder, ["run_report.json", "reports/quality_report.json",
                           "reports/rejected_records.json", "reports/duplicate_report.json"])
    if report.get("text_quality") is not None:
        validate_json(folder, ["reports/text_quality_report.json"])


def process(inputs, archive_dir, shared_dir, dry_run=False, **kwargs):
    # Reserve a version throughout processing, archival and publication.
    if not dry_run:
        Path(archive_dir).mkdir(parents=True, exist_ok=True)
        with FileLock(str(Path(archive_dir) / ".canonical.lock")):
            recover_publication(Path(shared_dir))
            return _process(inputs, archive_dir, shared_dir, dry_run, **kwargs)
    return _process(inputs, archive_dir, shared_dir, dry_run, **kwargs)


def _publish_canonical(run, shared, *, migrate_legacy=False):
    """Commit the complete version, then publish only the cleaned batch."""
    report = json.loads((run / "run_report.json").read_text(encoding="utf-8"))
    if report.get("base_source_dir") == "base_snapshot":
        commit_base(run, report)
        # The run retains the complete version for crash recovery; shared exposes
        # only the cleaned batch and its reports.
        with TemporaryDirectory(prefix=".canonical_publish_", dir=run.parent) as temporary:
            delivery = Path(temporary)
            for path in run.iterdir():
                if path.name == "base_snapshot":
                    continue
                if path.is_dir():
                    shutil.copytree(path, delivery / path.name)
                else:
                    shutil.copy2(path, delivery / path.name)
            return publish(delivery, shared, validate_batch, migrate_legacy=migrate_legacy)
    if report.get("merge_base") or report.get("reset_data") or report.get("base_dir"):
        existing = list(shared.glob("*_canonical.parquet"))
        if existing:
            current = latest(shared, "_canonical.parquet")
            wanted = version(report["output_file"][:-len("_canonical.parquet")])
            active = version(current.name[:-len("_canonical.parquet")])
            if wanted[0] != active[0] or wanted[1:] < active[1:]:
                raise ValueError("Refusing to publish an older or unrelated canonical version")
            if wanted == active and (run / report["output_file"]).read_bytes() != current.read_bytes():
                raise ValueError("Canonical version already exists with different content")
    filename = report["output_file"]
    # Keep immutable version copies for canonicalization only, including archives
    # produced before this history mirror was restored. Unversioned standalone
    # outputs continue to publish without entering the baseline history.
    if re.fullmatch(r".+_v\d+\.\d+_canonical\.parquet", filename):
        commit_base(run, {"base_dir": str(run.parent / "base_data"), "base_outputs": [filename]})
    return publish(run, shared, validate_run, migrate_legacy=migrate_legacy)


def validate_batch(folder):
    report = json.loads((folder / "run_report.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(folder / report["output_file"])
    if list(frame.columns) != COLUMNS or len(frame) != report["output_rows"]:
        raise ValueError("Invalid canonical batch")
    validate_json(folder, ["reports/quality_report.json", "reports/rejected_records.json", "reports/duplicate_report.json"])


def publish_canonical(run, shared, *, migrate_legacy=False):
    """Retry publication without merging or modifying historical archives."""
    with FileLock(str(run.parent / ".canonical.lock")):
        recover_publication(shared)
        validate_run(run)
        return _publish_canonical(run, shared, migrate_legacy=migrate_legacy)


def _process(inputs: list[Path], archive_dir: Path, shared_dir: Path, dry_run: bool = False,
            *, migrate_legacy: bool = False, min_content_chars: int = 6,
            merge_base: bool = False, reset_data: bool = False, deleted_path=None,
            max_identity_rejections: int = 0, max_content_rejection_ratio: float = 0.05,
            content_rejection_tolerance: int = 5) -> dict:
    if reset_data and (merge_base or deleted_path is not None):
        raise ValueError("Reset data cannot be combined with base merging or deletion files")
    if isinstance(min_content_chars, bool) or not isinstance(min_content_chars, int) or min_content_chars < 1:
        raise ValueError("min_content_chars must be a positive integer")
    for name, value in (("max_identity_rejections", max_identity_rejections),
                        ("content_rejection_tolerance", content_rejection_tolerance)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if isinstance(max_content_rejection_ratio, bool) or not isinstance(max_content_rejection_ratio, (int, float)) \
            or not 0 <= max_content_rejection_ratio <= 1:
        raise ValueError("max_content_rejection_ratio must be between 0 and 1")
    archive_dir, shared_dir = archive_dir.resolve(), shared_dir.resolve()
    if archive_dir == shared_dir or archive_dir.is_relative_to(shared_dir) or shared_dir.is_relative_to(archive_dir):
        raise ValueError("Archive and shared roots must not overlap")
    run_dir = run_path(archive_dir)
    if not inputs:
        raise ValueError("At least one input is required")
    normalized_inputs = [path.resolve() for path in inputs]
    output_file = f"{batch_stem(inputs)}_canonical.parquet"
    if len(set(normalized_inputs)) != len(normalized_inputs):
        raise ValueError("Duplicate input paths are not allowed")
    if any(path.is_relative_to(shared_dir) or path.is_relative_to(archive_dir) for path in normalized_inputs):
        raise ValueError("Output must not overwrite an input or its containing directory")
    # Check once before reading datasets; the detector is reused for every row.
    detector()
    raw_frames = []
    for path in inputs:
        LOG.info("Reading %s", path)
        raw_frames.append(standardize(pd.read_parquet(path)))
    raw = pd.concat(raw_frames, ignore_index=True) if len(raw_frames) > 1 else raw_frames[0]
    # Reports use zero-based source row positions, regardless of Parquet index labels.
    raw = raw.reset_index(drop=True)
    clean_rows, rejected, accepted_indices = [], [], []
    text_quality = {"normalized_rows": 0, "language_reassigned_rows": 0,
                    "uncertain_language_rows": 0, "replacement_character_rows": 0,
                    "html_like_text_rows": 0,
                    "warnings": []}
    for index, row in raw.iterrows():
        category = "content"
        try:
            item = row.copy()
            item["title"] = clean_text(item["title"])
            item["content"] = clean_text(item["content"])
            if item["title"] != row["title"] or item["content"] != row["content"]:
                text_quality["normalized_rows"] += 1
            reason = content_rejection(item["content"], min_content_chars)
            if reason:
                raise ValueError(reason)
            detected = detect_language(item["title"], item["content"])
            if detected["language"] is None:
                raise ValueError("Language cannot be determined from text")
            if detected["language"] != row["language"].strip().lower():
                text_quality["language_reassigned_rows"] += 1
            item["language"] = detected["language"]
            issues = []
            if detected["uncertain"]:
                text_quality["uncertain_language_rows"] += 1
                issues.append("uncertain_language")
            if "\ufffd" in item["title"] + item["content"]:
                text_quality["replacement_character_rows"] += 1
                issues.append("unicode_replacement_character")
            if re.search(r"</?[a-zA-Z][^>]*>", item["title"] + item["content"]):
                text_quality["html_like_text_rows"] += 1
                issues.append("html_like_text_after_cleaning")
            if issues:
                text_quality["warnings"].append({"row_index": int(index), "id": str(row["id"]),
                    "issues": issues, "detected_language": detected["language"],
                    "confidence": detected["confidence"]})
            # Identity/metadata errors mean the batch format is wrong, not that a
            # record is low value; the guard below stops instead of dropping them.
            category = "identity"
            item = canonicalize_row(item)
            item["search_text"] = item["content"]
            clean_rows.append(item)
            accepted_indices.append(int(index))
        except (ValueError, TypeError) as exc:
            rejected.append({"row_index": int(index), "id": str(row["id"]), "source_type": str(row["source_type"]),
                             "category": category, "reason": str(exc)})
    # Series.name retains source labels after rejection; deduplication must return
    # positions in accepted_indices rather than those sparse source labels.
    clean = pd.DataFrame(clean_rows, columns=COLUMNS).reset_index(drop=True)
    deleted = deleted_ids(deleted_path)
    if deleted_path is not None:
        for row in pd.read_parquet(deleted_path).to_dict('records'):
            if row.get('id') is not None and row.get('source_type') is not None:
                kind, identifier = identity_alias(row)
                deleted.add(document_id({**row, 'id': identifier, 'source_type': kind, 'document_id': None}))
    deleted_rows = 0
    if deleted:
        keep = [document_id(row) not in deleted for row in clean.to_dict("records")]
        deleted_rows += len(keep) - sum(keep)
        accepted_indices = [value for value, retain in zip(accepted_indices, keep) if retain]
        clean = clean.loc[keep].reset_index(drop=True)
    duplicate_stats = duplicate_report(clean)
    clean, removed_duplicates = remove_duplicates(clean)
    for entry in removed_duplicates:
        entry["row_index"] = accepted_indices[entry["row_index"]]
        entry["kept_row_index"] = accepted_indices[entry["kept_row_index"]]
    duplicate_stats["removed_rows"] = len(removed_duplicates)
    duplicate_stats["removed_records"] = removed_duplicates
    duplicate_stats["after_deduplication"] = duplicate_report(clean)
    duplicate_stats["policy"] = "normalized title+content+url only; keep first valid input row"
    incoming_rows = len(clean)
    base_file = None
    batch_clean = clean.copy()
    base_output_file = None
    if merge_base or reset_data:
        base_file = latest(archive_dir / "base_data", "_canonical.parquet")
        base_output_file = next_name(base_file, "_canonical.parquet") + "_canonical.parquet"
    if merge_base:
        base = standardize(pd.read_parquet(base_file))
        if deleted:
            keep = []
            for row in base.to_dict('records'):
                kind, identifier = identity_alias(row)
                alias = document_id({**row, 'id': identifier, 'source_type': kind})
                keep.append(document_id(row) not in deleted and alias not in deleted)
            deleted_rows += len(keep) - sum(keep)
            base = base.loc[keep].reset_index(drop=True)
        keys = ["source_type", "id"]
        if clean.duplicated(keys).any() or base.duplicated(keys).any():
            raise ValueError("Conflicting document identities; resolve duplicate source_type/id before merge")
        incoming_keys = set(map(tuple, clean[keys].itertuples(index=False, name=None)))
        retain = [identity_alias(row) not in incoming_keys for row in base.to_dict('records')]
        clean, merge_removed = remove_duplicates(pd.concat([clean, base.loc[retain]], ignore_index=True))
        duplicate_stats["merge_removed_rows"] = len(merge_removed)
        duplicate_stats["after_deduplication"] = duplicate_report(clean)
    report = {
        "pipeline_run_id": os.getenv("PIPELINE_RUN_ID"),
        "deleted_documents": len(deleted),
        "deleted_rows": deleted_rows,
        "input_files": [str(p) for p in inputs],
        "mode": "reset_data" if reset_data else "merge" if len(inputs) > 1 else "single",
        "input_rows": int(len(raw)),
        "incoming_clean_rows": incoming_rows,
        "base_input": str(base_file) if base_file else None,
        "base_dir": str(archive_dir / "base_data") if base_output_file else None,
        "base_output_file": base_output_file,
        "base_source_dir": "base_snapshot" if base_output_file else None,
        "base_outputs": [base_output_file] if base_output_file else [],
        "base_output_rows": int(len(clean)),
        "merge_base": merge_base,
        "reset_data": reset_data,
        "output_rows": int(len(batch_clean)),
        "rejected_rows": len(rejected),
        "rejection_reason_counts": dict(Counter(entry["reason"] for entry in rejected)),
        "rejected_samples": rejected[:5],
        "duplicate_removed_rows": len(removed_duplicates),
        "min_content_chars": min_content_chars,
        "text_quality": {key: value for key, value in text_quality.items() if key != "warnings"},
        "source_type_counts": batch_clean["source_type"].value_counts().to_dict() if not batch_clean.empty else {},
        "language_counts": batch_clean["language"].value_counts().to_dict() if not batch_clean.empty else {},
        "duplicates": duplicate_stats,
        "dry_run": dry_run,
        "run_dir": str(run_dir),
        "archive_dir": str(archive_dir),
        "shared_dir": str(shared_dir),
        "output_file": output_file,
        "output": str(shared_dir / output_file),
    }
    guard = rejection_guard(rejected, len(raw), max_identity_rejections=max_identity_rejections,
                            max_content_rejection_ratio=max_content_rejection_ratio,
                            content_rejection_tolerance=content_rejection_tolerance)
    report["rejection_guard"] = guard
    if guard["violations"] and not dry_run:
        samples = "; ".join(f"{entry['id']} ({entry['source_type']}): {entry['reason']}"
                            for entry in rejected[:5])
        raise ValueError("Rejection guard stopped canonicalization; nothing was archived or published. "
                         + " ".join(guard["violations"]) + f" Samples: {samples}")
    if reset_data and incoming_rows == 0:
        raise ValueError("Reset data requires at least one accepted record; output not published")
    if incoming_rows == 0 and len(raw) and not deleted_rows:
        raise ValueError("All records were rejected; output not published")
    if not dry_run:
        with archive_run(run_dir, validate_run) as staging:
            batch_clean.to_parquet(staging / output_file, index=False, engine="pyarrow")
            if base_output_file:
                (staging / "base_snapshot").mkdir()
                clean.to_parquet(staging / "base_snapshot" / base_output_file, index=False, engine="pyarrow")
            write_json(staging / "reports/quality_report.json", report)
            write_json(staging / "reports/rejected_records.json", rejected)
            write_json(staging / "reports/duplicate_report.json", duplicate_stats)
            write_json(staging / "reports/text_quality_report.json", text_quality)
            write_json(staging / "run_report.json", report)
        try:
            report["publication"] = _publish_canonical(run_dir, shared_dir, migrate_legacy=migrate_legacy)
        except Exception as exc:
            raise RuntimeError(f"Archive complete at {run_dir}; publication failed. "
                               f"Retry with --publish-run {run_dir}: {exc}") from exc
    return report
