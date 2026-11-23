"""Configurable query routing and result selection; no model or network calls."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import math
import re
import unicodedata


DEFAULTS = {
    "enabled": True, "deduplicate": True, "diversify_documents": True,
    "max_chunks_per_document": 1, "candidate_multiplier": 5,
    "recent_candidates": True, "recent_730_limit": 20, "recent_365_limit": 10,
    "metadata_ranking": True, "freshness_half_life_days": 365,
    "freshness_weight": 0.0, "page_weight": 0.10,
}

RECENT_RERANKER_EXTRA_LIMIT = 10


def reranker_candidates(results, policy, routing, limit, now):
    """Keep primary relevance candidates, then admit up to ten recent extras."""
    primary, dropped = select(results, policy, limit=limit)
    if not policy["recent_candidates"] or routing["temporal_guard"] != "default":
        return primary, primary, [], dropped
    keys = {item["chunk_id"] for item in primary}
    cutoff = now - timedelta(days=730)
    recent = []
    for item in results:
        if item["chunk_id"] in keys:
            continue
        date, _ = ranking_date(item.get("payload", {}))
        if date is not None and cutoff <= date <= now:
            recent.append(item)
    combined, extra_dropped = select(primary + recent, policy,
                                     limit=len(primary) + RECENT_RERANKER_EXTRA_LIMIT)
    extras = combined[len(primary):]
    return combined, primary, extras, dropped + extra_dropped


def settings(raw=None):
    raw = {} if raw is None else raw
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError("ranking must be a mapping containing known policy options")
    result = {**DEFAULTS, **raw}
    for key, default in DEFAULTS.items():
        value = result[key]
        if isinstance(default, bool):
            valid = isinstance(value, bool)
        elif isinstance(default, int):
            valid = isinstance(value, int) and not isinstance(value, bool) and value > 0
        else:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1
        if not valid:
            raise ValueError(f"Invalid ranking.{key}")
    return result


def normalize(value):
    return " ".join(unicodedata.normalize("NFC", str(value or "")).casefold().split())


def route(query):
    text = normalize(query)
    guard = "default"
    if re.search(r"\b(?:19|20)\d{2}\b", text):
        guard = "explicit_year"
    elif any(cue in text for cue in ("konfrans", "tədbir", "yarış", "conference", "конференц", "keçirilmiş", "keçirilən görüş")):
        guard = "specific_event"
    elif any(cue in text for cue in ("əvvəlki", "keçmiş", "tarixç", "unec-in tarixi", "unec tarixi",
                                    "universitetin tarixi", "history", "historical", "бывш", "истори")):
        guard = "historical"
    groups = (
        ("admission", ("qəbul", "abituriyent", "admission", "поступлен")),
        ("catalog", ("ixtisas", "proqram", "specialit", "program", "специальност")),
        ("digital", ("elektron", "onlayn", "kabinet", "online", "digital", "электрон")),
        ("structure", ("rektor", "dekan", "rəhbər", "struktur", "rector", "dean", "ректор")),
        ("location", ("ünvan", "yerləş", "address", "located", "адрес")),
    )
    intent = next((name for name, cues in groups if any(cue in text for cue in cues)), "general")
    windows = [730] if intent in {"admission", "catalog", "structure"} else [730, 365] if intent in {"digital", "location"} else []
    return {"intent": intent, "temporal_guard": guard,
            "recent_windows": windows if guard == "default" else [],
            "metadata_eligible": guard == "default" and intent in {"admission", "catalog", "structure", "digital"}}


def ranking_date(payload):
    """Current reference content uses modification; news uses publication only."""
    current = normalize(payload.get("source_type")) in {"page", "form", "research", "legal"}
    for key in (("modified", "date") if current else ("date",)):
        value = payload.get(key)
        if value is None:
            continue
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            parsed = parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
            return parsed, key
        except (ValueError, TypeError):
            pass
    return None, None


def timestamp(payload):
    return ranking_date(payload)[0]


def recent_document_ids(records, days, now):
    cutoff = now - timedelta(days=days)
    return sorted({str(row["document_id"]) for row in records
                   if (date := timestamp(row)) is not None and cutoff <= date <= now})


def select(results, policy, *, limit=None, diversify=True):
    selected, dropped, seen, counts = [], [], set(), Counter()
    for item in results:
        payload = item.get("payload", {})
        chunk = item.get("chunk_id") or payload.get("chunk_id") or item.get("id")
        text, url = normalize(payload.get("chunk_text")), str(payload.get("url") or "").strip()
        # Keep URL path case; blank URLs or empty text are not evidence of duplicates.
        key = (url, text) if url and text else ("chunk", chunk) if chunk else None
        doc = payload.get("document_id") or chunk
        if policy["deduplicate"] and key is not None and key in seen:
            dropped.append({"chunk_id": chunk, "reason": "duplicate_url_text"})
            continue
        if diversify and policy["diversify_documents"] and doc and counts[doc] >= policy["max_chunks_per_document"]:
            dropped.append({"chunk_id": chunk, "reason": "document_quota"})
            continue
        selected.append(item)
        if key is not None:
            seen.add(key)
        if doc:
            counts[doc] += 1
        if limit is not None and len(selected) >= limit:
            break
    return selected, dropped


def postrank(results, routing, policy, now):
    # Keep page-source postranking; temporal relevance is handled by the reranker.
    apply = policy["metadata_ranking"] and routing["temporal_guard"] == "default"
    ranked = []
    for index, item in enumerate(results):
        payload = item.get("payload", {})
        date, date_field = ranking_date(payload)
        age = max((now - date).total_seconds() / 86400, 0) if date is not None and date <= now else None
        fresh = 0.0  # Date now enters the cross-encoder, never an additive bonus.
        base = 1 - index / max(len(results) - 1, 1)
        freshness_bonus = 0.0
        page_bonus = (policy["page_weight"] if apply and routing["metadata_eligible"]
                      and payload.get("source_type") == "page" else 0.0)
        score = base + freshness_bonus + page_bonus
        ranked.append({**item, "postrank_score": score, "freshness_score": fresh,
                       "ranking_date": date.isoformat() if date else None, "ranking_date_field": date_field,
                       "age_days": age, "reranker_base_score": base,
                       "freshness_bonus": freshness_bonus, "page_bonus": page_bonus,
                       "metadata_ranking_applied": apply})
    ranked.sort(key=lambda item: -item["postrank_score"])
    return [{**item, "postrank_rank": rank} for rank, item in enumerate(ranked, 1)]
