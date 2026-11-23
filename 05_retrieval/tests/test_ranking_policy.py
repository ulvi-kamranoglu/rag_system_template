from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
import pytest
from retrieval.ranking_policy import settings, route, select, postrank, recent_document_ids, ranking_date
from retrieval.search_flow import run_trace
from retrieval.hybrid_fusion import ReciprocalRankFusion
from retrieval.search.policy import reranker_candidates


def test_recent_extra_pool_keeps_twenty_primary_and_caps_ten_unique_extras():
    now = datetime(2026, 10, 31, tzinfo=timezone.utc)
    primary = [hit(f"primary{i}", date="2010-01-01") for i in range(20)]
    # Duplicate document/text candidates must not consume the extra allowance.
    recent = [hit("duplicate-doc", doc="primary0", date=now.isoformat()),
              hit("duplicate-text", url=primary[0]["payload"]["url"], text="primary0", date=now.isoformat())]
    recent += [hit(f"recent{i}", date=now.isoformat()) for i in range(15)]
    combined, base, extras, _ = reranker_candidates(primary + recent, settings(), route("Who is the rector?"), 20, now)
    assert base == primary
    assert len(combined) == 30 and len(extras) == 10
    assert [row["chunk_id"] for row in extras] == [f"recent{i}" for i in range(10)]
    assert len({row["chunk_id"] for row in combined}) == 30


def test_recent_extras_use_valid_730_day_dates_without_reordering_primary():
    now = datetime(2026, 10, 31, tzinfo=timezone.utc)
    values = [hit("primary", date="2010-01-01"), hit("missing"), hit("invalid", date="bad"),
              hit("future", date=(now + timedelta(days=1)).isoformat()),
              hit("too-old", date=(now - timedelta(days=731)).isoformat()),
              hit("boundary", date=(now - timedelta(days=730)).isoformat()),
              hit("updated", source_type="research", date="2010-01-01", modified=now.isoformat())]
    combined, _, extras, _ = reranker_candidates(values, settings(), route("Who is the rector?"), 1, now)
    assert [row["chunk_id"] for row in combined] == ["primary", "boundary", "updated"]
    assert len(extras) == 2


@pytest.mark.parametrize("query", ["Who was rector in 2016?", "UNEC history", "Rektor konfransda ne deyib?"])
def test_historical_and_event_queries_receive_no_recent_extras(query):
    now = datetime.now(timezone.utc)
    rows = [hit("old"), hit("new", date=now.isoformat())]
    combined, primary, extras, _ = reranker_candidates(rows, settings(), route(query), 1, now)
    assert combined == primary == rows[:1] and extras == []


def test_flow_sends_recent_extras_to_same_reranker_and_preserves_final_limit():
    now = datetime.now(timezone.utc).isoformat()
    old = [hit(f"old{i}", date="2010-01-01") for i in range(3)]
    new = hit("recent", date=now)
    retriever, _ = fake_retriever(settings({"recent_candidates": True}), old + [new], old)
    seen = []
    def rerank(query, candidates, limit):
        seen.extend(candidates)
        return [{**row, "reranker_score": 1} for row in reversed(candidates)]
    retriever.reranker.rerank = rerank
    report = run_trace(retriever, "Who is the rector?", dense_limit=4, bm25_limit=3,
                       reranker_input_limit=2, final_limit=1)
    assert len(seen) == 3
    assert report["limits"]["recent_extra_count"] == 1
    assert report["stages"]["recent_extras"][0]["chunk_id"] == "recent"
    assert report["stages"]["final"][0]["chunk_id"] == "recent"


def hit(key, doc=None, url=None, text=None, **metadata):
    return {"chunk_id": key, "score": 1.0, "payload": {
        "chunk_id": key, "document_id": doc or key, "url": url or f"https://example.org/{key}",
        "chunk_text": text or key, "source_type": "post", **metadata}}


def test_dedup_and_document_quota_preserve_order_and_refill():
    values = [hit("one", url="https://example.org/a", text="Same   text"),
              hit("copy", url="https://example.org/a", text="same text"),
              hit("extra", doc="one"), hit("two"), hit("three")]
    selected, removed = select(values, settings(), limit=3)
    assert [item["chunk_id"] for item in selected] == ["one", "two", "three"]
    assert [item["reason"] for item in removed] == ["duplicate_url_text", "document_quota"]


def test_blank_urls_and_different_chunks_are_not_duplicates():
    rows = [hit("one", text="same"), hit("two", text="same")]
    for row in rows:
        row["payload"]["url"] = ""
    assert len(select(rows, settings())[0]) == 2
    rows = [hit("one", url="https://example.org/a", text="first"), hit("two", url="https://example.org/a", text="second")]
    assert len(select(rows, settings())[0]) == 2


def test_explicit_year_and_event_disable_recency():
    assert route("UNEC-in rektoru kimdir?")["recent_windows"] == [730]
    for query in ("2014-cü ildə UNEC-in rektoru kim idi?", "Rektor konfransda nə deyib?"):
        routed = route(query)
        assert routed["recent_windows"] == []
        assert not routed["metadata_eligible"]


def test_recency_uses_publication_date_and_excludes_future():
    now = datetime(2026, 10, 26, tzinfo=timezone.utc)
    rows = [{"document_id": "old", "date": "2020-01-01", "modified": "2026-10-25"},
            {"document_id": "new", "date": "2026-10-25"},
            {"document_id": "future", "date": "2027-01-01"}]
    assert recent_document_ids(rows, 365, now) == ["new"]


def test_temporal_guard_preserves_reranker_order():
    now = datetime(2026, 10, 26, tzinfo=timezone.utc)
    rows = [hit("old", date="2020-01-01"), hit("new", date="2026-10-25", source_type="page")]
    ranked = postrank(rows, route("2014-cü ildə rektor kim idi?"), settings(), now)
    assert [row["chunk_id"] for row in ranked] == ["old", "new"]
    assert all(not row["metadata_ranking_applied"] for row in ranked)


def test_invalid_policy_fails_before_search():
    with pytest.raises(ValueError):
        settings({"freshness_weight": float("nan")})
    with pytest.raises(ValueError):
        settings({"max_chunks_per_document": 0})


def fake_retriever(policy, dense, bm25):
    calls = []
    def search(**kwargs):
        calls.append(kwargs)
        allowed = kwargs.get("document_ids")
        return [row for row in dense if allowed is None or row["payload"]["document_id"] in allowed][:kwargs["limit"]]
    def rerank(query, candidates, limit):
        return [{**row, "reranker_score": 1 - index * 0.1, "reranker_rank": index + 1}
                for index, row in enumerate(candidates[:limit])]
    retriever = SimpleNamespace(ranking_policy=policy, embedder=SimpleNamespace(embed_query=lambda q: [1, 0]),
        vector_index=SimpleNamespace(search=search), bm25_search=lambda q, limit: bm25[:limit],
        bm25_index=SimpleNamespace(records=[row["payload"] for row in dense]),
        fusion=ReciprocalRankFusion(), reranker=SimpleNamespace(rerank=rerank))
    return retriever, calls


def test_disabled_policy_keeps_legacy_fusion_and_final_order():
    dense = [hit("one"), hit("two"), hit("three")]
    retriever, calls = fake_retriever(settings({"enabled": False}), dense, [dense[1]])
    report = run_trace(retriever, "rektor kimdir?", dense_limit=3, bm25_limit=1, reranker_input_limit=3, final_limit=2)
    expected = ReciprocalRankFusion().fuse(dense, [dense[1]], limit=3)
    assert [row["chunk_id"] for row in report["stages"]["final"]] == [row["chunk_id"] for row in expected[:2]]
    assert len(calls) == 1


def test_batched_passage_selection_matches_serial_and_avoids_empty_retry():
    dense = [hit("dense", doc="known")]
    lexical = [hit("lexical", doc="missing"), hit("empty", doc="empty")]
    chosen = hit("best", doc="missing")
    def make():
        retriever, _ = fake_retriever(settings(), dense, lexical)
        def vector_search(**kw):
            docs = kw.get("document_ids")
            return [chosen] if docs == ["missing"] else [] if docs else dense
        retriever.vector_index.search = vector_search
        return retriever
    serial = make()
    expected = run_trace(serial, "exam", dense_limit=2, bm25_limit=2, reranker_input_limit=3, final_limit=3)
    batched = make()
    seen = []
    def batch(**kw):
        seen.append(kw["document_ids"])
        return {"missing": [chosen], "empty": []}
    original_search = batched.vector_index.search
    def no_serial_lookup(**kw):
        assert kw.get("document_ids") is None
        return original_search(**kw)
    batched.vector_index.search = no_serial_lookup
    batched.vector_index.search_documents = batch
    actual = run_trace(batched, "exam", dense_limit=2, bm25_limit=2, reranker_input_limit=3, final_limit=3)
    assert seen == [["missing", "empty"]]
    assert actual["stages"] == expected["stages"]


def test_enabled_flow_uses_date_filtered_branch_and_unique_documents():
    date = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    dense = [hit("one", doc="doc", date=date), hit("extra", doc="doc", date=date), hit("two", date=date)]
    retriever, calls = fake_retriever(settings(), dense, [dense[-1]])
    report = run_trace(retriever, "UNEC-in rektoru kimdir?", dense_limit=2, bm25_limit=2, reranker_input_limit=5, final_limit=5)
    assert "recent_730" in report["stages"]
    assert any("document_ids" in call for call in calls)
    final = report["stages"]["final"]
    assert len({row["payload"]["document_id"] for row in final}) == len(final)
    assert all(row["passage_selection"] == "dense_best" for row in final)


def test_diagnostics_and_production_trace_share_results():
    from retrieval.diagnostics import diagnose
    dense = [hit("one"), hit("two")]
    retriever, _ = fake_retriever(settings({"recent_candidates": False}), dense, dense)
    options = dict(dense_limit=2, bm25_limit=2, reranker_input_limit=2, final_limit=2)
    retriever.trace_search = lambda query, **kwargs: run_trace(retriever, query, **kwargs)
    expected = run_trace(retriever, "ordinary question", **options)["stages"]["final"]
    actual = diagnose(retriever, "ordinary question", **options)["stages"]["final"]
    assert [row["chunk_id"] for row in actual] == [row["chunk_id"] for row in expected]


@pytest.mark.parametrize("source", ["page", "form", "research", "legal"])
def test_current_sources_use_modified_date(source):
    value, field = ranking_date({"source_type": source, "date": "2015-01-01", "modified": "2026-10-26"})
    assert value == datetime(2026, 10, 26, tzinfo=timezone.utc)
    assert field == "modified"


def test_old_news_cannot_look_new_after_modification():
    value, field = ranking_date({"source_type": "post", "date": "2015-01-01", "modified": "2026-10-26"})
    assert value.year == 2015 and field == "date"
    assert ranking_date({"source_type": "post", "modified": "2026-10-26"}) == (None, None)


def test_bad_modified_date_falls_back_to_publication():
    value, field = ranking_date({"source_type": "form", "date": "2026-10-26", "modified": "invalid"})
    assert value.year == 2026 and field == "date"


def test_no_additive_date_bonus_even_for_legacy_nonzero_weights():
    now = datetime(2026, 10, 27, tzinfo=timezone.utc)
    rows = [hit("old", date="2010-01-01"), hit("new", date=now.isoformat())]
    result = postrank(rows, route("General question"), settings({"freshness_weight":.20}), now)
    assert [r["chunk_id"] for r in result] == ["old", "new"]
    assert all(r["freshness_bonus"] == 0 for r in result)


@pytest.mark.parametrize("query", ["2016-cı ildə qəbul qaydaları", "UNEC-in tarixi", "UNEC history"])
def test_historical_queries_receive_no_date_bonus(query):
    now = datetime(2026, 10, 27, tzinfo=timezone.utc)
    result = postrank([hit("one", date=now.isoformat())], route(query), settings(), now)[0]
    assert result["freshness_bonus"] == 0


@pytest.mark.parametrize("metadata", [{}, {"date":"invalid"}, {"date":"2027-01-01"},
    {"source_type":"page","modified":"2027-01-01","date":"2020-01-01"}])
def test_missing_invalid_and_future_dates_get_no_bonus(metadata):
    now = datetime(2026, 10, 27, tzinfo=timezone.utc)
    result = postrank([hit("one", **metadata)], route("Ümumi sual"), settings(), now)[0]
    assert result["freshness_bonus"] == 0


def test_zero_weight_disables_only_freshness_and_modified_is_used_in_recent_pool():
    now = datetime(2026, 10, 27, tzinfo=timezone.utc)
    row = hit("one", source_type="page", date="2010-01-01", modified=now.isoformat())
    result = postrank([row], route("Rektor kimdir?"), settings({"freshness_weight":0}), now)[0]
    assert result["freshness_bonus"] == 0 and result["page_bonus"] == .10
    assert recent_document_ids([row["payload"]], 365, now) == ["one"]
