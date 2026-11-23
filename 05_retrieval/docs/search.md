# Search and ranking

## Configurable ranking policy

`ranking` in `config/config.yaml` controls the shared production/diagnostic flow.
Set `enabled: false` to compare against the previous 05_retrieval/reranker ordering.
Alternatively pass `retrieval-search --legacy-ranking ...` for a single baseline
invocation without editing YAML. Save baseline and policy reports under different
filenames and compare both answer relevance and diagnostic timings on the same
set of questions (including historical and event-specific questions).
Each feature can also be disabled separately. No corpus or vector rebuild is
required; start a new search process after changing configuration.

With the policy enabled, retrieval overfetches by `candidate_multiplier` and
selects diverse documents from each branch. The global dense and BM25 branches
are always kept. A conservative keyword router adds 730-day dense candidates for
admission, catalog and structure questions, and 730/365-day candidates for digital
and location questions. These are additional RRF branches; a document found in
multiple branches gains multiple rank contributions. Reference content (`page`,
`form`, `research`, `legal`) uses valid `modified`, falling back to `date` when
modification is missing/invalid. Other sources use publication `date` only:
editing an old news article does not make its event recent. Missing/invalid/future dates gain no
recency advantage. No preference is given to newly joined datasets or research.

Recent branches use document IDs from the active local BM25 corpus and filter
Qdrant by those IDs. FAISS supports the same filter using a full sorted search,
which can be slower. The router is heuristic; unrecognized queries still use the
global branches. Explicit 19xx/20xx years, historical cues and recognized specific
event cues disable both recent branches and metadata reranking.

Candidate selection removes equal nonempty URL + NFC/casefold/whitespace-normalized
chunk text duplicates and limits chunks per document (default 1). URL paths stay
case-sensitive. With a one-chunk quota, the reranker receives the best dense chunk
for each selected document; BM25-only documents may require additional filtered
vector searches. With a larger quota, multiple selected passages are retained.
Limits apply to a bounded candidate pool: fewer final results may be returned
when insufficient distinct candidates are available.

All selected passages are reranked before final limiting. `reranker.date_context: true`
adds document date metadata before the passage text and, for current questions,
today's UTC date plus a preference for current relevant evidence to the model query.
Explicit-year, historical and recognized specific-event questions keep their original
query; document dates remain visible as evidence. Future/invalid dates are marked unknown.
The resulting `reranker_score` is returned unchanged: there is no additive date bonus.
Historical `freshness_weight` / `freshness_half_life_days` settings are accepted only
for compatibility and have no scoring effect. Diagnostics retains `freshness_bonus: 0`
for older report consumers and exposes the actual `reranker_date_context` input.

BGE is not an instruction-following LLM or a model trained specifically for temporal ranking.
This contextualization makes dates available to the cross-encoder; it does not guarantee
that a newer document will outrank an older document. Live before/after evaluation is required.
Set `reranker.date_context: false` for the previous query/passage representation.
`--legacy-ranking` alone does not disable the new reranker date context.
The postrank base remains normalized reranker **rank**; the existing page-source
bonus keeps its previous eligibility. No static rank column is written.
Restart a long-running API after changing code. Diagnostics exposes
raw/selected branches, fusion, document-selected passages, reranked, postrank and
final stages, routing decisions and removed candidate reasons. It runs exactly
the same flow as regular `search`; datasets are not deduplicated or changed.

Run tests manually:

```powershell
python -m pytest 05_retrieval/tests/test_ranking_policy.py 05_retrieval/tests/test_retrieval_diagnostics.py 05_retrieval/tests/test_retrieval_settings.py -q
```

## Diagnose search ranking

```powershell
retrieval-search --dataset chatbot_dataset_v1.4 --query "UNEC-in rektoru kimdir?" --limit 5 --diagnose --focus-source research --output diagnostics/rektor-v1.4.json
```

Terminal output shows five previews per stage and ranks/counts for the selected
source_type. Full dense/BM25, fusion candidates, reranked candidates and final
payloads are saved only when `--output` is supplied; existing reports are not
overwritten. `--focus-source` marks results without filtering retrieval.
Missing focus rows do not prove those documents are absent from the index.
`--dense-limit`, `--bm25-limit` and `--rerank-limit` override candidate counts
for this invocation; omitted flags preserve configured search limits.
Diagnosis does not change corpus deduplication or ranking policy. Run local fake
retriever tests with `python -m pytest 05_retrieval/tests/test_retrieval_diagnostics.py -q`.

## Search performance

Qdrant startup prepares a keyword payload index on `document_id` for the existing
document filters. Initial index creation and incremental upserts also ensure it
exists. This is an additive index: it does not recreate the collection or change
vectors, payloads, candidate limits, RRF, temporal rules, or document selection.
The first startup against an older collection needs permission to create this
payload index. A conflicting field type fails explicitly instead of being replaced.

Collection existence is checked at startup, not before every internal vector
query. If a collection disappears later, the query's 404 becomes a clear missing
collection error; authentication and server errors propagate unchanged.
Document-specific passage lookups use one batch HTTP call with an independent
filtered top-1 query for each missing document. This retains the original
per-document selection semantics, including empty results; FAISS and other
backends without batch support retain the serial path.

Recent-document lookup uses sorted timestamps prepared with each loaded corpus
snapshot. Binary searches retain the exact inclusive cutoff and current-time
bound, including exclusion of future documents. Reloading a published corpus
rebuilds the structure; there is no stale daily result cache.

`reranker.batch_size` controls inference grouping (default `4`). It keeps all
configured candidates and the complete existing date-aware query/passage inputs.
Different GPU batch groupings can produce small floating-point score differences;
check final result agreement on representative questions when changing it.

