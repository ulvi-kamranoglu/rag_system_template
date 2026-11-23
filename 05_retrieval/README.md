# Retrieval

For current-information queries, reranker input includes the primary configured
candidate set (normally 20), plus at most 10 additional candidates dated within
the last 730 days from the complete fused pool. Extras follow fusion order and
respect document quotas and text deduplication. Missing, invalid and future dates
are excluded. The date uses the same publication/modification rules as reranking.
Explicit-year, historical and event queries receive no extras. Disabling ranking
or `ranking.recent_candidates` also disables this expansion. All candidates are
reranked together; the final result limit remains unchanged (normally 8).
Diagnostics expose `fusion_primary`, `recent_extras` and the combined `fusion`.

Hazır embedding-lərin indekslənməsi və Qdrant/FAISS + BM25 əsasında hybrid axtarış.
Bütün terminal əmrləri `Chatbot_projects` kökündən icra edilir.

## Struktur

```text
05_retrieval/
  pyproject.toml
  src/retrieval/
    backends/        # Qdrant, FAISS, BM25
    search/          # Retriever, fusion, reranker, ranking, diagnostics
    indexing/        # Build, sync, archive/publication
    artifacts.py
    config.py
    cli.py
  config/config.yaml # active checkout configuration
  tests/
  docs/
```

Python paketindəki əvvəlki modul adları uyğunluq girişləridir: məsələn, `hybrid_retriever.py`,
`pipeline.py`, `vector_index.py`, `sync_pipeline.py` və `storage.py`.
İcra məntiqi yeni qovluqlardadır; köhnə və yeni import yolları eyni modul obyektinə yönəlir.
Bu, mövcud assistant, koordinator, test və terminal əmrlərini saxlayır.

## Məsuliyyətlər

| Sahə | Əsas modullar |
|---|---|
| Backend-lər | `backends/qdrant.py`, `faiss.py`, `bm25.py` |
| Axtarış axını | `search/retriever.py`, `flow.py` |
| Sıralama | `search/fusion.py`, `reranker.py`, `policy.py` |
| Diaqnostika | `search/diagnostics.py` |
| İlkin indekslər | `indexing/qdrant.py`, `bm25.py`, `faiss.py` |
| Yeniləmə | `indexing/sync.py`, `delta.py` |
| Lokal arxiv/retry | `indexing/storage.py` |

Axtarış payload hazırlığını indeks yaradan pipeline-dan import etmir;
ortaq əməliyyatlar `artifacts.py` daxilindədir.
Bu struktur dəyişikliyi score, ranking qaydası, vektor və ya dataset-i dəyişmir.

## Quraşdırma

```powershell
python -m pip install -r 05_retrieval/requirements.txt
retrieval-pipeline --help
retrieval-search --help
```

Quraşdırma servis başlatmır və indeks yaratmır. GPU asılılığı layihənin
`requirements-cuda130.txt` faylı ilə ayrıca idarə edilir.

## Əmrlər

```powershell
# İlkin indekslər — uyğun hazır embedding dataset-i üçün:
retrieval-pipeline --dataset chatbot_dataset_v1.3 --stage qdrant
retrieval-pipeline --dataset chatbot_dataset_v1.3 --stage bm25

# Axtarış — aktiv dataset avtomatik seçilir:
retrieval-search --query "UNEC-in rektoru kimdir?" --limit 5

# Yeni JSON faylına tam diaqnostika:
retrieval-search --query "UNEC-in rektoru kimdir?" --limit 5 --diagnose --output diagnostics/new-search.json
```

| Stage | Davranış |
|---|---|
| `qdrant` | İlkin tam build; seçilmiş collection-u silib yenidən yaradır |
| `bm25` | Hazır embedding mapping-dən lokal corpus qurur |
| `faiss` | Alternativ lokal dense indeks qurur |
| `sync` | Tam yeni snapshot-u mövcud Qdrant/BM25 ilə uyğunlaşdırır; collection-u sıfırlamır |
| `delta` | Ayrı NEW/UPDATED və deletion müqaviləsi tələb edən delta axını |

Gündəlik `collect` və `join` axınları [koordinatorla](../docs/pipeline.md) işləyir;
onların son mərhələsi `sync`-dir. İlkin `qdrant` build-i gündəlik merge əmri deyil.

Köhnə girişlər də işləyir:

```powershell
python -m retrieval --help
python -m retrieval.pipeline --help
python -m retrieval.sync_pipeline --help
python -m retrieval.storage --publish-run "C:/path/to/archive_data/05_indexes/run_..."
```

## Konfiqurasiya və data

Default: `05_retrieval/config/config.yaml`. CLI əvvəl override-ları təyin edir,
sonra stage/retriever modullarını import edir.

- Config: `--config` / `RETRIEVAL_CONFIG`.
- Layihə kökü: `--project-root` / `CHATBOT_PROJECT_ROOT`.
- Dataset: CLI → environment → YAML → aktiv dataset pointer-i → tək embedding mapping.
- Qdrant: `QDRANT_URL`, `QDRANT_COLLECTION`; backend: `VECTOR_BACKEND`.
- Giriş: `shared_data/04_embeddings`; lokal çıxış: `shared_data/05_indexes`.
- İcra arxivləri: `archive_data/05_indexes/run_...`.

Qdrant vektorları servis storage-indədir. Lokal arxiv Qdrant snapshot-u deyil.
Search query embedder və reranker yükləyir; corpus embedding-lərini təkrar hesablamır.

## Testlər və detallı təlimatlar

```powershell
python -m pytest
```

Default suite servissizdir. Model, servis və xüsusi indeks tələb edən testlər
`integration` marker-i altındadır.

- [Axtarış, ranking və diaqnostika](docs/search.md)
- [Storage, indekslər və publication retry](docs/storage.md)
