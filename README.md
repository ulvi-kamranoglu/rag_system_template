# UNEC AI Assistant — Server API

Server delivery for university question answering with source references.
Includes the API, retrieval, Ollama integration and complete data update pipeline.
User interfaces are implemented by the integrating team.

## Start here

1. [Server installation and deployment](docs/deployment.md).
2. [API contract and frontend examples](docs/api.md).
3. [Data update operations](docs/pipeline.md).
4. [Architecture and delivery boundaries](docs/architecture.md).

Python **3.12** is the reference environment. All Python components use one venv.
Qdrant and Ollama run as separate services. Their databases/model weights and
the Hugging Face model cache must be provisioned on the target server.

```text
unec_ai_assistant_api/
├── assistant_app/             HTTP API and response generation
├── 01_data_collection/        University source collection
├── 02_data_canonicalization/  Cleaning and canonical dataset merging
├── 03_data_chunking/          Search passage preparation
├── 04_embedding/              Embeddings and vector reuse
├── 05_retrieval/              Qdrant/BM25 retrieval and synchronization
├── orchestration/            Execution, locks and checkpoints
├── storage_common/           Storage and versioning utilities
├── shared_data/              Current datasets and retrieval artifacts
├── archive_data/             Five stage archives and base datasets
├── docs/                     Integration, deployment and data contracts
├── scripts/                  Administration and verification
├── deploy/                   Service configuration examples
├── tests/                    Cross-component tests
├── pipeline.yaml             Ingestion configuration
└── run_pipeline.py           Administrator/scheduler entrypoint
```

## HTTP interface

| Endpoint | Purpose |
|---|---|
| `GET /health` | Loading/error status and startup readiness flags |
| `GET /readyz` | 200 after initial retriever and LLM readiness; otherwise 503 |
| `POST /ask` | Complete answer, source references and timings |
| `POST /ask/stream` | NDJSON answer events and final answer with sources |
| `GET /docs` | Interactive API documentation |
| `GET /openapi.json` | Machine-readable HTTP schema |

`ASSISTANT_API_KEY` optionally protects both answer endpoints. Set it before
exposing the API to clients. Authentication and CORS are in the API guide.

## Included data

The delivery's active snapshot is `chatbot_dataset_v1.4`. Current stage outputs
and existing archives under the five numbered directories are included, about
1 GiB total. Follow the active pointer after future updates. Historical reports
retain original paths as provenance; they are not server configuration.
Collector baseline/snapshot state is included. Completed development retry
pointers and transient run history are not carried into the delivery.

On a **new empty Qdrant target**, use `python scripts/bootstrap_indexes.py` to
index included vectors without re-embedding. It refuses to replace an existing
collection. Do not rebuild the development instance to test this delivery.

## Verification

```text
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/check_environment.py
python run_pipeline.py --plan
```

Default tests do not require live Qdrant/Ollama or model inference. The environment
check is read-only: no downloads, index writes or ingestion. The API currently
serializes generation per process; simultaneous-user capacity requires load tests.
