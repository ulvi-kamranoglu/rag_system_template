# Server architecture

```text
Client -> HTTP API -> routing -> Qdrant/BM25 retrieval -> Ollama -> answer + sources

Scheduler / administrator -> collection (optional) -> canonicalization
    -> chunking -> incremental embedding -> Qdrant/BM25 synchronization
```

The HTTP API reads published artifacts. It does not launch ingestion per question.
Ingestion uses the administrator CLI or scheduler. This release has no public
dataset-upload/deletion HTTP endpoint; administration remains on the server.
Frontend applications integrate with [the conversation API](api.md).

All Python packages share one environment. Qdrant and Ollama are external services.
Embedding and reranking weights are needed in the target model cache. The API
requires a published BM25 corpus and matching Qdrant collection; ingestion also
requires canonical/chunk/vector base snapshots.

`shared_data` contains active artifacts; the five `archive_data` stage directories
contain historical outputs and base versions. `runtime_data` stores new locks,
checkpoints and run logs. Collector baseline/snapshot state is separately stored
in `01_data_collection/data/state` and must be backed up.

Paths resolve against this checkout's `pipeline.yaml`. For wheel installations,
set `CHATBOT_PROJECT_ROOT` and explicit YAML paths. Historical reports can contain
original Windows paths. Retrieval validation accepts Windows/POSIX metadata
filenames. Old histories are provenance, not resumable jobs on another server.

The adapter notices published dataset changes, reloads search state and reuses
models. Restart the API after model or server setting changes. Qdrant writes and
local publication are not one transaction: use a maintenance window if strict
consistency is required and back up Qdrant/local artifacts together.

Generation is serialized within each process. Extra Uvicorn workers duplicate
model state and do not provide coordinated GPU scheduling. Start with one process.

The server delivery contains no UI package/assets. Core data and ranking logic
is retained. API additions are typed OpenAPI models, request validation, optional
API-key authentication, configurable CORS and a startup readiness endpoint.
