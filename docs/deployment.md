# Server installation and handover

## Prerequisites

Python 3.12 is the reference environment. Provision Qdrant with persistent storage
and Ollama with model storage. NVIDIA execution needs a compatible driver and
PyTorch build. CPU operation has different latency and may require the reranker
FP16 setting to be disabled. No virtual environment or model cache is bundled.
Included datasets/archives are approximately 1 GiB, not a Qdrant database backup.

## Install one environment

Linux:

```bash
cd /srv/unec_ai_assistant_api
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# Optional NVIDIA build; review compatibility with the target driver first.
python -m pip install -r requirements-cuda130.txt
python -m pip install -r requirements.txt
cp .env.example .env
```

Windows PowerShell:

```powershell
Set-Location C:\services\unec_ai_assistant_api
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-cuda130.txt
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

CUDA installation is separate and optional. Install an approved alternative
PyTorch build first if needed. Root requirements install all local modules into
the same environment; editable checkout paths must remain available. Package and
model downloads need network access unless caches are provisioned.

## Configure the runtime

Edit `.env`: Qdrant URL/collection, Ollama URL/model, API host/port, API key and
browser origins. The API binds loopback by default. Expose another interface
only with suitable gateway/firewall configuration. Keep Qdrant and Ollama private;
the current Qdrant backend expects a trusted reachable URL and does not configure
Qdrant authentication headers. Use an unpinned `PIPELINE_DATASET` to follow updates.

Active settings are in each module's `config/` directory. Embedding model and
dimension must match bundled vectors. Changing model IDs requires compatible
re-embedding/reindexing, not merely a new server setting.

The API, root administrator entrypoint and helper scripts read `.env`. Standalone
stage CLIs can require exported environment variables or explicit configuration;
do not assume every command reads `.env` automatically.

Provision the configured Ollama model:

```text
ollama pull gemma4:e4b
ollama create unec-assistant -f assistant_app/src/assistant_app/llm/Modelfile
ollama list
```

The installed Ollama must support and have access to that base model. Embedding
and reranking weights download on first use unless pre-provisioned.

## Initialize a NEW Qdrant target

```text
python scripts/check_environment.py
python scripts/bootstrap_indexes.py
python scripts/check_environment.py
```

The first check will report a missing collection on a new target. Bootstrap uses
included vectors and the existing indexing stages, without re-embedding. It refuses
an existing collection, preventing accidental replacement. Do not run the API or
ingestion during bootstrap. If interrupted, inspect logs/target before retrying;
a partial collection is not automatically deleted. A separately restored matching
Qdrant snapshot does not need bootstrap. The environment check compares IDs,
counts and vector configuration, not every vector value or payload field.

## Start the API

```text
python -m assistant_app.server
```

Wait for `/readyz` HTTP 200; first startup can take longer due to model downloads
and warm-up. `/docs` and `/openapi.json` are available before model readiness.
Use [the integration contract](api.md) for complete and streaming responses.

`deploy/unec-assistant.service` is a Linux service template. Adjust user/paths and
have the server team register it. It is not installed automatically. Start one
process; extra workers duplicate model state. Configure Qdrant/Ollama service
dependencies and restarts in the target infrastructure.

Use a TLS reverse proxy. `deploy/nginx-location.conf` is a location example for
an existing HTTPS server block. Add user authentication, rate limits and request
timeouts at the gateway. Set `ASSISTANT_API_KEY` when other applications connect.
For the example's `/assistant/` public prefix, set `SERVER_ROOT_PATH=/assistant`
so interactive API documentation uses the correct public URLs. Leave it unset
when the API is exposed at the domain root. If the gateway keeps the service key,
configure it to inject `X-API-Key` after authenticating the user.

## Data updates

```text
python run_pipeline.py --plan
python run_pipeline.py --mode join --input shared_data/01_raw/new_dataset.parquet
python run_pipeline.py --mode collect
```

Select one mode. `join` takes approved administrator data; `collect` needs source
access. Do not import the included sample raw file unintentionally. Explicit
deleted records are handled by the pipeline. A scheduler can call `--mode collect
--resume-pending`. The Windows task helper is provided, not automatically installed.
Read [pipeline operations](pipeline.md), including collection limitations.

Qdrant/local publication is not one transaction. Use a maintenance window if
strict consistency is required. Back up both sides before updates.

## Acceptance and maintenance

```text
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Default tests use substitutes for live services. On the real server, additionally
verify readiness, source-backed answers, streaming, approved incremental updates
and deletion, recovery and the intended concurrent load. Unit tests do not establish
production capacity. Review logs and agree access/retention for recorded conversations.

Back up active files, five stage archive directories, collector state, new
checkpoints and Qdrant together. Keep `.env` secrets out of Git. Historical report
paths are provenance, not commands to execute on the new server. For rollback,
restore compatible code, local data and Qdrant as a matched set.
