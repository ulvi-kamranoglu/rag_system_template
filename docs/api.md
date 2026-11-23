# API integration contract

All JSON is UTF-8. Paths are relative to your API base URL. OpenAPI is available
at `/openapi.json`; interactive documentation is at `/docs`.

## Authentication and origins

When `ASSISTANT_API_KEY` is set, send `X-API-Key` on `/ask` and `/ask/stream`.
Missing/wrong keys return HTTP 401 with `{"detail":"Invalid or missing API key"}`.
An empty setting disables key authentication. Health/docs routes are public;
restrict them at the gateway if necessary. Keep a shared service key in the
integrating backend/gateway, not publicly distributed JavaScript. The gateway
is responsible for end-user authentication; this key is service authentication.

`CORS_ORIGINS` is a comma-separated browser-origin allowlist. It is empty by
default. Same-origin and server-to-server requests do not require CORS. CORS
does not authenticate requests; cookie credentials are not enabled.

## POST /ask

```json
{
  "question": "UNEC-in kitabxanasından necə istifadə edə bilərəm?",
  "history": [
    {"role": "user", "content": "Kitabxana haqqında məlumat ver."},
    {"role": "assistant", "content": "Əvvəlki cavabın mətni."}
  ]
}
```

`question` is required, trimmed, nonblank and at most 8,000 characters. `history`
defaults to `[]`, accepts at most 20 messages with `user`/`assistant` roles, and
each message contains 1–16,000 characters. Configured recent turns are used to
interpret follow-ups. Clients supply history; no server session ID is issued.

HTTP 200 example (illustrative content and timings):

```json
{
  "answer": "Mənbələrə əsasən hazırlanmış cavab.",
  "sources": [{"title": "Mənbə", "url": "https://unec.edu.az/", "date": "2026-10-07", "source": "UNEC website"}],
  "search_query": "UNEC-in kitabxanasından necə istifadə edə bilərəm?",
  "timings": {"translate": 1.2, "search": 0.6, "llm": 3.1},
  "intent": "unec", "language": "az", "show_sources": true
}
```

Timings are seconds. `translate` covers routing/query rewriting, not just language
translation; timings exclude some queueing/network time. `intent` normally is
`unec`, `greeting`, `thanks` or `off_topic`; language normally is `az`, `en` or `ru`.
Each source includes `title`, `url`, `date` and `source` (the dataset source field).
`source` is an empty string when missing. URL/date may be empty. Only create links for nonempty URLs and honor `show_sources`.

Invalid input returns HTTP 422 with FastAPI's `detail` list. Loading, retrieval
or generation failures return HTTP 503 with `{"error":"..."}`.

## POST /ask/stream

Same request and authentication. Returns `application/x-ndjson; charset=utf-8`:
one JSON event per line, **not SSE** (no `data:` prefix).

```jsonl
{"type":"query","text":"Dəqiqləşdirilmiş sual"}
{"type":"token","text":"Cavabın "}
{"type":"token","text":"davamı."}
{"type":"done","answer":"Cavabın davamı.","sources":[],"timings":{"translate":1.2,"search":0.6,"llm":3.1},"search_query":"Sual","intent":"unec","language":"az","show_sources":true}
```

`query` is optional and appears when the search query changes. `token` can occur
zero or many times. `done.answer` is authoritative; replace accumulated text with
it because fixed replies can emit no tokens. Final metadata includes sources.
Failures emit `{"type":"error","message":"..."}` and close without `done`.

Authentication/input errors are HTTP 401/422 before streaming. After streaming
starts, application errors, including loading, are NDJSON events in HTTP 200.
Treat EOF without `done` as incomplete. Disconnection requests cancellation,
which takes effect on a subsequent callback. Decode UTF-8 incrementally and
retain partial lines across network chunks.

## JavaScript streaming example

```javascript
// Call a same-origin gateway; it injects X-API-Key server-side.
async function ask(question, history, onToken, signal) {
  const response = await fetch('/assistant/ask/stream', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question, history}), signal
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const reader = response.body.getReader(), decoder = new TextDecoder('utf-8');
  let buffer = '', result;
  function consume(line) {
    if (!line.trim()) return;
    const event = JSON.parse(line);
    if (event.type === 'token') onToken(event.text);
    if (event.type === 'error') throw new Error(event.message);
    if (event.type === 'done') result = event;
  }
  try {
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value, {stream: !done});
      let end;
      while ((end = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, end); buffer = buffer.slice(end + 1);
        consume(line);
      }
      if (done) break;
    }
    consume(buffer);
    if (!result) throw new Error('Incomplete answer stream');
    return result;
  } finally { await reader.cancel(); reader.releaseLock(); }
}
// AbortController.abort() stops the fetch when the user cancels generation.
```

## Pipeline administration

Set a separate `ASSISTANT_ADMIN_API_KEY` in the API process environment or root
`.env`, then restart the API. All `/admin/pipeline/*` routes require that key in
`X-API-Key`. The ordinary chat key does not grant administrative access unless
you deliberately configure both values identically. An unset admin key disables
the routes (503); a missing or incorrect key returns 401. Keep this key in an
administrator backend or use Swagger's Authorize control for local testing.

These operations update production datasets and indexes. They do not run inside
the HTTP process or hold an HTTP request open during collection/inference.

| Method and path | Behavior |
| --- | --- |
| `POST /admin/pipeline/collect` | Start collection and all subsequent stages. Optional repeated query `sources` (`wordpress/pages`, `wordpress/posts`, `instagram/<profile>`, `news`) collects only that subset; omitted means every enabled source. A source disabled in the collection config returns 422 before any job is created. |
| `GET /admin/pipeline/sources` | Collection sources from the collection config and whether each is enabled. |
| `POST /admin/pipeline/join` | Upload one raw Parquet dataset and start a batch. |
| `POST /admin/pipeline/reset-data` | Upload a complete replacement dataset, without retaining old records. |
| `GET /admin/pipeline/runs` | List API-submitted jobs, newest first. |
| `GET /admin/pipeline/runs/{run_id}` | Read job state and coordinator checkpoints. |
| `GET /admin/pipeline/runs/{run_id}/logs` | Read the last 64 KiB of worker output. |
| `POST /admin/pipeline/runs/{run_id}/resume` | Submit a new job resuming a failed/interrupted pending run. |

Join accepts `multipart/form-data` with one required field, `file`. In Swagger,
choose **Try it out**, select a `.parquet` file, then **Execute**. The previous
JSON filename request is no longer accepted. No deletion upload is accepted.

The file is decoded and must contain at least one row and the canonical input
columns `id`, `source_type`, `content`. Filenames use ASCII letters, numbers,
dots, underscores or hyphens, end in `.parquet`, and cannot end in
`_deleted.parquet`. Maximum file size is 512 MiB (413 if exceeded); invalid
Parquet/schema returns 422 without replacing the current raw input.

The validated original is committed first under
`archive_data/01_raw/uploads/run_.../<filename>`. Then its contents are published
to the `raw_dir` in `pipeline.yaml` (normally `shared_data/01_raw`), replacing
that directory's old contents. Only the uploaded file remains in that directory;
old deletion files do not carry into the new join. Publication uses the existing
journal/rollback mechanism. Previous raw contents without a run report are kept
in a hidden sibling backup unless their complete file inventory and SHA-256
hashes exactly match a committed upload archive. Verified archived uploads do
not create redundant `.01_raw.legacy_*` folders; unarchived or modified files
remain protected. `upload_archive` in the job response identifies the
archived original. A publication or worker startup failure does not remove it.

API submission and coordinator locks guard raw replacement. Busy/pending runs
return 409 without changing raw files. Files are streamed from the multipart
spool to disk and validated before starting the background pipeline. The 202
response follows archival/publication; collection and inference remain asynchronous.
The application size limit is checked after multipart parsing; configure request
limits at the reverse proxy as well. The nginx example permits multipart overhead.

Reset data uses the same `file` multipart field, admin key, upload validation,
512 MiB limit, raw archival/publication and job polling as join. Its job `mode`
is `reset_data`. It cleans only the uploaded rows, without merging old canonical
rows or applying default/explicit deletion files. The latest filename in `archive_data/02_clean/base_data`
determines the next version: `chatbot_dataset_v1.5_canonical.parquet` becomes
`chatbot_dataset_v1.6_canonical.parquet`, regardless of the upload filename.
A valid existing canonical base version is required. In every mode, the upload
name determines only batch filenames: `example_filename.parquet` becomes
`example_filename_canonical.parquet` in shared_data/02_clean and its run archive.
The full next version is stored separately in archive_data/02_clean/base_data.
Chunking and embedding outputs likewise contain only this batch. Join/collect
merge its chunks into the active index; reset replaces active index membership.

Chunking processes the entire replacement, embeddings are generated without
reusing old vectors, and snapshot synchronization removes Qdrant points absent
from the replacement and publishes a BM25 corpus containing only its chunks.
It does not delete/recreate the Qdrant collection. Historical run archives remain
available. Subsequent join/collect runs merge into this new replacement baseline.
Empty or entirely rejected replacement data does not overwrite the current
canonical snapshot. Reset has the same maintenance-window and checkpoint/resume
requirements as join; the pipeline is not an all-stages atomic transaction.

Start/resume returns HTTP 202 with `run_id` (an API job ID), `status`, `mode` and
submission metadata. Poll the status URL until `completed`, `failed` or
`interrupted`. A 202 acknowledges process submission, not successful pipeline
preflight. `queued` and `running` are nonterminal. `pipeline_run_id` is the
coordinator's separate identifier; `current_stage`, `completed_stages`,
`pipeline_status` and `dataset_name` expose its progress when available.
`pipeline_status` can be `complete` or `no_changes` on successful jobs.
On failure inspect `error` and logs. Add `?stage=collection`, `canonical`, `chunk`,
`embedding` or `retrieval` for the corresponding stage log (404 until available).

Concurrent submissions and new batches while a coordinator run is pending return
409. The coordinator lock also prevents overlap with CLI/scheduler executions;
if another entrypoint wins a race after submission, the accepted job fails safely.
The API does not queue multiple data updates. A resume returns a new API job ID
and retains the same coordinator checkpoints; `resumable` identifies a pending
run eligible for recovery. CLI-only pending runs must still be resumed through
the CLI. There is no automatic retry or cancellation endpoint.

Job requests, process identities, results and worker logs persist under
`<state_dir>/api_jobs`. They can be read after an API process restart. Workers
are separate processes; no model is loaded in the HTTP request handler.
Process identity checks detect exited/reused worker PIDs. Host shutdown, process
termination or a service manager stopping the entire service group can interrupt
execution. Inspect downstream processes and logs before resuming an interrupted
job. The supplied systemd unit's default control-group stop behavior terminates
workers too; detached processes alone do not provide an independent system service.

Run data updates in a maintenance window: Qdrant publication and local BM25
publication are not atomic, and embedding competes with answering for GPU memory.
After a successful update, restart the assistant API to load the new local
retrieval snapshot. The endpoint does not reload a live retriever automatically.

## Operations

`GET /health` returns 200 with `status` (`loading`/`error`/`ok`), `model`,
`retriever_ready`, `llm_ready`, `error`. `ok` means the retriever loaded; inspect
both readiness flags. `/readyz` returns 503 until retriever and LLM initially
become ready, then 200. These flags are not continuous downstream health probes.
If initial Ollama warm-up fails, answer requests can retry it, or restart the API
after Ollama is ready. Generation is serialized; concurrent requests may queue.
Use full-request timeouts and disable proxy buffering for streams. No idempotency
key is issued; do not blindly retry disconnected generation requests.
