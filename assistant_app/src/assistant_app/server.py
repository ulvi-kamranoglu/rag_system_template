"""HTTP API around pipeline.answer(): /health, /ask and /ask/stream (NDJSON)."""
import asyncio
from contextlib import asynccontextmanager
import json
import os
import threading

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse as BaseJSONResponse, StreamingResponse
import uvicorn

from .config import load_settings
from .schemas import AskRequest, AskResponse, ErrorResponse, HealthResponse, Message
from .security import require_api_key
from .admin import router as admin_router
SETTINGS = load_settings()

SERVER_HOST = SETTINGS.SERVER_HOST
SERVER_PORT = SETTINGS.SERVER_PORT

MSG_LOADING = "Server hələ yüklənir. Bir az gözləyib yenidən cəhd edin."
MSG_LOAD_FAILED = "Server düzgün başlamadı. Server pəncərəsindəki xətaya baxın."
MSG_LLM = (
    "Dil modeli cavab vermir (Ollama işləmir və ya vaxt bitdi). "
    "Ollama-nı işə salıb yenidən cəhd edin."
)
MSG_SEARCH = "Axtarış alınmadı. Yenidən cəhd edin."
MSG_UNKNOWN = "Gözlənilməz xəta baş verdi. Yenidən cəhd edin."


class State:
    pipeline = None  # the imported pipeline module, set once the retriever is loaded
    llm_ready = False
    load_error = None


state = State()

# One GPU: only one answer is generated at a time.
gpu_lock = threading.Lock()


class Cancelled(Exception):
    """Raised inside the pipeline callbacks when the client has gone away."""


# The cancel event of the request being answered in the current thread.
current_request = threading.local()


def mark_stopped_in_log(pipeline):
    """Makes pipeline's own log line say "stopped" when the user stopped the answer.

    pipeline.answer() sees our Cancelled as an LLM failure; only the label is changed here.
    """
    log_turn = pipeline.log_turn

    def log_turn_with_stop(record):
        cancel = getattr(current_request, "cancel", None)

        if cancel is not None and cancel.is_set() and not record.get("answer"):
            record = {**record, "error": "stopped"}

        log_turn(record)

    pipeline.log_turn = log_turn_with_stop


def load_models():
    """Imports the pipeline, loads the retriever and warms up the LLM (runs once)."""
    try:
        from assistant_app import pipeline
    except Exception as exc:
        state.load_error = f"{type(exc).__name__}: {exc}"
        print(f"XƏTA: pipeline yüklənmədi: {state.load_error}", flush=True)
        return

    mark_stopped_in_log(pipeline)

    try:
        pipeline.load()
        state.llm_ready = True
    except pipeline.LLMError as exc:
        # The retriever is loaded; the LLM is retried on the first question.
        print(f"XƏBƏRDARLIQ: LLM hazır deyil: {exc}", flush=True)
    except Exception as exc:
        state.load_error = f"{type(exc).__name__}: {exc}"
        print(f"XƏTA: retriever yüklənmədi: {state.load_error}", flush=True)
        return

    state.pipeline = pipeline
    print("Server hazırdır.", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded in a thread so /health can answer "loading" in the meantime.
    threading.Thread(target=load_models, daemon=True).start()
    try:
        yield
    finally:
        from .retrieval_adapter import close
        with gpu_lock:
            close()


class JSONResponse(BaseJSONResponse):
    # Windows PowerShell may otherwise decode UTF-8 JSON using a legacy charset.
    media_type = "application/json; charset=utf-8"


app = FastAPI(
    title="UNEC AI Assistant API", version="1.0.0",
    description="Source-grounded university question answering. Streaming uses NDJSON.",
    lifespan=lifespan, default_response_class=JSONResponse,
)
app.include_router(admin_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


def to_turns(history: list[Message]) -> list[dict]:
    """[{"role", "content"}, ...] -> [{"question", "answer"}, ...] as the pipeline expects."""
    turns = []
    question = None

    for message in history:
        if message.role == "user":
            question = message.content
        elif message.role == "assistant" and question is not None:
            turns.append({"question": question, "answer": message.content})
            question = None

    return turns


def routing_fields(result: dict) -> dict:
    """intent/language/show_sources from the pipeline, with defaults for a plain answer."""
    return {
        "intent": result.get("intent", "unec"),
        "language": result.get("language"),
        "show_sources": result.get("show_sources", True),
    }


def not_ready_message():
    if state.pipeline is not None:
        return None

    return MSG_LOAD_FAILED if state.load_error else MSG_LOADING


def error_message(exc: Exception) -> str:
    pipeline = state.pipeline

    if isinstance(exc, pipeline.LLMError):
        return MSG_LLM

    if isinstance(exc, pipeline.SearchError):
        return MSG_SEARCH

    return MSG_UNKNOWN


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health():
    if state.pipeline is not None:
        status = "ok"
    elif state.load_error:
        status = "error"
    else:
        status = "loading"

    return {
        "status": status,
        "model": getattr(state.pipeline, "LLM_MODEL", os.environ.get("LLM_MODEL")),
        "retriever_ready": state.pipeline is not None,
        "llm_ready": state.llm_ready,
        "error": state.load_error,
    }


@app.get("/readyz", response_model=HealthResponse,
         responses={503: {"model": HealthResponse}}, tags=["Operations"])
def ready():
    """Startup readiness, not a live probe of every downstream service."""
    result = health()
    if state.pipeline is None or not state.llm_ready:
        return JSONResponse(result, status_code=503)
    return result


@app.post("/ask", response_model=AskResponse,
          responses={503: {"model": ErrorResponse}},
          dependencies=[Depends(require_api_key)], tags=["Conversation"])
def ask(body: AskRequest):
    message = not_ready_message()

    if message:
        return JSONResponse({"error": message}, status_code=503)

    with gpu_lock:
        try:
            result = state.pipeline.answer(body.question.strip(), to_turns(body.history))
        except Exception as exc:
            return JSONResponse({"error": error_message(exc)}, status_code=503)

    if result["timings"]["llm"] > 0:
        state.llm_ready = True

    return {**result, **routing_fields(result)}


@app.post("/ask/stream", response_class=StreamingResponse,
          responses={200: {"description": "One JSON event per line: query, token, done or error.",
                           "content": {"application/x-ndjson": {"schema": {"type": "string"}}}}},
          dependencies=[Depends(require_api_key)], tags=["Conversation"])
async def ask_stream(body: AskRequest):
    question = body.question.strip()
    turns = to_turns(body.history)

    loop = asyncio.get_running_loop()
    events = asyncio.Queue()
    cancel = threading.Event()

    def put(event):
        try:
            loop.call_soon_threadsafe(events.put_nowait, event)
        except RuntimeError:
            pass  # the server is shutting down

    def on_query(query):
        if cancel.is_set():
            raise Cancelled

        if query and query != question:
            put({"type": "query", "text": query})

    def on_token(text):
        # Raising here closes the Ollama stream, so a stopped answer frees the GPU.
        if cancel.is_set():
            raise Cancelled

        put({"type": "token", "text": text})

    def work():
        current_request.cancel = cancel

        try:
            message = not_ready_message()

            if message:
                put({"type": "error", "message": message})
                return

            with gpu_lock:
                if cancel.is_set():
                    return

                result = state.pipeline.answer(
                    question, turns, on_query=on_query, on_token=on_token
                )

            if result["timings"]["llm"] > 0:
                state.llm_ready = True
            put(
                {
                    "type": "done",
                    # The final text: fixed replies are not streamed as tokens.
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "timings": result["timings"],
                    "search_query": result["search_query"],
                    **routing_fields(result),
                }
            )
        except Exception as exc:
            if not cancel.is_set():
                put({"type": "error", "message": error_message(exc)})
        finally:
            put(None)

    threading.Thread(target=work, daemon=True).start()

    async def lines():
        try:
            while (event := await events.get()) is not None:
                yield json.dumps(event, ensure_ascii=False) + "\n"
        finally:
            # Also reached when the client disconnects (stop button, closed app).
            cancel.set()

    return StreamingResponse(
        lines(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT,
                root_path=os.getenv("SERVER_ROOT_PATH", "").rstrip("/"))
