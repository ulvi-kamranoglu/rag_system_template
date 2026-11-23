"""Assistant YAML and optional environment overrides; no models are loaded."""
import os
from pathlib import Path
from types import SimpleNamespace
import yaml
from dotenv import load_dotenv
from pipeline_storage import default_config, workspace_root

ROOT = workspace_root(__file__)


def load_settings():
    load_dotenv(ROOT / ".env", override=False)
    selected = os.getenv("ASSISTANT_CONFIG")
    path = Path(selected).expanduser().resolve() if selected else default_config(__file__, "config/config.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Assistant configuration must be a YAML mapping")
    for section in ("server", "llm", "conversation"):
        if not isinstance(raw.get(section, {}), dict):
            raise ValueError(f"{section} must be a YAML mapping")
    fields = {
        "SERVER_HOST": ("server", "host", "127.0.0.1", str), "SERVER_PORT": ("server", "port", 8000, int),
        "OLLAMA_URL": ("llm", "url", "http://127.0.0.1:11434", str), "LLM_MODEL": ("llm", "model", "unec-assistant", str),
        "LLM_TIMEOUT": ("llm", "timeout", 180, float), "TEMPERATURE": ("llm", "temperature", 0.2, float),
        "NUM_CTX": ("llm", "num_ctx", 8192, int), "MAX_PASSAGE_CHARS": ("llm", "max_passage_chars", 1500, int),
        "TOP_K": ("conversation", "top_k", 8, int), "HISTORY_TURNS": ("conversation", "history_turns", 3, int)}
    values = {}
    for key, (section, name, default, convert) in fields.items():
        value = os.getenv(key, raw.get(section, {}).get(name, default))
        if isinstance(value, bool):
            raise ValueError(f"Invalid {key}")
        try:
            values[key] = convert(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {key}") from exc
    import math
    for key in ("SERVER_PORT", "LLM_TIMEOUT", "NUM_CTX", "MAX_PASSAGE_CHARS", "TOP_K"):
        if not math.isfinite(values[key]) or values[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if values["SERVER_PORT"] > 65535 or values["HISTORY_TURNS"] < 0 or not 0 <= values["TEMPERATURE"] <= 2:
        raise ValueError("Invalid port, history length or temperature")
    from urllib.parse import urlsplit
    url = urlsplit(values["OLLAMA_URL"])
    if url.scheme not in {"http", "https"} or not url.netloc:
        raise ValueError("OLLAMA_URL must be an HTTP(S) URL")
    values["OLLAMA_URL"] = values["OLLAMA_URL"].rstrip("/")
    if not values["LLM_MODEL"].strip():
        raise ValueError("LLM_MODEL must not be empty")
    log = Path(raw.get("conversation", {}).get("log_path", "logs/chat_log.jsonl"))
    values["LOG_PATH"] = log if log.is_absolute() else ROOT / log
    origins = os.getenv("CORS_ORIGINS")
    origins = origins.split(",") if origins is not None else raw.get("server", {}).get("cors_origins", [])
    if not isinstance(origins, list) or any(not isinstance(value, str) for value in origins):
        raise ValueError("server.cors_origins must be a list of strings")
    values["CORS_ORIGINS"] = [value.strip() for value in origins if value.strip()]
    return SimpleNamespace(**values)
