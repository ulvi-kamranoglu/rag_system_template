import json
import logging
import re

import requests
from .repetition import RepetitionGuard

from ..config import load_settings
SETTINGS = load_settings()

OLLAMA_URL = SETTINGS.OLLAMA_URL
LLM_MODEL = SETTINGS.LLM_MODEL
LLM_TIMEOUT = SETTINGS.LLM_TIMEOUT
TEMPERATURE = SETTINGS.TEMPERATURE
NUM_CTX = SETTINGS.NUM_CTX

KEEP_ALIVE = "30m"
MAX_PASSAGE_CHARS = SETTINGS.MAX_PASSAGE_CHARS

# The Modelfile tells the model to output exactly this when the passages have no answer.
NOT_FOUND = "NOT_FOUND"
# The start of an answer is held back until it is clear that it is not NOT_FOUND.
HOLD_CHARS = 20

# Keep grounding even if LLM_MODEL points to an existing Ollama model instead of
# the custom model created from the supplied Modelfile.
ANSWER_SYSTEM = """You are the UNEC student assistant. Use ONLY facts from the given passages, never add facts.
Write a clear, natural answer in the requested language (Azerbaijani, Russian or English).
For longer answers, use several short, coherent paragraphs separated by a blank line.
Keep each paragraph focused on one topic, usually 2-4 sentences; never produce a wall of text.
Use Markdown bullet lists for parallel items and numbered lists for ordered steps when helpful.
Use short headings only when a detailed answer covers several distinct topics.
Keep simple answers brief; do not add filler, repetition or unsupported details to create structure.
Keep names, dates and numbers exactly as written. Do not write URLs or links. Do not guess.
If the passages do not answer the question, output exactly NOT_FOUND and nothing else.
If passages contradict each other, prefer the most recent applicable date or explain the difference.
Never mention passage numbers. If rules apply to a specific group or year, state that scope;
never present group-specific or historical rules as general current requirements."""

# Full URL (http(s)://... or www....) -> group 1 is its bare domain.
URL_PATTERN = re.compile(
    r"(?:https?://(?:www\.)?|www\.)([a-z0-9-]+(?:\.[a-z0-9-]+)+)"
    r"(?:/\S*?)?(?=[.,;:!?)\]]*(?:\s|$))",
    re.IGNORECASE,
)


def chat(messages, temperature, on_token=None, num_predict=None, json_output=False) -> str:
    """Streams one /api/chat call; returns the full text with URLs reduced to domains."""
    options = {"temperature": temperature, "num_ctx": NUM_CTX}

    if num_predict is not None:
        options["num_predict"] = num_predict

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": True,
        "think": False,
        "keep_alive": KEEP_ALIVE,
        "options": options,
    }

    if json_output:
        payload["format"] = "json"

    parts = []
    guard = None if json_output else RepetitionGuard()

    def emit(text):
        text = URL_PATTERN.sub(r"\1", text)
        parts.append(text)

        if on_token and text:
            on_token(text)

    with requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        stream=True,
        timeout=LLM_TIMEOUT,
    ) as response:
        response.raise_for_status()

        pending = ""

        for line in response.iter_lines():
            if not line:
                continue

            data = json.loads(line)

            if "error" in data:
                raise RuntimeError(data["error"])

            content = data["message"]["content"]
            pending += guard.feed(content) if guard is not None else content

            # Emit whole words only, so a URL is cleaned before it is shown.
            cut = max(pending.rfind(" "), pending.rfind("\n")) + 1
            emit(pending[:cut])
            pending = pending[cut:]
            if guard is not None and guard.stopped:
                logging.getLogger(__name__).warning("Stopped a repetitive Ollama generation stream")
                # Closing the response also stops consuming the looping stream.
                break

        if guard is not None:
            pending += guard.finish()
        emit(pending)

    return "".join(parts).strip()


def warm_up():
    """Loads the model into memory so the first real answer is fast."""
    chat([{"role": "user", "content": "ok"}], temperature=0, num_predict=1)


def generate(
    question: str,
    hits: list[dict],
    history: list[dict],
    language: str,
    on_token=None,
) -> str:
    """Returns the answer, or NOT_FOUND (nothing streamed) if the passages have no answer."""
    passages = "\n\n".join(
        f"[{i}] {hit['title']} ({hit['date']})\n{hit['text'][:MAX_PASSAGE_CHARS]}"
        for i, hit in enumerate(hits, start=1)
    )

    messages = [{"role": "system", "content": ANSWER_SYSTEM}]

    for turn in history:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    messages.append(
        {
            "role": "user",
            "content": f"Passages:\n\n{passages}\n\nQuestion: {question}\n\n"
            f"Answer in {language}. If the passages do not contain the answer, "
            f"output only {NOT_FOUND}.",
        }
    )

    held = []
    released = False

    def gate(text):
        nonlocal released

        if released:
            on_token(text)
            return

        held.append(text)
        start = "".join(held).lstrip()

        if len(start) >= HOLD_CHARS and NOT_FOUND not in start[:HOLD_CHARS]:
            released = True
            on_token("".join(held))

    answer = chat(messages, TEMPERATURE, gate if on_token else None)

    if NOT_FOUND in answer[:HOLD_CHARS]:
        return NOT_FOUND

    if on_token and not released:
        on_token("".join(held))  # an answer shorter than HOLD_CHARS

    return answer
