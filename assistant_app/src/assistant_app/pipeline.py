"""Connection between the retriever and the LLM: answer(question, history)."""
from datetime import datetime
import json
import time

from .llm.client import LLM_MODEL, NOT_FOUND, chat, generate, warm_up
from .llm.replies import REPLIES
from .retrieval_adapter import search
from .config import load_settings

SETTINGS = load_settings()
TOP_K = SETTINGS.TOP_K
HISTORY_TURNS = SETTINGS.HISTORY_TURNS

LOG_PATH = SETTINGS.LOG_PATH

LANGUAGE_NAMES = {"az": "Azerbaijani", "ru": "Russian", "en": "English"}
INTENTS = {"greeting", "thanks", "off_topic", "unec"}

# Obvious messages are answered without the LLM router: text -> (intent, language).
KEYWORDS = {
    "salam": ("greeting", "az"),
    "salam aleykum": ("greeting", "az"),
    "hello": ("greeting", "en"),
    "hi": ("greeting", "en"),
    "привет": ("greeting", "ru"),
    "здравствуйте": ("greeting", "ru"),
    "sağ ol": ("thanks", "az"),
    "sağol": ("thanks", "az"),
    "təşəkkürlər": ("thanks", "az"),
    "thanks": ("thanks", "en"),
    "thank you": ("thanks", "en"),
    "спасибо": ("thanks", "ru"),
}

ROUTER_PROMPT = """You route messages for the virtual assistant of UNEC (Azerbaijan State University of Economics).
Reply with one JSON object and nothing else:
{"intent": "greeting" | "thanks" | "off_topic" | "unec", "language": "az" | "ru" | "en", "search_query": "..."}

intent:
- "greeting": the message only greets, asks how the assistant is, or asks who the assistant is.
- "thanks": the message only thanks or says goodbye.
- "unec": anything about UNEC, admission, studies, exams, programs, fees, scholarships, student life, dormitory, university structure, staff, contacts. A question about education that names no university is about UNEC.
- "off_topic": unrelated to UNEC or education: weather, coding, general chat, internals of other universities, etc.

language: the language the last message is written in. Short Latin-script words like "salam", "sağol", "necəsən" are Azerbaijani ("az"). Cyrillic is "ru".

search_query: for "unec", the last message as one standalone question. It MUST be written in Azerbaijani even when the message is in Russian or English: translate it. For every other intent, "".

Examples:
Last message: Какие документы нужны для магистратуры?
{"intent": "unec", "language": "ru", "search_query": "Magistratura üçün hansı sənədlər lazımdır?"}
Last message: What is the tuition fee?
{"intent": "unec", "language": "en", "search_query": "Təhsil haqqı nə qədərdir?"}
Last message: salam, necəsən?
{"intent": "greeting", "language": "az", "search_query": ""}

Use the conversation ONLY to resolve a follow-up that cannot be understood alone, such as "bəs bakalavr üçün?": fill in what it leaves out. If the last message can be understood alone, or is a greeting, or starts a new topic, ignore the conversation completely. Never add a topic the user did not ask about."""


class LLMError(Exception):
    pass


class SearchError(Exception):
    pass


def has_cyrillic(text: str) -> bool:
    return any("а" <= char <= "я" or char == "ё" for char in text.lower())


def fallback_route(question: str) -> dict:
    """Used when the router's JSON is unusable."""
    return {
        "intent": "unec",
        "language": "ru" if has_cyrillic(question) else "az",
        "search_query": question,
    }


def route(question: str, history: list[dict]) -> dict:
    """{"intent", "language", "search_query"}: what kind of message this is and what to search."""
    keyword = KEYWORDS.get(" ".join(question.lower().strip(" .,!?;:)(").split()))

    if keyword:
        return {"intent": keyword[0], "language": keyword[1], "search_query": ""}

    user = f"Last message: {question}"

    if history:
        conversation = "\n".join(
            f"User: {turn['question']}\nAssistant: {turn['answer'][:300]}"
            for turn in history
        )
        user = f"Conversation:\n{conversation}\n\n{user}"

    reply = chat(
        [
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0,
        json_output=True,
    )

    try:
        data = json.loads(reply)
        intent, language = data["intent"], data["language"]
        search_query = str(data.get("search_query") or "").strip()
    except (ValueError, KeyError, TypeError):
        return fallback_route(question)

    if intent not in INTENTS or language not in LANGUAGE_NAMES:
        return fallback_route(question)

    # The script settles what the model sometimes gets wrong: Russian is always Cyrillic.
    if has_cyrillic(question):
        language = "ru"
    elif language == "ru":
        language = "az"

    if intent != "unec":
        search_query = ""
    elif not search_query:
        search_query = question

    return {"intent": intent, "language": language, "search_query": search_query}


def log_turn(record: dict):
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **record}

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def answer(question: str, history: list[dict], on_query=None, on_token=None) -> dict:
    """history: [{"question", "answer"}, ...]; only the last HISTORY_TURNS are used.

    on_query(search_query) and on_token(text) are optional callbacks for live output;
    on_query is called only when the retriever is searched.
    Returns answer, sources, search_query ("" if nothing was searched), timings,
    intent, language and show_sources. Raises LLMError or SearchError.
    """
    history = history[-HISTORY_TURNS:] if HISTORY_TURNS > 0 else []

    result = {
        "answer": "",
        "sources": [],
        "search_query": question,
        "timings": {"translate": 0.0, "search": 0.0, "llm": 0.0},
        "intent": "unec",
        "language": None,
        "show_sources": False,
    }
    error = None

    try:
        start = time.perf_counter()

        try:
            routed = route(question, history)
        except Exception as exc:
            raise LLMError(str(exc)) from exc

        result.update(routed)
        result["timings"]["translate"] = time.perf_counter() - start

        if result["intent"] != "unec":
            result["answer"] = REPLIES[result["intent"]][result["language"]]

            if on_token:
                on_token(result["answer"])

            return result

        if on_query:
            on_query(result["search_query"])

        start = time.perf_counter()

        try:
            hits = search(result["search_query"], TOP_K)
        except Exception as exc:
            raise SearchError(str(exc)) from exc

        result["timings"]["search"] = time.perf_counter() - start

        for hit in hits:
            source = {"title": hit["title"], "url": hit["url"], "date": hit["date"],
                      "source": str(hit.get("source") or "")}

            if source not in result["sources"]:
                result["sources"].append(source)

        start = time.perf_counter()

        try:
            result["answer"] = generate(
                question, hits, history, LANGUAGE_NAMES[result["language"]], on_token
            )
        except Exception as exc:
            raise LLMError(str(exc)) from exc

        result["timings"]["llm"] = time.perf_counter() - start

        if result["answer"] == NOT_FOUND:
            result["answer"] = REPLIES["not_found"][result["language"]]

            if on_token:
                on_token(result["answer"])
        else:
            result["show_sources"] = True

        return result

    except (LLMError, SearchError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise

    finally:
        log_turn(
            {
                "question": question,
                "intent": result["intent"],
                "language": result["language"],
                "search_query": result["search_query"],
                "source_urls": [source["url"] for source in result["sources"]],
                "answer": result["answer"],
                "timings": result["timings"],
                "model": LLM_MODEL,
                "error": error,
            }
        )


def load():
    """Loads retriever models/index and the LLM once. Raises LLMError if Ollama is down."""
    search("UNEC", 1)

    try:
        warm_up()
    except Exception as exc:
        log_turn({"question": None, "model": LLM_MODEL, "error": f"warm-up: {exc}"})
        raise LLMError(str(exc)) from exc
