"""Deterministic text-based detection restricted to the supported languages."""
from functools import lru_cache
import logging
import shlex
import os
import sys

LOG = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def detector():
    try:
        from lingua import Language, LanguageDetectorBuilder
    except ImportError as exc:
        commands = [
            [sys.executable, "-m", "pip", "uninstall", "-y", "lingua", "lingua-language-detector"],
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "lingua-language-detector>=2,<3"],
        ]
        # Instructions use this process's interpreter; no automatic package installation.
        rendered = [("& " + " ".join("'" + part.replace("'", "''") + "'" for part in command)) if os.name == "nt"
                    else shlex.join(command) for command in commands]
        raise RuntimeError(
            "Dil detektoru yüklənmədi. lingua-language-detector paketi çatışmır, "
            "natamamdır və ya başqa lingua paketi ilə toqquşur. "
            "Bu mühitin terminalında ardıcıl icra edin:\n" + "\n".join(rendered)
        ) from exc
    result = LanguageDetectorBuilder.from_languages(
        Language.AZERBAIJANI, Language.ENGLISH, Language.RUSSIAN
    ).build()
    LOG.info("Dil aşkarlama importu və detektor: OK (az/en/ru)")
    return result


def detect_language(title: str, content: str) -> dict:
    text = f"{title}\n{content}".strip()
    if not any(ch.isalpha() for ch in text):
        return {"language": None, "confidence": 0.0, "uncertain": True}
    values = detector().compute_language_confidence_values(text)
    if not values:
        return {"language": None, "confidence": 0.0, "uncertain": True}
    best = values[0]
    margin = best.value - values[1].value if len(values) > 1 else best.value
    return {"language": best.language.iso_code_639_1.name.lower(),
            "confidence": float(best.value),
            "uncertain": best.value < 0.6 or margin < 0.15}
