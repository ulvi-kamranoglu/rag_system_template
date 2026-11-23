"""Checkout compatibility bridge; implementation lives under src/."""
from pathlib import Path as _Path
__path__ = [str(_Path(__file__).resolve().parent / "src" / "orchestration")]
