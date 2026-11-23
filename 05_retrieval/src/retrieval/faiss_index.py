"""Compatibility path; implementation: retrieval.backends.faiss."""
import sys
from importlib import import_module

_implementation = import_module("retrieval.backends.faiss")
sys.modules[__name__] = _implementation
