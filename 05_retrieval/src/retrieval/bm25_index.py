"""Compatibility path; implementation: retrieval.backends.bm25."""
import sys
from importlib import import_module

_implementation = import_module("retrieval.backends.bm25")
sys.modules[__name__] = _implementation
