"""Compatibility path; implementation: retrieval.search.reranker."""
import sys
from importlib import import_module

_implementation = import_module("retrieval.search.reranker")
sys.modules[__name__] = _implementation
