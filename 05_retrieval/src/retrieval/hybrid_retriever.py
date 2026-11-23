"""Compatibility path; implementation: retrieval.search.retriever."""
import sys
from importlib import import_module

_implementation = import_module("retrieval.search.retriever")
sys.modules[__name__] = _implementation
