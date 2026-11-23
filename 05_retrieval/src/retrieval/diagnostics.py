"""Compatibility path; implementation: retrieval.search.diagnostics."""
import sys
from importlib import import_module

_implementation = import_module("retrieval.search.diagnostics")
sys.modules[__name__] = _implementation
