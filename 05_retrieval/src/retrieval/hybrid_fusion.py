"""Compatibility path; implementation: retrieval.search.fusion."""
import sys
from importlib import import_module

_implementation = import_module("retrieval.search.fusion")
sys.modules[__name__] = _implementation
