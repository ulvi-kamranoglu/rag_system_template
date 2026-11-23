"""Compatibility path; implementation: retrieval.search.flow."""
import sys
from importlib import import_module

_implementation = import_module("retrieval.search.flow")
sys.modules[__name__] = _implementation
