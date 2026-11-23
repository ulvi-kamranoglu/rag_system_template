"""Compatibility path; implementation: retrieval.backends.qdrant."""
import sys
from importlib import import_module

_implementation = import_module("retrieval.backends.qdrant")
sys.modules[__name__] = _implementation
