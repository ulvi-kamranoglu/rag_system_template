"""Compatibility path; implementation: retrieval.search.policy."""
import sys
from importlib import import_module

_implementation = import_module("retrieval.search.policy")
sys.modules[__name__] = _implementation
