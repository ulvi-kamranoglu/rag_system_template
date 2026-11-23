"""Compatibility path; implementation: retrieval.indexing.sync."""
import sys
from importlib import import_module

_implementation = import_module("retrieval.indexing.sync")

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
