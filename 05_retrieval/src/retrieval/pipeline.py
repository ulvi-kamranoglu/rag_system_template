"""Compatibility path; implementation: retrieval.indexing.qdrant."""
import sys
from importlib import import_module

_implementation = import_module("retrieval.indexing.qdrant")

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
