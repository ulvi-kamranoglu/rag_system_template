"""Compatibility path; implementation: retrieval.indexing.storage."""
import sys
from importlib import import_module

_implementation = import_module("retrieval.indexing.storage")

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
