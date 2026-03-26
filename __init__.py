"""
Top-level Source package.

Provides a unified package structure so submodules can reference each other:
- ESE: code clustering and evaluation module
- live_code_bench: LiveCodeBench execution tooling
"""

from . import ESE
from . import live_code_bench

__all__ = [
    'ESE',
    'live_code_bench',
]

__version__ = "0.1.0"

