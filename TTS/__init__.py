"""
Cascade Test-Time Scaling framework.

An entropy-based multi-layer code generation framework that decides whether to
sample more by evaluating uncertainty layer by layer.
"""

from .cascade_tts import CascadeTTS

__all__ = ["CascadeTTS"]
__version__ = "1.0.0"


