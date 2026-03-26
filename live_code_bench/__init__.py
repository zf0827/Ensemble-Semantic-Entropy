"""
LiveCodeBench Code Generation Evaluation Framework

This package provides a clean and modular implementation for:
- Loading and formatting LiveCodeBench datasets
- Executing and evaluating code on test cases
- Running self-debug loops with LLM feedback
"""

from .LCB_bench import load_lcb_dataset, prepare_dspy_examples
from .Test_Utils import test_on_single_test, test_on_public_tests, test_on_private_tests
from .LCB_run import run_debug
from .utils import post_process_code, format_test_feedback

__all__ = [
    'load_lcb_dataset',
    'prepare_dspy_examples',
    'test_on_single_test',
    'test_on_public_tests',
    'test_on_private_tests',
    'run_debug',
    'post_process_code',
    'format_test_feedback',
]
