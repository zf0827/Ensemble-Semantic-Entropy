"""
Utility functions for code processing and test feedback formatting.
"""

import re
import json
from typing import Dict, List, Any, Tuple


def post_process_code(code: str) -> str:
    """
    Extract code from markdown code blocks or return as-is.

    Tries to extract code in order:
    1. ```python ... ``` blocks
    2. ``` ... ``` blocks (any language)
    3. <code>...</code> tags
    4. Return original code if no blocks found

    Args:
        code: Raw code string potentially containing markdown

    Returns:
        Cleaned code string
    """
    # Try matching ```python...``` blocks
    pattern_python = r'```python\s*\n(.*?)```'
    matches = re.findall(pattern_python, code, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # Try matching ```...``` blocks (any language)
    pattern_generic = r'```(?:.*?)\n(.*?)```'
    matches = re.findall(pattern_generic, code, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # Try matching <code>...</code> tags
    pattern_code_tag = r'<code>(.*?)</code>'
    matches = re.findall(pattern_code_tag, code, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # Return original code if no blocks found
    return code.strip()


def parse_value(value_str: str) -> Any:
    """
    Parse a string value into appropriate Python type (int, float, list, or str).

    Args:
        value_str: String to parse

    Returns:
        Parsed value in appropriate type
    """
    value_str = value_str.strip()

    # Try parsing as JSON (handles lists, dicts, etc.)
    if value_str.startswith(('[', '{')):
        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            pass

    # Try parsing as int
    try:
        return int(value_str)
    except ValueError:
        pass

    # Try parsing as float
    try:
        return float(value_str)
    except ValueError:
        pass

    # Return as string, removing quotes if present
    return value_str.strip('"\'')


def format_test_feedback(
    test_idx: int,
    test_input: Any,
    expected_output: Any,
    actual_output: Any,
    passed: bool,
    execution_time: float = None,
    error_msg: str = None
) -> str:
    """
    Format test execution result in a model-friendly way.

    Args:
        test_idx: Test case index
        test_input: Input to the test
        expected_output: Expected output
        actual_output: Actual output from code execution
        passed: Whether test passed
        execution_time: Execution time in seconds (optional)
        error_msg: Error message if test failed (optional)

    Returns:
        Formatted feedback string
    """
    feedback = f"[Test {test_idx}]\n"
    feedback += f"Input: {test_input}\n"
    feedback += f"Expected: {expected_output}\n"
    feedback += f"Actual: {actual_output}\n"

    if passed:
        feedback += "Status: PASSED ✓\n"
    else:
        feedback += "Status: FAILED ✗\n"
        if error_msg:
            feedback += f"Error: {error_msg}\n"

    if execution_time is not None:
        feedback += f"Time: {execution_time:.3f}s\n"

    return feedback


def format_test_results_summary(results: List[Tuple[bool, str, Any, float]]) -> str:
    """
    Format a summary of all test results.

    Args:
        results: List of (passed, error_msg, output_value, time_elapsed) tuples

    Returns:
        Formatted summary string
    """
    num_passed = sum(1 for r in results if r[0])
    num_total = len(results)

    summary = f"\n{'='*50}\n"
    summary += f"Test Results Summary\n"
    summary += f"{'='*50}\n"
    summary += f"Passed: {num_passed}/{num_total}\n"

    if num_passed < num_total:
        summary += f"\nFailed tests:\n"
        for idx, (passed, error_msg, _, _) in enumerate(results):
            if not passed:
                summary += f"  - Test {idx}: {error_msg}\n"

    summary += f"{'='*50}\n"

    return summary


def format_test_for_debugging(
    test_case: Dict,
    result: Tuple[bool, str, Any, float],
    is_stdin: bool
) -> str:
    """
    Format a single test case and its result for debugging purposes.

    Args:
        test_case: Test case dictionary
        result: Test result tuple (passed, error_msg, output, time)
        is_stdin: Whether test uses stdin/stdout

    Returns:
        Formatted debug string
    """
    passed, error_msg, output, time_elapsed = result

    debug_str = "\n" + "-"*50 + "\n"

    if is_stdin:
        debug_str += f"Input (stdin):\n{test_case['input']}\n"
        debug_str += f"Expected Output (stdout):\n{test_case['output']}\n"
    else:
        debug_str += f"Input: {test_case['input']}\n"
        debug_str += f"Expected Output: {test_case['output']}\n"

    debug_str += f"Actual Output: {output}\n"
    debug_str += f"Status: {'PASS' if passed else 'FAIL'}\n"
    debug_str += f"Time: {time_elapsed:.3f}s\n"

    if not passed:
        debug_str += f"Error: {error_msg}\n"

    debug_str += "-"*50 + "\n"

    return debug_str


# Model name mapping for convenience
MODEL_NAME_MAP = {
    "4o-mini": 'openai/gpt-4o-mini',
    "4o": 'openai/gpt-4o',
    "o1-mini": 'openai/o1-mini',
    "o1": 'openai/o1-preview',
    "o3-mini": 'openai/o3-mini',
    "o1-preview": 'openai/o1-preview',
}
