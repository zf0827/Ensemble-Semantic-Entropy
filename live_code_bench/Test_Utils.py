"""
Test execution utilities for LiveCodeBench.

This module provides optimized test execution functions:
- Test on single test case
- Test on multiple test cases (public/private)
- Multiprocessing-based safe execution with timeout
"""

import io
import sys
import time
import json
import copy
import signal
import multiprocessing
import contextlib
import faulthandler
from io import StringIO
from typing import Dict, List, Tuple, Any, Optional

from .utils import parse_value


# =============================================================================
# Test Input/Output Preparation
# =============================================================================

def prepare_test_io_functional(test_case: Dict, is_extracted: bool) -> Tuple[Any, Any]:
    """
    Parse functional test case input and output.

    Args:
        test_case: Test case dictionary with 'input' and optional 'output' keys
        is_extracted: Whether this is an extracted test from problem description

    Returns:
        Tuple of (parsed_input, parsed_output). parsed_output is None if 'output' key is missing.
    """
    if not is_extracted:
        # Direct JSON format: {"input": {...}, "output": ...}
        # Support test cases without 'output' field (for test generators)
        return test_case["input"], test_case.get("output", None)

    # Extracted test format - needs parsing
    input_str = test_case["input"]
    # Support test cases without 'output' field
    if "output" not in test_case:
        return input_str, None
    expected_output = test_case["output"].strip()

    # Parse input
    inputs = []
    if "=" in input_str:
        # Key-value format: "x=1, y=2" or "x=1"
        input_parts = [part.strip() for part in input_str.split(",")]
        for part in input_parts:
            if "=" in part:
                key, value = part.split("=", 1)
                inputs.append(parse_value(value.strip()))
    else:
        # Standard format with newlines
        for part in input_str.split("\n"):
            part = part.strip()
            if part:
                inputs.append(parse_value(part))

    # Parse expected output
    try:
        expected_output = json.loads(expected_output)
    except json.JSONDecodeError:
        expected_output = expected_output.strip()

    return inputs, expected_output


def prepare_test_io_stdin(test_case: Dict) -> Tuple[str, Optional[str]]:
    """
    Parse stdin/stdout test case input and output.

    Args:
        test_case: Test case dictionary with 'input' and optional 'output' keys

    Returns:
        Tuple of (input_string, output_string). output_string is None if 'output' key is missing.
    """
    test_input = test_case["input"]
    # Support test cases without 'output' field (for test generators)
    if "output" not in test_case:
        return test_input, None
    
    test_output = test_case["output"].strip()

    # Remove trailing '-' if present
    if test_output.endswith("-"):
        test_output = test_output[:test_output.rfind("-")].rstrip()

    return test_input, test_output


# =============================================================================
# Test Execution
# =============================================================================

def run_test_functional(
    completion: str,
    test_input: Any,
    test_output: Any,
    is_extracted: bool
) -> Tuple[bool, Any]:
    """
    Run a functional test (function call with arguments).

    Args:
        completion: Generated code
        test_input: Input arguments
        test_output: Expected output (None if not provided)
        is_extracted: Whether test is extracted from problem

    Returns:
        Tuple of (passed, actual_output). If test_output is None, always returns (True, actual_output).
    """
    namespace = {}

    # Execute the generated code
    try:
        exec(completion, namespace)
    except Exception as e:
        return False, f"Error: {str(e)}"

    # Get function name
    func_name = completion.split("(")[0].split()[-1]

    # Prepare arguments
    if not is_extracted:
        if isinstance(test_input, dict):
            args = []
            kwargs = test_input
        else:
            args = [test_input]
            kwargs = {}
    else:
        args = test_input if isinstance(test_input, list) else [test_input]
        kwargs = {}

    # Redirect stdout
    output = io.StringIO()
    sys.stdout = output

    try:
        result = namespace[func_name](*args, **kwargs)

        # If no expected output provided, just return the actual output (always pass)
        if test_output is None:
            return True, result

        if result != test_output:
            return False, result

        return True, result

    except Exception as e:
        return False, f"Error: {str(e)}"

    finally:
        sys.stdout = sys.__stdout__


def run_test_stdin(
    completion: str,
    test_input: str,
    test_output: Optional[str]
) -> Tuple[bool, str]:
    """
    Run a stdin/stdout test.

    Args:
        completion: Generated code
        test_input: Input string (for stdin)
        test_output: Expected output string (from stdout), None if not provided

    Returns:
        Tuple of (passed, actual_output). If test_output is None, always returns (True, actual_output).
    """
    # Redirect stdin
    sys.stdin = StringIO(test_input)

    # Redirect stdout
    output = StringIO()
    sys.stdout = output

    # Handle __main__ execution
    if '__name__ == "__main__"' in completion:
        completion = f'__name__ = "__main__"\n' + completion

    namespace = {}
    try:
        exec(completion, namespace)
    except Exception as e:
        return False, f"Error: {str(e)}"
    finally:
        sys.stdout = sys.__stdout__
        sys.stdin = sys.__stdin__

    actual_output = output.getvalue().strip()
    
    # If no expected output provided, just return the actual output (always pass)
    if test_output is None:
        return True, actual_output
    
    return actual_output == test_output, actual_output


# =============================================================================
# Safe Multiprocess Test Execution
# =============================================================================

def reliability_guard():
    """
    Disable dangerous functions to prevent generated code from interfering with test.

    WARNING: This is NOT a security sandbox. Do not run untrusted code outside a proper sandbox.
    """
    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None

    import subprocess
    subprocess.Popen = None


def _run_tests_worker(
    test_cases: List[Dict],
    completion: str,
    result_list: multiprocessing.Manager().list,
    is_extracted: bool,
    fast_check: bool
):
    """
    Worker function to run tests in a separate process.

    Args:
        test_cases: List of test case dictionaries
        completion: Generated code to test
        result_list: Shared list to store results
        is_extracted: Whether tests are extracted from problem
        fast_check: Whether to stop on first failure
    """
    reliability_guard()

    test_type = test_cases[0]["testtype"]

    # Process tests in reverse order to trigger fast_check on timeout tests earlier
    for test_case in test_cases[::-1]:
        time_start = time.time()

        try:
            if test_type == "functional":
                test_input, test_output = prepare_test_io_functional(test_case, is_extracted)
                passed, actual_output = run_test_functional(
                    completion,
                    copy.deepcopy(test_input),
                    copy.deepcopy(test_output),
                    is_extracted
                )
            else:  # stdin
                test_input, test_output = prepare_test_io_stdin(test_case)
                passed, actual_output = run_test_stdin(
                    completion,
                    copy.deepcopy(test_input),
                    copy.deepcopy(test_output)
                )

            time_elapsed = time.time() - time_start

            if test_output is None:
                # No expected output - just report the actual output
                error_msg = f"For test input: {test_input}. Actual output: {actual_output}."
            elif passed:
                error_msg = f"For test input: {test_input}. Expected output is: {test_output}, " \
                           f"your solution correctly passes this test with output {actual_output}."
            else:
                error_msg = f"For test input: {test_input}. Expected output is: {test_output}, " \
                           f"but got: {actual_output}."

        except Exception as e:
            passed = False
            error_msg = f"Error during test execution: {str(e)}"
            actual_output = f"Error: {str(e)}"
            time_elapsed = time.time() - time_start

        result_list.append((passed, error_msg, actual_output, time_elapsed))

        # Fast check: stop on first failure
        if fast_check and not passed:
            return


def test_on_single_test(
    code: str,
    test_case: Dict,
    timeout: float = 6.0,
    is_extracted: bool = False
) -> Tuple[bool, str, Any, float]:
    """
    Test code on a single test case with timeout protection.

    Args:
        code: Generated code to test
        test_case: Single test case dictionary
        timeout: Timeout in seconds
        is_extracted: Whether test is extracted from problem

    Returns:
        Tuple of (passed, error_message, output_value, time_elapsed)
    """
    manager = multiprocessing.Manager()
    result_list = manager.list()

    p = multiprocessing.Process(
        target=_run_tests_worker,
        args=([test_case], code, result_list, is_extracted, False)
    )
    p.start()
    p.join(timeout=timeout + 1)

    if p.is_alive():
        p.kill()
        return (False, "Time out!", "Error: Time out!", float("inf"))

    if len(result_list) == 0:
        return (False, "No result", "Error: No result", float("inf"))

    return result_list[0]


def test_on_public_tests(
    code: str,
    public_tests: List[Dict],
    timeout: float = 6.0,
    is_extracted: bool = True,
    fast_check: bool = True
) -> Dict[str, Any]:
    """
    Test code on public test cases.

    Args:
        code: Generated code to test
        public_tests: List of public test case dictionaries
        timeout: Timeout in seconds per test
        is_extracted: Whether tests are extracted from problem
        fast_check: Whether to stop on first failure

    Returns:
        Dictionary containing test results:
        - passed: bool, whether all tests passed
        - details: List[bool], pass/fail for each test
        - error_messages: List[str], error message for each test
        - output_values: List[Any], actual output for each test
        - time_elapsed: List[float], time for each test
    """
    manager = multiprocessing.Manager()
    result_list = manager.list()

    total_timeout = timeout * len(public_tests)
    if fast_check and total_timeout > 80:
        total_timeout = 80

    p = multiprocessing.Process(
        target=_run_tests_worker,
        args=(public_tests, code, result_list, is_extracted, fast_check)
    )
    p.start()
    p.join(timeout=total_timeout)

    if p.is_alive():
        p.kill()

    # Fill in timeout results for missing tests
    for i in range(len(public_tests) - len(result_list)):
        result_list.append((False, "Time out!", "Error: Time out!", float("inf")))

    # Parse results
    details = [r[0] for r in result_list]
    error_messages = [r[1] for r in result_list]
    output_values = [r[2] for r in result_list]
    time_elapsed = [r[3] for r in result_list]
    timeout_details = ["Timed out!" in msg for msg in error_messages]

    return {
        "passed": all(details),
        "details": details,
        "timeout_details": timeout_details,
        "error_messages": error_messages,
        "output_values": output_values,
        "time_elapsed": time_elapsed
    }


def test_on_private_tests(
    code: str,
    private_tests: List[Dict],
    timeout: float = 6.0,
    is_extracted: bool = False
) -> Dict[str, Any]:
    """
    Test code on private test cases.

    This is the same as test_on_public_tests but with different defaults.

    Args:
        code: Generated code to test
        private_tests: List of private test case dictionaries
        timeout: Timeout in seconds per test
        is_extracted: Whether tests are extracted (usually False for private)

    Returns:
        Dictionary containing test results (same format as test_on_public_tests)
    """
    return test_on_public_tests(
        code,
        private_tests,
        timeout=timeout,
        is_extracted=is_extracted,
        fast_check=True
    )
