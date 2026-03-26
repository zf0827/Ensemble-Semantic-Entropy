"""
DSPy Prompt Templates and Signatures for LiveCodeBench.

This module contains all DSPy signature definitions for:
- Code generation (stdin and functional types)
- Test generation
- Self-debugging
- Code selection
"""

import dspy


# =============================================================================
# Code Generation Signatures
# =============================================================================

class GenerateLCBCodeStdin(dspy.Signature):
    """Generate Python code that reads from stdin and writes to stdout."""

    prompt = dspy.InputField(format=str)
    code = dspy.OutputField(
        desc="Executable Python script generated from the given prompt. "
             "Define a `main()` function that reads from stdin, prints the required output, "
             "and include `if __name__ == \"__main__\": main()` at the end. "
             "Return ONLY the runnable code without extra comments or explanations."
    )


class GenerateLCBCodeFunctional(dspy.Signature):
    """Generate a Python function to solve the problem."""

    prompt = dspy.InputField(format=str)
    code = dspy.OutputField(
        desc="Executable Python function generated from the given prompt. "
             "DO NOT include anything other than function body! Give me only the function itself!"
    )


# =============================================================================
# Self-Debug Signatures
# =============================================================================

class SelfDebug(dspy.Signature):
    """Debug and correct code based on test feedback."""

    prompt = dspy.InputField(format=str)
    code = dspy.OutputField(
        desc="Here is the past history of your code and the test case feedback. "
             "Please reason why your code failed in the last round, and correct the code. "
             "Do not write non-code content in the code field.",
        max_length=2048
    )


# =============================================================================
# Test Generation Signatures
# =============================================================================

class GenerateTestsStdin(dspy.Signature):
    """Generate test cases for stdin/stdout style problems."""

    prompt = dspy.InputField(format=str)
    tests = dspy.OutputField(
        desc='Generate a complete set of potential test cases to test an AI-generated solution to the coding problem. '
             'Assume input will be used via stdin and output is captured from stdout. Cover: '
             '(i) Edge cases, such as empty string or arrays, '
             '(ii) Complex and difficult inputs, but do not include very long inputs. '
             '(iii) Other ones that can maximize the chance of catching a bug. '
             'You MUST return a JSON array (starting with [ and ending with ]) containing test case objects. '
             'Each object should have this exact format: '
             '{"input": <stdin_input_string>, "output": <expected_stdout_string>} '
             'For multi-line inputs or outputs, use \\n to represent newlines within the JSON string. '
             'Example: [{"input": "3\\n1 2 3", "output": "Yes"}, {"input": "5\\n5 4 3 2 1", "output": "No"}] '
             'Ensure the input and output are strings that match what would be provided via stdin/stdout. '
             'Do not wrap the JSON array in any object or add any field names like "tests". '
             'Do not include any additional text, explanations, or markdown code blocks. Return ONLY the JSON array.'
    )


class GenerateTestsFunctional(dspy.Signature):
    """Generate test cases for functional style problems."""

    prompt = dspy.InputField(format=str)
    tests = dspy.OutputField(
        desc='Generate a complete set of potential test cases to test an AI-generated solution to the coding problem. Cover: '
             '(i) Edge cases, such as empty string or arrays, '
             '(ii) Complex and difficult inputs, but do not include very long inputs. '
             '(iii) Other ones that can maximize the chance of catching a bug. '
             'You MUST return a JSON array (starting with [ and ending with ]) containing test case objects. '
             'Each object should have this exact format: '
             '{"input": <example_input>, "output": <expected_output>} '
             'Example of the expected format: [{"input": {"x": 1, "y": 2}, "output": 3}, {"input": {"x": 0, "y": 0}, "output": 0}] '
             'Ensure the input and output match the types and structure expected for the problem. '
             'Do not wrap the JSON array in any object or add any field names like "tests". '
             'Do not include any additional text, explanations, or markdown code blocks. Return ONLY the JSON array.'
    )


# =============================================================================
# Test Input Parsing
# =============================================================================

def post_process_tests_inputs_stdin(raw_text: str) -> list:
    """
    Parse LLM-generated test cases for stdin/stdout format.

    Args:
        raw_text: Raw text from LLM containing test cases

    Returns:
        List of test case dictionaries with 'input', 'output', 'testtype' keys
    """
    import json
    import re

    # Clean the input string
    cleaned_string = raw_text.strip().strip("```json").strip("```").strip()

    # Try to parse as JSON array (new format)
    if cleaned_string.startswith("[") and cleaned_string.endswith("]"):
        try:
            test_cases = json.loads(cleaned_string)
            formatted_tests = []

            for test_case in test_cases:
                if isinstance(test_case, dict) and "input" in test_case and "output" in test_case:
                    input_str = str(test_case["input"])
                    output_str = str(test_case["output"])

                    # Ensure strings end with newline
                    if not input_str.endswith("\n"):
                        input_str += "\n"
                    if not output_str.endswith("\n"):
                        output_str += "\n"

                    formatted_tests.append({
                        "input": input_str,
                        "output": output_str,
                        "testtype": "stdin",
                    })

            return formatted_tests
        except json.JSONDecodeError:
            pass

    # Fallback to text-based parsing
    blocks = raw_text.split("Input:")
    formatted_tests = []

    for block in blocks:
        if not block.strip():
            continue

        input_output = block.split("Output:")

        if len(input_output) == 2:
            input_value = input_output[0].strip()
            output_value = input_output[1].strip()

            formatted_tests.append({
                "input": input_value + "\n",
                "output": output_value + "\n",
                "testtype": "stdin",
            })

    return formatted_tests


def post_process_tests_inputs_functional(raw_text: str) -> list:
    """
    Parse LLM-generated test cases for functional format.

    Args:
        raw_text: Raw text from LLM containing test cases

    Returns:
        List of test case dictionaries with 'input', 'output', 'testtype' keys
    """
    import json
    import re

    # Clean the input string
    cleaned_string = raw_text.strip().strip("```json").strip("```").strip()

    # Evaluate Python expressions in JSON values (e.g., 10**9)
    def evaluate_python_expr(match):
        expr = match.group(0)
        try:
            value = eval(expr, {"__builtins__": {}})
            return str(value)
        except Exception:
            return expr

    # Match numeric expressions in JSON values
    pattern = r'(?<=:\s)(\d+(?:\s*(?:\*\*|[-+*/%])\s*\d+)+)(?=\s*[,\}\]])'
    cleaned_string = re.sub(pattern, evaluate_python_expr, cleaned_string)

    # Try to parse as JSON array
    if cleaned_string.startswith("[") and cleaned_string.endswith("]"):
        try:
            test_cases = json.loads(cleaned_string)
            for test_case in test_cases:
                test_case["testtype"] = "functional"
            return test_cases
        except json.JSONDecodeError:
            pass

    # Handle concatenated JSON objects
    json_pattern = re.compile(r'(\{\s*"input"\s*:.*?"output"\s*:.*?\})', re.DOTALL)
    matches = json_pattern.findall(cleaned_string)

    if matches:
        json_array_string = "[" + ",".join(matches) + "]"
        try:
            test_cases = json.loads(json_array_string)
            for test_case in test_cases:
                test_case["testtype"] = "functional"
            return test_cases
        except json.JSONDecodeError:
            pass

    # Fallback: line-by-line parsing
    test_cases = []
    for line in cleaned_string.split("\n"):
        try:
            test_case = json.loads(line)
            test_case["testtype"] = "functional"
            test_cases.append(test_case)
        except json.JSONDecodeError:
            continue

    return test_cases
