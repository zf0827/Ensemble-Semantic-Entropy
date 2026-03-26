"""
TestInputGenerator - test input generator.

Based on DSPy. Supports generating test cases from problem prompts and code.
"""

import dspy
import json
import re
import logging
import subprocess
import tempfile
import os
import ast
from typing import List, Dict, Any, Optional

# Configure logger.
logger = logging.getLogger(__name__)


def _convert_single_quotes_to_json(text: str) -> str:
    """
    Convert Python dict literals (single quotes) to JSON (double quotes).

    Args:
        text: string that may contain single quotes

    Returns:
        JSON-compatible string with double quotes

    Raises:
        ValueError: if conversion fails
    """
    # Try ast.literal_eval first to safely parse Python literals.
    # This handles single/double quotes, numbers, lists, dicts, etc.
    try:
        parsed = ast.literal_eval(text)
        # Convert the parsed object back to JSON.
        result = json.dumps(parsed)
        logger.debug(f"[_convert_single_quotes_to_json] Successfully converted using ast.literal_eval")
        return result
    except (ValueError, SyntaxError) as e:
        logger.debug(f"[_convert_single_quotes_to_json] ast.literal_eval failed: {e}, trying regex replacement")
        
        # If ast.literal_eval fails, fall back to regex replacement.
        # This method is more conservative, only replacing quotes for keys
        # and string values. It avoids matching quotes inside string content.
        #
        # Strategy: find all strings wrapped by single quotes (keys and values).
        # Pattern: match '...' while excluding escaped quotes (\').
        def replace_quotes(match):
            content = match.group(1)
            # Escape double quotes in content and wrap in double quotes.
            # Note: backslashes inside single-quoted strings need special handling.
            escaped_content = content.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped_content}"'
        
        # Match single-quoted strings: '...' or '...\'...'
        # This regex handles escaped single quotes.
        # (?:[^'\\]|\\.)* matches any character while handling escape sequences.
        pattern = r"'((?:[^'\\]|\\.)*)'"
        result = re.sub(pattern, replace_quotes, text)
        
        # Validate the converted string as JSON.
        try:
            json.loads(result)
            logger.debug(f"[_convert_single_quotes_to_json] Successfully converted using regex replacement")
            return result
        except json.JSONDecodeError as json_err:
            logger.warning(f"[_convert_single_quotes_to_json] Regex replacement result is not valid JSON: {json_err}")
            # If still failing, raise the original error.
            raise ValueError(f"Failed to convert single quotes to JSON: {e}") from e


class GenerateTestsFromPrompt_stdin(dspy.Signature):
    """Generate test cases from a prompt (stdin-style)."""
    prompt = dspy.InputField(format=str, desc="Problem prompt")
    tests = dspy.OutputField(
        desc='Generate a complete set of potential test cases to test an AI-generated solution to the coding problem. Assume input will be used via stdin and output is captured from stdout. Cover: (i) Edge cases, such as empty string or arrays, (ii) Complex and difficult inputs, but do not include very long inputs. (iii) Other ones that can maximize the chance of catching a bug. \
        You MUST return a JSON array (starting with [ and ending with ]) containing test case objects. Each object should have this exact format: \
        {"input": <stdin_input_string>, "output": <expected_stdout_string>} \
        For multi-line inputs or outputs, use \\n to represent newlines within the JSON string. \
        Example: [{"input": "3\\n1 2 3", "output": "Yes"}, {"input": "5\\n5 4 3 2 1", "output": "No"}] \
        Ensure the input and output are strings that match what would be provided via stdin/stdout. \
        Do not wrap the JSON array in any object or add any field names like "tests". Do not include any additional text, explanations, or markdown code blocks. Return ONLY the JSON array.'
    )


class GenerateTestsFromPrompt_func(dspy.Signature):
    """Generate test cases from a prompt (functional-style)."""
    prompt = dspy.InputField(format=str, desc="Problem prompt")
    tests = dspy.OutputField(
        desc='Generate a complete set of potential test cases to test an AI-generated solution to the coding problem. Cover: (i) Edge cases, such as empty string or arrays, (ii) Complex and difficult inputs, but do not include very long inputs. (iii) Other ones that can maximize the chance of catching a bug. \
        You MUST return a JSON array (starting with [ and ending with ]) containing test case objects. Each object should have this exact format: \
        {"input": <example_input>, "output": <expected_output>} \
        Example of the expected format: [{"input": {"x": 1, "y": 2}, "output": 3}, {"input": {"x": 0, "y": 0}, "output": 0}] \
        Ensure the input and output match the types and structure expected for the problem. \
        Do not wrap the JSON array in any object or add any field names like "tests". Do not include any additional text, explanations, or markdown code blocks. Return ONLY the JSON array.'
    )


class GenerateTestGeneratorCode_stdin(dspy.Signature):
    """Generate test generator code from a prompt (stdin-style)."""
    prompt = dspy.InputField(format=str, desc="Problem prompt")
    generator_code = dspy.OutputField(
        desc='Generate a complete Python program that generates test inputs for the coding problem. The program should: \
        (1) Be a standalone Python script that requires no input and can be executed directly. \
        (2) Use random methods (e.g., random.randint, random.choice, random.shuffle) to generate test inputs that conform to the problem requirements. \
        (3) Generate exactly 5 test cases. \
        (4) Output a JSON array string containing exactly 5 test case objects. Each object should have this exact format: \
        {"input": <stdin_input_string>} \
        Note: Only generate the input field, NOT the output field. \
        For multi-line inputs, use \\n to represent newlines within the JSON string. \
        Example output format: [{"input": "3\\n1 2 3"}, {"input": "5\\n5 4 3 2 1"}, {"input": "10\\n1 1 1 1 1"}, {"input": "0"}, {"input": "100\\n" + " ".join([str(i) for i in range(100)])}] \
        The program should use random methods to generate diverse test inputs that cover edge cases (empty inputs, boundary values) and complex cases, but avoid very long inputs. \
        The program should print the JSON array string to stdout when executed. \
        Do not include any markdown code blocks, explanations, or additional text. Return ONLY the Python code.'
    )


class GenerateTestGeneratorCode_func(dspy.Signature):
    """Generate test generator code from a prompt (functional-style)."""
    prompt = dspy.InputField(format=str, desc="Problem prompt")
    generator_code = dspy.OutputField(
        desc='Generate a complete Python program that generates test inputs for the coding problem. The program should: \
        (1) Be a standalone Python script that requires no input and can be executed directly. \
        (2) Use random methods (e.g., random.randint, random.choice, random.shuffle) to generate test inputs that conform to the problem requirements. \
        (3) Generate exactly 5 test cases. \
        (4) Output a JSON array string containing exactly 5 test case objects. Each object should have this exact format: \
        {"input": <example_input>} \
        Note: Only generate the input field, NOT the output field. \
        The input should match the types and structure expected for the problem (e.g., dictionaries, lists, numbers, strings). \
        Example output format: [{"input": {"x": 1, "y": 2}}, {"input": {"x": 0, "y": 0}}, {"input": {"arr": [1, 2, 3]}}, {"input": {"s": ""}}, {"input": {"n": 100}}] \
        The program should use random methods to generate diverse test inputs that cover edge cases (empty inputs, boundary values) and complex cases, but avoid very long inputs. \
        The program should print the JSON array string to stdout when executed. \
        Do not include any markdown code blocks, explanations, or additional text. Return ONLY the Python code.'
    )


class GenerateDifferentiatingTestsFromCodes_stdin(dspy.Signature):
    """Generate differentiating tests from a prompt and two codes (stdin-style)."""
    prompt = dspy.InputField(format=str, desc="Problem prompt")
    codeA = dspy.InputField(format=str, desc="First code")
    codeB = dspy.InputField(format=str, desc="Second code")
    current_test_results = dspy.InputField(
        format=str, 
        desc="Execution results of both codes on the current tests, format: [Test i][Input]...[Output]..."
    )
    tests = dspy.OutputField(
        desc='Generate test cases that can differentiate between the two code solutions. The current tests (shown in current_test_results) produce identical outputs for both codes. Generate new test cases that are likely to produce different outputs. \
        You MUST return a JSON array (starting with [ and ending with ]) containing test case objects. Each object should have this exact format: \
        {"input": <stdin_input_string>, "output": <expected_stdout_string>} \
        For multi-line inputs or outputs, use \\n to represent newlines within the JSON string. \
        Example: [{"input": "3\\n1 2 3", "output": "Yes"}, {"input": "5\\n5 4 3 2 1", "output": "No"}] \
        Ensure the input and output are strings that match what would be provided via stdin/stdout. \
        Do not wrap the JSON array in any object or add any field names like "tests". Do not include any additional text, explanations, or markdown code blocks. Return ONLY the JSON array.'
    )


class GenerateDifferentiatingTestsFromCodes_func(dspy.Signature):
    """Generate differentiating tests from a prompt and two codes (functional-style)."""
    prompt = dspy.InputField(format=str, desc="Problem prompt")
    codeA = dspy.InputField(format=str, desc="First code")
    codeB = dspy.InputField(format=str, desc="Second code")
    current_test_results = dspy.InputField(
        format=str, 
        desc="Execution results of both codes on the current tests, format: [Test i][Input]...[Output]..."
    )
    tests = dspy.OutputField(
        desc='Generate test cases that can differentiate between the two code solutions. The current tests (shown in current_test_results) produce identical outputs for both codes. Generate new test cases that are likely to produce different outputs. \
        You MUST return a JSON array (starting with [ and ending with ]) containing test case objects. Each object should have this exact format: \
        {"input": <example_input>, "output": <expected_output>} \
        Example of the expected format: [{"input": {"x": 1, "y": 2}, "output": 3}, {"input": {"x": 0, "y": 0}, "output": 0}] \
        Ensure the input and output match the types and structure expected for the problem. \
        Do not wrap the JSON array in any object or add any field names like "tests". Do not include any additional text, explanations, or markdown code blocks. Return ONLY the JSON array.'
    )


def post_process_tests_inputs(raw_text: str, is_stdin: bool) -> List[Dict[str, Any]]:
    """
    Parse and format LLM-generated test case text.

    Args:
        raw_text: raw test case string from the LLM
        is_stdin: whether the problem uses stdin/stdout style

    Returns:
        A list of formatted test case dicts with input/output/testtype fields.
    """
    test_cases = None  # Initialize test_cases to avoid UnboundLocalError
    if is_stdin: 
        logger.debug(f"[post_process_tests_inputs] raw_text: {raw_text}")
        
        # Step 1: Clean the input string by removing markdown markers and extra spaces.
        cleaned_string = raw_text.strip().strip("```json").strip("```").strip()
        
        # Step 2: Try to parse as a JSON array (new format).
        try:
            if cleaned_string.startswith("[") and cleaned_string.endswith("]"):
                logger.debug(f"[post_process_tests_inputs] Attempting to parse as JSON array")
                try:
                    test_cases = json.loads(cleaned_string)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try converting single quotes to double quotes
                    logger.debug(f"[post_process_tests_inputs] JSON parse failed, trying single quote conversion")
                    cleaned_string = _convert_single_quotes_to_json(cleaned_string)
                    test_cases = json.loads(cleaned_string)
                
                # Process each test case to ensure proper format.
                formatted_tests = []
                for test_case in test_cases:
                    if isinstance(test_case, dict) and "input" in test_case:
                        # Get input string.
                        # JSON parsing automatically converts \\n to actual newlines.
                        input_str = str(test_case["input"])
                        
                        # Get output string if present; otherwise use empty string.
                        output_str = str(test_case.get("output", ""))
                        
                        # Clean up any malformed strings from prior parsing errors.
                        # Remove artifacts like trailing quotes and JSON syntax.
                        input_str = input_str.rstrip('",\n ').rstrip('"')
                        output_str = output_str.rstrip('",\n ').rstrip('"')
                        
                        # Ensure strings end with newline.
                        if not input_str.endswith("\n"):
                            input_str += "\n"
                        if output_str and not output_str.endswith("\n"):
                            output_str += "\n"
                        
                        formatted_tests.append({
                            "input": input_str,
                            "output": output_str,
                            "testtype": "stdin",
                        })
                
                logger.debug(f"[post_process_tests_inputs] formatted_tests from JSON: {formatted_tests}")
                return formatted_tests
        except json.JSONDecodeError as e:
            logger.debug(f"[post_process_tests_inputs] Failed to parse as JSON array: {e}, falling back to text parsing")
        
        # Step 3: Fallback to legacy text parsing (backward compatibility).
        blocks = raw_text.split("Input:")
        formatted_tests = []

        for block in blocks:
            if not block.strip():
                continue

            input_output = block.split("Output:")

            if len(input_output) == 2:
                input_value = input_output[0].strip()
                output_value = input_output[1].strip()

                formatted_tests.append(
                    {
                        "input": input_value + "\n",
                        "output": output_value + "\n",
                        "testtype": "stdin",
                    }
                )
        logger.debug(f"[post_process_tests_inputs] formatted_tests from text parsing: {formatted_tests}")
        return formatted_tests 
    else:
        logger.debug(f"[post_process_tests_inputs] raw_text: {raw_text}")
        # Step 1: Clean the input string by removing markdown markers and extra spaces.
        cleaned_string = raw_text.strip().strip("```json").strip("```").strip()
        
        # Step 1.5: Convert Python expressions to values before JSON parsing.
        # This handles cases like 10**9, 10**9-1, 10**9+7, 111+444+777, etc.
        def evaluate_python_expr(match):
            expr = match.group(0)
            try:
                # Use eval to compute numeric expressions safely.
                value = eval(expr, {"__builtins__": {}})
                logger.debug(f"[post_process_tests_inputs] Converted expression '{expr}' to {value}")
                return str(value)
            except Exception as e:
                logger.warning(f"[post_process_tests_inputs] Failed to evaluate expression '{expr}': {e}")
                return expr
        
        # Match numeric expressions in JSON values (after colons).
        # This pattern matches expressions that appear as JSON values:
        # - After "output": or "input":
        # - Contains numbers with operators (+, -, *, /, **, %)
        # - Stops at comma or closing brace/bracket
        # Strategy: find expressions in JSON value positions and evaluate them.
        # This matches: number (operator number)+ where operators are +, -, *, /, **, %
        pattern = r'(?<=:\s)(\d+(?:\s*(?:\*\*|[-+*/%])\s*\d+)+)(?=\s*[,\}\]])'
        cleaned_string = re.sub(pattern, evaluate_python_expr, cleaned_string)

        # Step 2: Check if it's a JSON array.
        logger.debug(f"[post_process_tests_inputs] cleaned_string: {cleaned_string}")
        if cleaned_string.startswith("[") and cleaned_string.endswith("]"):
            logger.debug(f"[post_process_tests_inputs] cleaned_string is a JSON array")
            try:
                test_cases = json.loads(cleaned_string)
            except json.JSONDecodeError:
                # If JSON parsing fails, try converting single quotes to double quotes.
                logger.debug(f"[post_process_tests_inputs] JSON parse failed, trying single quote conversion")
                cleaned_string = _convert_single_quotes_to_json(cleaned_string)
                test_cases = json.loads(cleaned_string)
            for test_case in test_cases:
                test_case["testtype"] = "functional"
            logger.debug(f"[post_process_tests_inputs] test_cases: {test_cases}")
            return test_cases

        # Step 3: Handle concatenated JSON objects without commas.
        # Regex handles whitespace between { and "input".
        # Pattern explanation:
        # \{\s* - opening brace with optional whitespace
        # ["']input["']\s*: - "input" or 'input' with optional whitespace and colon
        # .*? - non-greedy match of any chars
        # ["']output["']\s*: - "output" or 'output' with optional whitespace and colon
        # .*? - non-greedy match of any chars
        # \} - closing brace
        # Supports both single and double quotes.
        json_pattern = re.compile(r'(\{\s*["\']input["\']\s*:.*?["\']output["\']\s*:.*?\})', re.DOTALL)
        matches = json_pattern.findall(cleaned_string)

        if matches:
            # Combine matches into a valid JSON array by inserting commas.
            json_array_string = "[" + ",".join(matches) + "]"
            try:
                try:
                    test_cases = json.loads(json_array_string)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try converting single quotes to double quotes.
                    logger.debug(f"[post_process_tests_inputs] JSON parse failed for concatenated objects, trying single quote conversion")
                    json_array_string = _convert_single_quotes_to_json(json_array_string)
                    test_cases = json.loads(json_array_string)
                for test_case in test_cases:
                    test_case[
                        "testtype"
                    ] = "functional"  # Add testtype for each case.
                logger.debug(f"[post_process_tests_inputs] test_cases from regex extraction: {test_cases}")
                return test_cases
            except json.JSONDecodeError as e:
                logger.debug(f"[post_process_tests_inputs] Error parsing concatenated JSON: {e}")
                print(f"Error parsing concatenated JSON: {e}")

        # If no matches are found, fall back to line-by-line parsing.
        cleaned_lines = cleaned_string.split("\n")
        test_cases = []
        for line in cleaned_lines:
            if not line.strip():
                continue
            try:
                try:
                    test_case = json.loads(line)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try converting single quotes to double quotes.
                    logger.debug(f"[post_process_tests_inputs] JSON parse failed for line, trying single quote conversion")
                    line = _convert_single_quotes_to_json(line)
                    test_case = json.loads(line)
                test_case["testtype"] = "functional"
                test_cases.append(test_case)
            except (json.JSONDecodeError, ValueError, SyntaxError) as e:
                logger.debug(f"[post_process_tests_inputs] Error parsing JSON line: {e}")
                print(f"Error parsing JSON line: {e}")
                with open("DEBUG NOT JSON RETURN TEST.txt", "a") as log_file:
                    log_file.write(f"{line}\n")
                logger.debug(f"[post_process_tests_inputs] line: {line}")
                continue

        logger.debug(f"[post_process_tests_inputs] test_cases: {test_cases}")
        return test_cases


class TestInputGenerator(dspy.Module):
    """
    Test input generator (vanilla).

    Generates test cases from problem prompts and code.
    """
    
    def __init__(self, lm: Optional[dspy.LM] = None):
        """
        Initialize the test input generator.

        Args:
            lm: optional language model used for all generations
        """
        super().__init__()
        
        # Initialize generation programs.
        self.stdin_test_gen = dspy.ChainOfThought(GenerateTestsFromPrompt_stdin)
        self.func_test_gen = dspy.ChainOfThought(GenerateTestsFromPrompt_func)
        self.stdin_diff_test_gen = dspy.ChainOfThought(GenerateDifferentiatingTestsFromCodes_stdin)
        self.func_diff_test_gen = dspy.ChainOfThought(GenerateDifferentiatingTestsFromCodes_func)
        
        # Store the language model (if provided).
        self.lm = lm
    
    def forward(
        self, 
        prompt: str, 
        is_stdin: bool,
        codes: Optional[List[str]] = None,
        current_test_results: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> dspy.Prediction:
        """
        Generate test cases.

        Args:
            prompt: problem prompt
            is_stdin: whether the problem uses stdin/stdout
            codes: optional code list; if two are provided, generate differentiating tests
            current_test_results: current test results (for differentiating tests)
            temperature: generation temperature
            **kwargs: other parameters

        Returns:
            dspy.Prediction with a tests field
        """
        config = {}
        if temperature is not None:
            config["temperature"] = temperature
        
        # Decide whether to generate differentiating tests.
        if codes is not None and len(codes) >= 2 and current_test_results is not None:
            # Generate differentiating tests.
            codeA = codes[0]
            codeB = codes[1]
            
            if is_stdin:
                gen_prog = self.stdin_diff_test_gen
            else:
                gen_prog = self.func_diff_test_gen
            
            # Use the provided LM (if any).
            if self.lm is not None:
                with dspy.context(lm=self.lm):
                    result = gen_prog(
                        prompt=prompt,
                        codeA=codeA,
                        codeB=codeB,
                        current_test_results=current_test_results,
                        config=config if config else None
                    )
            else:
                result = gen_prog(
                    prompt=prompt,
                    codeA=codeA,
                    codeB=codeB,
                    current_test_results=current_test_results,
                    config=config if config else None
                )
        else:
            # Generate tests from the prompt.
            if is_stdin:
                gen_prog = self.stdin_test_gen
            else:
                gen_prog = self.func_test_gen
            # Use the provided LM (if any).
            if self.lm is not None:
                with dspy.context(lm=self.lm):
                    result = gen_prog(
                        prompt=prompt,
                        config=config if config else None
                    )
            else:
                result = gen_prog(
                    prompt=prompt,
                    config=config if config else None
                )
        
        return result
    
    def generate_tests_from_prompt(
        self,
        prompt: str,
        is_stdin: bool,
        temperature: Optional[float] = None,
        parse_output: bool = True
    ) -> Any:
        """
        Convenience method to generate tests from a prompt.

        Args:
            prompt: problem prompt
            is_stdin: whether the problem uses stdin/stdout
            temperature: generation temperature
            parse_output: parse output into structured format if True

        Returns:
            Parsed test cases if parse_output=True; otherwise raw JSON string.
        """
        result = self(
            prompt=prompt,
            is_stdin=is_stdin,
            temperature=temperature
        )
        
        if parse_output:
            return post_process_tests_inputs(result.tests, is_stdin)
        else:
            return result.tests
    
    def generate_differentiating_tests(
        self,
        prompt: str,
        codeA: str,
        codeB: str,
        current_test_results: str,
        is_stdin: bool,
        temperature: Optional[float] = None,
        parse_output: bool = True
    ) -> Any:
        """
        Convenience method to generate differentiating test cases.

        Args:
            prompt: problem prompt
            codeA: first code
            codeB: second code
            current_test_results: current test results string
            is_stdin: whether the problem uses stdin/stdout
            temperature: generation temperature
            parse_output: parse output into structured format if True

        Returns:
            Parsed test cases if parse_output=True; otherwise raw JSON string.
        """
        result = self(
            prompt=prompt,
            is_stdin=is_stdin,
            codes=[codeA, codeB],
            current_test_results=current_test_results,
            temperature=temperature
        )
        
        if parse_output:
            return post_process_tests_inputs(result.tests, is_stdin)
        else:
            return result.tests


class TestInputGenerator_v2(dspy.Module):
    """
    Test input generator - v2.

    Generates test cases by producing and executing Python code.
    """
    
    def __init__(self, lm: Optional[dspy.LM] = None):
        """
        Initialize test input generator v2.

        Args:
            lm: optional language model used for all generations
        """
        super().__init__()
        
        # Initialize generation programs.
        self.stdin_code_gen = dspy.ChainOfThought(GenerateTestGeneratorCode_stdin)
        self.func_code_gen = dspy.ChainOfThought(GenerateTestGeneratorCode_func)
        
        # Store the language model (if provided).
        self.lm = lm
    
    def _extract_code_from_output(self, raw_output: str) -> str:
        """
        Extract Python code from LLM output.

        Args:
            raw_output: raw output from the LLM

        Returns:
            Extracted Python code string
        """
        # Remove markdown code block markers.
        cleaned = raw_output.strip()
        
        # If ```python or ``` markers exist, extract code within them.
        if "```python" in cleaned:
            start_idx = cleaned.find("```python") + len("```python")
            end_idx = cleaned.find("```", start_idx)
            if end_idx != -1:
                cleaned = cleaned[start_idx:end_idx].strip()
        elif "```" in cleaned:
            start_idx = cleaned.find("```") + 3
            end_idx = cleaned.rfind("```")
            if end_idx != -1 and end_idx > start_idx:
                cleaned = cleaned[start_idx:end_idx].strip()
        
        return cleaned
    
    def _execute_generator_code(self, code: str, timeout: int = 30) -> str:
        """
        Execute generated Python code and return stdout.

        Args:
            code: Python code to execute
            timeout: execution timeout in seconds

        Returns:
            stdout output from execution

        Raises:
            RuntimeError: if execution fails or times out
        """
        # Create a temporary file to store code.
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute code.
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                error_msg = f"Code execution failed with return code {result.returncode}\n"
                error_msg += f"STDERR: {result.stderr}\n"
                error_msg += f"STDOUT: {result.stdout}"
                logger.error(f"[TestInputGenerator_v2] {error_msg}")
                raise RuntimeError(error_msg)
            
            return result.stdout.strip()
        finally:
            # Clean up temporary file.
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def generate_tests_from_prompt(
        self,
        prompt: str,
        is_stdin: bool,
        temperature: Optional[float] = None,
        timeout: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Generate tests from a prompt by creating and running Python code.

        Args:
            prompt: problem prompt
            is_stdin: whether the problem uses stdin/stdout
            temperature: generation temperature
            timeout: execution timeout in seconds

        Returns:
            Parsed test cases; each is a dict with input/output/testtype fields.
        """
        config = {}
        if temperature is not None:
            config["temperature"] = temperature
        
        # Select the appropriate generator.
        if is_stdin:
            gen_prog = self.stdin_code_gen
        else:
            gen_prog = self.func_code_gen
        
        # Generate code.
        if self.lm is not None:
            with dspy.context(lm=self.lm):
                result = gen_prog(
                    prompt=prompt,
                    config=config if config else None
                )
        else:
            result = gen_prog(
                prompt=prompt,
                config=config if config else None
            )
        
        # Extract code.
        generator_code = self._extract_code_from_output(result.generator_code)
        logger.debug(f"[TestInputGenerator_v2] Generated code:\n{generator_code}")
        
        # Execute code.
        try:
            output = self._execute_generator_code(generator_code, timeout=timeout)
            logger.debug(f"[TestInputGenerator_v2] Code output: {output}")
        except Exception as e:
            logger.error(f"[TestInputGenerator_v2] Failed to execute code: {e}")
            raise
        
        # Process output.
        logger.info("[TestInputGenerator_v2] Parsing test cases from output...")
        try:
            test_cases = post_process_tests_inputs(output, is_stdin)
            logger.info(f"[TestInputGenerator_v2] Successfully parsed {len(test_cases)} test cases")
            for i, test_case in enumerate(test_cases):
                logger.debug(f"[TestInputGenerator_v2] Test case {i}: {test_case}")
        except Exception as e:
            logger.error(f"[TestInputGenerator_v2] Failed to parse test cases: {e}")
            logger.error(f"[TestInputGenerator_v2] Raw output that failed to parse:\n{output}")
            raise
        
        return test_cases

