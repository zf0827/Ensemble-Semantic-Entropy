"""
Main execution logic for running LLM code generation and debugging.

This module provides:
- Simple code generation (naive)
- Self-debug loop with feedback from test results
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import os
import dspy
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from tqdm import tqdm

# set up logger
logger = logging.getLogger(__name__)

from .DSPY_PROMPT import (
    GenerateLCBCodeStdin,
    GenerateLCBCodeFunctional,
    SelfDebug
)
from .Test_Utils import test_on_public_tests, test_on_private_tests
from .utils import post_process_code, format_test_results_summary


# =============================================================================
# Helper Functions for Logprob Extraction
# =============================================================================

def _extract_logprobs(lm: dspy.LM) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
    """
    Extract prompt_tokens, completion_tokens, logprob and norm_logprob from lm.history[-1].
    
    Args:
        lm: DSPy language model
        
    Returns:
        Tuple of (prompt_tokens, completion_tokens, logprob, norm_logprob)
        - prompt_tokens: Number of input tokens
        - completion_tokens: Number of output tokens
        - logprob: Sum of logprobs for all completion tokens
        - norm_logprob: logprob / completion_tokens
    """
    try:
        last = lm.history[-1]
        raw_resp = last.get("response")
        
        if raw_resp is None:
            return None, None, None, None
        
        # Get token counts from usage
        usage = last.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        if completion_tokens == 0:
            return prompt_tokens, completion_tokens, None, None
        
        # Extract logprobs from LiteLLM ModelResponse
        logprob_sum = 0.0
        token_count = 0
        
        if hasattr(raw_resp, 'choices') and len(raw_resp.choices) > 0:
            choice = raw_resp.choices[0]
            if hasattr(choice, 'logprobs') and choice.logprobs:
                # Extract logprob for each token
                if hasattr(choice.logprobs, 'content'):
                    for token_logprob in choice.logprobs.content:
                        token_logprob_val = None
                        if hasattr(token_logprob, 'logprob'):
                            token_logprob_val = token_logprob.logprob
                        elif isinstance(token_logprob, dict) and 'logprob' in token_logprob:
                            token_logprob_val = token_logprob['logprob']
                        
                        if token_logprob_val is not None:
                            logprob_sum += token_logprob_val
                            token_count += 1
        
        # Return None if no logprob was successfully extracted
        if token_count == 0:
            return prompt_tokens, completion_tokens, None, None
        logger.info(f"logprob_sum: {logprob_sum}, token_count: {token_count}, completion_tokens: {completion_tokens}")
        # Calculate norm_logprob
        norm_logprob = logprob_sum / token_count if token_count > 0 else None
        
        return prompt_tokens, completion_tokens, logprob_sum, norm_logprob
    except Exception as e:
        logger.debug(f"Failed to extract logprobs: {e}")
        return None, None, None, None


# =============================================================================
# Naive Code Generation (No Debug Loop)
# =============================================================================

class NaiveCodeGenerator(dspy.Module):
    """Generate code without any debug loop."""

    def __init__(self, use_cot: bool = False):
        """
        Args:
            use_cot: Whether to use chain-of-thought reasoning
        """
        super().__init__()
        self.use_cot = use_cot

    def forward(self, prompt: str, is_stdin: bool, lm: Optional[dspy.LM] = None) -> dspy.Prediction:
        """
        Generate code for the problem.

        Args:
            prompt: Problem description
            is_stdin: Whether problem uses stdin/stdout
            lm: DSPy language model (optional, will use context if not provided)

        Returns:
            DSPy Prediction with 'code', 'logprob', and 'norm_logprob' fields
        """
        if is_stdin:
            signature = GenerateLCBCodeStdin
        else:
            signature = GenerateLCBCodeFunctional

        # Get current lm (from context or parameter)
        if lm is None:
            lm = dspy.settings.lm
        
            with dspy.settings.context(lm=lm, track_usage=True):
                if self.use_cot:
                    result = dspy.ChainOfThought(signature)(prompt=prompt)
                else:
                    result = dspy.Predict(signature)(prompt=prompt)
            
            code_text = result.code
            
            # Extract prompt_tokens, completion_tokens, logprob and norm_logprob
            prompt_tokens, completion_tokens, logprob, norm_logprob = _extract_logprobs(lm)
        
        return dspy.Prediction(
            code=code_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            logprob=logprob,
            norm_logprob=norm_logprob
        )


# =============================================================================
# Self-Debug with Test Feedback
# =============================================================================

def run_debug(
    problem: dspy.Example,
    public_tests: List[Dict],
    lm: dspy.LM,
    num_rounds: int = 3,
    timeout: float = 6.0,
    use_cot_initial: bool = True,
    verbose: bool = False
) -> Tuple[str, List[Dict]]:
    """
    Run self-debug loop: generate code, test, and refine based on feedback.

    Args:
        problem: DSPy Example containing problem info
        public_tests: List of public test cases
        lm: DSPy language model
        num_rounds: Number of debug iterations
        timeout: Timeout for each test
        use_cot_initial: Whether to use CoT for initial generation
        verbose: Whether to print debug info

    Returns:
        Tuple of (final_code, debug_history)
        - final_code: Best code after debugging
        - debug_history: List of dicts with 'code', 'test_results', 'passed' for each round
    """
    prompt = problem.prompt
    is_stdin = problem.is_stdin
    task_id = problem.task_id

    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"Task: {task_id}")
        logger.info(f"Type: {'stdin/stdout' if is_stdin else 'functional'}")
        logger.info(f"{'='*60}\n")

    debug_history = []

    # Create progress bar for debug rounds (only in verbose mode and single-process)
    # In multiprocessing mode, we disable progress bar to avoid conflicts
    use_pbar = verbose
    if use_pbar:
        # Truncate task_id if too long for display
        display_id = task_id[:25] + '...' if len(task_id) > 25 else task_id
        pbar = tqdm(total=num_rounds, desc=f"Debug {display_id}", unit="round", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                   leave=False)  # leave=False to clean up after completion
    else:
        pbar = None

    try:
        with dspy.context(lm=lm):
            # Round 1: Initial generation
            if verbose:
                logger.info(f"[Round 1/{num_rounds}] Generating initial solution...")
            if pbar:
                pbar.set_description(f"Round 1/{num_rounds}: Generating")

            if is_stdin:
                signature = GenerateLCBCodeStdin
            else:
                signature = GenerateLCBCodeFunctional

            with dspy.settings.context(lm=lm, track_usage=True):
                if use_cot_initial:
                    result = dspy.ChainOfThought(signature)(prompt=prompt)
                else:
                    result = dspy.Predict(signature)(prompt=prompt)
                
                code_text = result.code
                
                # Extract prompt_tokens, completion_tokens, logprob and norm_logprob
                prompt_tokens, completion_tokens, logprob, norm_logprob = _extract_logprobs(lm)

            code = post_process_code(code_text)
            if verbose:
                logger.info(code)

            # Test initial code
            if pbar:
                pbar.set_description(f"Round 1/{num_rounds}: Testing")
            test_results = test_on_public_tests(
                code,
                public_tests,
                timeout=timeout,
                is_extracted=True,
                fast_check=False
            )

            debug_history.append({
                "round": 1,
                "code": code,
                "test_results": test_results,
                "passed": test_results["passed"],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "logprob": logprob,
                "norm_logprob": norm_logprob
            })

            num_passed = sum(test_results['details'])
            num_total = len(test_results['details'])
            if pbar:
                pbar.set_postfix({
                    'Tests': f"{num_passed}/{num_total}",
                    'Status': '✓ PASS' if test_results["passed"] else '✗ FAIL'
                })
                pbar.update(1)

            if verbose:
                logger.info(f"Initial tests: {num_passed}/{num_total} passed")

            if test_results["passed"]:
                if verbose:
                    logger.info("All tests passed! Returning solution.")
                return code, debug_history

            # Rounds 2-N: Self-debug loop
            for round_idx in range(2, num_rounds + 1):
                if verbose:
                    logger.info(f"\n[Round {round_idx}/{num_rounds}] Debugging solution...")
                if pbar:
                    pbar.set_description(f"Round {round_idx}/{num_rounds}: Debugging")

                # Prepare feedback
                feedback = _prepare_debug_feedback(
                    prompt,
                    code,
                    test_results,
                    public_tests
                )

                # Generate improved code
                with dspy.settings.context(lm=lm, track_usage=True):
                    debug_result = dspy.Predict(SelfDebug)(prompt=feedback)
                    code_text = debug_result.code
                    
                    # Extract prompt_tokens, completion_tokens, logprob and norm_logprob
                    prompt_tokens, completion_tokens, logprob, norm_logprob = _extract_logprobs(lm)
                
                code = post_process_code(code_text)
                if verbose:
                    logger.info(code)
                
                # Test improved code
                if pbar:
                    pbar.set_description(f"Round {round_idx}/{num_rounds}: Testing")
                test_results = test_on_public_tests(
                    code,
                    public_tests,
                    timeout=timeout,
                    is_extracted=True,
                    fast_check=False
                )

                debug_history.append({
                    "round": round_idx,
                    "code": code,
                    "test_results": test_results,
                    "passed": test_results["passed"],
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "logprob": logprob,
                    "norm_logprob": norm_logprob
                })

                num_passed = sum(test_results['details'])
                num_total = len(test_results['details'])
                if pbar:
                    pbar.set_postfix({
                        'Tests': f"{num_passed}/{num_total}",
                        'Status': '✓ PASS' if test_results["passed"] else '✗ FAIL'
                    })
                    pbar.update(1)

                if verbose:
                    logger.info(f"Tests: {num_passed}/{num_total} passed")

                if test_results["passed"]:
                    if verbose:
                        logger.info("All tests passed! Returning solution.")
                    return code, debug_history

        # Return best code even if not all tests passed
        if verbose:
            logger.info("\nMax rounds reached. Returning best solution.")

        return code, debug_history
    
    finally:
        if pbar:
            pbar.close()


def _prepare_debug_feedback(
    prompt: str,
    code: str,
    test_results: Dict,
    test_cases: List[Dict]
) -> str:
    """
    Prepare feedback string for debug prompt.

    Args:
        prompt: Original problem prompt
        code: Current code
        test_results: Test results from test_on_public_tests
        test_cases: List of test case dictionaries

    Returns:
        Formatted feedback string
    """
    feedback = f"# Problem\n{prompt}\n\n"
    feedback += f"# Your Previous Code\n```python\n{code}\n```\n\n"
    feedback += "# Test Results\n"

    details = test_results["details"]
    error_messages = test_results["error_messages"]

    for idx, (passed, error_msg, test_case) in enumerate(zip(details, error_messages, test_cases)):
        status = "✓ PASS" if passed else "✗ FAIL"
        feedback += f"\nTest {idx + 1}: {status}\n"

        if not passed:
            feedback += f"Input: {test_case.get('input', 'N/A')}\n"
            feedback += f"Expected: {test_case.get('output', 'N/A')}\n"
            feedback += f"Error: {error_msg}\n"

    feedback += f"\n# Task\n"
    feedback += f"Analyze why your code failed and provide a corrected version. "
    feedback += f"Focus on fixing the specific test failures shown above."

    # logger.info(feedback)
    return feedback


# =============================================================================
# Batch Evaluation Helper
# =============================================================================

def evaluate_on_problem_worker(arguments: Tuple) -> Dict[str, Any]:
    """
    Worker function: Run debug on a single problem and evaluate on private tests.
    
    Args:
        arguments: Tuple containing (example, public_tests, private_tests, lm_config, num_rounds, timeout, use_cot_initial)
    
    Returns:
        Result dictionary containing:
        - question_id: Question ID
        - debug_trace: List of code, logprob, norm_logprob for each round
        - is_passed_public: Whether passed public tests
        - passed: Whether passed private tests
        - score: Private test score (number of passed tests / total private tests)
    """
    example, public_tests, private_tests, lm_config, num_rounds, timeout, use_cot_initial = arguments
    
    try:
        # Recreate LM object in subprocess (to avoid serialization issues)
        model_name = lm_config["model_name"]
        kwargs = {k: v for k, v in lm_config.items() if k != "model_name"}
        lm = dspy.LM(model_name, **kwargs)
        
        # Run debug process (verbose=False in worker to avoid progress bar conflicts)
        final_code, debug_history = run_debug(
            problem=example,
            public_tests=public_tests,
            lm=lm,
            num_rounds=num_rounds,
            timeout=timeout,
            use_cot_initial=use_cot_initial,
            verbose=False  # Disable verbose in worker to avoid progress bar conflicts
        )
        
        # Check if passed public tests
        is_passed_public = debug_history[-1]["passed"]
        
        # Evaluate final code on private tests
        private_test_results = test_on_private_tests(
            code=final_code,
            private_tests=private_tests,
            timeout=timeout,
            is_extracted=not example.is_stdin
        )
        
        # Calculate private test score
        private_details = private_test_results["details"]
        num_private_tests = len(private_details)
        num_passed_private = sum(private_details)
        score = num_passed_private / num_private_tests if num_private_tests > 0 else 0.0
        passed = private_test_results["passed"]
        
        # Construct debug_trace
        debug_trace = []
        for round_data in debug_history:
            debug_trace.append({
                "code": round_data["code"],
                "prompt_tokens": round_data.get("prompt_tokens"),
                "completion_tokens": round_data.get("completion_tokens"),
                "logprob": round_data.get("logprob"),
                "norm_logprob": round_data.get("norm_logprob")
            })
        
        return {
            "question_id": example.task_id,
            "debug_trace": debug_trace,
            "is_passed_public": is_passed_public,
            "passed": passed,
            "score": score,
            "success": True
        }
    
    except Exception as e:
        logger.error(f"Error processing question {example.task_id}: {str(e)}")
        return {
            "question_id": example.task_id,
            "debug_trace": [],
            "is_passed_public": False,
            "passed": False,
            "score": 0.0,
            "success": False,
            "error": str(e)
        }


def write_result_to_file(result: Dict[str, Any], output_path: str, lock: Any):
    """
    Write a single result to JSON file (thread-safe).
    
    Args:
        result: Result dictionary
        output_path: Output file path
        lock: Process lock
    """
    # Only save required fields
    output_result = {
        "question_id": result["question_id"],
        "debug_trace": result["debug_trace"],
        "is_passed_public": result["is_passed_public"],
        "passed": result["passed"],
        "score": result["score"]
    }
    
    with lock:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(output_result, ensure_ascii=False) + '\n')
            f.flush()  # Flush immediately to disk


def evaluate_on_dataset(
    dataset: List[dspy.Example],
    public_tests_dict: Dict[str, List[Dict]],
    private_tests_dict: Dict[str, List[Dict]],
    lm_config: Dict[str, Any],
    num_rounds: int = 3,
    timeout: float = 6.0,
    use_cot_initial: bool = True,
    num_workers: int = 1,
    output_path: Optional[str] = None,
    verbose: bool = False
) -> List[Dict]:
    """
    Evaluate run_debug on dataset in multiprocessing mode.

    Args:
        dataset: List of DSPy Examples
        public_tests_dict: Dict mapping task_id to public test cases
        private_tests_dict: Dict mapping task_id to private test cases
        lm_config: Dict containing LM configuration with 'model_name' and other kwargs
        num_rounds: Number of debug rounds
        timeout: Timeout per test
        use_cot_initial: Whether to use CoT for initial generation
        num_workers: Number of parallel workers
        output_path: Optional path to write results incrementally (JSONL format)
        verbose: Whether to print progress

    Returns:
        List of result dictionaries with:
        - question_id: Question ID
        - debug_trace: List of code, logprob, norm_logprob for each round
        - is_passed_public: Whether passed public tests
        - passed: Whether passed private tests
        - score: Private test score (number of passed tests / total private tests)
    """
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info("Starting batch evaluation on dataset")
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Number of workers: {num_workers}")
        if output_path:
            logger.info(f"Output path: {output_path}")
        logger.info(f"{'='*60}\n")
    
    # Create output directory (if output path is specified)
    if output_path:
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Clear or create output file
        open(output_path, 'w').close()
    
    # Prepare task arguments
    tasks = []
    for example in dataset:
        task_id = example.task_id
        
        if task_id not in public_tests_dict:
            if verbose:
                logger.warning(f"Warning: Question {task_id} has no public tests, skipping")
            continue
        
        if task_id not in private_tests_dict:
            if verbose:
                logger.warning(f"Warning: Question {task_id} has no private tests, skipping")
            continue
        
        public_tests = public_tests_dict[task_id]
        private_tests = private_tests_dict[task_id]
        
        tasks.append((example, public_tests, private_tests, lm_config, num_rounds, timeout, use_cot_initial))
    
    if verbose:
        logger.info(f"Valid tasks: {len(tasks)}\n")
    
    # Create process lock for file writing
    manager = Manager()
    lock = manager.Lock()
    
    # Use multiprocessing for evaluation
    results = []
    success_count = 0
    public_pass_count = 0
    private_pass_count = 0
    
    # Create progress bar with more detailed information
    pbar_desc = "Evaluating" if verbose else "Progress"
    with tqdm(total=len(tasks), desc=pbar_desc, unit="task", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(evaluate_on_problem_worker, task): task[0].task_id
                for task in tasks
            }
            
            for future in as_completed(futures):
                task_id = futures[future]
                result = future.result()
                results.append(result)
                
                # Write to file immediately (if output path is specified)
                if output_path:
                    write_result_to_file(result, output_path, lock)
                
                # Update statistics
                if result.get('success', False):
                    success_count += 1
                    if result.get('is_passed_public', False):
                        public_pass_count += 1
                    if result.get('passed', False):
                        private_pass_count += 1
                
                # Update progress bar with detailed information
                current_accuracy = (private_pass_count / len(results) * 100) if len(results) > 0 else 0.0
                pbar.set_postfix({
                    'Public': f"{public_pass_count}/{len(results)}",
                    'Private': f"{private_pass_count}/{len(results)}",
                    'Acc': f"{current_accuracy:.1f}%",
                    'Task': task_id[:20] + '...' if len(task_id) > 20 else task_id
                })
                pbar.update(1)
    
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing completed: {success_count}/{len(tasks)} successful")
        if output_path:
            logger.info(f"Results saved to: {output_path}")
        logger.info(f"{'='*60}\n")
    
    return results
