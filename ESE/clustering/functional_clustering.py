"""
Functional test clustering based on execution results.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import random
import dspy

# Import base class and utilities (relative imports).
from .clustering_base import BaseClusteringMethod
from .clustering_utils import cluster_by_equivalence

# Import test generators (from test_generator subpackage).
from .test_generator import TestInputGenerator, TestInputGenerator_v2

# Import cache manager.
from .test_cache_manager import TestCacheManager

# Import test execution utility (supports two path layouts).
try:
    # When source/ is on the path (e.g., cascade_tts.py)
    from live_code_bench import test_on_single_test
except ImportError:
    # When Source/ is on the path (e.g., calc_entropy.py, example_usage.py)
    from source.live_code_bench.Test_Utils import test_on_single_test

logger = logging.getLogger(__name__)


class FunctionalClustering(BaseClusteringMethod):
    """
    Cluster code using functional tests (code-aware test generation).

    References:
    - Source/dev_exp/TestGen.py (TestInputGenerator)
    - Source/live_code_bench/Test_Utils.py (test_on_single_test)
    """
    
    def __init__(
        self,
        lm: Optional[dspy.LM] = None,
        num_tests: int = 5,
        test_timeout: float = 6.0,
        use_cache: bool = True,
        cache_path: str,
        **kwargs
    ):
        """
        Initialize functional test clustering.

        Args:
            lm: language model
            num_tests: number of tests to generate (max; if cache has more, sample)
            test_timeout: per-test timeout (seconds)
            use_cache: whether to use cache (True: read, False: regenerate and write)
            cache_path: cache file path
        """
        super().__init__(**kwargs)
        self.test_generator = TestInputGenerator(lm=lm)
        self.num_tests = num_tests
        self.test_timeout = test_timeout
        self.use_cache = use_cache
        self.cache_manager = TestCacheManager(cache_path=cache_path)
    
    def _generate_tests(
        self,
        problem_info: Dict[str, Any],
        temperature: float = 0.7,
        use_cache: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate test cases (with cache support).

        Args:
            problem_info: problem info (must include task_id and is_stdin)
            temperature: generation temperature
            use_cache: whether to use cache (None uses self.use_cache)

        Returns:
            List of test cases
        """
        if use_cache is None:
            use_cache = self.use_cache
        
        prompt = problem_info["prompt"]
        is_stdin = problem_info["is_stdin"]
        task_id = problem_info.get("task_id", "")
        
        if not task_id:
            logger.warning("No task_id in problem_info; cache will be disabled.")
            use_cache = False
        
        # If using cache, try loading from cache.
        if use_cache:
            cached_tests = self.cache_manager.load_tests(task_id, is_stdin)
            if cached_tests:
                logger.info(f"Loaded {len(cached_tests)} tests from cache.")
                # If cache has more tests, sample num_tests.
                if len(cached_tests) > self.num_tests:
                    selected_tests = random.sample(cached_tests, self.num_tests)
                    logger.info(f"Cache has {len(cached_tests)} tests; sampled {self.num_tests}.")
                    return selected_tests
                else:
                    logger.info(f"Using all {len(cached_tests)} cached tests.")
                    return cached_tests
            else:
                logger.info("No cached tests found; regenerating.")
        
        # Generate new test cases.
        try:
            tests = self.test_generator.generate_tests_from_prompt(
                prompt=prompt,
                is_stdin=is_stdin,
                temperature=temperature,
                parse_output=True
            )
            
            # If tests are generated and task_id exists, write to cache.
            if tests and task_id:
                self.cache_manager.save_tests(task_id, is_stdin, tests)
                logger.info(f"Generated {len(tests)} tests and wrote to cache.")
            
            return tests
        except Exception as e:
            logger.error(f"Failed to generate tests: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _execute_tests(
        self,
        codes: List[str],
        tests: List[Dict[str, Any]]
    ) -> Tuple[List[Tuple], List[int]]:
        """
        Execute tests on all code samples.

        Args:
            codes: code list
            tests: test case list

        Returns:
            (result vectors, valid test indices)
        """
        if not tests:
            return [], []
        
        # Execute all tests.
        all_results = []
        for code_idx, code in enumerate(codes):
            code_results = []
            for test_idx, test in enumerate(tests):
                try:
                    passed, error_msg, output_value, time_elapsed = test_on_single_test(
                        code=code,
                        test_case=test,
                        timeout=self.test_timeout,
                        is_extracted=False
                    )
                    # Convert output to string.
                    output_str = str(output_value)
                    code_results.append(output_str)
                except Exception as e:
                    error_str = f"Error: {str(e)}"
                    code_results.append(error_str)
            all_results.append(code_results)
        
        # Log all test results (before filtering).
        logger.info("All test results (before filtering):")
        for code_idx, code_results in enumerate(all_results):
            logger.info(f"  Code {code_idx} results: {code_results}")
        
        # Count errors per test.
        num_codes = len(codes)
        error_threshold = num_codes / 2  # Half of samples.
        
        # Normalize error results to "ERROR".
        for code_idx, code_results in enumerate(all_results):
            for test_idx in range(len(tests)):
                if "Error:" in code_results[test_idx]:
                    code_results[test_idx] = "ERROR"
        
        # Filter tests: if >= half of samples error, drop the test.
        valid_test_indices = []
        invalid_test_indices = []
        for test_idx in range(len(tests)):
            error_count = 0
            for code_results in all_results:
                if code_results[test_idx] == "ERROR":
                    error_count += 1
            
            if error_count >= error_threshold:
                # >= half of samples errored, drop this test.
                invalid_test_indices.append(test_idx)
                logger.info(f"Test {test_idx}: {error_count}/{num_codes} samples errored; dropping test.")
            else:
                # Keep test.
                valid_test_indices.append(test_idx)
                if error_count > 0:
                    logger.info(
                        f"Test {test_idx}: {error_count}/{num_codes} samples errored; keeping test (errors set to ERROR)."
                    )
        
        # Build filtered result vectors.
        filtered_vectors = []
        for code_idx, code_results in enumerate(all_results):
            filtered_vector = tuple(
                code_results[i] for i in valid_test_indices
            )
            filtered_vectors.append(filtered_vector)
        
        return filtered_vectors, valid_test_indices
    
    def cluster(
        self,
        codes: List[Dict[str, Any]],
        problem_info: Dict[str, Any],
        cluster_algorithm: str = 'dfs',
        temperature: float = 0.7,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> List[int]:
        """
        Cluster code samples.

        Args:
            codes: code list
            problem_info: problem info (must include task_id and is_stdin)
            cluster_algorithm: clustering algorithm
            temperature: test generation temperature
            use_cache: whether to use cache (None uses self.use_cache)
            **kwargs: other parameters

        Returns:
            Cluster ID list
        """
        self._validate_inputs(codes, problem_info)
        
        code_strings = [c["code"] for c in codes]
        
        # Generate tests (with cache support).
        tests = self._generate_tests(problem_info, temperature, use_cache=use_cache)
        
        if not tests:
            logger.warning("No tests generated, treating all codes as different clusters")
            return list(range(len(codes)))
        
        # Execute tests.
        result_vectors, valid_test_indices = self._execute_tests(code_strings, tests)
        
        if not valid_test_indices:
            logger.warning("No valid tests after filtering, treating all codes as different clusters")
            return list(range(len(codes)))
        
        # Cluster based on result vectors.
        def are_equivalent(i: int, j: int) -> bool:
            """Check if two result vectors are identical."""
            return result_vectors[i] == result_vectors[j]
        
        return cluster_by_equivalence(code_strings, are_equivalent, cluster_algorithm)


class FunctionalVanillaClustering(BaseClusteringMethod):
    """
    Cluster code using functional tests (random test generation).

    References:
    - Source/dev_exp/TestGen.py (TestInputGenerator_v2)
    - Source/live_code_bench/Test_Utils.py (test_on_single_test)
    """
    
    def __init__(
        self,
        lm: Optional[dspy.LM] = None,
        test_timeout: float = 6.0,
        generator_timeout: int = 30,
        **kwargs
    ):
        """
        Initialize functional test clustering (vanilla version).

        Args:
            lm: language model
            test_timeout: per-test timeout (seconds)
            generator_timeout: test generator execution timeout (seconds)
        """
        super().__init__(**kwargs)
        self.test_generator = TestInputGenerator_v2(lm=lm)
        self.test_timeout = test_timeout
        self.generator_timeout = generator_timeout
    
    def _generate_tests(
        self,
        problem_info: Dict[str, Any],
        temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Generate test cases.

        Args:
            problem_info: problem info
            temperature: generation temperature

        Returns:
            List of test cases
        """
        prompt = problem_info["prompt"]
        is_stdin = problem_info["is_stdin"]
        
        try:
            tests = self.test_generator.generate_tests_from_prompt(
                prompt=prompt,
                is_stdin=is_stdin,
                temperature=temperature,
                timeout=self.generator_timeout
            )
            return tests
        except Exception as e:
            logger.error(f"Failed to generate tests: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _execute_tests(
        self,
        codes: List[str],
        tests: List[Dict[str, Any]]
    ) -> Tuple[List[Tuple], List[int]]:
        """
        Execute tests on all code samples.

        Args:
            codes: code list
            tests: test case list

        Returns:
            (result vectors, valid test indices)
        """
        if not tests:
            return [], []
        
        # Execute all tests.
        all_results = []
        for code_idx, code in enumerate(codes):
            code_results = []
            for test_idx, test in enumerate(tests):
                try:
                    passed, error_msg, output_value, time_elapsed = test_on_single_test(
                        code=code,
                        test_case=test,
                        timeout=self.test_timeout,
                        is_extracted=False
                    )
                    # Convert output to string.
                    output_str = str(output_value)
                    code_results.append(output_str)
                except Exception as e:
                    error_str = f"Error: {str(e)}"
                    code_results.append(error_str)
            all_results.append(code_results)
        
        # Log all test results (before filtering).
        logger.info("All test results (before filtering, vanilla):")
        for code_idx, code_results in enumerate(all_results):
            logger.info(f"  Code {code_idx} results: {code_results}")
        
        # Count errors per test.
        num_codes = len(codes)
        error_threshold = num_codes / 2  # Half of samples.
        
        # Normalize error results to "ERROR".
        for code_idx, code_results in enumerate(all_results):
            for test_idx in range(len(tests)):
                if "Error:" in code_results[test_idx]:
                    code_results[test_idx] = "ERROR"
        
        # Filter tests: if >= half of samples error, drop the test.
        valid_test_indices = []
        invalid_test_indices = []
        for test_idx in range(len(tests)):
            error_count = 0
            for code_results in all_results:
                if code_results[test_idx] == "ERROR":
                    error_count += 1
            
            if error_count >= error_threshold:
                # >= half of samples errored, drop this test.
                invalid_test_indices.append(test_idx)
                logger.info(f"Test {test_idx}: {error_count}/{num_codes} samples errored; dropping test.")
            else:
                # Keep test.
                valid_test_indices.append(test_idx)
                if error_count > 0:
                    logger.info(
                        f"Test {test_idx}: {error_count}/{num_codes} samples errored; keeping test (errors set to ERROR)."
                    )
        
        # Build filtered result vectors.
        filtered_vectors = []
        for code_idx, code_results in enumerate(all_results):
            filtered_vector = tuple(
                code_results[i] for i in valid_test_indices
            )
            filtered_vectors.append(filtered_vector)
        
        return filtered_vectors, valid_test_indices
    
    def cluster(
        self,
        codes: List[Dict[str, Any]],
        problem_info: Dict[str, Any],
        cluster_algorithm: str = 'dfs',
        temperature: float = 0.7,
        **kwargs
    ) -> List[int]:
        """
        Cluster code samples.

        Args:
            codes: code list
            problem_info: problem info
            cluster_algorithm: clustering algorithm
            temperature: test generation temperature
            **kwargs: other parameters

        Returns:
            Cluster ID list
        """
        self._validate_inputs(codes, problem_info)
        
        code_strings = [c["code"] for c in codes]
        
        # Generate tests.
        tests = self._generate_tests(problem_info, temperature)
        
        if not tests:
            logger.warning("No tests generated, treating all codes as different clusters")
            return list(range(len(codes)))
        
        # Execute tests.
        result_vectors, valid_test_indices = self._execute_tests(code_strings, tests)
        
        if not valid_test_indices:
            logger.warning("No valid tests after filtering, treating all codes as different clusters")
            return list(range(len(codes)))
        
        # Cluster based on result vectors.
        def are_equivalent(i: int, j: int) -> bool:
            """Check if two result vectors are identical."""
            return result_vectors[i] == result_vectors[j]
        
        return cluster_by_equivalence(code_strings, are_equivalent, cluster_algorithm)

