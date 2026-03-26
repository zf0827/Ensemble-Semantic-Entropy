"""
Baseline evaluation framework.

Implements four baseline methods to evaluate code generation:
1. vanilla: select one sample and return correctness of the first code
2. major_voting: cluster all first codes and pick a random one from the largest cluster
3. pass_at_n: return whether any final code passes
4. pass_at_n_oneshot: return whether any first code passes
"""

import json
import os
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Import clustering module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ESE.clustering.clustering_interface import cluster_codes
import dspy

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Baselines:
    """Baseline evaluation framework."""
    
    def __init__(
        self,
        config_path: str,
        result_base_path: str,
        difficulty: str = "easy",
        clustering_method: str = "embed",
        cluster_algorithm: str = "dfs",
        questions_json_path: Optional[str] = None,
        **clustering_kwargs
    ):
        """
        Initialize the baseline framework.
        
        Args:
            config_path: config file path
            result_base_path: base path for results (e.g., source/live_code_bench/result)
            difficulty: difficulty level ("easy", "medium", "hard")
            clustering_method: clustering method name
            cluster_algorithm: clustering algorithm ("dfs" or "greedy")
            questions_json_path: question metadata JSON path (optional)
            **clustering_kwargs: extra clustering parameters
        """
        self.config_path = config_path
        self.result_base_path = result_base_path
        self.difficulty = difficulty
        self.clustering_method = clustering_method
        self.cluster_algorithm = cluster_algorithm
        self.clustering_kwargs = clustering_kwargs
        self.questions_json_path = questions_json_path
        
        # Question metadata cache (lazy load)
        self._questions_dict = None
        
        # Load config
        self.config = self._load_config()
        self.layers = self.config["layers"]
        
        # Use only the first layer
        if not self.layers:
            raise ValueError("No layer configuration found in config file")
        
        self.first_layer = self.layers[0]
        
        # Validate config
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the config file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    
    def _validate_config(self):
        """Validate the config file."""
        required_keys = ["layer_idx", "samples"]
        for key in required_keys:
            if key not in self.first_layer:
                raise ValueError(f"First layer config missing required key: {key}")
    
    def _create_lm_from_config(self):
        """
        Create a language model instance from config (for functional clustering).
        
        Returns:
            dspy.LM instance
        """
        lm_config = {
            "model_name": "openai/qcoder",
            "api_base": "http://localhost:8001/v1",
            "api_key": "your-api-key-here",
            "timeout": 120,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "max_tokens": 8192,
            "logprobs": True,
            "top_logprobs": 0,
            "cache": True
        }
        
        # Build kwargs (filter None values)
        kwargs = {}
        if lm_config.get("api_base") is not None:
            kwargs["api_base"] = lm_config["api_base"]
        if lm_config.get("api_key") is not None:
            kwargs["api_key"] = lm_config["api_key"]
        if lm_config.get("timeout") is not None:
            kwargs["timeout"] = lm_config["timeout"]
        if lm_config.get("max_tokens") is not None:
            kwargs["max_tokens"] = lm_config["max_tokens"]
        if lm_config.get("temperature") is not None:
            kwargs["temperature"] = lm_config["temperature"]
        kwargs["cache"] = lm_config.get("cache", True)
        
        # Create LM instance
        lm = dspy.LM(lm_config["model_name"], **kwargs)
        logger.info(f"Language model configured: {lm_config['model_name']}")
        
        return lm
    
    def _load_questions_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Load question metadata.
        
        Returns:
            Dict mapping task_id to question info.
        """
        if self._questions_dict is not None:
            return self._questions_dict
        
        if self.questions_json_path is None:
            logger.warning("Question metadata path not provided; cannot load")
            return {}
        
        if not os.path.exists(self.questions_json_path):
            logger.warning(f"Question metadata file not found: {self.questions_json_path}")
            return {}
        
        logger.info(f"Loading question metadata: {self.questions_json_path}")
        
        try:
            with open(self.questions_json_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            # Convert to dict for quick lookup
            self._questions_dict = {q["task_id"]: q for q in questions}
            
            logger.info(f"Loaded metadata for {len(self._questions_dict)} questions")
            return self._questions_dict
        except Exception as e:
            logger.error(f"Failed to load question metadata: {e}")
            return {}
    
    def get_all_question_ids(self) -> List[str]:
        """
        Collect all available question_ids from the dataset.
        
        Returns:
            Deduplicated list of question_ids.
        """
        question_ids_set = set()
        samples_config = self.first_layer["samples"]
        
        # Traverse samples for each model
        for model_name, rounds in samples_config.items():
            for round_idx in rounds:
                # Build file path
                file_path = os.path.join(
                    self.result_base_path,
                    self.difficulty,
                    model_name,
                    f"round_{round_idx}.jsonl"
                )
                
                # Read file and collect question_ids
                if not os.path.exists(file_path):
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if "question_id" in data:
                                question_id = str(data["question_id"])
                                question_ids_set.add(question_id)
                        except json.JSONDecodeError:
                            continue
        
        question_ids = sorted(list(question_ids_set))
        return question_ids
    
    def _load_all_samples(self, question_id: str) -> List[Dict[str, Any]]:
        """
        Load all samples from the first layer (both first and last code).
        
        Args:
            question_id: question ID
        
        Returns:
            List of samples, each containing:
                - first_code: first code (debug_trace[0])
                - first_logprob: logprob of first code
                - first_norm_logprob: normalized logprob of first code
                - last_code: last code (debug_trace[-1])
                - first_passed: first code passed (passed AND len(debug_trace)==1)
                - last_passed: last code passed (passed)
                - passed: passed private tests
                - is_passed_public: passed public tests
                - score: private test score
                - source: origin (model_name/round_idx)
        """
        samples_config = self.first_layer["samples"]
        all_samples = []
        
        # Traverse samples for each model
        for model_name, rounds in samples_config.items():
            for round_idx in rounds:
                # Build file path
                file_path = os.path.join(
                    self.result_base_path,
                    self.difficulty,
                    model_name,
                    f"round_{round_idx}.jsonl"
                )
                
                # Read file and locate the question
                if not os.path.exists(file_path):
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        if data["question_id"] == question_id:
                            debug_trace = data.get("debug_trace", [])
                            if not debug_trace:
                                continue
                            
                            # Extract first and last code
                            first_trace = debug_trace[0]
                            last_trace = debug_trace[-1]
                            
                            first_code = first_trace.get("code", "")
                            last_code = last_trace.get("code", "")
                            
                            # Extract logprob/norm_logprob from first_trace (no backtracking)
                            first_logprob = first_trace.get("logprob")
                            first_norm_logprob = first_trace.get("norm_logprob")
                            
                            # If logprob missing, use defaults
                            if first_logprob is None:
                                first_logprob = 0.0
                            if first_norm_logprob is None:
                                first_norm_logprob = 0.0
                            
                            # First code correctness: passed AND len(debug_trace)==1
                            first_passed = data.get("passed", False) and len(debug_trace) == 1
                            
                            # Last code correctness: passed
                            last_passed = data.get("passed", False)
                            
                            sample = {
                                "first_code": first_code,
                                "last_code": last_code,
                                "first_logprob": first_logprob,
                                "first_norm_logprob": first_norm_logprob,
                                "first_passed": first_passed,
                                "last_passed": last_passed,
                                "passed": data.get("passed", False),
                                "is_passed_public": data.get("is_passed_public", False),
                                "score": data.get("score", 0.0),
                                "source": f"{model_name}/round_{round_idx}",
                                "debug_trace_length": len(debug_trace)
                            }
                            
                            all_samples.append(sample)
                            break
        
        return all_samples
    
    def method_vanilla(self, question_id: str) -> Dict[str, Any]:
        """
        Method 1: vanilla.
        Select one sample and return correctness of the first code.
        
        Args:
            question_id: question ID
        
        Returns:
            Result dict with:
                - question_id
                - method
                - passed
                - score
                - source
        """
        samples = self._load_all_samples(question_id)
        
        if len(samples) == 0:
            return {
                "question_id": question_id,
                "method": "vanilla",
                "passed": False,
                "score": 0.0,
                "source": "none"
            }
        
        # Randomly select one sample
        selected_sample = random.choice(samples)
        
        return {
            "question_id": question_id,
            "method": "vanilla",
            "passed": selected_sample["first_passed"],
            "score": selected_sample["score"],
            "source": selected_sample["source"]
        }
    
    def method_major_voting(
        self,
        question_id: str,
        problem_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Method 2: major_voting.
        Cluster all first codes and randomly choose one from the largest cluster.
        
        Args:
            question_id: question ID
            problem_info: question metadata (optional, for clustering)
        
        Returns:
            Result dict
        """
        samples = self._load_all_samples(question_id)
        
        if len(samples) == 0:
            return {
                "question_id": question_id,
                "method": "major_voting",
                "passed": False,
                "score": 0.0,
                "source": "none"
            }
        
        # Collect all first codes (with logprob/norm_logprob)
        first_codes = []
        for sample in samples:
            code_dict = {
                "code": sample["first_code"],
                "logprob": sample["first_logprob"],
                "norm_logprob": sample["first_norm_logprob"],
                "first_passed": sample["first_passed"],
                "score": sample["score"],
                "source": sample["source"]
            }
            first_codes.append(code_dict)
        
        # No code available -> return failure
        if len(first_codes) == 0:
            return {
                "question_id": question_id,
                "method": "major_voting",
                "passed": False,
                "score": 0.0,
                "source": "none"
            }
        
        # Prepare clustering kwargs
        clustering_kwargs = self.clustering_kwargs.copy()
        
        # If using functional or functional_vanilla
        if self.clustering_method in ["functional", "functional_vanilla"]:
            # Check cache usage (default True)
            use_cache = clustering_kwargs.get("use_cache", True)
            
            # With cache, LM is optional (tests already cached).
            # Without cache or when generating new tests, LM is required.
            if "lm" not in clustering_kwargs:
                if not use_cache:
                    # Without cache, must create LM to generate tests
                    clustering_kwargs["lm"] = self._create_lm_from_config()
                # If use_cache=True, lm can be None (FunctionalClustering allows it).
                # If cache lacks tests, an error is expected.
        
        # Load problem_info if not provided
        if problem_info is None:
            questions_dict = self._load_questions_info()
            if question_id in questions_dict:
                question_info = questions_dict[question_id]
                problem_info = {
                    "prompt": question_info.get("prompt", ""),
                    "is_stdin": question_info.get("is_stdin", False),
                    "task_id": question_id
                }
            else:
                problem_info = {
                    "prompt": "",
                    "is_stdin": False,
                    "task_id": question_id
                }
        
        # Run clustering (pass code/logprob/norm_logprob)
        cluster_ids = cluster_codes(
            codes=first_codes,
            problem_info=problem_info,
            method=self.clustering_method,
            cluster_algorithm=self.cluster_algorithm,
            **clustering_kwargs
        )
        
        # Count cluster sizes
        cluster_counter = Counter(cluster_ids)
        
        # Find the largest cluster
        if len(cluster_counter) == 0:
            return {
                "question_id": question_id,
                "method": "major_voting",
                "passed": False,
                "score": 0.0,
                "source": "none"
            }
        
        largest_cluster_id = cluster_counter.most_common(1)[0][0]
        
        # Indices of samples in the largest cluster
        largest_cluster_indices = [
            i for i, cid in enumerate(cluster_ids) if cid == largest_cluster_id
        ]
        
        # Randomly choose one
        selected_idx = random.choice(largest_cluster_indices)
        selected_sample = first_codes[selected_idx]
        
        return {
            "question_id": question_id,
            "method": "major_voting",
            "passed": selected_sample["first_passed"],
            "score": selected_sample["score"],
            "source": selected_sample["source"],
            "num_clusters": len(cluster_counter),
            "largest_cluster_size": len(largest_cluster_indices)
        }
    
    def method_pass_at_n(self, question_id: str) -> Dict[str, Any]:
        """
        Method 3: pass_at_n.
        Return whether any final code passes.
        
        Args:
            question_id: question ID
        
        Returns:
            Result dict
        """
        samples = self._load_all_samples(question_id)
        
        if len(samples) == 0:
            return {
                "question_id": question_id,
                "method": "pass_at_n",
                "passed": False,
                "score": 0.0,
                "num_samples": 0
            }
        
        # Check whether any final code passes
        any_passed = any(sample["last_passed"] for sample in samples)
        
        # Compute average score
        avg_score = np.mean([sample["score"] for sample in samples]) if samples else 0.0
        
        return {
            "question_id": question_id,
            "method": "pass_at_n",
            "passed": any_passed,
            "score": avg_score,
            "num_samples": len(samples),
            "num_passed": sum(1 for sample in samples if sample["last_passed"])
        }
    
    def method_pass_at_n_oneshot(self, question_id: str) -> Dict[str, Any]:
        """
        Method 4: pass_at_n_oneshot.
        Return whether any first code passes.
        
        Args:
            question_id: question ID
        
        Returns:
            Result dict
        """
        samples = self._load_all_samples(question_id)
        
        if len(samples) == 0:
            return {
                "question_id": question_id,
                "method": "pass_at_n_oneshot",
                "passed": False,
                "score": 0.0,
                "num_samples": 0
            }
        
        # Check whether any first code passes
        any_passed = any(sample["first_passed"] for sample in samples)
        
        # Compute average score
        avg_score = np.mean([sample["score"] for sample in samples]) if samples else 0.0
        
        return {
            "question_id": question_id,
            "method": "pass_at_n_oneshot",
            "passed": any_passed,
            "score": avg_score,
            "num_samples": len(samples),
            "num_passed": sum(1 for sample in samples if sample["first_passed"])
        }
    
    def process_question(
        self,
        question_id: str,
        method: str,
        problem_info: Optional[Dict[str, Any]] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single question.
        
        Args:
            question_id: question ID
            method: method name ("vanilla", "major_voting", "pass_at_n", "pass_at_n_oneshot")
            problem_info: question metadata (optional, for clustering)
            verbose: whether to print detailed output
        
        Returns:
            Result dict
        """
        if method == "vanilla":
            return self.method_vanilla(question_id)
        elif method == "major_voting":
            return self.method_major_voting(question_id, problem_info)
        elif method == "pass_at_n":
            return self.method_pass_at_n(question_id)
        elif method == "pass_at_n_oneshot":
            return self.method_pass_at_n_oneshot(question_id)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def process_questions(
        self,
        question_ids: List[str],
        method: str,
        verbose: bool = True,
        max_workers: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch.
        
        Args:
            question_ids: list of question IDs
            method: method name
            verbose: whether to print detailed output
            max_workers: max worker threads (default 1 for serial)
        
        Returns:
            List of results
        """
        if max_workers <= 1:
            # Serial processing
            results = []
            
            for i, question_id in enumerate(question_ids):
                if verbose:
                    print(f"\n处理问题 {i+1}/{len(question_ids)}: {question_id}")
                
                result = self.process_question(question_id, method, verbose=verbose)
                results.append(result)
        else:
            # Parallel processing
            results = [None] * len(question_ids)
            print_lock = Lock() if verbose else None
            
            def process_with_index(index_and_id):
                """Processing function with index for parallel execution"""
                index, question_id = index_and_id
                try:
                    # In parallel, guard verbose output with a lock
                    if verbose and print_lock:
                        with print_lock:
                            print(f"\n处理问题 {index+1}/{len(question_ids)}: {question_id}")
                    
                    result = self.process_question(question_id, method, verbose=False)
                    return index, result
                except Exception as e:
                    logger.error(f"Error processing question {question_id}: {e}")
                    if verbose and print_lock:
                        with print_lock:
                            print(f"Error: Error processing question {question_id}: {e}")
                    # Return error result
                    return index, {
                        "question_id": question_id,
                        "method": method,
                        "passed": False,
                        "score": 0.0,
                        "source": "none",
                        "error": str(e)
                    }
            
            # Parallel processing with thread pool
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(process_with_index, (i, qid)): i 
                    for i, qid in enumerate(question_ids)
                }
                
                # Collect results
                completed = 0
                for future in as_completed(future_to_index):
                    completed += 1
                    try:
                        index, result = future.result()
                        results[index] = result
                        if verbose and print_lock:
                            with print_lock:
                                print(f"Completed {completed}/{len(question_ids)} questions")
                    except Exception as e:
                        index = future_to_index[future]
                        question_id = question_ids[index]
                        logger.error(f"Error getting results for question {question_id}: {e}")
                        if verbose and print_lock:
                            with print_lock:
                                print(f"Error: Error getting results for question {question_id}: {e}")
                        # Set error result
                        results[index] = {
                            "question_id": question_id,
                            "method": method,
                            "passed": False,
                            "score": 0.0,
                            "source": "none",
                            "error": str(e)
                        }
        
        return results
    
    def get_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute summary statistics.
        
        Args:
            results: list of results
        
        Returns:
            Statistics dict
        """
        if not results:
            return {}
        
        total = len(results)
        passed = sum(1 for r in results if r.get("passed", False))
        accuracy = passed / total if total > 0 else 0.0
        
        # Compute average score
        scores = [r.get("score", 0.0) for r in results]
        avg_score = np.mean(scores) if scores else 0.0
        
        stats = {
            "total_questions": total,
            "passed": passed,
            "accuracy": accuracy,
            "avg_score": avg_score
        }
        
        # Add extra stats if available
        if results and "num_clusters" in results[0]:
            avg_clusters = np.mean([r.get("num_clusters", 0) for r in results])
            stats["avg_num_clusters"] = avg_clusters
        
        if results and "num_samples" in results[0]:
            avg_samples = np.mean([r.get("num_samples", 0) for r in results])
            stats["avg_num_samples"] = avg_samples
        
        return stats


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Baselines")
    parser.add_argument(
        "--config",
        type=str,
        help="Config file path"
    )
    parser.add_argument(
        "--result_base",
        type=str,
        help="Base path for result files"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard"],
        help="Difficulty level"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["vanilla", "major_voting", "pass_at_n", "pass_at_n_oneshot"],
        help="Baseline method"
    )
    parser.add_argument(
        "--clustering_method",
        type=str,
        default="embed",
        help="Clustering method (for major_voting)"
    )
    parser.add_argument(
        "--cluster_algorithm",
        type=str,
        default="dfs",
        choices=["dfs", "greedy"],
        help="Clustering algorithm"
    )
    parser.add_argument(
        "--question_ids",
        type=str,
        nargs="+",
        help="List of question IDs to process"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Similarity threshold (for embed and bleu)"
    )
    parser.add_argument(
        "--questions_json",
        type=str,
        help="Question metadata JSON path"
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        default=True,
        help="Use test case cache (functional methods, default True)"
    )
    parser.add_argument(
        "--no_cache",
        action="store_false",
        dest="use_cache",
        help="Disable test case cache (functional methods)"
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        help="Test case cache file path (functional methods)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Max worker threads (default 1 for serial)"
    )
    
    args = parser.parse_args()
    
    # Prepare clustering kwargs
    clustering_kwargs = {}
    if args.threshold is not None:
        clustering_kwargs["threshold"] = args.threshold
    if args.clustering_method in ["functional", "functional_vanilla"]:
        clustering_kwargs["use_cache"] = args.use_cache
        clustering_kwargs["cache_path"] = args.cache_path
    
    # Initialize framework
    baselines = Baselines(
        config_path=args.config,
        result_base_path=args.result_base,
        difficulty=args.difficulty,
        clustering_method=args.clustering_method,
        cluster_algorithm=args.cluster_algorithm,
        questions_json_path=args.questions_json,
        **clustering_kwargs
    )
    
    # Process questions
    if args.question_ids:
        question_ids = args.question_ids
    else:
        # If question_ids not specified, collect all from dataset
        print("No question_ids specified, collecting all question IDs from the dataset...")
        question_ids = baselines.get_all_question_ids()
        print(f"Found {len(question_ids)} questions")
    
    if question_ids:
        results = baselines.process_questions(
            question_ids=question_ids,
            method=args.method,
            verbose=True,
            max_workers=args.max_workers
        )
        
        # Print statistics
        stats = baselines.get_statistics(results)
        print("\n" + "="*60)
        print(f"Method: {args.method}")
        print("Statistics:")
        print("="*60)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        print("Warning: No question IDs found. Please check dataset path and configuration.")


if __name__ == "__main__":
    main()
