"""
Cascade Test-Time Scaling framework.

An entropy-based multi-layer code generation framework that decides whether to
sample more by evaluating uncertainty layer by layer.
"""

import json
import os
import random
import shutil
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Import clustering and entropy modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ESE.clustering.clustering_interface import cluster_codes
from ESE.entropy.semantic_entropy import semantic_entropy
import dspy

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CascadeTTS:
    """Cascade Test-Time Scaling framework."""
    
    def __init__(
        self,
        config_path: str,
        result_base_path: str,
        difficulty: str = "easy",
        clustering_method: str = "embed",
        metric: str = "SE",
        use_norm_logprob: bool = False,
        cluster_algorithm: str = "dfs",
        exp_name: Optional[str] = None,
        questions_json_path: Optional[str] = None,
        **clustering_kwargs
    ):
        """
        Initialize the Cascade TTS framework.
        
        Args:
            config_path: config file path
            result_base_path: base results path (e.g., source/live_code_bench/result)
            difficulty: difficulty level ("easy", "medium", "hard")
            clustering_method: clustering method name
            metric: entropy metric ("PE_MC", "PE_Rao", "SE", "DSE")
            use_norm_logprob: whether to use normalized log-probabilities
            cluster_algorithm: clustering algorithm ("dfs" or "greedy")
            exp_name: experiment name for saving results
            questions_json_path: question metadata JSON path
            **clustering_kwargs: extra clustering parameters
        """
        self.config_path = config_path
        self.result_base_path = result_base_path
        self.difficulty = difficulty
        self.clustering_method = clustering_method
        self.metric = metric
        self.use_norm_logprob = use_norm_logprob
        self.cluster_algorithm = cluster_algorithm
        self.clustering_kwargs = clustering_kwargs
        self.exp_name = exp_name
        self.questions_json_path = questions_json_path
        
        # Question metadata cache (lazy load)
        self._questions_dict = None
        
        # Load config
        self.config = self._load_config()
        self.layers_count = self.config["layers_count"]
        self.layers = self.config["layers"]
        
        # Validate config
        self._validate_config()
        
        # If exp_name is provided, create experiment directory and save config
        if self.exp_name:
            self._setup_experiment_dir()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the config file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    
    def _validate_config(self):
        """Validate the config file."""
        if len(self.layers) != self.layers_count:
            raise ValueError(
                f"Config file layers_count={self.layers_count} does not match actual number of layers={len(self.layers)}"
            )
        
        for layer in self.layers:
            required_keys = ["layer_idx", "samples", "alpha", "beta", "threshold"]
            for key in required_keys:
                if key not in layer:
                    raise ValueError(f"Layer {layer.get('layer_idx', '?')} missing required key: {key}")
    
    def _setup_experiment_dir(self):
        """Set up experiment directory and save config."""
        # Get current file directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create experiment output directory
        self.exp_dir = os.path.join(current_dir, "result", self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Copy config and add experiment parameters
        exp_config = self.config.copy()
        exp_config["experiment"] = {
            "exp_name": self.exp_name,
            "difficulty": self.difficulty,
            "clustering_method": self.clustering_method,
            "metric": self.metric,
            "use_norm_logprob": self.use_norm_logprob,
            "cluster_algorithm": self.cluster_algorithm,
            "clustering_kwargs": self.clustering_kwargs,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save config file
        config_save_path = os.path.join(self.exp_dir, "config.json")
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(exp_config, f, indent=2, ensure_ascii=False)
        
    
    def _create_lm_from_config(self):
        """
        Create a language model instance from config.
        
        Returns:
            dspy.LM instance
        """
        # Config mirrors calc_entropy.py and is hardcoded here
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
        logger.info(f"API Base: {kwargs.get('api_base', 'N/A')}")
        
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
        
        # Traverse layer configs to collect all models and rounds
        for layer_config in self.layers:
            samples_config = layer_config["samples"]
            
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
                                    # Ensure question_id is a string
                                    question_id = str(data["question_id"])
                                    question_ids_set.add(question_id)
                            except json.JSONDecodeError:
                                continue
        
        question_ids = sorted(list(question_ids_set))
        return question_ids
    
    def _load_samples_from_layer(
        self,
        layer_config: Dict[str, Any],
        question_id: str
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Load samples from a specific layer.
        
        Args:
            layer_config: layer config
            question_id: question ID
        
        Returns:
            (codes, total_count, passed_public_count)
            - codes: codes that passed public tests
            - total_count: total sample count
            - passed_public_count: count that passed public tests
        """
        samples_config = layer_config["samples"]
        all_codes = []
        total_count = 0
        
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
                            total_count += 1
                            
                            # Extract last code from debug_trace
                            debug_trace = data.get("debug_trace", [])
                            if not debug_trace:
                                continue
                            
                            last_trace = debug_trace[-1]
                            code = last_trace.get("code", "")
                            
                            # Extract logprob/norm_logprob
                            # If missing on last sample, search backwards
                            logprob = None
                            norm_logprob = None
                            
                            for trace in reversed(debug_trace):
                                if logprob is None and trace.get("logprob") is not None:
                                    logprob = trace["logprob"]
                                if norm_logprob is None and trace.get("norm_logprob") is not None:
                                    norm_logprob = trace["norm_logprob"]
                                if logprob is not None and norm_logprob is not None:
                                    break
                            
                            # If logprob missing, use defaults
                            if logprob is None:
                                logprob = 0.0
                            if norm_logprob is None:
                                norm_logprob = 0.0
                            
                            # Check public test status
                            is_passed_public = data.get("is_passed_public", False)
                            
                            code_dict = {
                                "code": code,
                                "logprob": logprob,
                                "norm_logprob": norm_logprob,
                                "is_passed_public": is_passed_public,
                                "passed": data.get("passed", False),
                                "score": data.get("score", 0.0),
                                "source": f"{model_name}/round_{round_idx}"
                            }
                            
                            # Keep only codes that passed public tests
                            if is_passed_public:
                                all_codes.append(code_dict)
                            
                            break
        
        passed_public_count = len(all_codes)
        return all_codes, total_count, passed_public_count
    
    def _compute_entropy_and_score(
        self,
        codes: List[Dict[str, Any]],
        problem_info: Dict[str, Any],
        alpha: float,
        beta: float,
        total_count: int,
        passed_public_count: int
    ) -> Tuple[float, float, List[int], Dict[str, float]]:
        """
        Compute entropy and the combined score.
        
        Args:
            codes: list of code samples
            problem_info: question metadata
            alpha: entropy weight
            beta: public_fail_rate weight
            total_count: total sample count
            passed_public_count: count that passed public tests
        
        Returns:
            (score, entropy, cluster_ids, entropy_dict)
        """
        if len(codes) == 0:
            return float('inf'), 0.0, [], {}
        
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
        
        # Run clustering
        cluster_ids = cluster_codes(
            codes=codes,
            problem_info=problem_info,
            method=self.clustering_method,
            cluster_algorithm=self.cluster_algorithm,
            **clustering_kwargs
        )
        
        # Compute four entropy metrics
        entropy_dict = semantic_entropy(
            codes=codes,
            cluster_ids=cluster_ids,
            use_norm_logprob=self.use_norm_logprob
        )
        
        # Select the requested entropy metric
        entropy = entropy_dict[self.metric]
        
        # Compute public_fail_rate
        public_fail_rate = (total_count - passed_public_count) / total_count if total_count > 0 else 0.0
        
        # Compute combined score
        score = entropy * alpha + public_fail_rate * beta
        
        return score, entropy, cluster_ids, entropy_dict
    
    def _select_from_largest_cluster(
        self,
        codes: List[Dict[str, Any]],
        cluster_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Randomly select a sample from the largest cluster.
        
        Args:
            codes: list of codes
            cluster_ids: list of cluster IDs
        
        Returns:
            Selected code dict
        """
        if len(codes) == 0:
            return None
        
        # Count cluster sizes
        cluster_counter = Counter(cluster_ids)
        
        # Find the largest cluster
        largest_cluster_id = cluster_counter.most_common(1)[0][0]
        
        # Collect samples in the largest cluster
        largest_cluster_indices = [
            i for i, cid in enumerate(cluster_ids) if cid == largest_cluster_id
        ]
        
        # Randomly select one
        selected_idx = random.choice(largest_cluster_indices)
        
        return codes[selected_idx]
    
    def process_question(
        self,
        question_id: str,
        problem_info: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single question through cascade layers.
        
        Args:
            question_id: question ID
            problem_info: question metadata (optional, for clustering)
            verbose: whether to print detailed output
        
        Returns:
            Result dict with:
                - question_id
                - selected_code
                - final_score
                - final_layer
                - layer_details
        """
        if problem_info is None:
            # Try loading question metadata from file
            questions_dict = self._load_questions_info()
            if question_id in questions_dict:
                question_info = questions_dict[question_id]
                problem_info = {
                    "prompt": question_info.get("prompt", ""),
                    "is_stdin": question_info.get("is_stdin", False),
                    "task_id": question_id
                }
            else:
                # If not found, fall back to defaults
                logger.warning(f"Question {question_id} not found, using defaults")
                problem_info = {
                    "prompt": "",
                    "is_stdin": False,
                    "task_id": question_id
                }
        
        layer_details = []
        all_codes_per_layer = []  # Codes per layer for result construction
        all_cluster_ids_per_layer = []  # Cluster IDs per layer
        
        for layer_idx, layer_config in enumerate(self.layers):
            if verbose:
                print(f"\n=== Processing layer {layer_idx} ===")
            
            # Load samples for this layer
            codes, total_count, passed_public_count = self._load_samples_from_layer(
                layer_config, question_id
            )
            
            if verbose:
                print(f"Total samples: {total_count}")
                print(f"Passed public test: {passed_public_count}")
                print(f"Filtered samples: {len(codes)}")
            
            # If no samples passed public tests, skip this layer
            if len(codes) == 0:
                if verbose:
                    print("No samples passed public tests, skipping this layer")
                
                layer_details.append({
                    "layer_idx": layer_idx,
                    "total_count": total_count,
                    "passed_public_count": passed_public_count,
                    "filtered_count": 0,
                    "score": float('inf'),
                    "entropy": 0.0,
                    "public_fail_rate": 1.0,
                    "entropy_dict": {},
                    "num_clusters": 0,
                    "decision": "skip_no_samples"
                })
                all_codes_per_layer.append([])
                all_cluster_ids_per_layer.append([])
                continue
            
            # Compute entropy and score
            score, entropy, cluster_ids, entropy_dict = self._compute_entropy_and_score(
                codes=codes,
                problem_info=problem_info,
                alpha=layer_config["alpha"],
                beta=layer_config["beta"],
                total_count=total_count,
                passed_public_count=passed_public_count
            )
            
            all_codes_per_layer.append(codes)
            all_cluster_ids_per_layer.append(cluster_ids)
            
            public_fail_rate = (total_count - passed_public_count) / total_count
            num_clusters = len(set(cluster_ids))
            
            if verbose:
                print(f"Entropy metric ({self.metric}): {entropy:.4f}")
                print(f"Public fail rate: {public_fail_rate:.4f}")
                print(f"Combined score: {score:.4f}")
                print(f"Number of clusters: {num_clusters}")
                print(f"Threshold: {layer_config['threshold']}")
            
            layer_detail = {
                "layer_idx": layer_idx,
                "total_count": total_count,
                "passed_public_count": passed_public_count,
                "filtered_count": len(codes),
                "score": score,
                "entropy": entropy,
                "public_fail_rate": public_fail_rate,
                "entropy_dict": entropy_dict,
                "num_clusters": num_clusters,
                "threshold": layer_config["threshold"]
            }
            
            # Decide whether to move to the next layer
            is_last_layer = (layer_idx == self.layers_count - 1)
            
            # Compute largest-cluster info for the new stop condition
            cluster_counter = Counter(cluster_ids)
            largest_cluster_id, largest_cluster_size = cluster_counter.most_common(1)[0] if cluster_counter else (None, 0)
            largest_cluster_indices = [i for i, cid in enumerate(cluster_ids) if cid == largest_cluster_id] if largest_cluster_id is not None else []
            unique_models_in_largest = len(set(codes[i].get("source", "").split("/")[0] for i in largest_cluster_indices)) if largest_cluster_indices else 0
            
            # New stop condition: all three conditions satisfied
            should_stop = (largest_cluster_size > 0.3 * total_count and 
                          largest_cluster_size > 0.6 * passed_public_count and 
                          unique_models_in_largest > 1)
            
            if is_last_layer or should_stop:
            # if is_last_layer or score <= layer_config["threshold"]:
                # Stop at current layer and select from largest cluster
                selected_code = self._select_from_largest_cluster(codes, cluster_ids)
                
                layer_detail["decision"] = "stop"
                layer_details.append(layer_detail)
                
                if verbose:
                    print(f"Decision: stop at layer {layer_idx}")
                    if is_last_layer:
                        print("Reason: reached the last layer")
                    else:
                        print(f"Reason: new stop condition satisfied (largest cluster={largest_cluster_size}, number of models={unique_models_in_largest})")
                
                return {
                    "question_id": question_id,
                    "selected_code": selected_code["code"] if selected_code else "",
                    "final_score": selected_code["score"] if selected_code else 0.0,
                    "final_passed": selected_code["passed"] if selected_code else False,
                    "final_layer": layer_idx,
                    "layer_details": layer_details,
                    "source": selected_code["source"] if selected_code else "none",
                    "_codes_per_layer": all_codes_per_layer,
                    "_cluster_ids_per_layer": all_cluster_ids_per_layer
                }
            else:
                # Continue to next layer
                layer_detail["decision"] = "continue"
                layer_details.append(layer_detail)
                
                if verbose:
                    print(f"Decision: continue to next layer (largest cluster={largest_cluster_size}, number of models={unique_models_in_largest})")
        
        # If all layers are skipped, return empty result
        return {
            "question_id": question_id,
            "selected_code": "",
            "final_score": 0.0,
            "final_passed": False,
            "final_layer": -1,
            "layer_details": layer_details,
            "source": "none",
            "_codes_per_layer": all_codes_per_layer,
            "_cluster_ids_per_layer": all_cluster_ids_per_layer
        }
    
    def _format_result_for_save(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format result for saving.
        
        Args:
            result: raw result dict
        
        Returns:
            Formatted result dict
        """
        codes_per_layer = result.get("_codes_per_layer", [])
        cluster_ids_per_layer = result.get("_cluster_ids_per_layer", [])
        
        layers = []
        for layer_idx, layer_detail in enumerate(result["layer_details"]):
            codes = codes_per_layer[layer_idx] if layer_idx < len(codes_per_layer) else []
            cluster_ids = cluster_ids_per_layer[layer_idx] if layer_idx < len(cluster_ids_per_layer) else []
            
            # Extract passed status for each sample
            passed_list = [code.get("passed", False) for code in codes]
            
            layer_info = {
                "layer_idx": layer_idx,
                "passed_public": layer_detail.get("passed_public_count", 0),
                "passed": passed_list,
                "cluster_ids": cluster_ids,
                "entropy": layer_detail.get("entropy", 0.0),
                "score": layer_detail.get("score", 0.0),
                "stop": layer_detail.get("decision") == "stop"
            }
            layers.append(layer_info)
        
        formatted_result = {
            "question_id": result["question_id"],
            "layers": layers,
            "score": result.get("final_score", 0.0),
            "passed": result.get("final_passed", False)
        }
        
        return formatted_result
    
    def process_questions(
        self,
        question_ids: List[str],
        output_path: Optional[str] = None,
        verbose: bool = True,
        max_workers: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch.
        
        Args:
            question_ids: list of question IDs
            output_path: output file path (optional; auto-saved if exp_name is set)
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
                    print(f"\n{'='*60}")
                    print(f"Processing question {i+1}/{len(question_ids)}: {question_id}")
                    print(f"{'='*60}")
                
                result = self.process_question(question_id, verbose=verbose)
                results.append(result)
        else:
            # Parallel processing
            results = [None] * len(question_ids)
            print_lock = Lock() if verbose else None
            
            def process_with_index(index_and_id):
                index, question_id = index_and_id
                try:
                    # In parallel, guard verbose output with a lock
                    if verbose and print_lock:
                        with print_lock:
                            print(f"\n{'='*60}")
                            print(f"Processing question {index+1}/{len(question_ids)}: {question_id}")
                            print(f"{'='*60}")
                    
                    result = self.process_question(question_id, verbose=False)
                    return index, result
                except Exception as e:
                    logger.error(f"Error processing question {question_id}: {e}")
                    if verbose and print_lock:
                        with print_lock:
                            print(f"Error processing question {question_id}: {e}")
                    # Return error result
                    return index, {
                        "question_id": question_id,
                        "selected_code": "",
                        "final_score": 0.0,
                        "final_passed": False,
                        "final_layer": -1,
                        "layer_details": [],
                        "source": "none",
                        "_codes_per_layer": [],
                        "_cluster_ids_per_layer": [],
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
                                print(f"Error getting results for question {question_id}: {e}")
                        # Set error result
                        results[index] = {
                            "question_id": question_id,
                            "selected_code": "",
                            "final_score": 0.0,
                            "final_passed": False,
                            "final_layer": -1,
                            "layer_details": [],
                            "source": "none",
                            "_codes_per_layer": [],
                            "_cluster_ids_per_layer": [],
                            "error": str(e)
                        }
        
        # If exp_name is set, save to experiment directory
        if self.exp_name:
            result_path = os.path.join(self.exp_dir, "result.jsonl")
            with open(result_path, 'w', encoding='utf-8') as f:
                for result in results:
                    formatted_result = self._format_result_for_save(result)
                    f.write(json.dumps(formatted_result, ensure_ascii=False) + '\n')
            
            if verbose:
                print(f"\nResults saved to: {result_path}")
        
        # If output_path is provided, save full results too
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    # Remove temporary fields
                    result_copy = result.copy()
                    result_copy.pop("_codes_per_layer", None)
                    result_copy.pop("_cluster_ids_per_layer", None)
                    f.write(json.dumps(result_copy, ensure_ascii=False) + '\n')
            
            if verbose:
                print(f"\nResults saved to: {output_path}")
        
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
        
        # Count stops per layer
        layer_stops = Counter([r["final_layer"] for r in results])
        
        # Compute average score
        avg_score = np.mean([r["final_score"] for r in results])
        
        # Compute average entropy per layer
        layer_entropies = {}
        for layer_idx in range(self.layers_count):
            entropies = []
            for r in results:
                if layer_idx < len(r["layer_details"]):
                    layer_detail = r["layer_details"][layer_idx]
                    if "entropy" in layer_detail:
                        entropies.append(layer_detail["entropy"])
            if entropies:
                layer_entropies[layer_idx] = np.mean(entropies)
        
        return {
            "total_questions": len(results),
            "layer_stops": dict(layer_stops),
            "avg_final_score": avg_score,
            "layer_avg_entropies": layer_entropies
        }


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cas Framework")
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
        "--clustering_method",
        type=str,
        default="embed",
        help="Clustering method"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="SE",
        choices=["PE_MC", "PE_Rao", "SE", "DSE"],
        help="Entropy metric"
    )
    parser.add_argument(
        "--use_norm_logprob",
        action="store_true",
        help="Use normalized log-probabilities"
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
        "--output",
        type=str,
        help="Output file path"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Similarity threshold (for embed and bleu)"
    )
    parser.add_argument(
        "--exp",
        type=str,
        help="Experiment name; saves to source/TTS/result/<exp_name>/"
    )
    parser.add_argument(
        "--questions_json",
        type=str,
        help="Question metadata JSON path (e.g., lcb_release_v2_all_questions.json)"
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
    cascade_tts = CascadeTTS(
        config_path=args.config,
        result_base_path=args.result_base,
        difficulty=args.difficulty,
        clustering_method=args.clustering_method,
        metric=args.metric,
        use_norm_logprob=args.use_norm_logprob,
        cluster_algorithm=args.cluster_algorithm,
        exp_name=args.exp,
        questions_json_path=args.questions_json,
        **clustering_kwargs
    )
    
    # Process questions
    if args.question_ids:
        question_ids = args.question_ids
    else:
        # If question_ids not specified, collect all from dataset
        question_ids = cascade_tts.get_all_question_ids()
    
    if question_ids:
        results = cascade_tts.process_questions(
            question_ids=question_ids,
            output_path=args.output,
            verbose=True,
            max_workers=args.max_workers
        )
        
        # Print statistics
        stats = cascade_tts.get_statistics(results)
        print("\n" + "="*60)
        print("="*60)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:


if __name__ == "__main__":
    main()


