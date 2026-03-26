"""
Semantic entropy evaluation script.

Based on Plan.md Step 3:
1. Load sampling results for the specified model and difficulty (rounds 0, 5, 10, 15, 19).
2. Load question metadata from lcb_release_v2_all_questions.json.
3. Cluster code samples and compute four entropy metrics.
4. Save results to source/ESE/result/<difficulty>/<model_name>/<method>/result.jsonl.

Clustering is done via the unified cluster_codes interface, which creates a
clusterer instance on each call.
"""

import json
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from source.ESE.clustering import cluster_codes
from source.ESE.clustering.utils import HuggingfaceModel
from source.ESE.entropy import semantic_entropy
import dspy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _create_lm_from_config():
    """
    Create a language model instance from config.
    
    Returns:
        dspy.LM instance
    """
    # Config mirrors config.json, hardcoded here
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
    logger.info(f"API base: {kwargs.get('api_base', 'N/A')}")
    
    return lm




def load_questions_info(json_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load question metadata.
    
    Args:
        json_path: path to lcb_release_v2_all_questions.json
    
    Returns:
        Dict mapping task_id to question info.
    """
    logger.info(f"Loading question metadata: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    # Convert to dict for quick lookup
    questions_dict = {q["task_id"]: q for q in questions}
    
    logger.info(f"Loaded {len(questions_dict)} questions")
    return questions_dict


def load_model_samples(
    model_name: str,
    difficulty: str,
    round_ids: List[int],
    result_dir: str = "source/live_code_bench/result"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load samples for a given model, difficulty, and rounds.
    
    Args:
        model_name: model name (maps to result_dir/<difficulty>/<model_name>/)
        difficulty: difficulty level ("easy", "medium", "hard")
        round_ids: list of rounds to load (e.g., [0, 5, 10, 15, 19])
        result_dir: results directory
    
    Returns:
        Dict mapping question_id to its samples.
    """
    logger.info(f"Loading samples for model '{model_name}', difficulty '{difficulty}'...")
    
    samples_by_question = {}
    
    for round_id in round_ids:
        round_file = os.path.join(
            result_dir, difficulty, model_name, f"round_{round_id}.jsonl"
        )
        
        if not os.path.exists(round_file):
            logger.warning(f"File not found, skip: {round_file}")
            continue
        
        logger.info(f"  Reading round_{round_id}.jsonl")
        
        with open(round_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                question_id = entry["question_id"]
                
                # Extract last code, probabilities, and score
                if "debug_trace" in entry and len(entry["debug_trace"]) > 0:
                    last_code = entry["debug_trace"][-1]
                    
                    # If last code has no logprob, search backwards
                    logprob = last_code.get("logprob")
                    norm_logprob = last_code.get("norm_logprob")
                    
                    # If logprob is None, find the first available logprob backwards
                    if logprob is None:
                        for code_item in reversed(entry["debug_trace"]):
                            if code_item.get("logprob") is not None:
                                logprob = code_item.get("logprob")
                                break
                    
                    # If norm_logprob is None, find the first available norm_logprob backwards
                    if norm_logprob is None:
                        for code_item in reversed(entry["debug_trace"]):
                            if code_item.get("norm_logprob") is not None:
                                norm_logprob = code_item.get("norm_logprob")
                                break
                    
                    # Build sample record
                    sample = {
                        "code": last_code["code"],
                        "logprob": logprob,
                        "norm_logprob": norm_logprob,
                        "score": entry.get("score", 0.0),
                        "round_id": round_id
                    }
                    
                    if question_id not in samples_by_question:
                        samples_by_question[question_id] = []
                    
                    samples_by_question[question_id].append(sample)
    
    logger.info(f"Loaded samples for {len(samples_by_question)} questions")
    return samples_by_question


def load_ensemble_samples(
    ensemble_list: List[str],
    difficulty: str,
    round_ids: List[int],
    result_dir: str = "source/live_code_bench/result"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load ensemble samples from multiple models.
    
    Args:
        ensemble_list: model list; each round_id maps to a model (e.g., ["qw", "qw", "qw14", "qw14", "qw14"])
        difficulty: difficulty level ("easy", "medium", "hard")
        round_ids: list of rounds to load (e.g., [0, 5, 10, 15, 19])
        result_dir: results directory
    
    Returns:
        Dict mapping question_id to its samples (from different models).
    """
    logger.info("Loading ensemble samples...")
    logger.info(f"  Ensemble list: {ensemble_list}")
    logger.info(f"  Rounds: {round_ids}")
    
    if len(ensemble_list) != len(round_ids):
        raise ValueError(f"ensemble_list length ({len(ensemble_list)}) must equal round_ids length ({len(round_ids)})")
    
    samples_by_question = {}
    
    for round_id, model_name in zip(round_ids, ensemble_list):
        round_file = os.path.join(
            result_dir, difficulty, model_name, f"round_{round_id}.jsonl"
        )
        
        if not os.path.exists(round_file):
            logger.warning(f"File not found, skip: {round_file}")
            continue
        
        logger.info(f"  Reading round_{round_id}.jsonl from model '{model_name}'")
        
        with open(round_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                question_id = entry["question_id"]
                
                # Extract last code, probabilities, and score
                if "debug_trace" in entry and len(entry["debug_trace"]) > 0:
                    last_code = entry["debug_trace"][-1]
                    
                    # If last code has no logprob, search backwards
                    logprob = last_code.get("logprob")
                    norm_logprob = last_code.get("norm_logprob")
                    
                    # If logprob is None, find the first available logprob backwards
                    if logprob is None:
                        for code_item in reversed(entry["debug_trace"]):
                            if code_item.get("logprob") is not None:
                                logprob = code_item.get("logprob")
                                break
                    
                    # If norm_logprob is None, find the first available norm_logprob backwards
                    if norm_logprob is None:
                        for code_item in reversed(entry["debug_trace"]):
                            if code_item.get("norm_logprob") is not None:
                                norm_logprob = code_item.get("norm_logprob")
                                break
                    
                    # Build sample record with model info
                    sample = {
                        "code": last_code["code"],
                        "logprob": logprob,
                        "norm_logprob": norm_logprob,
                        "score": entry.get("score", 0.0),
                        "round_id": round_id,
                        "model": model_name  # Source model
                    }
                    
                    if question_id not in samples_by_question:
                        samples_by_question[question_id] = []
                    
                    samples_by_question[question_id].append(sample)
    
    logger.info(f"Loaded ensemble samples for {len(samples_by_question)} questions")
    return samples_by_question


def process_single_question(
    question_id: str,
    codes: List[Dict[str, Any]],
    problem_info: Dict[str, Any],
    clustering_method: str,
    cluster_algorithm: str = 'dfs',
    use_norm_logprob: bool = False,
    **cluster_kwargs
) -> Optional[Dict[str, Any]]:
    """
    Process a single question: cluster codes and compute entropy.
    
    Args:
        question_id: question ID
        codes: list of code samples
        problem_info: question metadata
        clustering_method: clustering method name
        cluster_algorithm: clustering algorithm
        use_norm_logprob: whether to use normalized log-probabilities
        **cluster_kwargs: extra clustering parameters
    
    Returns:
        Dict with clustering/entropy info, or None on failure.
    """
    try:
        # Check how many samples have logprob
        prob_key = "norm_logprob" if use_norm_logprob else "logprob"
        has_logprob_count = sum(1 for code in codes if code.get(prob_key) is not None)
        
        if len(codes) == 0:
            logger.warning(f"Question {question_id} has no samples, skip")
            return None
        
        # Log if some samples lack logprob
        if has_logprob_count < len(codes):
            logger.info(
                f"Question {question_id}: {len(codes) - has_logprob_count} samples lack logprob; "
                f"only DSE will be computed"
            )
        
        # functional/functional_vanilla require an LM
        if clustering_method in ["functional", "functional_vanilla"]:
            if "lm" not in cluster_kwargs:
                lm = _create_lm_from_config()
                cluster_kwargs["lm"] = lm
        
        # For nlg_llm, reuse a preloaded model if provided.
        # NOTE: The model should be loaded and passed in evaluate_dataset.
        
        # Run clustering (uses all samples; only code is needed)
        cluster_ids = cluster_codes(
            codes=codes,
            problem_info=problem_info,
            method=clustering_method,
            cluster_algorithm=cluster_algorithm,
            **cluster_kwargs
        )
        
        # Compute entropy (semantic_entropy handles missing logprob)
        entropy_result = semantic_entropy(
            codes=codes,
            cluster_ids=cluster_ids,
            use_norm_logprob=use_norm_logprob
        )
        
        # Compute average score (use all samples)
        scores = [c.get("score", 0.0) for c in codes]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Build result
        result = {
            "problem_id": question_id,
            "num_samples": len(codes),
            "cluster_ids": cluster_ids,
            "num_clusters": len(set(cluster_ids)),
            "PE_MC": entropy_result["PE_MC"],
            "PE_Rao": entropy_result["PE_Rao"],
            "SE": entropy_result["SE"],
            "DSE": entropy_result["DSE"],
            "avg_score": avg_score,
            "scores": scores
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing question {question_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_dataset(
    model: str,
    difficulty: str,
    clustering_method: str = "embed",
    round_ids: Optional[List[int]] = None,
    cluster_algorithm: str = "dfs",
    use_norm_logprob: bool = False,
    output_dir: str = "source/ESE/result",
    questions_json: str = "lcb_release_v2_all_questions.json",
    result_dir: str = "source/live_code_bench/result",
    num_workers: int = 1,
    ensemble_name: Optional[str] = None,
    ensemble_list: Optional[List[str]] = None,
    **cluster_kwargs
):
    """
    Evaluate the full dataset.
    
    Args:
        model: model name (maps to result_dir/<difficulty>/<model>/), unused in ensemble mode
        difficulty: difficulty level ("easy", "medium", "hard")
        clustering_method: clustering method name
        round_ids: rounds to load, defaults to [0, 5, 10, 15, 19]
        cluster_algorithm: clustering algorithm
        use_norm_logprob: whether to use normalized log-probabilities
        output_dir: output directory
        questions_json: question metadata JSON path
        result_dir: model results directory
        num_workers: worker count (reserved; parallelism not implemented)
        ensemble_name: ensemble experiment name; enables ensemble mode when set
        ensemble_list: ensemble model list; required if ensemble_name is set
        **cluster_kwargs: extra clustering parameters
    """
    if round_ids is None:
        round_ids = [0, 5, 10, 15, 19]
    
    # Check ensemble mode
    is_ensemble = ensemble_name is not None
    
    logger.info("=" * 80)
    logger.info("Starting evaluation")
    logger.info("=" * 80)
    if is_ensemble:
        logger.info("Mode: Ensemble")
        logger.info(f"Experiment name: {ensemble_name}")
        logger.info(f"Ensemble list: {ensemble_list}")
    else:
        logger.info("Mode: Single model")
        logger.info(f"Model: {model}")
    logger.info(f"Difficulty: {difficulty}")
    logger.info(f"Clustering method: {clustering_method}")
    logger.info(f"Rounds: {round_ids}")
    logger.info(f"Clustering algorithm: {cluster_algorithm}")
    logger.info("=" * 80)
    
    # 1. Load question metadata
    questions_dict = load_questions_info(questions_json)
    
    # 2. Load samples
    if is_ensemble:
        if ensemble_list is None:
            raise ValueError("ensemble_list must be provided in ensemble mode")
        if len(ensemble_list) != len(round_ids):
            raise ValueError(f"ensemble_list length ({len(ensemble_list)}) must equal round_ids length ({len(round_ids)})")
        samples_by_question = load_ensemble_samples(
            ensemble_list=ensemble_list,
            difficulty=difficulty,
            round_ids=round_ids,
            result_dir=result_dir
        )
    else:
        samples_by_question = load_model_samples(
            model_name=model,
            difficulty=difficulty,
            round_ids=round_ids,
            result_dir=result_dir
        )
    
    # Filter: keep only questions present in questions_dict
    valid_questions = {}
    for qid, samples in samples_by_question.items():
        if qid in questions_dict:
            valid_questions[qid] = samples
        else:
            logger.warning(f"Question {qid} not in metadata, skip")
    
    logger.info(f"Valid questions: {len(valid_questions)}")
    
    if len(valid_questions) == 0:
        logger.error("No valid questions, exiting")
        return
    
    # 2.5 Preload embed model (avoid per-call loading)
    if clustering_method == "embed" and "HF_embed" not in cluster_kwargs:
        # Get model name
        model_name = cluster_kwargs.get("model_name", "Salesforce/SFR-Embedding-Code-400M_R")
        
        logger.info(f"Loading embed model: {model_name}")
        try:
            if model_name == "Salesforce/SFR-Embedding-Code-400M_R":
                embed_model = SentenceTransformer(model_name, trust_remote_code=True)
            else:
                embed_model = SentenceTransformer(model_name, trust_remote_code=True)
            cluster_kwargs["HF_embed"] = embed_model
            logger.info("Embed model loaded")
        except Exception as e:
            logger.error(f"Failed to load embed model: {e}")
            logger.info("Will use model_name and load per call")
            # On failure, keep model_name for per-call loading
    
    # 2.6 Preload DeBERTa for nlg_deberta (avoid per-call loading)
    if clustering_method == "nlg_deberta" and "HF_deberta" not in cluster_kwargs:
        DEVICE = "cuda:7"
        
        try:
            deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli", device_map=DEVICE)
            deberta_model = AutoModelForSequenceClassification.from_pretrained(
                "microsoft/deberta-v2-xlarge-mnli", device_map=DEVICE)
            cluster_kwargs["HF_deberta"] = {
                "model": deberta_model,
                "tokenizer": deberta_tokenizer
            }
            logger.info("DeBERTa model loaded")
        except Exception as e:
            logger.error(f"Failed to load DeBERTa model: {e}")
            logger.info("Will use default loading per call")
            # On failure, allow per-call loading
    
    # 2.7 Preload LLM for nlg_llm (avoid per-call loading)
    if clustering_method == "nlg_llm" and "HF_llm" not in cluster_kwargs:
        # Get model name
        model_name = cluster_kwargs.get("model_name")
        if model_name is None:
            # Use default model name
            model_name = "Qwen/Qwen3-8B"
            cluster_kwargs["model_name"] = model_name
        
        logger.info(f"Loading LLM model: {model_name}")
        try:
            llm_model = HuggingfaceModel(
                model_name, 
                stop_sequences='default', 
                max_new_tokens=30
            )
            cluster_kwargs["HF_llm"] = llm_model
            logger.info("LLM model loaded")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            logger.info("Will use model_name and load per call")
            # On failure, keep model_name for per-call loading
    
    # 3. Process all questions
    results = []
    
    logger.info(f"Processing {len(valid_questions)} questions...")
    
    for question_id, codes in tqdm(valid_questions.items(), desc="处理题目"):
        # Get question info
        problem_info = questions_dict[question_id]
        
        # Process a single question
        result = process_single_question(
            question_id=question_id,
            codes=codes,
            problem_info=problem_info,
            clustering_method=clustering_method,
            cluster_algorithm=cluster_algorithm,
            use_norm_logprob=use_norm_logprob,
            **cluster_kwargs
        )
        
        if result is not None:
            results.append(result)
    
    logger.info(f"Processed {len(results)}/{len(valid_questions)} questions successfully")
    
    # 5. Save results
    if is_ensemble:
        output_path = os.path.join(output_dir, difficulty, ensemble_name, clustering_method)
    else:
        output_path = os.path.join(output_dir, difficulty, model, clustering_method)
    os.makedirs(output_path, exist_ok=True)
    
    output_file = os.path.join(output_path, "result.jsonl")
    logger.info(f"Saving results to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 6. If ensemble mode, save experiment config
    if is_ensemble:
        config_file = os.path.join(output_path, "config.txt")
        logger.info(f"Saving experiment config to: {config_file}")
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(f"Experiment name: {ensemble_name}\n")
            f.write(f"Difficulty: {difficulty}\n")
            f.write(f"Clustering method: {clustering_method}\n")
            f.write(f"Ensemble list: {ensemble_list}\n")
            f.write(f"Rounds: {round_ids}\n")
            f.write(f"Clustering algorithm: {cluster_algorithm}\n")
            f.write(f"Use normalized logprob: {use_norm_logprob}\n")
    
    logger.info("=" * 80)
    logger.info("Evaluation complete!")
    logger.info("=" * 80)
    
    # Print stats
    logger.info("\nStatistics:")
    logger.info(f"  Total questions: {len(results)}")
    
    if results:
        avg_pe_mc = sum(r["PE_MC"] for r in results) / len(results)
        avg_pe_rao = sum(r["PE_Rao"] for r in results) / len(results)
        avg_se = sum(r["SE"] for r in results) / len(results)
        avg_dse = sum(r["DSE"] for r in results) / len(results)
        avg_score = sum(r["avg_score"] for r in results) / len(results)
        avg_clusters = sum(r["num_clusters"] for r in results) / len(results)
        
        logger.info(f"  Avg PE_MC: {avg_pe_mc:.4f}")
        logger.info(f"  Avg PE_Rao: {avg_pe_rao:.4f}")
        logger.info(f"  Avg SE: {avg_se:.4f}")
        logger.info(f"  Avg DSE: {avg_dse:.4f}")
        logger.info(f"  Avg score: {avg_score:.4f}")
        logger.info(f"  Avg clusters: {avg_clusters:.2f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="计算语义熵评估 - 对指定模型和难度的数据集进行聚类和熵计算"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=False,
        help="Model name; reads from result_dir/<difficulty>/<model_name>/ (not needed in ensemble mode)"
    )
    
    parser.add_argument(
        "--difficulty", "-d",
        type=str,
        required=True,
        choices=["easy", "medium", "hard"],
        help="Difficulty level"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="embed",
        choices=["nlg_deberta", "nlg_llm", "embed", "bleu", "symbolic", 
                 "functional", "functional_vanilla"],
        help="Clustering method"
    )
    
    parser.add_argument(
        "--rounds",
        type=str,
        default="0,5,10,15,19",
        help="Rounds to read, comma-separated (default: 0,5,10,15,19)"
    )
    
    parser.add_argument(
        "--cluster-algorithm",
        type=str,
        default="dfs",
        choices=["dfs", "greedy"],
        help="Clustering algorithm"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Similarity threshold (for embed and bleu)"
    )
    
    parser.add_argument(
        "--use-norm-logprob",
        action="store_true",
        help="Use normalized log-probabilities"
    )
    
    parser.add_argument(
        "--questions-json",
        type=str,
        default="lcb_release_v2_all_questions.json",
        help="Question metadata JSON path"
    )
    
    parser.add_argument(
        "--result-dir",
        type=str,
        default="source/live_code_bench/result",
        help="Model results directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="source/ESE/result",
        help="Output directory"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        help="LLM model name (for nlg_llm)"
    )
    
    parser.add_argument(
        "--strict-entailment",
        action="store_true",
        help="Use strict entailment (for nlg methods)"
    )
    
    parser.add_argument(
        "--ensemble",
        type=str,
        help="Ensemble experiment name; enables ensemble mode with multiple models"
    )
    
    parser.add_argument(
        "--ensemble-list",
        type=str,
        help="Ensemble model list, comma-separated (e.g., qw14,qw14,qcoder,qcoder,qcoder). Defaults if not provided."
    )
    
    args = parser.parse_args()
    
    # Validate args: must choose either ensemble or single model
    if args.ensemble is None and args.model is None:
        parser.error("Must provide either --model or --ensemble")
    if args.ensemble is not None and args.model is not None:
        logger.warning("Both --model and --ensemble provided; using ensemble and ignoring --model")
    
    # Parse rounds
    round_ids = [int(r.strip()) for r in args.rounds.split(",")]
    
    # Handle ensemble mode
    ensemble_name = args.ensemble
    ensemble_list = None
    if ensemble_name is not None:
        # Parse ensemble_list
        if args.ensemble_list is not None:
            ensemble_list = [m.strip() for m in args.ensemble_list.split(",")]
        else:
            # Use defaults if not provided
            ensemble_list = ["qw14", "qw14", "qcoder", "qcoder", "qcoder"]
            logger.info("--ensemble-list not provided; using defaults")
        
        # Ensure ensemble_list matches round_ids length
        if len(ensemble_list) != len(round_ids):
            # Truncate or pad with the last element
            if len(ensemble_list) < len(round_ids):
                ensemble_list = ensemble_list + [ensemble_list[-1]] * (len(round_ids) - len(ensemble_list))
            else:
                ensemble_list = ensemble_list[:len(round_ids)]
        
        logger.info("Ensemble mode enabled")
        logger.info(f"  Experiment name: {ensemble_name}")
        logger.info(f"  Ensemble list: {ensemble_list}")
        logger.info(f"  Rounds: {round_ids}")
    
    # Build clustering kwargs
    cluster_kwargs = {}
    
    if args.method in ["embed", "bleu"]:
        cluster_kwargs["threshold"] = args.threshold
    
    if args.method in ["nlg_deberta", "nlg_llm"]:
        cluster_kwargs["strict_entailment"] = args.strict_entailment
    
    if args.method == "nlg_llm":
        # Use provided model_name or fall back to the default
        if args.model_name is not None:
            cluster_kwargs["model_name"] = args.model_name
        else:
            # Hardcoded default consistent with run_calc.py
            cluster_kwargs["model_name"] = "Qwen/Qwen3-8B"
    
    # Run evaluation
    # model is unused in ensemble mode; pass a placeholder
    model_param = args.model if args.model is not None else "ensemble"
    evaluate_dataset(
        model=model_param,
        difficulty=args.difficulty,
        clustering_method=args.method,
        round_ids=round_ids,
        cluster_algorithm=args.cluster_algorithm,
        use_norm_logprob=args.use_norm_logprob,
        output_dir=args.output_dir,
        questions_json=args.questions_json,
        result_dir=args.result_dir,
        ensemble_name=ensemble_name,
        ensemble_list=ensemble_list,
        **cluster_kwargs
    )


if __name__ == "__main__":
    main()

