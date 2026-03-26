"""
Interface for running self-debug evaluation on LiveCodeBench dataset.

This module provides a command-line interface to:
1. Load model configuration from config.json
2. Run multiple iterations of self-debug evaluation
3. Save results to organized output directories
"""

import json
import os
import shutil
import argparse
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .LCB_bench import (
    load_lcb_dataset,
    prepare_dspy_examples,
    extract_public_tests,
    extract_private_tests
)
from .LCB_run import evaluate_on_dataset

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config.json file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def save_config_backup(config_path: str, backup_path: str):
    """
    Save a backup copy of the configuration file.
    
    Args:
        config_path: Path to original config.json
        backup_path: Path to save backup
    """
    shutil.copy2(config_path, backup_path)
    logger.info(f"Saved config backup to: {backup_path}")


def update_config_temperature(config_path: str, temperature: float):
    """
    Update config.json with new temperature.
    
    Args:
        config_path: Path to config.json
        temperature: New temperature value
    """
    config = load_config(config_path)
    
    if "generate_lm" not in config:
        raise ValueError("Configuration file is missing 'generate_lm' field")
    
    config["generate_lm"]["temperature"] = temperature
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Updated config.json: temperature={temperature}")


def restore_config(config_path: str, backup_path: str):
    """
    Restore config.json from backup.
    
    Args:
        config_path: Path to config.json
        backup_path: Path to backup file
    """
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, config_path)
        logger.info(f"Restored config.json from backup")
    else:
        logger.warning(f"Backup file not found: {backup_path}")


def build_lm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build LM configuration dictionary from config.json.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LM configuration dictionary for evaluate_on_dataset
        (contains model_name and all other LM parameters)
    """
    if "generate_lm" not in config:
        raise ValueError("Configuration file is missing 'generate_lm' field")
    
    # Copy the entire generate_lm section
    # This includes model_name and all other parameters (api_base, api_key, etc.)
    lm_config = config["generate_lm"].copy()
    
    return lm_config


def full_evaluation(
    model_name: str,
    num_workers: int = 1,
    base_temperature: float = 0.7,
    config_path: str = None,
    dataset_difficulty: Optional[str] = None,
    num_rounds: int = 3,
    timeout: float = 6.0,
    use_cot_initial: bool = True,
    verbose: bool = True,
    num_iters: int = 20
):
    """
    Run full evaluation with multiple iterations, each with incrementing temperature.
    
    This function sequentially executes num_iters iterations (iter 0 to num_iters-1),
    where each iteration uses temperature = base_temperature + 0.01 * i.
    
    Args:
        model_name: Model name (for output file naming)
        num_workers: Number of parallel workers
        base_temperature: Base temperature value (default: 0.7)
        config_path: Path to config.json (default: config.json in current directory)
        dataset_difficulty: Dataset difficulty filter ("easy", "medium", "hard", None for all)
        num_rounds: Number of debug rounds
        timeout: Timeout per test
        use_cot_initial: Whether to use CoT for initial generation
        verbose: Whether to print progress
        num_iters: Number of iterations to run (default: 20)
    """
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info("Starting Full Evaluation")
        logger.info(f"Model: {model_name}")
        logger.info(f"Total iterations: {num_iters}")
        logger.info(f"Base temperature: {base_temperature}")
        logger.info(f"Temperature range: {base_temperature} to {base_temperature + 0.01 * (num_iters - 1)}")
        logger.info(f"{'='*60}\n")
    
    results = []
    
    for i in range(num_iters):
        iter_num = i
        temperature = base_temperature + 0.01 * i
        
        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Progress: Iteration {i+1}/{num_iters}")
            logger.info(f"Iteration number: {iter_num}")
            logger.info(f"Temperature: {temperature:.3f}")
            logger.info(f"{'='*60}\n")
        
        try:
            result = run_evaluation(
                model_name=model_name,
                iter_num=iter_num,
                num_workers=num_workers,
                temperature=temperature,
                config_path=config_path,
                dataset_difficulty=dataset_difficulty,
                num_rounds=num_rounds,
                timeout=timeout,
                use_cot_initial=use_cot_initial,
                verbose=verbose
            )
            results.append(result)
            
            if verbose:
                logger.info(f"Iteration {iter_num} completed successfully")
        
        except Exception as e:
            logger.error(f"Error in iteration {iter_num}: {str(e)}")
            if verbose:
                logger.error(f"Continuing with next iteration...")
            # Continue to next iteration even if this one failed
            continue
    
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info("Full Evaluation Completed!")
        logger.info(f"Successfully completed {len(results)}/{num_iters} iterations")
        logger.info(f"{'='*60}\n")
    
    return results


def run_evaluation(
    model_name: str,
    iter_num: int,
    num_workers: int = 1,
    temperature: float = 0.7,
    config_path: str = None,
    dataset_difficulty: Optional[str] = None,
    num_rounds: int = 3,
    timeout: float = 6.0,
    use_cot_initial: bool = True,
    verbose: bool = True
):
    """
    Run self-debug evaluation on LiveCodeBench dataset.
    
    Args:
        model_name: Model name (for output file naming)
        iter_num: Iteration number (for output file naming)
        num_workers: Number of parallel workers
        temperature: Temperature for model generation
        config_path: Path to config.json (default: config.json in current directory)
        dataset_difficulty: Dataset difficulty filter ("easy", "medium", "hard", None for all)
        num_rounds: Number of debug rounds
        timeout: Timeout per test
        use_cot_initial: Whether to use CoT for initial generation
        verbose: Whether to print progress
    """
    # Set default config path
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.json")
    
    # Create backup of config.json
    backup_path = config_path + ".backup"
    save_config_backup(config_path, backup_path)
    
    try:
        # Update config.json with new temperature (if needed)
        update_config_temperature(config_path, temperature)
        
        # Load updated configuration
        config = load_config(config_path)
        lm_config = build_lm_config(config)
        
        # Load dataset
        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info("Loading LiveCodeBench dataset...")
            if dataset_difficulty:
                logger.info(f"Difficulty filter: {dataset_difficulty}")
            logger.info(f"{'='*60}\n")
        
        dataset = load_lcb_dataset(difficulty=dataset_difficulty)
        examples = prepare_dspy_examples(dataset)
        public_tests_dict = extract_public_tests(dataset)
        private_tests_dict = extract_private_tests(dataset)
        
        if verbose:
            logger.info(f"Loaded {len(examples)} problems")
        
        # Prepare output path (use model_name directly for file system)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Ensure model_name is safe for file paths by replacing problematic characters
        safe_model_name = model_name.replace('/', '_').replace('\\', '_')
        # Use dataset_difficulty in path, default to "all" if None
        difficulty_dir = dataset_difficulty if dataset_difficulty else "all"
        result_dir = os.path.join(current_dir, "result", difficulty_dir, safe_model_name)
        os.makedirs(result_dir, exist_ok=True)
        
        output_path = os.path.join(result_dir, f"round_{iter_num}.json")
        
        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting evaluation")
            logger.info(f"Model: {model_name}")
            logger.info(f"Iteration: {iter_num}")
            logger.info(f"Workers: {num_workers}")
            logger.info(f"Temperature: {temperature}")
            logger.info(f"Output: {output_path}")
            logger.info(f"{'='*60}\n")
        
        # Run evaluation
        results = evaluate_on_dataset(
            dataset=examples,
            public_tests_dict=public_tests_dict,
            private_tests_dict=private_tests_dict,
            lm_config=lm_config,
            num_rounds=num_rounds,
            timeout=timeout,
            use_cot_initial=use_cot_initial,
            num_workers=num_workers,
            output_path=output_path,
            verbose=verbose
        )
        
        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info("Evaluation completed!")
            logger.info(f"Results saved to: {output_path}")
            logger.info(f"{'='*60}\n")
        
        return results
    
    finally:
        # Restore original config.json
        restore_config(config_path, backup_path)


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run self-debug evaluation on LiveCodeBench dataset"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name to use (e.g., 'openai/qcoder')"
    )
    
    parser.add_argument(
        "--iter",
        type=int,
        default=None,
        help="Iteration number (for output file naming, required when --full is not set)"
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full evaluation with 20 iterations (iter 0-19) with incrementing temperature"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for model generation (default: 0.7). When --full is set, this is the base temperature."
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config.json (default: config.json in current directory)"
    )
    
    parser.add_argument(
        "--dataset_difficulty",
        type=str,
        default=None,
        choices=["easy", "medium", "hard"],
        help="Filter dataset by difficulty (default: None, all difficulties)"
    )
    
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=3,
        help="Number of debug rounds (default: 3)"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=6.0,
        help="Timeout per test in seconds (default: 6.0)"
    )
    
    parser.add_argument(
        "--no_cot",
        action="store_true",
        help="Disable chain-of-thought for initial generation"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.full and args.iter is None:
        parser.error("--iter is required when --full is not set")
    
    # Run evaluation
    if args.full:
        # Run full evaluation with 20 iterations
        full_evaluation(
            model_name=args.model_name,
            num_workers=args.num_workers,
            base_temperature=args.temperature,
            config_path=args.config_path,
            dataset_difficulty=args.dataset_difficulty,
            num_rounds=args.num_rounds,
            timeout=args.timeout,
            use_cot_initial=not args.no_cot,
            verbose=not args.quiet,
            num_iters=20
        )
    else:
        # Run single evaluation
        run_evaluation(
            model_name=args.model_name,
            iter_num=args.iter,
            num_workers=args.num_workers,
            temperature=args.temperature,
            config_path=args.config_path,
            dataset_difficulty=args.dataset_difficulty,
            num_rounds=args.num_rounds,
            timeout=args.timeout,
            use_cot_initial=not args.no_cot,
            verbose=not args.quiet
        )


if __name__ == "__main__":
    main()

