"""
Unified clustering interface for code clustering entry points.
"""

from typing import List, Dict, Any, Optional

# Import all clustering methods (relative imports).
from .nlg_clustering import NLGClusteringDeberta, NLGClusteringLLM
from .embed_clustering import EmbedClustering
from .symbolic_clustering import SymbolicClustering
from .bleu_clustering import BLEUClustering
from .functional_clustering import FunctionalClustering, FunctionalVanillaClustering


# Method registry.
METHOD_REGISTRY = {
    "nlg_deberta": NLGClusteringDeberta,
    "nlg_llm": NLGClusteringLLM,
    "embed": EmbedClustering,
    "symbolic": SymbolicClustering,
    "bleu": BLEUClustering,
    "functional": FunctionalClustering,
    "functional_vanilla": FunctionalVanillaClustering,
}


def cluster_codes(
    codes: List[Dict[str, Any]],
    problem_info: Dict[str, Any],
    method: str,
    cluster_algorithm: str = 'dfs',
    **kwargs
) -> List[int]:
    """
    Unified code clustering interface.

    Args:
        codes: list of dicts, each containing:
            - code: str, code string
            - logprob: float, log-probability
            - norm_logprob: float, normalized log-probability

        problem_info: problem metadata dict containing:
            - prompt: str, problem prompt
            - is_stdin: bool, stdin-style or not
            - task_id: str, task ID

        method: clustering method name, options:
            - "nlg_deberta": NLG with Deberta model
            - "nlg_llm": NLG with LLM
            - "embed": code embedding clustering
            - "symbolic": symbolic execution clustering (functional only)
            - "bleu": CodeBLEU clustering
            - "functional": functional test clustering (code-aware)
            - "functional_vanilla": functional test clustering (random)

        cluster_algorithm: clustering algorithm, "dfs" or "greedy"

        **kwargs: parameters passed to specific methods:
            - threshold: float, similarity threshold (embed, bleu)
            - lm: dspy.LM, language model (nlg_llm, functional, functional_vanilla)
            - HF_deberta: dict with 'model' and 'tokenizer' (nlg_deberta, preferred)
            - HF_embed: SentenceTransformer instance (embed, preferred)
            - HF_llm: HuggingfaceModel instance (nlg_llm, preferred)
            - model_name: str, model name (nlg_llm, embed)
            - temperature: float, generation temperature (functional, functional_vanilla)
            - strict_entailment: bool (nlg_deberta, nlg_llm)
            - timeout: int (symbolic)
            - lang: str, programming language (bleu)
            - entailment_cache_id: str (nlg_llm)
            - entailment_cache_only: bool (nlg_llm)

    Returns:
        Cluster ID list aligned with codes; same ID means same cluster.

    Raises:
        ValueError: if method is unsupported
        ValueError: if input parameters are invalid

    Examples:
        >>> codes = [
        ...     {"code": "def add(a, b):\\n    return a + b", "logprob": -1.0, "norm_logprob": -0.1},
        ...     {"code": "def add(x, y):\\n    return x + y", "logprob": -1.2, "norm_logprob": -0.12},
        ... ]
        >>> problem_info = {
        ...     "prompt": "Write a function to add two numbers",
        ...     "is_stdin": False,
        ...     "task_id": "test_1"
        ... }
        >>> cluster_ids = cluster_codes(codes, problem_info, method="embed", threshold=0.9)
        >>> print(cluster_ids)  # [0, 0] - two codes are clustered together
    """
    # Check method support.
    if method not in METHOD_REGISTRY:
        supported_methods = ", ".join(METHOD_REGISTRY.keys())
        raise ValueError(
            f"Unknown clustering method: {method}. "
            f"Supported methods are: {supported_methods}"
        )
    
    # Get clustering method class.
    clustering_class = METHOD_REGISTRY[method]
    
    # Extract init parameters based on method type.
    init_kwargs = {}
    
    if method == "nlg_deberta":
        # NLGClusteringDeberta can accept a preloaded model.
        if "HF_deberta" in kwargs:
            init_kwargs["HF_deberta"] = kwargs.pop("HF_deberta")
    
    elif method == "nlg_llm":
        # NLGClusteringLLM requires special init parameters.
        # Prefer a preloaded model; otherwise use model_name.
        if "HF_llm" in kwargs:
            init_kwargs["HF_llm"] = kwargs.pop("HF_llm")
        elif "model_name" in kwargs:
            init_kwargs["model_name"] = kwargs.pop("model_name")
        else:
            raise ValueError("nlg_llm method requires either 'HF_llm' or 'model_name' parameter")
        
        if "entailment_cache_id" in kwargs:
            init_kwargs["entailment_cache_id"] = kwargs.pop("entailment_cache_id")
        
        if "entailment_cache_only" in kwargs:
            init_kwargs["entailment_cache_only"] = kwargs.pop("entailment_cache_only")
    
    elif method == "embed":
        # EmbedClustering can take a model name or a preloaded model.
        # Prefer a preloaded model; otherwise use model_name.
        if "HF_embed" in kwargs:
            init_kwargs["HF_embed"] = kwargs.pop("HF_embed")
        elif "model_name" in kwargs:
            init_kwargs["model_name"] = kwargs.pop("model_name")
    
    elif method == "symbolic":
        # SymbolicClustering supports timeout and other parameters.
        if "timeout" in kwargs:
            init_kwargs["timeout"] = kwargs.pop("timeout")
        if "max_iterations" in kwargs:
            init_kwargs["max_iterations"] = kwargs.pop("max_iterations")
        if "per_path_timeout" in kwargs:
            init_kwargs["per_path_timeout"] = kwargs.pop("per_path_timeout")
    
    elif method == "bleu":
        # BLEUClustering supports programming language setting.
        if "lang" in kwargs:
            init_kwargs["lang"] = kwargs.pop("lang")
    
    elif method in ["functional", "functional_vanilla"]:
        # FunctionalClustering supports LM and timeout parameters.
        if "lm" in kwargs:
            init_kwargs["lm"] = kwargs.pop("lm")
        if "test_timeout" in kwargs:
            init_kwargs["test_timeout"] = kwargs.pop("test_timeout")
        
        if method == "functional":
            if "num_tests" in kwargs:
                init_kwargs["num_tests"] = kwargs.pop("num_tests")
            if "use_cache" in kwargs:
                init_kwargs["use_cache"] = kwargs.pop("use_cache")
            if "cache_path" in kwargs:
                init_kwargs["cache_path"] = kwargs.pop("cache_path")
        elif method == "functional_vanilla":
            if "generator_timeout" in kwargs:
                init_kwargs["generator_timeout"] = kwargs.pop("generator_timeout")
    
    # Instantiate clustering method.
    try:
        clustering_method = clustering_class(**init_kwargs)
    except Exception as e:
        raise ValueError(f"Failed to initialize clustering method '{method}': {str(e)}")
    
    # Execute clustering.
    try:
        cluster_ids = clustering_method.cluster(
            codes=codes,
            problem_info=problem_info,
            cluster_algorithm=cluster_algorithm,
            **kwargs
        )
        return cluster_ids
    except Exception as e:
        raise RuntimeError(f"Clustering with method '{method}' failed: {str(e)}")


def get_available_methods() -> List[str]:
    """
    Get all available clustering methods.

    Returns:
        List of method names
    """
    return list(METHOD_REGISTRY.keys())


def get_method_info(method: str) -> Dict[str, Any]:
    """
    Get clustering method information.

    Args:
        method: method name

    Returns:
        Dict containing:
            - name: method name
            - class: method class
            - description: method description

    Raises:
        ValueError: if the method does not exist
    """
    if method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method}")
    
    method_class = METHOD_REGISTRY[method]
    
    descriptions = {
        "nlg_deberta": "NLI clustering with a Deberta model",
        "nlg_llm": "NLI clustering with an LLM",
        "embed": "Similarity clustering using code embeddings",
        "symbolic": "Equivalence clustering via symbolic execution (functional only)",
        "bleu": "Similarity clustering using CodeBLEU",
        "functional": "Clustering via functional tests (code-aware test generation)",
        "functional_vanilla": "Clustering via functional tests (random test generation)",
    }
    
    return {
        "name": method,
        "class": method_class,
        "description": descriptions.get(method, ""),
    }

