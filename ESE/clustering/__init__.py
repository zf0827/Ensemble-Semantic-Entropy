"""
Code clustering module.

Provides multiple code clustering methods:
- NLG clustering (natural language inference)
- Code embedding clustering
- Symbolic execution clustering
- CodeBLEU clustering
- Functional test clustering

Main APIs:
    cluster_codes(): unified clustering interface
    get_available_methods(): list all available clustering methods
    get_method_info(): get detailed info for a clustering method

Example:
    >>> from ESE.clustering import cluster_codes
    >>> codes = [
    ...     {"code": "def add(a, b): return a + b", "logprob": -1.0, "norm_logprob": -0.1},
    ...     {"code": "def add(x, y): return x + y", "logprob": -1.2, "norm_logprob": -0.12},
    ... ]
    >>> problem_info = {
    ...     "prompt": "Write a function to add two numbers",
    ...     "is_stdin": False,
    ...     "task_id": "test_1"
    ... }
    >>> cluster_ids = cluster_codes(codes, problem_info, method="embed", threshold=0.9)
"""

# Import unified interface.
from .clustering_interface import (
    cluster_codes,
    get_available_methods,
    get_method_info,
)

# Import all clustering method classes (for advanced use).
from .nlg_clustering import (
    NLGClusteringDeberta,
    NLGClusteringLLM,
)
from .embed_clustering import EmbedClustering
from .symbolic_clustering import SymbolicClustering
from .bleu_clustering import BLEUClustering
from .functional_clustering import (
    FunctionalClustering,
    FunctionalVanillaClustering,
)

# Import base class and utilities.
from .clustering_base import BaseClusteringMethod
from .clustering_utils import (
    cluster_by_equivalence,
    cluster_by_similarity,
    detect_code_type,
    extract_function_signature,
    extract_function_name,
    build_similarity_matrix,
)

__all__ = [
    # Main interfaces
    "cluster_codes",
    "get_available_methods",
    "get_method_info",
    
    # Clustering method classes
    "NLGClusteringDeberta",
    "NLGClusteringLLM",
    "EmbedClustering",
    "SymbolicClustering",
    "BLEUClustering",
    "FunctionalClustering",
    "FunctionalVanillaClustering",
    
    # Base class and utilities
    "BaseClusteringMethod",
    "cluster_by_equivalence",
    "cluster_by_similarity",
    "detect_code_type",
    "extract_function_signature",
    "extract_function_name",
    "build_similarity_matrix",
]

__version__ = "0.1.0"

