"""
CodeBLEU clustering based on CodeBLEU similarity.
"""

import logging
from typing import List, Dict, Any
import numpy as np

from codebleu import calc_codebleu

# Use relative imports.
from .clustering_base import BaseClusteringMethod
from .clustering_utils import cluster_by_similarity, build_similarity_matrix

# Configure logging to reduce tree-sitter build output noise.
logging.getLogger('tree_sitter').setLevel(logging.ERROR)


def calculate_codebleu_similarity(
    code_reference: str, 
    code_candidate: str, 
    lang: str = "python"
) -> float:
    """
    Compute CodeBLEU similarity between two code snippets.

    Reference: Source/dev_exp/bleu.py (lines 7-28)

    Args:
        code_reference: reference code
        code_candidate: candidate code to compare
        lang: programming language

    Returns:
        CodeBLEU similarity score (0-1)
    """
    try:
        result = calc_codebleu(
            references=[code_reference], 
            predictions=[code_candidate], 
            lang=lang,
            weights=(0.25, 0.25, 0.25, 0.25),
            tokenizer=None
        )
        return result['codebleu']
    except Exception:
        # If computation fails, return 0.
        return 0.0


class BLEUClustering(BaseClusteringMethod):
    """
    Cluster code using CodeBLEU.

    Reference: Source/dev_exp/bleu.py
    """
    
    def __init__(self, lang: str = "python", **kwargs):
        """
        Initialize the CodeBLEU clustering method.

        Args:
            lang: programming language
        """
        super().__init__(**kwargs)
        self.lang = lang
    
    def cluster(
        self,
        codes: List[Dict[str, Any]],
        problem_info: Dict[str, Any],
        cluster_algorithm: str = 'dfs',
        threshold: float = 0.7,
        **kwargs
    ) -> List[int]:
        """
        Cluster code samples.

        Args:
            codes: code list
            problem_info: problem metadata
            cluster_algorithm: clustering algorithm
            threshold: similarity threshold; above means equivalent
            **kwargs: other parameters

        Returns:
            Cluster ID list
        """
        self._validate_inputs(codes, problem_info)
        
        code_strings = [c["code"] for c in codes]
        
        def bidirectional_codebleu(code1: str, code2: str) -> float:
            """
            Compute bidirectional CodeBLEU similarity.

            Args:
                code1: first code
                code2: second code

            Returns:
                Bidirectional average similarity
            """
            bleu_1_2 = calculate_codebleu_similarity(code1, code2, self.lang)
            bleu_2_1 = calculate_codebleu_similarity(code2, code1, self.lang)
            return (bleu_1_2 + bleu_2_1) / 2.0
        
        # Build similarity matrix.
        similarity_matrix = build_similarity_matrix(code_strings, bidirectional_codebleu)
        
        # Cluster based on similarity matrix.
        return cluster_by_similarity(
            code_strings,
            similarity_matrix,
            threshold,
            cluster_algorithm
        )

