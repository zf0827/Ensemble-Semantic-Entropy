"""
Code embedding clustering based on embedding similarity.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# Use relative imports.
from .clustering_base import BaseClusteringMethod
from .clustering_utils import cluster_by_similarity


class CodeEmbedding:
    """
    Code embedding model.

    Uses Salesforce/SFR-Embedding-Code-400M_R.
    """
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        HF_embed: Optional[SentenceTransformer] = None
    ):
        """
        Initialize the code embedding model.

        Args:
            model_name: embedding model name (used if HF_embed is not provided)
            HF_embed: preloaded SentenceTransformer instance (preferred if provided)
        """
        if HF_embed is not None:
            # Use the provided preloaded model.
            self.model = HF_embed
        elif model_name is not None:
            # Load model by name.
            if model_name == "Salesforce/SFR-Embedding-Code-400M_R":
                self.model = SentenceTransformer(model_name, trust_remote_code=True)
            else:
                self.model = SentenceTransformer(model_name, trust_remote_code=True)
        else:
            # Use default model name.
            model_name = "Salesforce/SFR-Embedding-Code-400M_R"
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
    
    def get_similarity_matrix(self, code_list: List[str]) -> np.ndarray:
        """
        Compute the similarity matrix for a list of code snippets.

        Args:
            code_list: list of code strings

        Returns:
            Similarity matrix with shape (n, n)
        """
        # Encode code and normalize embeddings.
        embeddings = self.model.encode(code_list, normalize_embeddings=True)
        
        # Compute similarity matrix (cosine similarity).
        similarity_matrix = self.model.similarity(embeddings, embeddings)
        
        # Convert to numpy array.
        if hasattr(similarity_matrix, 'numpy'):
            similarity_matrix = similarity_matrix.numpy()
        elif hasattr(similarity_matrix, 'cpu'):
            similarity_matrix = similarity_matrix.cpu().numpy()
        else:
            similarity_matrix = np.array(similarity_matrix)
        
        return similarity_matrix


class EmbedClustering(BaseClusteringMethod):
    """
    Cluster code using embeddings.
    """
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        HF_embed: Optional[SentenceTransformer] = None,
        **kwargs
    ):
        """
        Initialize the embedding-based clustering method.

        Args:
            model_name: embedding model name (used if HF_embed not provided)
            HF_embed: preloaded SentenceTransformer instance (preferred if provided)
        """
        super().__init__(**kwargs)
        self.embedding_model = CodeEmbedding(
            model_name=model_name,
            HF_embed=HF_embed
        )
    
    def cluster(
        self,
        codes: List[Dict[str, Any]],
        problem_info: Dict[str, Any],
        cluster_algorithm: str = 'dfs',
        threshold: float = 0.5,
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
        
        # Compute similarity matrix.
        similarity_matrix = self.embedding_model.get_similarity_matrix(code_strings)
        
        # Cluster based on similarity matrix.
        return cluster_by_similarity(
            code_strings, 
            similarity_matrix, 
            threshold, 
            cluster_algorithm
        )

