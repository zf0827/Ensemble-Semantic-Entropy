"""
Semantic entropy calculation module.

Compute four entropy metrics based on clustering results:
1. Predictive Entropy (MC) - Monte Carlo estimate
2. Predictive Entropy (Rao) - normalized version
3. Semantic Entropy (Standard/Soft)
4. Discrete Semantic Entropy (Frequency/Hard)

All calculations are performed in log space for numerical stability.
"""

import numpy as np
from scipy.special import logsumexp
from typing import List, Dict, Any, Optional


def calc_predictive_entropy_mc(logprobs: np.ndarray) -> float:
    """
    Compute Predictive Entropy (MC - Monte Carlo Estimate).

    Meaning: negative mean log-likelihood of generated sequences. This is the
    simplest estimate and does not require normalization.

    Args:
        logprobs: array of length N with log-probabilities of N generations

    Returns:
        Predictive Entropy (MC)

    Formula:
        PE(MC) = -mean(logprobs)
    """
    if len(logprobs) == 0:
        return 0.0
    return float(-np.mean(logprobs))


def calc_predictive_entropy_rao(logprobs: np.ndarray) -> float:
    """
    Compute Predictive Entropy (Rao - Normalized).

    Meaning: treat the N samples as a distribution and compute its Shannon
    entropy. Since log-probs do not necessarily sum to 1, normalization is
    required.

    Args:
        logprobs: array of length N with log-probabilities of N generations

    Returns:
        Predictive Entropy (Rao)

    Formula:
        Z = logsumexp(logprobs)
        norm_logprobs = logprobs - Z
        PE(Rao) = -sum(exp(norm_logprobs) * norm_logprobs)
    """
    if len(logprobs) == 0:
        return 0.0
    
    # 1. Normalize in log space
    Z = logsumexp(logprobs)
    norm_logprobs = logprobs - Z
    
    # 2. Calculate entropy: -sum(p * log_p)
    probs = np.exp(norm_logprobs)
    entropy = -np.sum(probs * norm_logprobs)
    
    return float(entropy)


def calc_semantic_entropy(logprobs: np.ndarray, cluster_ids: np.ndarray) -> float:
    """
    Compute Semantic Entropy (Standard/Soft).

    Meaning: aggregate probabilities within each semantic cluster, then compute
    entropy over clusters. This leverages model confidence.

    Args:
        logprobs: array of length N with log-probabilities of N generations
        cluster_ids: array of length N with semantic cluster labels

    Returns:
        Semantic Entropy

    Formula:
        For each cluster c:
            cluster_logprob_c = logsumexp(logprobs[i] for i in cluster c)
        Z = logsumexp(all cluster_logprobs)
        norm_cluster_logprobs = cluster_logprobs - Z
        SE = -sum(exp(norm_cluster_logprobs) * norm_cluster_logprobs)
    """
    if len(logprobs) == 0:
        return 0.0
    
    if len(logprobs) != len(cluster_ids):
        raise ValueError(
            f"logprobs and cluster_ids must have the same length. "
            f"Got {len(logprobs)} logprobs and {len(cluster_ids)} cluster_ids"
        )
    
    # 1. Grouping and aggregating
    unique_clusters = np.unique(cluster_ids)
    cluster_logprobs = []
    
    for cid in unique_clusters:
        # Find all logprobs that belong to this cluster.
        indices = np.where(cluster_ids == cid)[0]
        current_cluster_logprobs = logprobs[indices]
        # Aggregate within the cluster.
        cluster_val = logsumexp(current_cluster_logprobs)
        cluster_logprobs.append(cluster_val)
    
    cluster_logprobs = np.array(cluster_logprobs)
    
    # 2. Normalize across clusters
    Z = logsumexp(cluster_logprobs)
    norm_cluster_logprobs = cluster_logprobs - Z
    
    # 3. Calculate entropy
    cluster_probs = np.exp(norm_cluster_logprobs)
    entropy = -np.sum(cluster_probs * norm_cluster_logprobs)
    
    return float(entropy)


def calc_discrete_semantic_entropy(cluster_ids: np.ndarray) -> float:
    """
    Compute Discrete Semantic Entropy (Frequency/Hard).

    Meaning: ignore logprobs and use only vote counts, assuming equal weight
    for each sample.

    Args:
        cluster_ids: array of length N with semantic cluster labels

    Returns:
        Discrete Semantic Entropy

    Formula:
        counts = count of each cluster_id
        probs = counts / N
        DSE = -sum(probs * log(probs))
    """
    if len(cluster_ids) == 0:
        return 0.0
    
    # 1. Counts
    _, counts = np.unique(cluster_ids, return_counts=True)
    N = len(cluster_ids)
    
    # 2. Probabilities
    probs = counts / N
    
    # 3. Entropy
    # Add epsilon to avoid log(0) if needed.
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    return float(entropy)


def semantic_entropy(
    codes: List[Dict[str, Any]],
    cluster_ids: List[int],
    use_norm_logprob: bool = False
) -> Dict[str, Optional[float]]:
    """
    Compute four entropy metrics: Predictive Entropy (MC), Predictive Entropy
    (Rao), Semantic Entropy, and Discrete Semantic Entropy.

    Args:
        codes: list of dicts, each containing:
            - code: str, code string
            - logprob: float, log-probability (optional; if missing, PE/SE are None)
            - norm_logprob: float, normalized log-probability (optional)
        cluster_ids: list of cluster IDs aligned with codes; same ID => same cluster
        use_norm_logprob: whether to use normalized log-probabilities

    Returns:
        dict with four entropy values:
            - PE_MC: float or None, Predictive Entropy (MC)
            - PE_Rao: float or None, Predictive Entropy (Rao)
            - SE: float or None, Semantic Entropy (Standard/Soft)
            - DSE: float, Discrete Semantic Entropy (Frequency/Hard)

    Examples:
        >>> codes = [
        ...     {"code": "def add(a, b): return a + b", "logprob": -1.0, "norm_logprob": -0.1},
        ...     {"code": "def add(x, y): return x + y", "logprob": -1.2, "norm_logprob": -0.12},
        ... ]
        >>> cluster_ids = [0, 0]  # Two codes are in the same cluster.
        >>> result = semantic_entropy(codes, cluster_ids)
        >>> print(result["PE_MC"], result["PE_Rao"], result["SE"], result["DSE"])
    """
    if len(codes) != len(cluster_ids):
        raise ValueError(
            f"codes and cluster_ids must have the same length. "
            f"Got {len(codes)} codes and {len(cluster_ids)} cluster_ids"
        )
    
    if len(codes) == 0:
        return {
            "PE_MC": 0.0,
            "PE_Rao": 0.0,
            "SE": 0.0,
            "DSE": 0.0
        }
    
    # Collect log-probabilities per code (ignore None).
    prob_key = "norm_logprob" if use_norm_logprob else "logprob"
    valid_codes_with_logprob = []
    valid_cluster_ids_for_logprob = []
    
    for code_dict, cid in zip(codes, cluster_ids):
        if code_dict.get(prob_key) is not None:
            valid_codes_with_logprob.append(code_dict)
            valid_cluster_ids_for_logprob.append(cid)
    
    cluster_ids_array = np.array(cluster_ids)
    
    # Compute DSE using all samples.
    dse = calc_discrete_semantic_entropy(cluster_ids_array)
    
    # If there are samples with logprobs, compute PE and SE.
    if len(valid_codes_with_logprob) > 0:
        logprobs = np.array([c[prob_key] for c in valid_codes_with_logprob])
        valid_cluster_ids_array = np.array(valid_cluster_ids_for_logprob)
        
        pe_mc = calc_predictive_entropy_mc(logprobs)
        pe_rao = calc_predictive_entropy_rao(logprobs)
        se = calc_semantic_entropy(logprobs, valid_cluster_ids_array)
    else:
        # No logprobs available, set PE and SE to None.
        pe_mc = None
        pe_rao = None
        se = None
    
    return {
        "PE_MC": pe_mc,
        "PE_Rao": pe_rao,
        "SE": se,
        "DSE": dse
    }

