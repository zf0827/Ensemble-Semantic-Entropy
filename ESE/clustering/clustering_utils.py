"""
Common utilities for clustering.
"""

import ast
from typing import List, Callable, Optional
import numpy as np


def greedy_clustering(strings_list: List[str], are_equivalent: Callable[[int, int], bool]) -> List[int]:
    """
    Greedy clustering algorithm.

    Args:
        strings_list: list of strings (used for length)
        are_equivalent: equivalence function, takes two indices and returns True/False

    Returns:
        Cluster ID list
    """
    # Initialise all ids with -1.
    N = len(strings_list)
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    for i in range(N):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, N):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(i, j):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids


def dfs_clustering(strings_list: List[str], are_equivalent: Callable[[int, int], bool]) -> List[int]:
    """
    DFS clustering algorithm.

    Args:
        strings_list: list of strings (used for length)
        are_equivalent: equivalence function, takes two indices and returns True/False

    Returns:
        Cluster ID list
    """
    N = len(strings_list)
    visited = [False] * N
    semantic_set_ids = [-1] * N
    current_group = 0
    
    def dfs(node):
        stack = [node]
        while stack:
            v = stack.pop()
            for neighbor in range(N):
                if are_equivalent(v, neighbor) and not visited[neighbor]:
                    visited[neighbor] = True
                    semantic_set_ids[neighbor] = current_group
                    stack.append(neighbor)
    
    for i in range(N):
        if not visited[i]:
            visited[i] = True
            semantic_set_ids[i] = current_group
            dfs(i)
            current_group += 1
            
    return semantic_set_ids


def cluster_by_equivalence(
    codes: List[str],
    are_equivalent_fn: Callable[[int, int], bool],
    method: str = 'dfs'
) -> List[int]:
    """
    Cluster based on an equivalence relation.

    Args:
        codes: list of code strings
        are_equivalent_fn: equivalence function, takes two indices and returns True/False
        method: clustering algorithm, 'dfs' or 'greedy'

    Returns:
        Cluster ID list

    Raises:
        ValueError: if method is not 'dfs' or 'greedy'
    """
    if method == 'dfs':
        return dfs_clustering(codes, are_equivalent_fn)
    elif method == 'greedy':
        return greedy_clustering(codes, are_equivalent_fn)
    else:
        raise ValueError(f"Unknown clustering method: {method}. Must be 'dfs' or 'greedy'")


def cluster_by_similarity(
    codes: List[str],
    similarity_matrix: np.ndarray,
    threshold: float,
    method: str = 'dfs'
) -> List[int]:
    """
    Cluster based on a similarity matrix.

    Args:
        codes: list of code strings
        similarity_matrix: similarity matrix with shape (n, n)
        threshold: similarity threshold; values above are considered equivalent
        method: clustering algorithm, 'dfs' or 'greedy'

    Returns:
        Cluster ID list

    Raises:
        ValueError: if similarity matrix shape is invalid or method is invalid
    """
    n = len(codes)
    if similarity_matrix.shape != (n, n):
        raise ValueError(f"Similarity matrix shape {similarity_matrix.shape} doesn't match codes length {n}")
    
    def are_equivalent(i: int, j: int) -> bool:
        return similarity_matrix[i, j] > threshold
    
    return cluster_by_equivalence(codes, are_equivalent, method)


def detect_code_type(code: str) -> str:
    """
    Detect code type.

    Args:
        code: code string

    Returns:
        Code type: "functional" or "stdin"
    """
    # Check for stdin-related indicators.
    stdin_indicators = [
        'sys.stdin',
        'input(',
        '__name__ == "__main__"',
        'if __name__'
    ]
    
    for indicator in stdin_indicators:
        if indicator in code:
            return "stdin"
    
    # Default to functional type.
    return "functional"


def extract_function_signature(code: str) -> Optional[str]:
    """
    Extract function signature from code.

    Args:
        code: code string

    Returns:
        Function signature string, or None if not found
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract function name and arguments.
                func_name = node.name
                args = []
                for arg in node.args.args:
                    args.append(arg.arg)
                return f"{func_name}({', '.join(args)})"
        return None
    except SyntaxError:
        return None


def extract_function_name(code: str) -> Optional[str]:
    """
    Extract the first function name from code.

    Args:
        code: code string

    Returns:
        Function name, or None if not found
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
        return None
    except SyntaxError:
        return None


def build_similarity_matrix(
    codes: List[str],
    similarity_fn: Callable[[str, str], float]
) -> np.ndarray:
    """
    Build a similarity matrix.

    Args:
        codes: list of code strings
        similarity_fn: similarity function returning a score for two code strings

    Returns:
        Similarity matrix with shape (n, n)
    """
    n = len(codes)
    similarity_matrix = np.eye(n)  # Diagonal is 1.
    
    for i in range(n):
        for j in range(i + 1, n):
            similarity = similarity_fn(codes[i], codes[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    return similarity_matrix

