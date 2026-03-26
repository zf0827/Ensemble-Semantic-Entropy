"""
Base class definition for all clustering methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union


class BaseClusteringMethod(ABC):
    """
    Abstract base class for code clustering methods.

    All clustering methods should inherit from this class and implement
    the cluster method.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the clustering method.

        Args:
            **kwargs: subclass-specific parameters
        """
        pass
    
    @abstractmethod
    def cluster(
        self, 
        codes: List[Dict[str, Any]], 
        problem_info: Dict[str, Any],
        cluster_algorithm: str = 'dfs',
        **kwargs
    ) -> List[int]:
        """
        Cluster code samples.

        Args:
            codes: list of dicts, each containing:
                - code: str, code string
                - logprob: float, log-probability
                - norm_logprob: float, normalized log-probability
            problem_info: problem metadata dict containing:
                - prompt: str, problem prompt
                - is_stdin: bool, stdin-style or not
                - task_id: str, task ID
            cluster_algorithm: clustering algorithm, 'dfs' or 'greedy'
            **kwargs: subclass-specific parameters

        Returns:
            Cluster ID list aligned with codes; same ID means same cluster.

        Raises:
            ValueError: if input parameters are invalid
            NotImplementedError: if subclass does not implement this method
        """
        raise NotImplementedError("Subclass must implement cluster() method")
    
    def _validate_inputs(
        self, 
        codes: List[Dict[str, Any]], 
        problem_info: Dict[str, Any]
    ):
        """
        Validate input parameters.

        Args:
            codes: code list
            problem_info: problem metadata

        Raises:
            ValueError: if input parameters are invalid
        """
        if not codes:
            raise ValueError("codes列表不能为空")
        
        # Validate codes format.
        for i, code_dict in enumerate(codes):
            if not isinstance(code_dict, dict):
                raise ValueError(f"codes[{i}]必须是字典类型")
            if "code" not in code_dict:
                raise ValueError(f"codes[{i}]缺少'code'字段")
            if not isinstance(code_dict["code"], str):
                raise ValueError(f"codes[{i}]['code']必须是字符串类型")
        
        # Validate problem_info format.
        if not isinstance(problem_info, dict):
            raise ValueError("problem_info必须是字典类型")
        
        required_fields = ["prompt", "is_stdin", "task_id"]
        for field in required_fields:
            if field not in problem_info:
                raise ValueError(f"problem_info缺少'{field}'字段")
        
        if not isinstance(problem_info["is_stdin"], bool):
            raise ValueError("problem_info['is_stdin']必须是布尔类型")

