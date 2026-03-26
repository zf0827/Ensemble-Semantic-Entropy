"""
Symbolic execution clustering for code equivalence.
"""

import ast
import subprocess
import tempfile
import os
from typing import List, Dict, Any, Optional, Tuple

# Use relative imports.
from .clustering_base import BaseClusteringMethod
from .clustering_utils import cluster_by_equivalence


class CrosshairConfig:
    """CrossHair configuration."""
    def __init__(
        self,
        timeout: int = 30,
        max_iterations: Optional[int] = None,
        per_path_timeout: Optional[int] = None
    ):
        """
        Args:
            timeout: total timeout (seconds)
            max_iterations: max iterations
            per_path_timeout: per-path timeout (seconds)
        """
        self.timeout = timeout
        self.max_iterations = max_iterations
        self.per_path_timeout = per_path_timeout


def extract_function_name(code: str) -> Optional[str]:
    """Extract the first function name from code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
    except SyntaxError:
        pass
    return None


def extract_function_definitions(code: str) -> str:
    """Extract all function definitions (excluding import statements)."""
    try:
        tree = ast.parse(code)
        function_defs = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_defs.append(ast.unparse(node))
        
        return '\n\n'.join(function_defs) if function_defs else code
    except Exception:
        return code


def rename_function_in_code(code: str, old_name: str, new_name: str) -> str:
    """Rename a function and its calls using AST."""
    try:
        tree = ast.parse(code)
        
        class FunctionRenamer(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if node.name == old_name:
                    node.name = new_name
                return self.generic_visit(node)
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == old_name:
                    node.func.id = new_name
                return self.generic_visit(node)
        
        renamer = FunctionRenamer()
        new_tree = renamer.visit(tree)
        return ast.unparse(new_tree)
    except Exception:
        return code


def create_check_code(code1: str, code2: str) -> str:
    """
    Build code for CrossHair checking.

    Args:
        code1: first code string
        code2: second code string

    Returns:
        Assembled check code string

    Raises:
        ValueError: if function names cannot be extracted
    """
    # Extract function names.
    func1_name = extract_function_name(code1)
    func2_name = extract_function_name(code2)
    
    if not func1_name or not func2_name:
        raise ValueError("Failed to extract function names; ensure the code defines a function.")
    
    # Rename functions to avoid conflicts.
    renamed_code1 = rename_function_in_code(code1, func1_name, "fun1")
    renamed_code2 = rename_function_in_code(code2, func2_name, "fun2")
    
    # Extract function definitions.
    fun1_def = extract_function_definitions(renamed_code1)
    fun2_def = extract_function_definitions(renamed_code2)
    
    # Build check code.
    check_code = f"""from typing import *

# Function 1
{fun1_def}

# Function 2
{fun2_def}

def check(*args, **kwargs):
    '''Use CrossHair to check whether fun1 and fun2 are equivalent.'''
    result1 = fun1(*args, **kwargs)
    result2 = fun2(*args, **kwargs)
    assert result1 == result2, f"Results differ: {{result1}} != {{result2}}"
    return True
"""
    
    return check_code


def run_crosshair_check(check_code: str, config: CrosshairConfig) -> Tuple[bool, str]:
    """
    Run CrossHair check.

    Args:
        check_code: code string to check
        config: CrossHair configuration

    Returns:
        (is_equivalent, result_message)
    """
    temp_file = None
    try:
        # Create temporary file.
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(check_code)
            temp_file = f.name
        
        # Build crosshair command.
        cmd = ['crosshair', 'check', temp_file]
        
        # Add optional parameters.
        if config.max_iterations is not None:
            cmd.extend(['--max_iterations', str(config.max_iterations)])
        if config.per_path_timeout is not None:
            cmd.extend(['--per_path_timeout', str(config.per_path_timeout)])
        
        # Run crosshair.
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout
        )
        
        # Check result.
        if result.returncode == 0:
            return True, "CrossHair check passed; functions are equivalent."
        else:
            error_output = result.stderr + result.stdout
            return False, f"CrossHair found a counterexample: {error_output[:500]}"
            
    except subprocess.TimeoutExpired:
        return False, f"CrossHair check timed out ({config.timeout}s)."
    except FileNotFoundError:
        return False, "CrossHair is not installed; run: pip install crosshair-tool"
    except Exception as e:
        return False, f"CrossHair check error: {str(e)}"
    finally:
        # Clean up temporary file.
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass


def check_equivalence(
    code1: str,
    code2: str,
    config: Optional[CrosshairConfig] = None
) -> bool:
    """
    Check whether two code snippets are equivalent (main API).

    Args:
        code1: first code string
        code2: second code string
        config: CrossHair configuration; uses defaults if None

    Returns:
        True if equivalent, False otherwise or if the check fails
    """
    if config is None:
        config = CrosshairConfig()
    
    try:
        check_code = create_check_code(code1, code2)
        is_equivalent, _ = run_crosshair_check(check_code, config)
        return is_equivalent
    except Exception:
        return False


class SymbolicClustering(BaseClusteringMethod):
    """
    Cluster code using symbolic execution.

    Note: only supports functional-style code.

    Reference: Source/dev_exp/symb.py
    """
    
    def __init__(
        self,
        timeout: int = 30,
        max_iterations: Optional[int] = None,
        per_path_timeout: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize symbolic execution clustering.

        Args:
            timeout: total timeout (seconds)
            max_iterations: max iterations
            per_path_timeout: per-path timeout (seconds)
        """
        super().__init__(**kwargs)
        self.config = CrosshairConfig(
            timeout=timeout,
            max_iterations=max_iterations,
            per_path_timeout=per_path_timeout
        )
    
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
            codes: code list
            problem_info: problem metadata
            cluster_algorithm: clustering algorithm
            **kwargs: other parameters

        Returns:
            Cluster ID list

        Raises:
            ValueError: if code type is not functional
        """
        self._validate_inputs(codes, problem_info)
        
        # Check code type.
        if problem_info.get("is_stdin", False):
            raise ValueError("Symbolic clustering only supports functional type code")
        
        code_strings = [c["code"] for c in codes]
        
        def are_equivalent(i: int, j: int) -> bool:
            """Check whether two code snippets are equivalent."""
            code1 = code_strings[i]
            code2 = code_strings[j]
            
            try:
                # Use CrossHair to check equivalence.
                is_eq = check_equivalence(code1, code2, self.config)
                return is_eq
            except Exception:
                # If the check fails, treat as not equivalent.
                return False
        
        return cluster_by_equivalence(code_strings, are_equivalent, cluster_algorithm)

