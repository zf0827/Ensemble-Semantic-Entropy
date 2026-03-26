"""
LiveCodeBench dataset loading and formatting utilities.

This module provides functions to:
- Load LiveCodeBench dataset from HuggingFace
- Decode and prepare test cases (public and private)
- Convert to DSPy Example format
"""

import json
import base64
import zlib
import pickle
import logging
from typing import List, Dict, Optional
from datetime import datetime

import dspy
from datasets import load_dataset

# set up logger
logger = logging.getLogger(__name__)


def _has_test_type(tests_json: str, test_type: str) -> bool:
    """
    Check if any test in the test list has the specified testtype.

    Args:
        tests_json: JSON string containing list of tests
        test_type: Type to check for (e.g., "stdin", "functional")

    Returns:
        True if any test has the specified type
    """
    test_list = json.loads(tests_json)
    return any(test.get("testtype") == test_type for test in test_list)


def _decode_private_tests(encoded_data: str) -> List[Dict]:
    """
    Decode private test cases from base64-encoded zlib-compressed pickle.

    Args:
        encoded_data: Base64-encoded string

    Returns:
        List of test case dictionaries
    """
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    return json.loads(original_data)


def load_lcb_dataset(
    difficulty: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    version: str = "release_v2"
) -> List[Dict]:
    """
    Load and filter LiveCodeBench dataset.

    Args:
        difficulty: Filter by difficulty ("easy", "medium", "hard"). None for all.
        start_date: Start date for contest (format: "YYYY-MM-DD"). None for no lower bound.
        end_date: End date for contest (format: "YYYY-MM-DD"). None for no upper bound.
        version: Dataset version tag (default: "release_v2")

    Returns:
        List of dataset entries (raw format from HuggingFace)
    """
    # Load dataset

    lcb_path = "livecodebench/code_generation_lite"
    dataset = load_dataset(
        lcb_path,
        version_tag=version,
        split="test",
        trust_remote_code=True
    )

    # Filter by difficulty
    if difficulty is not None:
        dataset = [entry for entry in dataset if entry["difficulty"] == difficulty]

    # Filter by date range
    if start_date is not None:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [
            entry for entry in dataset
            if start_dt <= datetime.fromisoformat(entry["contest_date"])
        ]

    if end_date is not None:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [
            entry for entry in dataset
            if datetime.fromisoformat(entry["contest_date"]) <= end_dt
        ]

    # Decode private test cases
    for entry in dataset:
        try:
            decoded_private = _decode_private_tests(entry["private_test_cases"])
            entry["private_test_cases"] = decoded_private
        except Exception as e:
            logger.warning(f"Failed to decode private tests for {entry.get('question_id')}: {e}")
            entry["private_test_cases"] = []

    return dataset


def prepare_dspy_examples(dataset: List[Dict]) -> List[dspy.Example]:
    """
    Convert raw dataset entries to DSPy Example format.

    Args:
        dataset: List of raw dataset entries from load_lcb_dataset()

    Returns:
        List of DSPy Examples with fields:
        - prompt: Question content
        - test: Private test cases
        - public_test_cases: Public test cases
        - task_id: Question ID
        - is_stdin: Whether problem uses stdin/stdout
        - entry_point: Starter code
    """
    examples = []

    for entry in dataset:
        is_stdin = _has_test_type(entry["public_test_cases"], "stdin")

        example = dspy.Example(
            prompt=entry["question_content"],
            test=entry["private_test_cases"],
            public_test_cases=json.loads(entry["public_test_cases"]),
            task_id=entry["question_id"],
            is_stdin=is_stdin,
            entry_point=entry["starter_code"],
            canonical_solution=""  # LCB lite doesn't have canonical solution
        ).with_inputs(
            "prompt",
            "test",
            "public_test_cases",
            "task_id",
            "is_stdin",
            "entry_point",
            "canonical_solution"
        )

        examples.append(example)

    return examples


def extract_public_tests(dataset: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Extract public test cases indexed by task_id.

    Args:
        dataset: List of raw dataset entries

    Returns:
        Dictionary mapping task_id to list of public test dictionaries
    """
    public_tests = {}

    for entry in dataset:
        task_id = entry["question_id"]
        public_tests[task_id] = json.loads(entry["public_test_cases"])

    return public_tests


def extract_private_tests(dataset: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Extract private test cases indexed by task_id.

    Args:
        dataset: List of raw dataset entries

    Returns:
        Dictionary mapping task_id to list of private test dictionaries
    """
    private_tests = {}

    for entry in dataset:
        task_id = entry["question_id"]
        private_tests[task_id] = entry["private_test_cases"]

    return private_tests
