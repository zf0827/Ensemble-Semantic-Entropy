"""
TestCacheManager - test case cache manager.

Handles JSONL read/write and persists test case cache.
"""

import json
import os
import logging
import fcntl
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class TestCacheManager:
    """
    Test cache manager.

    Reads and writes test caches from JSONL.
    Format: one JSON object per problem per line
    {"task_id": "123", "is_stdin": true, "generated_tests": [...]}
    """
    
    def __init__(self, cache_path: str = "source/ESE/TestInput.jsonl"):
        """
        Initialize the cache manager.

        Args:
            cache_path: cache file path
        """
        self.cache_path = cache_path
        # Ensure cache directory exists.
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def load_tests(self, task_id: str, is_stdin: bool) -> Optional[List[Dict[str, Any]]]:
        """
        Load tests for a task from JSONL.

        Args:
            task_id: task ID
            is_stdin: whether the task is stdin-style

        Returns:
            List of test cases or None if not found
        """
        if not os.path.exists(self.cache_path):
            return None
        
        try:
            cache = self._load_cache()
            key = self._make_key(task_id, is_stdin)
            
            if key in cache:
                tests = cache[key].get("generated_tests", [])
                if tests:
                    logger.info(f"Loaded {len(tests)} tests from cache for task {task_id}")
                    return tests
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to load tests from cache: {e}")
            return None
    
    def save_tests(self, task_id: str, is_stdin: bool, tests: List[Dict[str, Any]]):
        """
        Append tests to JSONL (append to existing list if task exists).

        Args:
            task_id: task ID
            is_stdin: whether the task is stdin-style
            tests: list of tests to save
        """
        if not tests:
            return
        
        try:
            # Use file lock to protect concurrent writes.
            with open(self.cache_path, 'a+') as f:
                # Acquire exclusive lock.
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                try:
                    # Seek to start and read existing content.
                    f.seek(0)
                    cache = {}
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                entry_task_id = self._get_task_id(entry)
                                # Validate task_id is not empty.
                                if not entry_task_id:
                                    logger.warning(f"Skipping cache entry with empty task_id: {entry}")
                                    continue
                                key = self._make_key(
                                    entry_task_id,
                                    entry.get("is_stdin", False)
                                )
                                # Normalize to task_id format.
                                if "question_id" in entry and "task_id" not in entry:
                                    entry["task_id"] = entry.pop("question_id")
                                # If key exists, log warning (should not happen).
                                if key in cache:
                                    logger.warning(f"Duplicate cache key detected: {key}, overwriting with new entry")
                                cache[key] = entry
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse cache line: {e}, line: {line[:100]}")
                                continue
                    
                    # Update or add a new entry.
                    # Validate input task_id is not empty.
                    if not task_id:
                        logger.error(f"Cannot save tests with empty task_id")
                        return
                    key = self._make_key(task_id, is_stdin)
                    if key in cache:
                        # Append tests (deduplicate).
                        existing_tests = cache[key].get("generated_tests", [])
                        existing_tests_set = {json.dumps(t, sort_keys=True) for t in existing_tests}
                        new_tests = [
                            t for t in tests 
                            if json.dumps(t, sort_keys=True) not in existing_tests_set
                        ]
                        cache[key]["generated_tests"].extend(new_tests)
                        logger.info(f"Appended {len(new_tests)} new tests for task {task_id}")
                    else:
                        # New entry.
                        cache[key] = {
                            "task_id": task_id,
                            "is_stdin": is_stdin,
                            "generated_tests": tests
                        }
                        logger.info(f"Created new cache entry with {len(tests)} tests for task {task_id}")
                    
                    # Write back file (overwrite).
                    f.seek(0)
                    f.truncate()
                    for entry in cache.values():
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
                finally:
                    # Release lock.
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        except Exception as e:
            logger.error(f"Failed to save tests to cache: {e}")
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the entire JSONL into memory.

        Returns:
            Cache dict keyed by "task_id:is_stdin"
        """
        cache = {}
        
        if not os.path.exists(self.cache_path):
            return cache
        
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            entry_task_id = self._get_task_id(entry)
                            # Validate task_id is not empty.
                            if not entry_task_id:
                                logger.warning(f"Skipping cache entry with empty task_id: {entry}")
                                continue
                            key = self._make_key(
                                entry_task_id,
                                entry.get("is_stdin", False)
                            )
                            # Normalize to task_id format.
                            if "question_id" in entry and "task_id" not in entry:
                                entry["task_id"] = entry.pop("question_id")
                            # If key exists, log warning (should not happen).
                            if key in cache:
                                logger.warning(f"Duplicate cache key detected in _load_cache: {key}, overwriting with new entry")
                            cache[key] = entry
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse cache line: {e}, line: {line[:100]}")
                            continue
        
        except Exception as e:
            logger.error(f"Failed to load cache file: {e}")
        
        return cache
    
    def _get_task_id(self, entry: Dict[str, Any]) -> str:
        """
        Get task_id from entry with backward compatibility (legacy question_id).

        Args:
            entry: cache entry

        Returns:
            task_id string, or empty string if missing
        """
        # Prefer task_id, fallback to question_id (backward compatibility).
        task_id = entry.get("task_id") or entry.get("question_id", "")
        # Ensure string return type.
        return str(task_id) if task_id else ""
    
    def _make_key(self, task_id: str, is_stdin: bool) -> str:
        """
        Build cache key.

        Args:
            task_id: task ID
            is_stdin: whether the task is stdin-style

        Returns:
            Cache key string
        """
        return f"{task_id}:{is_stdin}"



