from unittest.mock import MagicMock

import openai.types
import pytest

from lmwrapper.caching import clear_cache_dir
from lmwrapper.openai_wrapper import get_open_ai_lm
from lmwrapper.openai_wrapper.batching import OpenAiBatchManager
from lmwrapper.sqlcache import SqlBackedCache
from lmwrapper.structs import LmPrompt

sample_file_resp = {
    "id": "file-mockid",
    "bytes": 272,
    "created_at": 1718670611,
    "filename": "upload",
    "object": "file",
    "purpose": "batch",
    "status": "processed",
    "status_details": None,
}


mock_batch_data = {
    "id": "batch_mockbatch",
    "completion_window": "24h",
    "created_at": 1718670788,
    "endpoint": "/v1/completions",
    "input_file_id": "file-mockid",
    "object": "batch",
    "status": "validating",
    "cancelled_at": None,
    "cancelling_at": None,
    "completed_at": None,
    "error_file_id": None,
    "errors": None,
    "expired_at": None,
    "expires_at": 1718757188,
    "failed_at": None,
    "finalizing_at": None,
    "in_progress_at": None,
    "metadata": None,
    "output_file_id": None,
    "request_counts": {"completed": 0, "failed": 0, "total": 0},
}


def test_batch_starting():
    clear_cache_dir()
    cache = SqlBackedCache(lm=get_open_ai_lm())
    orig_api = cache._lm._api

    mock_api = MagicMock(wraps=orig_api)
    cache._lm._api = mock_api

    calls = 0

    def mock_files_create(**kwargs):
        nonlocal calls
        calls += 1
        assert len(kwargs) == 2
        print("Files create")
        print(kwargs)
        assert kwargs["purpose"] == "batch"
        assert kwargs["file"] is not None
        print(kwargs["file"].getvalue().decode())
        # og = orig_api.files.create(**kwargs)
        # print(og)
        # print(og.dict())
        return openai.types.FileObject.model_validate(sample_file_resp)

    def mock_batches_create(**kwargs):
        nonlocal calls
        calls += 1
        print("Batches create")
        print(kwargs)
        assert kwargs["input_file_id"] == sample_file_resp["id"]
        return openai.types.Batch.model_validate(mock_batch_data)

    mock_api.files.create = mock_files_create
    mock_api.batches.create = mock_batches_create

    batch_manager = OpenAiBatchManager(
        [
            LmPrompt("hello", cache=True),
        ],
        cache=cache,
    )
    batch_manager.start_batch()
    assert calls == 2


def test_batch_starting_connection_error():
    clear_cache_dir()
    cache = SqlBackedCache(lm=get_open_ai_lm())
    orig_api = cache._lm._api

    mock_api = MagicMock(wraps=orig_api)
    cache._lm._api = mock_api

    calls = 0

    def mock_files_create(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise openai.APIConnectionError(request=MagicMock())
        assert len(kwargs) == 2
        print("Files create")
        print(kwargs)
        assert kwargs["purpose"] == "batch"
        assert kwargs["file"] is not None
        print(kwargs["file"].getvalue().decode())
        return openai.types.FileObject.model_validate(sample_file_resp)

    def mock_batches_create(**kwargs):
        nonlocal calls
        calls += 1
        print("Batches create")
        print(kwargs)
        assert kwargs["input_file_id"] == sample_file_resp["id"]
        return openai.types.Batch.model_validate(mock_batch_data)

    mock_api.files.create = mock_files_create
    mock_api.batches.create = mock_batches_create

    batch_manager = OpenAiBatchManager(
        [
            LmPrompt("hello", cache=True),
        ],
        cache=cache,
    )
    batch_manager.start_batch()
    assert calls == 3


def test_batch_dup_prompts():
    """Test that duplicate prompts are deduplicated at API level while maintaining order for results.
    4 prompts with 3 unique values should only send 3 requests to OpenAI API."""
    clear_cache_dir()
    cache = SqlBackedCache(lm=get_open_ai_lm())
    orig_api = cache._lm._api

    mock_api = MagicMock(wraps=orig_api)
    cache._lm._api = mock_api

    calls = 0

    def mock_files_create(**kwargs):
        nonlocal calls
        calls += 1
        assert len(kwargs) == 2
        print("Files create")
        print(kwargs)
        assert kwargs["purpose"] == "batch"
        assert kwargs["file"] is not None
        file_text = kwargs["file"].getvalue().decode()
        file_lines = [line for line in file_text.split("\n") if line.strip()]
        assert len(file_lines) == 3, "should have 3 unique prompts: hello, Goodbye, Yo"
        
        # Verify that the JSONL contains the expected unique prompts
        import json
        custom_ids = set()
        for line in file_lines:
            data = json.loads(line)
            custom_ids.add(data["custom_id"])
        # Should have 3 unique custom_ids corresponding to our 3 unique prompts
        assert len(custom_ids) == 3, f"Expected 3 unique custom_ids, got {len(custom_ids)}"
        return openai.types.FileObject.model_validate(sample_file_resp)

    def mock_batches_create(**kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["input_file_id"] == sample_file_resp["id"]
        return openai.types.Batch.model_validate(mock_batch_data)

    mock_api.files.create = mock_files_create
    mock_api.batches.create = mock_batches_create

    batch_manager = OpenAiBatchManager(
        [
            LmPrompt("hello", cache=True),
            LmPrompt("Goodbye", cache=True),
            LmPrompt("hello", cache=True),
            LmPrompt("Yo", cache=True),
        ],
        cache=cache,
    )
    batch_manager.start_batch()
    assert calls == 2, "unexpeced number of calls"
