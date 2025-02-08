"""
Tests to verify that any examples given in user-facing documentation
will execute.
"""

import concurrent.futures
import os
import re
import sys
import traceback
import types
from pathlib import Path

import pytest

from lmwrapper.caching import clear_cache_dir

cur_file = Path(__file__).parent.absolute()


def extract_code_blocks(file):
    with open(file) as f:
        content = f.read()
    # This regex captures an optional skip marker and then the code block.
    # It matches:
    #   <!-- skip test -->      (always skip)
    #   <!-- skip gh-action --!> (skip only on GitHub Actions)
    pattern = re.compile(
        r"(<!--\s*(?P<skip>skip(?:\s+(?P<type>test|gh-action))?)\s*(?:-->|--!>)\s*)?"
        r"```python\r?\n(?P<code>.*?)```",
        re.DOTALL,
    )
    blocks = []
    for m in pattern.finditer(content):
        skip = m.group("skip")
        skip_type = m.group("type")
        code = m.group("code")
        # Always skip blocks marked with "skip test"
        if skip and skip_type == "test":
            continue
        # Mark blocks for GitHub Actions skip if "gh-action" is specified
        skip_gh_action = bool(skip and skip_type == "gh-action")
        blocks.append((code, skip_gh_action))
    return blocks


def run_code(code):
    module_name = "dynamic_module"
    dynamic_module = types.ModuleType(module_name)
    exec_globals = dynamic_module.__dict__

    try:
        exec(code, exec_globals)
        return None
    except Exception:
        return sys.exc_info()


@pytest.mark.parametrize(
    "code, skip_gh_action",
    extract_code_blocks(cur_file / "../README.md"),
)
def test_readme_code(code, skip_gh_action):
    # If the block is marked to skip on GitHub Actions and we're on GitHub Actions, skip the test.
    if skip_gh_action and os.environ.get("GITHUB_ACTIONS") == "true":
        pytest.skip("Skipping this test on GitHub Actions")

    clear_cache_dir()
    print("### CODE BLOCK")
    print(code)
    print("### OUTPUT")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_code, code)
        try:
            result = future.result(timeout=300)  # 5 minutes timeout
        except concurrent.futures.TimeoutError:
            print("Execution timed out")
            pytest.fail("Timeout exceeded for docs test")

    clear_cache_dir()

    if result is not None:
        exc_type, exc_value, exc_traceback = result
        print("Exception occurred:")
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        pytest.fail(f"Exception occurred: {exc_type.__name__}: {exc_value}")
