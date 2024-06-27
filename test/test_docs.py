"""
Tests to verify that any examples given in user-facing documentation
will execute.
"""

import concurrent.futures
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
    pattern = re.compile(r"(<!-- skip ?test -->\s*)?```python\r?\n(.*?)```", re.DOTALL)
    blocks = pattern.findall(content)
    return [code for skip, code in blocks if not skip]


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
    "code",
    extract_code_blocks(cur_file / "../README.md"),
)
def test_readme_code(code):
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
