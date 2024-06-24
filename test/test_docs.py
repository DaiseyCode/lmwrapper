"""
Tests to verify that any examples given in user-facing documentation
will execute.
"""

import re
import sys
import threading
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


@pytest.mark.parametrize(
    "code",
    extract_code_blocks(cur_file / "../README.md"),
)
def test_readme_code(code):
    clear_cache_dir()
    print("### CODE BLOCK")
    print(code)
    print("### OUTPUT")

    # Create a new module
    module_name = "dynamic_module"
    dynamic_module = types.ModuleType(module_name)
    # Set the module's __dict__ as the global context for exec
    exec_globals = dynamic_module.__dict__

    # Define a function to run the exec statement and capture exceptions
    def run_exec():
        try:
            exec(code, exec_globals)
        except Exception:
            return sys.exc_info()
        return None

    # Use a queue to get the result from the thread
    import queue

    result_queue = queue.Queue()

    def thread_function():
        result = run_exec()
        result_queue.put(result)

    exec_thread = threading.Thread(target=thread_function)
    exec_thread.start()
    timeout = 60 * 5
    exec_thread.join(timeout)

    clear_cache_dir()

    if exec_thread.is_alive():
        print("Execution timed out")
        pytest.fail("timeout exceeded for docs text")

    # Check if there was an exception
    try:
        result = result_queue.get_nowait()
        if result is not None:
            exc_type, exc_value, exc_traceback = result
            print("Exception occurred:")
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            pytest.fail(f"Exception occurred: {exc_type.__name__}: {exc_value}")
    except queue.Empty:
        print("No result from thread (possible deadlock or infinite loop)")
        pytest.fail("No result from thread (possible deadlock or infinite loop)")
