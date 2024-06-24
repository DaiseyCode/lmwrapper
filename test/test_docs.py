"""
Tests to verify that any examples given in user-facing documentation
will execute.
"""

import re
from pathlib import Path

import pytest
import types
import threading

from lmwrapper.caching import clear_cache_dir

cur_file = Path(__file__).parent.absolute()


def extract_code_blocks(file):
    with open(file) as f:
        content = f.read()
    pattern = re.compile(
        r"(<!-- skip ?test -->\s*)?```python\r?\n(.*?)```", re.DOTALL
    )
    blocks = pattern.findall(content)
    return [code for skip, code in blocks if not skip]


"""
Tests to verify that any examples given in user-facing documentation
will execute.
"""

import re
from pathlib import Path

import pytest
import types
import threading

from lmwrapper.caching import clear_cache_dir

cur_file = Path(__file__).parent.absolute()


def extract_code_blocks(file):
    with open(file) as f:
        content = f.read()
    pattern = re.compile(
        r"(<!-- skip ?test -->\s*)?```python\r?\n(.*?)```", re.DOTALL
    )
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
    module_name = 'dynamic_module'
    dynamic_module = types.ModuleType(module_name)
    # Set the module's __dict__ as the global context for exec
    exec_globals = dynamic_module.__dict__

    # Define a function to run the exec statement
    def run_exec():
        exec(code, exec_globals)
    exec_thread = threading.Thread(target=run_exec)
    exec_thread.start()
    timeout = 60 * 5
    exec_thread.join(timeout)
    clear_cache_dir()
    if exec_thread.is_alive():
        print("Execution timed out")
        pytest.fail("timeout exceeded for docs text")
