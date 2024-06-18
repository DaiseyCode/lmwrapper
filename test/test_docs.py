"""
Tests to verify that any examples given in user-facing documentation
will execute.
"""

import re
from pathlib import Path

import pytest

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
    print("### CODE BLOCK")
    print(code)
    print("### OUTPUT")
    exec(code)
