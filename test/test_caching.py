import dataclasses
import tempfile
from pathlib import Path

from lmwrapper.caching import set_cache_dir
from lmwrapper.huggingface_wrapper.wrapper import get_huggingface_lm
from lmwrapper.structs import LmPrompt


def test_set_cache_dir():
    # Make a temporary directory for use as a cache directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        assert len(list(tmpdirname.rglob("*"))) == 0
        set_cache_dir(tmpdirname)
        lm = get_huggingface_lm("gpt2")
        prompt = LmPrompt(
            "Write a story about fish:",
            max_tokens=10,
            temperature=1.0,
            cache=False,
        )
        r1 = lm.predict(prompt)
        prompt = dataclasses.replace(prompt, cache=True)
        r2 = lm.predict(prompt)
        assert r1.completion_text != r2.completion_text
        lm2 = get_huggingface_lm("gpt2")
        r3 = lm2.predict(prompt)
        assert r2.completion_text == r3.completion_text
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        set_cache_dir(tmpdirname)
        lm2 = get_huggingface_lm("gpt2")
        r4 = lm2.predict(prompt)
        assert r3.completion_text != r4.completion_text
