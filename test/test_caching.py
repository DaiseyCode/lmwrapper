import dataclasses
import multiprocessing
import os
import random
import tempfile
from pathlib import Path

import pytest

from lmwrapper.abstract_predictor import get_mock_predictor
from lmwrapper.caching import set_cache_dir
from lmwrapper.huggingface_wrapper.wrapper import get_huggingface_lm
from lmwrapper.structs import LmPrompt

IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


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


def test_cache_stress_random():
    prompts = [LmPrompt(f"Prompt {i}", cache=True) for i in range(100)]
    import random

    lm = get_mock_predictor()
    for _ in range(10_000):
        prompt = random.choice(prompts)
        lm.predict(prompt)


@pytest.mark.skipif(IS_GITHUB_ACTIONS, reason="No multithread maybe makes it happier")
def test_cache_stress_random_multithread():
    prompts = [LmPrompt(f"Prompt {i}", cache=True) for i in range(100)]
    import random

    lm = get_mock_predictor()
    import threading

    def worker():
        for _ in range(1_000):
            prompt = random.choice(prompts)
            pred = lm.predict(prompt)
            assert pred.completion_text == prompt.get_text_as_string_default_form()

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()


def _worker_multiproc():
    lm = get_mock_predictor()
    prompts = [LmPrompt(f"Prompt {i}", cache=True) for i in range(100)]
    num_hits = 0
    for _ in range(10_000):
        prompt = random.choice(prompts)
        pred = lm.predict(prompt)
        assert pred.completion_text == prompt.get_text_as_string_default_form()
        if pred.was_cached:
            num_hits += 1
    assert num_hits > 9000


@pytest.mark.skipif(IS_GITHUB_ACTIONS, reason="Multiprocessing seems to cause issues")
def test_cache_stress_random_multiproc():
    procs = [multiprocessing.Process(target=_worker_multiproc) for _ in range(5)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
