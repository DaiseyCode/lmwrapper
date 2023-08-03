from lmwrapper.huggingface_wrapper import get_huggingface_lm
from lmwrapper.openai_wrapper import get_open_ai_lm
import pytest

from lmwrapper.structs import LmPrompt
import math

ALL_MODELS = [
    get_open_ai_lm(),
    get_huggingface_lm('gpt2'),
]


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_simple_pred(lm):
    out = lm.predict(
        LmPrompt(
            "Here is a story. Once upon a",
            max_tokens=1,
            logprobs=5,
            cache=False,
            num_completions=1,
            echo=False
        ))
    assert out.completion_text.strip() == "time"
    print(out)
    assert lm.remove_special_chars_from_tokens(out.completion_tokens) == [" time"]
    assert len(out.completion_logprobs) == 1
    assert math.exp(out.completion_logprobs[0]) >= 0.9


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_simple_pred_cache(lm):
    runtimes = []
    import time
    for i in range(2):
        start = time.time()
        out = lm.predict(
            LmPrompt(
                "Once upon a",
                max_tokens=1,
                logprobs=5,
                cache=True,
                num_completions=1,
                echo=False
            )
        )
        end = time.time()
        assert out.completion_text.strip() == "time"
        runtimes.append(end - start)


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_echo(lm):
    out = lm.predict(
        LmPrompt(
            "Once upon a",
            max_tokens=1,
            logprobs=5,
            cache=False,
            num_completions=1,
            echo=True
        )
    )
    print(out.get_full_text())
    assert out.get_full_text().strip() == "Once upon a time"
    assert out.completion_text.strip() == "time"
    assert lm.remove_special_chars_from_tokens(out.prompt_tokens) == ['Once', ' upon', ' a']
    assert len(out.prompt_logprobs) == 3
    assert len(out.prompt_logprobs) == 3
    assert len(out.full_logprobs) == 4
    assert (
        lm.remove_special_chars_from_tokens(out.get_full_tokens())
        == ['Once', ' upon', ' a', ' time']
    )


