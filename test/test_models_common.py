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
            cache=False,
        ))
    assert out.completion_text.strip() == "time"


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_simple_pred_lp(lm):
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


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_low_prob_in_weird_sentence(lm):
    weird = lm.predict(
        LmPrompt(
            "The Empire State Building is in New run and is my favorite",
            max_tokens=1,
            logprobs=5,
            cache=False,
            num_completions=1,
            echo=True
        )
    )
    normal = lm.predict(
        LmPrompt(
            "The Empire State Building is in New York and is my favorite",
            max_tokens=1,
            logprobs=5,
            cache=False,
            num_completions=1,
            echo=True
        )
    )
    no_space = lm.remove_special_chars_from_tokens(weird.prompt_tokens)
    assert (
        no_space
        == ["The", " Empire", " State", " Building", " is", " in", " New", " run", " and", " is", " my", " favorite"]
    )
    assert len(weird.prompt_logprobs) == len(weird.prompt_tokens)
    weird_idx = no_space.index(' run')
    assert math.exp(weird.prompt_logprobs[weird_idx]) < 0.001
    assert math.exp(normal.prompt_logprobs[weird_idx]) > .5
    assert (
        math.exp(weird.prompt_logprobs[weird_idx])
        < math.exp(normal.prompt_logprobs[weird_idx])
    )
    assert (
            math.exp(weird.prompt_logprobs[weird_idx - 1])
            == pytest.approx(math.exp(normal.prompt_logprobs[weird_idx - 1]), rel=1e-5)
    )


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_no_gen(lm):
    val = lm.predict(
        LmPrompt(
            "I like pie",
            max_tokens=0,
            logprobs=5,
            cache=False,
            num_completions=1,
            echo=True
        )
    )
    assert len(val.prompt_tokens) == 3
    assert len(val.prompt_logprobs) == 3
    assert len(val.completion_tokens) == 0
    assert len(val.completion_text) == 0
    assert len(val.completion_logprobs) == 0


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_many_gen(lm):
    val = lm.predict(
        LmPrompt(
            "Write a story about a pirate:",
            max_tokens=5,
            logprobs=1,
            cache=False
        )
    )
    assert len(val.completion_tokens) == 5


@pytest.mark.parametrize("lm", ALL_MODELS)
@pytest.mark.skip(reason="OpenAI will insert an <|endoftext|> when doing"
                         "unconditional generation and need to look into if also"
                         "happens with the chat models and how to handle it")
def test_unconditional_gen(lm):
    # TODO: handle for openai
    val = lm.predict(
        LmPrompt(
            "",
            max_tokens=2,
            logprobs=1,
            cache=False,
            echo=True,
        )
    )
    assert len(val.prompt_tokens) == 0
    assert len(val.prompt_logprobs) == 0
    assert len(val.completion_tokens) == 2
    assert len(val.completion_text) > 2
    assert len(val.completion_logprobs) == 2