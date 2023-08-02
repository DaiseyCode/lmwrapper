import pytest

from lmwrapper.huggingface_wrapper import get_huggingface_lm
from lmwrapper.structs import LmPrompt


@pytest.fixture(scope="module")
def lm():
    #return get_huggingface_lm("fxmarty/tiny-llama-fast-tokenizer")
    return get_huggingface_lm("gpt2")


def test_simple_pred(lm):
    out = lm.predict(
        LmPrompt(
            "Once upon a",
            max_tokens=1,
            cache=False,
            temperature=0,
        ))
    assert out.completion_text.strip() == "time"
    print(out)
    assert out.completion_tokens == ["Ġtime"]
    #assert math.exp(out.completion_logprobs[0]) >= 0.9
