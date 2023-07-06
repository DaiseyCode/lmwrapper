import math
import warnings

import pytest

from lmwrapper.caching import clear_cache_dir
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from lmwrapper.structs import LmPrompt, LmChatDialog


def play_with_probs():
    lm = get_open_ai_lm()
    out = lm.predict(
        LmPrompt(
            "Once upon",
            max_tokens=2,
            logprobs=5,
            cache=True,
            num_completions=1,
            echo=False
        ))
    print(out)
    print(out._get_completion_token_index())
    print(out.completion_tokens)
    print(out.completion_token_offsets)
    print(out.completion_logprobs)
    print(out.prompt_tokens)


def test_simple_pred():
    lm = get_open_ai_lm()
    out = lm.predict(
        LmPrompt(
            "Once upon a",
            max_tokens=1,
            logprobs=5,
            cache=False,
            num_completions=1,
            echo=False
        ))
    assert out.completion_text.strip() == "time"
    print(out)
    assert out.completion_tokens == [" time"]
    assert math.exp(out.completion_logprobs[0]) >= 0.9


def test_simple_pred_cache():
    lm = get_open_ai_lm()
    for i in range(2):
        out = lm.predict(
            LmPrompt(
                "Once upon a",
                max_tokens=1,
                logprobs=5,
                cache=True,
                num_completions=1,
                echo=False
            ))
        assert out.completion_text.strip() == "time"
        print(out)


def test_echo():
    lm = get_open_ai_lm()
    out = lm.predict(
        LmPrompt(
            "Once upon a",
            max_tokens=1,
            logprobs=5,
            cache=True,
            num_completions=1,
            echo=True
        )
    )
    print(out.get_full_text())
    assert out.get_full_text().strip() == "Once upon a time"
    assert out.completion_text.strip() == "time"
    assert out.prompt_tokens == ['Once', ' upon', ' a']
    assert len(out.prompt_logprobs) == 3
    assert len(out.prompt_logprobs) == 3
    assert len(out.full_logprobs) == 4
    assert out.get_full_tokens() == ['Once', ' upon', ' a', ' time']




def test_too_large_logprob():
    """Expect a warning to be thrown when logprobs is greater than 5 (which
    is the limit the openai api supports)"""
    lm = get_open_ai_lm()
    with pytest.warns(None) as called_warnings:
        lm.predict(
            LmPrompt(
                "Once",
                max_tokens=1,
                logprobs=5,
                cache=False,
                num_completions=1,
                echo=False
            )
        )
    assert len(called_warnings) == 0
    with pytest.warns(UserWarning):
        lm.predict(
            LmPrompt(
                "Once",
                max_tokens=1,
                logprobs=10,
                cache=False,
                num_completions=1,
                echo=False
            )
        )



def test_simple_chat_mode():
    lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo)
    out = lm.predict(LmPrompt(
        "What is 2+2? Answer with just one number.",
        max_tokens=1,
        num_completions=1,
        cache=False,
    ))
    assert out.completion_text.strip() == "4"


def test_simple_chat_mode_multiturn():
    lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo)
    prompt = [
        "What is 2+2? Answer with just one number.",
        "4",
        "What is 3+2?"
    ]
    assert LmChatDialog(prompt).as_dicts() == [
        {'role': 'user', 'content': 'What is 2+2? Answer with just one number.'},
        {'role': 'system', 'content': '4'},
        {'role': 'user', 'content': 'What is 3+2?'}
    ]
    out = lm.predict(LmPrompt(
        prompt,
        max_tokens=1,
        num_completions=1,
        cache=False,
    ))
    assert out.completion_text.strip() == "5"


def main():
    play_with_probs()
    exit()
    lm = get_open_ai_lm()
    clear_cache_dir()
    text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_tokens=1, logprobs=10, cache=True
        ))
    print(text.completion_text)
    lm = get_open_ai_lm()
    new_text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_tokens=1, logprobs=10, cache=True
        ))
    print(new_text.completion_text)
    assert text.completion_text == new_text.completion_text
    exit()
    lm = get_open_ai_lm()
    text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_tokens=1, logprobs=10,
        ))
    print(text.completion_text)


if __name__ == "__main__":
    main()
