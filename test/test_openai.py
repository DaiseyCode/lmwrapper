import sys
import tempfile
import pickle
import warnings
from typing import TypeVar
from unittest.mock import MagicMock

import pytest

from lmwrapper.caching import clear_cache_dir
from lmwrapper.openai_wrapper.wrapper import (
    OpenAiInstantiationHook,
    OpenAiModelNames,
    OpenAIPredictor,
    get_open_ai_lm,
    parse_backoff_time,
)
from lmwrapper.structs import LmChatDialog, LmPrompt


def play_with_probs():
    lm = get_open_ai_lm()
    out = lm.predict(
        LmPrompt(
            "Once upon",
            max_tokens=2,
            logprobs=5,
            cache=True,
            echo=False,
        ),
    )
    print(out)
    print(out._get_completion_token_index())
    print(out.completion_tokens)
    print(out.completion_token_offsets)
    print(out.completion_logprobs)
    print(out.prompt_tokens)


def test_with_probs_gpt35():
    lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo_instruct)
    out = lm.predict(
        LmPrompt(
            "Respond with true or false:",
            max_tokens=2,
            logprobs=5,
            cache=False,
            echo=False,
            temperature=1,
        ),
    )
    print(out)
    print(out.top_token_logprobs)
    # print(out._get_completion_token_index())
    # print(out.completion_tokens)
    # print(out.completion_token_offsets)
    # print(out.completion_logprobs)
    # print(out.prompt_tokens)


def test_too_large_logprob():
    """
    Expect a warning to be thrown when logprobs is greater than 5 (which
    is the limit the openai api supports)
    """
    lm = get_open_ai_lm()
    with warnings.catch_warnings():
        lm.predict(
            LmPrompt(
                "Once",
                max_tokens=1,
                logprobs=5,
                cache=False,
                echo=False,
            ),
        )

    with pytest.raises(ValueError):
        lm.predict(
            LmPrompt(
                "Once",
                max_tokens=1,
                logprobs=10,
                cache=False,
                echo=False,
            ),
        )


def test_simple_chat_mode():
    lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo)
    out = lm.predict(
        LmPrompt(
            "What is 2+2? Answer with just one number.",
            max_tokens=1,
            temperature=0.0,
            cache=False,
        ),
    )
    assert out.completion_text.strip() == "4"


def test_o1_mode():
    lm = get_open_ai_lm(OpenAiModelNames.o4_mini)
    out = lm.predict(
        LmPrompt(
            "What is 2+2? Answer with just one number.",
            max_tokens=1000,
            cache=False,
            logprobs=0,
        ),
    )
    assert out.completion_text.strip() == "4"


def test_gpt_5_1_mode():
    lm = get_open_ai_lm(OpenAiModelNames.gpt_5_1)
    out = lm.predict(
        LmPrompt(
            "What is 2+2? Answer with just one number.",
            max_tokens=1000,
            cache=False,
            logprobs=0,
        ),
    )
    assert out.completion_text.strip() == "4"


def test_o1_mode_permissive():
    lm = get_open_ai_lm(OpenAiModelNames.o4_mini)
    out = lm.predict(
        LmPrompt(
            "What is 2+2? Answer with just one number.",
            max_tokens=1000,
            cache=False,
            logprobs=1,
        ),
    )
    assert out.completion_text.strip() == "4"


def test_chat_nologprob_exception():
    lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo)
    out = lm.predict(
        LmPrompt(
            "What is 2+2? Answer with just one number.",
            max_tokens=1,
            temperature=0.0,
            cache=False,
            logprobs=0,
        ),
    )
    with pytest.raises(
        ValueError,
        match="Response does not contain top_logprobs.",
    ) as exc_info:
        out.top_token_logprobs

    assert type(exc_info.value) is ValueError


def test_simple_chat_mode_multiturn():
    lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo)
    prompt = [
        "What is 2+2? Answer with just one number.",
        "4",
        "What is 3+2?",
    ]
    assert LmChatDialog(prompt).as_dicts() == [
        {"role": "user", "content": "What is 2+2? Answer with just one number."},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "What is 3+2?"},
    ]
    out = lm.predict(
        LmPrompt(
            prompt,
            max_tokens=1,
            cache=False,
        ),
    )
    assert out.completion_text.strip() == "5"


def test_multiturn_chat_logprobs():
    lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo)
    prompt = [
        "What is 2+2? Answer with one number followed by a period.",
        "4.",
        "What is 3+2?",
    ]
    assert LmChatDialog(prompt).as_dicts() == [
        {
            "role": "user",
            "content": "What is 2+2? Answer with one number followed by a period.",
        },
        {"role": "assistant", "content": "4."},
        {"role": "user", "content": "What is 3+2?"},
    ]
    out = lm.predict(
        LmPrompt(
            prompt,
            max_tokens=2,
            temperature=0,
            cache=False,
            logprobs=2,
        ),
    )
    assert out.completion_text.strip() == "5."

    # even at t=0, the logprobs have high variance
    # perhaps anti-reverse engineering measures?

    assert out.top_token_logprobs == [
        {
            "5": pytest.approx(-0.00012689977, abs=4),
            "The": pytest.approx(-9.452922, abs=4),
        },
        {
            ".": pytest.approx(-4.2391708e-05, abs=4),
            "<|end|>": pytest.approx(-10.109505, abs=4),
        },
    ]


def test_ratelimit():
    try:
        OpenAIPredictor.configure_global_ratelimit(1, per_seconds=2)
        assert OpenAIPredictor._wait_ratelimit() == 0.0
        assert OpenAIPredictor._wait_ratelimit() == pytest.approx(2, rel=0.1)
    finally:
        OpenAIPredictor.configure_global_ratelimit(None)  # teardown


def main():
    play_with_probs()
    sys.exit()
    lm = get_open_ai_lm()
    clear_cache_dir()
    text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_tokens=1,
            logprobs=10,
            cache=True,
        ),
    )
    print(text.completion_text)
    lm = get_open_ai_lm()
    new_text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_tokens=1,
            logprobs=10,
            cache=True,
        ),
    )
    print(new_text.completion_text)
    assert text.completion_text == new_text.completion_text
    sys.exit()
    lm = get_open_ai_lm()
    text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_tokens=1,
            logprobs=10,
        ),
    )
    print(text.completion_text)


def test_tokenizer_gpt3():
    lm = get_open_ai_lm(
        OpenAiModelNames.gpt_3_5_turbo_instruct,
    )
    assert lm.token_limit == 4096
    assert lm.estimate_tokens_in_prompt(LmPrompt("Once", max_tokens=10)) == 1
    assert (
        lm.estimate_tokens_in_prompt(
            LmPrompt(
                "Once upon a time there was a magical place with a Spunctulus that was"
                " 3281 years old",
                max_tokens=10,
            ),
        )
        == 21
    )
    long_prompt = " ".join(["once"] * 2000)
    estimate = lm.estimate_tokens_in_prompt(
        LmPrompt(
            long_prompt,
            max_tokens=10,
        ),
    )
    assert estimate == 2000
    assert not lm.could_completion_go_over_token_limit(
        LmPrompt(
            " ".join(["once"] * 4000),
            max_tokens=10,
        ),
    )
    assert lm.could_completion_go_over_token_limit(
        LmPrompt(
            " ".join(["once"] * 4090),
            max_tokens=50,
        ),
    )


def test_tokenizer_chat():
    lm = get_open_ai_lm(
        OpenAiModelNames.gpt_3_5_turbo,
    )
    assert lm.token_limit == 4096
    assert (
        5
        < lm.estimate_tokens_in_prompt(
            LmPrompt(
                "Once upon a time",
                max_tokens=10,
            ),
        )
        < 4 + 12
    )


def test_instantiation_hook():
    was_called = False

    try:

        class SimpleHook(OpenAiInstantiationHook):
            def before_init(
                self,
                new_predictor: OpenAIPredictor,
                api,
                engine_name: str,
                chat_mode: bool,
                cache_outputs_default: bool,
                retry_on_rate_limit: bool,
            ):
                nonlocal was_called
                was_called = True
                assert engine_name == OpenAiModelNames.gpt_3_5_turbo_instruct

        OpenAIPredictor.add_instantiation_hook(SimpleHook())
        assert not was_called
        get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo_instruct)
        assert was_called
    finally:
        OpenAIPredictor._instantiation_hooks = []


@pytest.mark.skip("We don't need to do this usually")
@pytest.mark.parametrize(
    "model_name",
    [
        OpenAiModelNames.gpt_4_turbo,
        OpenAiModelNames.gpt_4o,
    ],
)
def test_simple_chat_mode_multiturn_4turbo(model_name):
    lm = get_open_ai_lm(model_name)
    prompt = [
        "What is 2+2? Answer with just one number.",
        "4",
        "What is 3+2?",
    ]
    assert LmChatDialog(prompt).as_dicts() == [
        {"role": "user", "content": "What is 2+2? Answer with just one number."},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "What is 3+2?"},
    ]
    out = lm.predict(
        LmPrompt(
            prompt,
            max_tokens=1,
            cache=False,
        ),
    )
    assert out.completion_text.strip() == "5"


def test_tokenizer():
    lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo)
    tokens = lm.tokenize("I like pie")
    assert tokens == ["I", " like", " pie"]


def test_backoff_parse():
    mock_exception = MagicMock()
    mock_exception.message = (
        "Rate limit reached for gpt-3.5-turbo-instruct in organization on tokens per"
        " min (TPM): Limit 90000, Used 89007, Requested 2360. Please try again in"
        " 1.42s. Visit https://platform.openai.com/account/rate-limits to learn more."
    )
    backoff = parse_backoff_time(mock_exception)
    assert backoff == 2


def test_backoff_parse2():
    mock_exception = MagicMock()
    mock_exception.message = (
        "Rate limit reached for gpt-3.5-turbo-instruct in organization org on tokens"
        " per min (TPM): Limit 90000, Used 89007, Requested 2360. Please try again in"
        " 1.42s. Visit https://platform.openai.com/account/rate-limits to learn more."
    )
    backoff = parse_backoff_time(mock_exception)
    assert backoff == 2


def test_backoff_parse3():
    mock_exception = MagicMock()
    mock_exception.message = (
        "Rate limit reached for gpt-3.5-turbo-instruct in organization org on tokens"
        " per min (TPM): Limit 90000, Used 89007, Requested 2360. Please try again in"
        " 5s. Visit https://platform.openai.com/account/rate-limits to learn more."
    )
    backoff = parse_backoff_time(mock_exception)
    assert backoff == 5

T = TypeVar("T")


def pickle_roundtrip(obj: T) -> T:
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        # Serialize the info to the tempfile
        pickle.dump(obj, temp_file)
        temp_file.seek(0)  # Reset file pointer to the beginning
        # Deserialize the info from the tempfile
        new_obj = pickle.load(temp_file)
    return new_obj


def test_pickle_model_info():
    info = OpenAiModelNames.gpt_4o_mini
    assert info == pickle_roundtrip(info)


def test_pickle_model():
    lm = get_open_ai_lm(OpenAiModelNames.gpt_4o_mini)
    pred1 = lm.predict(LmPrompt(
        "Give 10 random numbers",
        max_tokens=10, temperature=0.0, cache=False,
    ))
    lm2 = pickle_roundtrip(lm)
    pred2 = lm2.predict(
        LmPrompt(
            "Give 10 random numbers",
            max_tokens=10, temperature=0.0, cache=False,
        ),
    )
    assert pred1.completion_text == pred2.completion_text


if __name__ == "__main__":
    main()
