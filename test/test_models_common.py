import dataclasses
import warnings
import math
import pickle
import threading
import time

import numpy as np
import pytest

from lmwrapper.abstract_predictor import LmPredictor, LmReasoningStyle
from lmwrapper.batch_config import CompletionWindow
from lmwrapper.caching import cache_dir, clear_cache_dir
from lmwrapper.huggingface_wrapper import get_huggingface_lm
from lmwrapper.huggingface_wrapper.wrapper import get_huggingface_lm
from lmwrapper.openai_wrapper import get_open_ai_lm
from lmwrapper.openai_wrapper.wrapper import OpenAiModelNames, get_open_ai_lm
from lmwrapper.structs import LmPrompt, LmChatTurn, ChatGptRoles, LmChatDialog
from test.test_params import DEFAULT_SMALL
from lmwrapper.claude_wrapper.wrapper import get_claude_lm, ClaudeModelNames
from functools import wraps


def skip_if_no_token_ops(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        model = next(
            (arg for arg in (*args, *kwargs.values()) 
             if hasattr(arg, 'supports_token_operations')),
            None
        )
        if model and not model.supports_token_operations:
            pytest.skip("Model does not support token operations")
        return func(*args, **kwargs)
    return wrapper


MODEL_NAMES = {
    "3_5_turbo_instruct": get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo_instruct),
    "small_hug": get_huggingface_lm(DEFAULT_SMALL),
    "4o_mini": get_open_ai_lm(OpenAiModelNames.gpt_4o_mini),
    "4_1_nano": get_open_ai_lm(OpenAiModelNames.gpt_4_1_nano),
    "3_5_haiku": get_claude_lm(ClaudeModelNames.claude_3_5_haiku),
    "3_5_turbo": get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo),
    "gpt_5_nano": get_open_ai_lm(OpenAiModelNames.gpt_5_nano),
    "gpt_5_1": get_open_ai_lm(OpenAiModelNames.gpt_5_1),
    "claude_4_5_sonnet": get_claude_lm(ClaudeModelNames.claude_4_5_sonnet),
    "claude_4_5_haiku": get_claude_lm(ClaudeModelNames.claude_4_5_haiku),
    #"qwen25_500M_instruct": get_huggingface_lm("Qwen/Qwen2.5-0.5B-Instruct"),
}


ALL_MODELS = [
    MODEL_NAMES["3_5_turbo_instruct"],
    MODEL_NAMES["small_hug"],
    MODEL_NAMES["4o_mini"],
    MODEL_NAMES["claude_4_5_haiku"],
]

ALL_MODELS_OLD = [
    MODEL_NAMES["3_5_turbo_instruct"],
    MODEL_NAMES["small_hug"],
    MODEL_NAMES["3_5_turbo"]
]

COMPLETION_MODELS = [
    MODEL_NAMES["3_5_turbo_instruct"],
    MODEL_NAMES["small_hug"],
]

CHAT_MODELS = [
    #MODEL_NAMES["4_1_nano"],
    MODEL_NAMES["gpt_5_nano"],
    MODEL_NAMES["claude_4_5_haiku"],
    #MODEL_NAMES["qwen25_500M_instruct"]
]


ECHOABLE_MODELS = [
    # get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo_instruct),
    # Won't work with now that echo disabled
    get_huggingface_lm("gpt2"),
]

random_prompt = ("Give a random base-64 guid (answer with only the guid). "
                 "Then give 10 truely random words:")


def get_model_name(model):
    # Find the name by checking which value in MODEL_NAMES matches the model
    for name, value in MODEL_NAMES.items():
        if value is model:  # Using 'is' for identity comparison
            return value.get_model_cache_key()
    return "unknown_model"  # Fallback name


@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
def test_simple_pred(lm):
    for i in range(5):
        out = lm.predict(
            LmPrompt(
                "Give a one word completion. Answer with only the next word: "
                "'Here is a fairytale story. Once upon a",
                max_tokens=1,
                cache=False,
            ),
        )
        print(out.completion_text)
        if out.completion_text.strip() == "time":
            break
    else:
        assert False, "Failed to get the right completion"


@skip_if_no_token_ops
@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
def test_simple_pred_lp(lm):
    out = lm.predict(
        LmPrompt(
            "Give a one word completion. Answer with only the next word: "
            "'Here is a fairytale story. Once upon a",
            max_tokens=1,
            logprobs=1,
            cache=False,
            echo=False,
            temperature=0,
        ),
    )
    assert out.completion_text.strip() == "time"
    print(out)
    assert lm.remove_special_chars_from_tokens(out.completion_tokens) in (
        [" time"],
        ["time"],
    )
    assert len(out.completion_logprobs) == 1
    assert math.exp(out.completion_logprobs[0]) >= 0.80
    # Make sure prob not broken when temperature=0
    out = lm.predict(
        LmPrompt(
            "Give a one word completion: 'Here is a story. "
            "Once upon a time there was a",
            max_tokens=1,
            logprobs=1,
            cache=False,
            echo=False,
            temperature=0,
        ),
    )
    assert math.exp(out.completion_logprobs[0]) <= 0.95
    # Lowish temp
    out = lm.predict(
        LmPrompt(
            "Give a one word completion: 'Here is a fairytale story. Once upon a",
            max_tokens=1,
            logprobs=1,
            cache=False,
            echo=False,
            temperature=0.2,
        ),
    )
    assert out.completion_text.strip() == "time"
    print(out)
    assert lm.remove_special_chars_from_tokens(out.completion_tokens) in (
        [" time"],
        ["time"],
    )
    assert len(out.completion_logprobs) == 1
    assert math.exp(out.completion_logprobs[0]) >= 0.80


@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
def test_simple_pred_cache(lm):
    runtimes = []
    import time

    prompt = LmPrompt(
        "Give a one word completion: 'Here is a story. Once upon a",
        max_tokens=1,
        logprobs=1,
        cache=True,
        echo=False,
        temperature=0.0,
    )
    lm.remove_prompt_from_cache(prompt)

    for _i in range(2):
        start = time.time()
        out = lm.predict(prompt)
        end = time.time()
        assert out.completion_text.strip() == "time"
        runtimes.append(end - start)

    assert runtimes[0] > runtimes[1] * 5


@pytest.mark.parametrize("lm", ECHOABLE_MODELS, ids=get_model_name)
@skip_if_no_token_ops
def test_echo(lm):
    out = lm.predict(
        LmPrompt(
            "Once upon a",
            max_tokens=1,
            logprobs=1,
            cache=False,
            echo=True,
        ),
    )
    print(out.get_full_text())
    assert out.get_full_text().strip() == "Once upon a time"
    assert out.completion_text.strip() == "time"
    assert lm.remove_special_chars_from_tokens(out.prompt_tokens) == [
        "Once",
        " upon",
        " a",
    ]
    assert len(out.prompt_logprobs) == 3
    assert len(out.prompt_logprobs) == 3
    assert len(out.full_logprobs) == 4
    assert lm.remove_special_chars_from_tokens(out.get_full_tokens()) == [
        "Once",
        " upon",
        " a",
        " time",
    ]


@pytest.mark.parametrize("lm", ECHOABLE_MODELS, ids=get_model_name)
@skip_if_no_token_ops
def test_low_prob_in_weird_sentence(lm):
    weird = lm.predict(
        LmPrompt(
            "The Empire State Building is in New run and is my favorite",
            max_tokens=1,
            logprobs=1,
            cache=False,
            echo=True,
        ),
    )
    normal = lm.predict(
        LmPrompt(
            "The Empire State Building is in New York and is my favorite",
            max_tokens=1,
            logprobs=1,
            cache=False,
            echo=True,
        ),
    )
    no_space = lm.remove_special_chars_from_tokens(weird.prompt_tokens)
    assert no_space == [
        "The",
        " Empire",
        " State",
        " Building",
        " is",
        " in",
        " New",
        " run",
        " and",
        " is",
        " my",
        " favorite",
    ]
    assert len(weird.prompt_logprobs) == len(weird.prompt_tokens)
    weird_idx = no_space.index(" run")
    assert math.exp(weird.prompt_logprobs[weird_idx]) < 0.001
    assert math.exp(normal.prompt_logprobs[weird_idx]) > 0.5
    assert math.exp(weird.prompt_logprobs[weird_idx]) < math.exp(
        normal.prompt_logprobs[weird_idx],
    )
    assert math.exp(weird.prompt_logprobs[weird_idx - 1]) == pytest.approx(
        math.exp(normal.prompt_logprobs[weird_idx - 1]),
        rel=1e-5,
    )


@pytest.mark.parametrize("lm", ECHOABLE_MODELS, ids=get_model_name)
@skip_if_no_token_ops
def test_no_gen_with_echo(lm):
    val = lm.predict(
        LmPrompt(
            "I like pie",
            max_tokens=0,
            logprobs=1,
            cache=False,
            echo=True,
        ),
    )
    assert len(val.prompt_tokens) == 3
    assert len(val.prompt_logprobs) == 3
    assert len(val.completion_tokens) == 0
    assert len(val.completion_text) == 0
    assert len(val.completion_logprobs) == 0


@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
def test_many_gen(lm):
    val = lm.predict(
        LmPrompt(
            "Write a story about a pirate:",
            max_tokens=5,
            logprobs=1,
            cache=False,
        ),
    )
    assert val.usage_output_tokens == 5
    if lm.supports_token_operations:
        assert len(val.completion_tokens) == 5


@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
@pytest.mark.skip(
    reason=(
        "OpenAI will insert an <|endoftext|> when doing"
        "unconditional generation and need to look into if also"
        "happens with the chat models and how to handle it"
    ),
)
def test_unconditional_gen(lm):
    # TODO: handle for openai
    val = lm.predict(
        LmPrompt(
            "",
            max_tokens=2,
            logprobs=1,
            cache=False,
            echo=True,
        ),
    )
    assert len(val.prompt_tokens) == 0
    assert len(val.prompt_logprobs) == 0
    assert len(val.completion_tokens) == 2
    assert len(val.completion_text) > 2
    assert len(val.completion_logprobs) == 2


capital_prompt = (
    "The capital of Germany is the city Berlin. "
    "The capital of Spain is the city Madrid. "
    "The capital of UK is the city London. "
    "The capital of France"
)


@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
@skip_if_no_token_ops
def test_no_stopping_in_prompt(lm):
    capital_newlines = (
        "The capitol of Germany\n is the city Berlin.\n"
        "The capital of Spain\n is the city Madrid.\n"
        "The capital of UK\n is the city London.\n"
        "The capital of France\n"
    )
    # By having the last character of the prompt be a stop token
    # we ensure that stopping logic does not include the prompt
    new_line = lm.predict(
        LmPrompt(
            capital_newlines,
            stop=["\n"],
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    assert len(new_line.completion_tokens) == 4


@pytest.mark.parametrize("lm", COMPLETION_MODELS, ids=get_model_name)
def test_no_stopping_program(lm):
    prompt_text = "# Functions\ndef double(x):\n"
    resp = lm.predict(
        LmPrompt(
            prompt_text,
            max_tokens=50,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    assert "\ndef" in resp.completion_text
    print(resp.completion_text)
    resp = lm.predict(
        LmPrompt(
            prompt_text,
            stop=["\ndef"],
            max_tokens=50,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    assert "\ndef" not in resp.completion_text


@pytest.mark.parametrize("lm", COMPLETION_MODELS, ids=get_model_name)
def test_stopping_begin_tok_full_tok(lm):
    val_normal = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    print(val_normal.completion_text)
    assert "is the city Paris" in val_normal.completion_text
    assert len(val_normal.completion_tokens) == 4
    assert (
        lm.remove_special_chars_from_tokens(val_normal.completion_tokens)[-1]
        == " Paris"
    )
    val_no_pa = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
            stop=[" Paris"],
        ),
    )
    print(val_no_pa.completion_text)
    assert val_no_pa.completion_text == " is the city"
    assert len(val_no_pa.completion_tokens) == 3
    assert val_no_pa.completion_tokens[0] == val_normal.completion_tokens[0]
    assert val_no_pa.completion_tokens[1] == val_normal.completion_tokens[1]
    assert val_no_pa.completion_tokens[2] == val_normal.completion_tokens[2]


@pytest.mark.parametrize("lm", COMPLETION_MODELS, ids=get_model_name)
def test_stopping_begin_tok(lm):
    val_normal = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    print(val_normal.completion_text)
    assert "is the city Paris" in val_normal.completion_text
    assert len(val_normal.completion_tokens) == 4
    assert (
        lm.remove_special_chars_from_tokens(val_normal.completion_tokens)[-1]
        == " Paris"
    )
    # Chopping off first part of subtoken does not return token
    val_no_pa = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
            stop=[" Pa"],
        ),
    )
    print(val_no_pa.completion_text)
    assert val_no_pa.completion_text == " is the city"
    assert len(val_no_pa.completion_tokens) == 3
    assert val_no_pa.completion_tokens[0] == val_normal.completion_tokens[0]
    assert np.allclose(
        val_no_pa.completion_logprobs,
        val_normal.completion_logprobs[:-1],
        atol=0.001,
        rtol=0.001,
    )


@pytest.mark.parametrize("lm", COMPLETION_MODELS, ids=get_model_name)
def test_stopping_middle_tok(lm):
    val_normal = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    # Chopping off middle of subtoken returns token but cut
    val_no_ari = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
            stop=["ari"],
        ),
    )
    assert val_no_ari.completion_text == " is the city P"
    assert len(val_no_ari.completion_logprobs) == 4
    assert np.allclose(
        val_no_ari.completion_logprobs,
        val_normal.completion_logprobs,
        atol=0.001,
        rtol=0.001,
    )
    assert (
        lm.remove_special_chars_from_tokens(val_no_ari.completion_tokens)[-1]
        == " Paris"
    )


@pytest.mark.parametrize("lm", COMPLETION_MODELS, ids=get_model_name)
def test_stopping_end_tok(lm):
    val_normal = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    # Chopping off end of subtoken returns token but cut
    val_no_ris = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
            stop=["ris"],
        ),
    )
    assert val_no_ris.completion_text == " is the city Pa"
    assert len(val_no_ris.completion_logprobs) == 4
    assert np.allclose(
        val_no_ris.completion_logprobs,
        val_normal.completion_logprobs,
        atol=0.001,
        rtol=0.001,
    )
    assert (
        lm.remove_special_chars_from_tokens(val_no_ris.completion_tokens)[-1]
        == " Paris"
    )


@pytest.mark.parametrize("lm", COMPLETION_MODELS, ids=get_model_name)
def test_stopping_span_subtoks(lm):
    val_normal = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    # Chopping off between multiple subtokens
    val_no_ris = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=10,
            logprobs=1,
            temperature=0,
            cache=False,
            stop=["ity Paris"],
        ),
    )
    assert val_no_ris.completion_text == " is the c"
    assert len(val_no_ris.completion_logprobs) == 3
    assert np.allclose(
        val_no_ris.completion_logprobs,
        val_normal.completion_logprobs[:-1],
        atol=0.001,
        rtol=0.001,
    )
    assert (
        lm.remove_special_chars_from_tokens(val_no_ris.completion_tokens)[-1] == " city"
    )


@pytest.mark.parametrize("lm", COMPLETION_MODELS, ids=get_model_name)
def test_stopping_span_subtoks2(lm):
    val_normal = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    # Chopping off between multiple subtokens in middle
    val_no_ris = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=10,
            logprobs=1,
            temperature=0,
            cache=False,
            stop=["ity Par"],
        ),
    )
    assert val_no_ris.completion_text == " is the c"
    assert len(val_no_ris.completion_logprobs) == 3
    assert np.allclose(
        val_no_ris.completion_logprobs,
        val_normal.completion_logprobs[:-1],
        atol=0.001,
        rtol=0.001,
    )
    assert (
        lm.remove_special_chars_from_tokens(val_no_ris.completion_tokens)[-1] == " city"
    )


@pytest.mark.parametrize("lm", COMPLETION_MODELS, ids=get_model_name)
def test_stopping_span_subtoks_multiple(lm):
    val_normal = lm.predict(
        LmPrompt(
            capital_prompt,
            max_tokens=4,
            logprobs=1,
            temperature=0,
            cache=False,
        ),
    )
    for do_reverse in [True, False]:
        stop = ["ity Par", "ty P"]
        if do_reverse:
            stop.reverse()
        val_no_ris = lm.predict(
            LmPrompt(
                capital_prompt,
                max_tokens=10,
                logprobs=1,
                temperature=0,
                cache=False,
                stop=stop,
            ),
        )
        assert val_no_ris.completion_text == " is the c"
        assert len(val_no_ris.completion_logprobs) == 3
        assert np.allclose(
            val_no_ris.completion_logprobs,
            val_normal.completion_logprobs[:-1],
            atol=0.001,
            rtol=0.001,
        )
        assert (
            lm.remove_special_chars_from_tokens(val_no_ris.completion_tokens)[-1]
            == " city"
        )


@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
def test_remove_prompt_from_cache(lm):
    prompt = LmPrompt(
        random_prompt,
        max_tokens=100,
        temperature=1.0,
        cache=True,
    )
    r1 = lm.predict(prompt)
    r2 = lm.predict(prompt)
    assert r1.completion_text == r2.completion_text
    assert lm.remove_prompt_from_cache(prompt)
    r3 = lm.predict(prompt)
    if r1.completion_text != r3.completion_text:
        return  # Pass because it is different and uncached
    # Sometimes still flacky with just one prompt
    assert lm.remove_prompt_from_cache(prompt)
    r4 = lm.predict(prompt)
    assert (
        r1.completion_text != r3.completion_text
        or r1.completion_text != r4.completion_text
    )


@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
@skip_if_no_token_ops
def test_none_max_tokens(lm):
    prompt = LmPrompt(
        "Write a detailed story (three paragraphs) about a dog:",
        max_tokens=None,
        temperature=0.2,
        cache=False,
    )
    result = lm.predict(prompt)
    if lm.default_tokens_generated is not None:
        assert len(result.completion_tokens) == lm.default_tokens_generated
    else:
        assert len(result.completion_tokens) > 100


# @pytest.mark.parametrize("lm", ALL_MODELS)
# def test_need_tokens(lm):
#    prompt = LmPrompt(
#        capital_prompt,
#        max_tokens=4,
#        temperature=0.0,
#        cache=False,
#        potentially_need_tokens=True,
#        logprobs=0,
#    )
#    result = lm.predict(prompt)
#    tokens = lm.tokenize(capital_prompt)
#    assert tokens[:4] == ["The", " capital", " of", " Germany"]
#    assert result.prompt_tokens == tokens


@pytest.mark.parametrize("lm", ALL_MODELS_OLD, ids=get_model_name)
def test_response_to_dict_conversion(lm):
    prompt = LmPrompt(
        text=capital_prompt,
        max_tokens=4,
        stop=["\n"],
        logprobs=1,
        temperature=0,
        cache=False,
    )
    resp = lm.predict(prompt)
    resp_dict = resp.dict_serialize()
    expected = {
        "completion_text": " is the city Paris",
        "prompt": prompt.dict_serialize(),
        "was_cached": False,
        "completion_tokens": [
            " is",
            " the",
            " city",
            " Paris",
        ],
    }
    assert all(key in resp_dict for key in expected)
    assert all(resp_dict[key] == expected[key] for key in expected)


@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
def test_was_cached_marking(lm):
    prompt = LmPrompt(
        random_prompt,
        max_tokens=30,
        temperature=1.0,
        cache=False,
    )
    r1 = lm.predict(prompt)
    assert not r1.was_cached
    r2 = lm.predict(prompt)
    assert not r2.was_cached
    prompt = LmPrompt(
        random_prompt,
        max_tokens=100,
        temperature=1.0,
        cache=True,
    )
    lm.remove_prompt_from_cache(prompt)
    r3 = lm.predict(prompt)
    assert not r3.was_cached
    r4 = lm.predict(prompt)
    assert r4.was_cached
    assert not r3.was_cached
    assert r3.completion_text == r4.completion_text
    lm.remove_prompt_from_cache(prompt)
    r5 = lm.predict(prompt)
    assert not r5.was_cached


@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
def test_simple_pred_serialize(lm):
    pred = lm.predict(
        LmPrompt(
            "Here is a story. Once upon a",
            max_tokens=3,
            cache=False,
        ),
    )
    pred_dict = pred.dict_serialize()
    from pprint import pprint

    pprint(pred_dict)


@pytest.mark.skip(
    reason="Huggingface does not support completion_token_offsets currently",
)
@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
def test_token_offsets(lm):
    prompt = "A B C D E F G H"
    pred = lm.predict(
        LmPrompt(
            prompt,
            max_tokens=3,
            cache=False,
            temperature=0,
        ),
    )
    assert pred.completion_text == " I J K"
    assert pred.completion_tokens == [" I", " J", " K"]
    base_len = len(prompt)
    assert pred.completion_token_offsets == [base_len + 0, base_len + 2, base_len + 4]


@pytest.mark.parametrize("lm", ALL_MODELS_OLD, ids=get_model_name)
def test_token_seq_probs(lm):
    prompt = "A B C D E F G H"
    pred = lm.predict(
        LmPrompt(
            prompt,
            max_tokens=3,
            cache=False,
            temperature=0,
            logprobs=1,
        ),
    )
    is_chat = lm.model_name() not in [m.model_name() for m in COMPLETION_MODELS]
    if not is_chat:
        expected_tokens = [" I", " J", " K"]
    else:
        # When in chat the next turn doesn't start with a space
        expected_tokens = ["I", " J", " K"]
    print(f"{pred.completion_text=}")
    print(f"{expected_tokens=}")
    assert pred.completion_text == "".join(expected_tokens)
    assert pred.completion_tokens == expected_tokens
    top_probs = pred.top_token_logprobs
    assert len(top_probs) == 3
    for i, expected in enumerate(expected_tokens):
        assert isinstance(top_probs[i][expected], float)
        assert top_probs[i][expected] == pred.completion_logprobs[i]
        assert top_probs[i][expected] < 0


@pytest.mark.parametrize("lm", ECHOABLE_MODELS, ids=get_model_name)
def test_echo_many_toks(lm):
    out = lm.predict(
        LmPrompt(
            "Once upon a",
            max_tokens=7,
            logprobs=1,
            cache=False,
            echo=True,
            temperature=0,
        ),
    )
    assert len(out.full_logprobs) == len(out.get_full_tokens()) == 3 + 7
    print(out.get_full_tokens())
    print([math.exp(lp) for lp in out.full_logprobs])
    assert math.exp(out.full_logprobs[2]) > 0.98  # "a" after "Once upon" is high prob
    # Now try the non-echo version
    out2 = lm.predict(
        LmPrompt(
            "Once upon a",
            max_tokens=7,
            logprobs=1,
            cache=False,
            echo=False,
            temperature=0,
        ),
    )
    assert len(out2.completion_logprobs) == len(out2.completion_tokens) == 7
    assert np.allclose(
        np.exp(out2.completion_logprobs),
        np.exp(out.completion_logprobs),
        atol=0.001,
    )


@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
def test_predict_many(lm):
    pred = lm.predict_many(
        [
            LmPrompt(
                "A B C D E F G H",
                max_tokens=3,
                cache=False,
                temperature=0,
                logprobs=1,
            ),
            LmPrompt(
                "A B C D E F G H I J",
                max_tokens=3,
                cache=False,
                temperature=0,
                logprobs=1,
            ),
        ],
        completion_window=CompletionWindow.ASAP,
    )
    resps = list(pred)
    assert len(resps) == 2


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_predict_many_cached(lm):
    clear_cache_dir()
    prompts = [
        LmPrompt(
            t,
            max_tokens=3,
            cache=True,
            temperature=0,
            logprobs=1,
        )
        for t in ["A", "A B", "A B C"]
    ]
    preds = [lm.predict(prompt) for prompt in prompts]
    if hasattr(lm, "_api"):
        old_api = lm._api
        lm._api = None  # It's cached. No requests should be made
    else:
        old_api = None

    # Start the thread and set a timeout
    many = None

    try:

        def run_predict_many():
            nonlocal many
            now = time.time()
            many = lm.predict_many(
                prompts,
                completion_window=CompletionWindow.BATCH_ANY,
            )
            print("Delta", time.time() - now)

        thread = threading.Thread(target=run_predict_many)
        thread.start()
        thread.join(timeout=0.05)
        if thread.is_alive():
            pytest.fail("predict_many call timed out")
        else:
            resps = list(many)
            assert len(resps) == 3
            assert all(resp.was_cached for resp in resps)
            assert all(
                resp.completion_text == pred.completion_text
                for resp, pred in zip(resps, preds, strict=False)
            )
    finally:
        if old_api:
            lm._api = old_api


@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
def test_num_completions_two(lm):
    print(lm.get_model_cache_key())
    clear_cache_dir()
    prompt = LmPrompt(
        random_prompt,
        max_tokens=25,
        cache=False,
        temperature=1,
        logprobs=1,
        num_completions=2,
    )
    pred = lm.predict(prompt)
    assert len(pred) == 2
    assert isinstance(pred, list)
    assert 6 < pred[0].usage_output_tokens <= 25
    assert 6 < pred[1].usage_output_tokens <= 25
    assert pred[0].completion_text != pred[1].completion_text
    pred2 = lm.predict(prompt)
    assert pred[0].completion_text != pred2[0].completion_text
    assert pred[1].completion_text != pred2[1].completion_text


@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
def test_num_completions_one_list(lm):
    clear_cache_dir()
    prompt = LmPrompt(
        random_prompt,
        max_tokens=25,
        cache=False,
        temperature=1,
        logprobs=1,
        num_completions=1,
    )
    pred = lm.predict(prompt)
    assert isinstance(pred, list)
    assert len(pred) == 1
    assert 10 < pred[0].usage_output_tokens <= 25
    pred2 = lm.predict(prompt)
    assert pred[0].completion_text != pred2[0].completion_text


@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
def test_object_size_is_reasonable(lm):
    clear_cache_dir()
    num_actual_tokens = 400
    prompt = LmPrompt(
        "Write a 1000 word long detailed story about a dog:",
        max_tokens=num_actual_tokens,
        cache=False,
        temperature=0,
        logprobs=1,
    )
    pred = lm.predict(prompt)
    assert (
        pred.usage_output_tokens == num_actual_tokens
    ), f"got {pred.usage_output_tokens} tokens"
    if lm.supports_token_operations:
        prompt_tokens = lm.tokenize(prompt.text)
        assert 9 < len(prompt_tokens) < 20
        total_tokens = len(prompt_tokens) + len(pred.completion_tokens)
    else:
        total_tokens = 20 + num_actual_tokens
    # Optimal per token might be around 5 bytes (5 characters)
    # for the token text, maybe like 4 bytes for the logprob,
    # and maybe like 8 bytes for other random stuff. We will allow
    # a generous margin over this since hasn't been optimized
    # We'll also allow some prompt/other stuff static overhead
    acceptable_bytes_per_token = 17 * 10  # Ideally would like to get this down.
    #  The logprob choice objects are really big
    acceptable_static_overhead = 512
    acceptable_bytes = (
        total_tokens * acceptable_bytes_per_token + acceptable_static_overhead
    )
    used_bytes = len(pickle.dumps(pred))
    print(
        f"Used bytes: {used_bytes}, total tokens: {total_tokens}. Acceptable:"
        f" {acceptable_bytes}",
    )
    assert used_bytes < acceptable_bytes
    # Make sure the cache is reasonable size
    num_runs = 3

    def read_cache_size():
        d = cache_dir()
        return sum(f.stat().st_size for f in d.glob("**/*") if f.is_file())

    for i in range(num_runs):
        prompt = LmPrompt(
            f"Write a {1000 + i} word long detailed story about a dog:",
            max_tokens=num_actual_tokens,
            cache=True,
            temperature=0,
            logprobs=1,
        )
        pred = lm.predict(prompt)
        assert pred.usage_output_tokens == num_actual_tokens
    print("Cache size", read_cache_size())
    assert read_cache_size() < (acceptable_bytes * num_runs) * 2


@pytest.mark.parametrize("lm", CHAT_MODELS, ids=get_model_name)
def test_cast_convo(lm):
    pred = lm.predict(
        [
            "What is 2+2",
            "4",
            "What is 4+8",
            "12",
            "What is 3+5",
            "8",
            "What is 3+2",
        ],
    )
    assert pred.completion_text.strip() == "5"


@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
def test_num_completions_two_cached(lm):
    clear_cache_dir()
    prompt = LmPrompt(
        random_prompt,
        max_tokens=25,
        cache=True,
        temperature=1.0,
        logprobs=1,
        num_completions=2,
    )
    pred = lm.predict(prompt)
    assert len(pred) == 2
    assert isinstance(pred, list)
    assert all(not p.was_cached for p in pred)
    assert pred[0].completion_text != pred[1].completion_text
    print("pred 0", pred[0].completion_text)
    print("pred 1", pred[1].completion_text)
    pred2 = lm.predict(prompt)
    assert len(pred2) == 2
    assert all(p.was_cached for p in pred2)
    assert pred[0].completion_text == pred2[0].completion_text
    assert pred[1].completion_text == pred2[1].completion_text
    # Try with just a single completion
    prompt = dataclasses.replace(prompt, num_completions=None)
    pred3 = lm.predict(prompt)
    assert pred3.was_cached
    assert pred3.completion_text == pred2[0].completion_text == pred[0].completion_text
    # Try predict an extra one. The first 2 should be cached
    prompt = dataclasses.replace(prompt, num_completions=3)
    pred4 = lm.predict(prompt)
    assert pred4[0].was_cached
    assert pred4[1].was_cached
    assert not pred4[2].was_cached
    assert pred4[0].completion_text == pred[0].completion_text
    assert pred4[1].completion_text == pred[1].completion_text
    assert pred4[2].completion_text != pred[0].completion_text
    assert pred4[2].completion_text != pred[1].completion_text


@pytest.mark.parametrize("lm", CHAT_MODELS, ids=get_model_name)
def test_system_prompt(lm: LmPredictor):
    pred = lm.predict(
        LmPrompt(
            [
                LmChatTurn(
                    ChatGptRoles.system, 
                    "Always answer with only a single word in all capital letters."
                ),
                LmChatTurn(ChatGptRoles.user, "what is the capital of france"),
            ],
            max_tokens=10 if not lm.reasoning_style == LmReasoningStyle.ALWAYS_THINK else 1000,
            cache=False,
        ),
    )
    print("completion text")
    print(pred.completion_text)
    assert pred.completion_text.strip() == "PARIS"


@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
def test_metadata_predict(lm):
    # Test with string metadata
    prompt = LmPrompt(
        "Count from 1 to 5:",
        max_tokens=10,
        cache=False,
        temperature=0,
        metadata="test_metadata_string"
    )
    pred = lm.predict(prompt)
    assert pred.prompt.metadata == "test_metadata_string"
    
    # Test with dictionary metadata
    metadata_dict = {"label": "counting", "id": 12345}
    prompt = LmPrompt(
        "Count from 1 to 5:",
        max_tokens=10,
        cache=False,
        temperature=0,
        metadata=metadata_dict
    )
    pred = lm.predict(prompt)
    assert pred.prompt.metadata == metadata_dict
    assert pred.prompt.metadata["label"] == "counting"
    assert pred.prompt.metadata["id"] == 12345


@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
def test_metadata_predict_many(lm):
    # Create prompts with different metadata
    prompts = [
        LmPrompt(
            f"What is {i} + {i}?",
            max_tokens=5,
            cache=False,
            temperature=0,
            metadata={"question_id": i, "expected_answer": i*2}
        )
        for i in range(1, 4)
    ]
    
    # Use predict_many and verify metadata is preserved
    results = list(lm.predict_many(prompts, CompletionWindow.ASAP))
    
    for i, result in enumerate(results):
        expected_id = i + 1
        expected_answer = expected_id * 2
        assert result.prompt.metadata["question_id"] == expected_id
        assert result.prompt.metadata["expected_answer"] == expected_answer


@pytest.mark.parametrize("lm", ALL_MODELS, ids=get_model_name)
def test_metadata_with_cache_hit(lm: LmPredictor):
    """Test that metadata is preserved when there's a cache hit."""
    # First prompt with metadata to populate the cache
    prompt1 = LmPrompt(
        "What is 5 + 5?",
        max_tokens=5,
        cache=True,
        temperature=0,
        metadata={"original": True, "id": 1}
    )
    result1 = lm.predict(prompt1)
    
    # Same prompt with different metadata should use cache but keep the new metadata
    prompt2 = LmPrompt(
        "What is 5 + 5?",
        max_tokens=5,
        cache=True,
        temperature=0,
        metadata={"original": False, "id": 2}
    )
    result2 = lm.predict(prompt2)
    
    # Check that the result is from cache but has the new metadata
    assert result2.was_cached
    assert result2.prompt.metadata["original"] == False
    assert result2.prompt.metadata["id"] == 2


@pytest.mark.parametrize(
    "lm", CHAT_MODELS,
    ids=get_model_name
)
def test_prefilled_prompt(lm):
    if not lm.supports_prefilled_chat:
        pytest.skip("Model does not support prefilled chat")
    pred = lm.predict(LmPrompt([
        "What is the capital of France?",
        "Oui oui, the capital of France is Paris. The most famous landmark is the Eiffel",
    ], max_tokens=1, cache=False))
    assert pred.completion_text.strip() == "Tower"


@pytest.mark.parametrize("lm", CHAT_MODELS, ids=get_model_name)
def test_make_reply_prompt(lm):
    # Initial conversation
    # Skip temperature for ALWAYS_THINK models as they don't support it
    # ALWAYS_THINK models need more tokens for reasoning
    prompt_kwargs = {
        "max_tokens": 1000 if lm.reasoning_style == LmReasoningStyle.ALWAYS_THINK else 10,
        "cache": False
    }
    if lm.reasoning_style != LmReasoningStyle.ALWAYS_THINK:
        prompt_kwargs["temperature"] = 0
    initial_prompt = LmPrompt(
        "What is the capital of France?", **prompt_kwargs)
    pred = lm.predict(initial_prompt)
    assert "paris" in pred.completion_text.strip().lower()
    new_prompt = pred.make_reply_prompt(
        "What is its most famous landmark?",
        max_tokens=100,
    )
    assert new_prompt.text == LmChatDialog([
        LmChatTurn(ChatGptRoles.user, "What is the capital of France?"),
        LmChatTurn(ChatGptRoles.assistant, pred.completion_text),
        LmChatTurn(ChatGptRoles.user, "What is its most famous landmark?"),
    ])
    new_prompt = pred.make_reply_prompt(
        [
            "What is its most famous landmark?",
            "The Eiffel Tower",
            "How tall is it?",
        ],
        max_tokens=100,
    )
    assert new_prompt.text == LmChatDialog([
        LmChatTurn(ChatGptRoles.user, "What is the capital of France?"),
        LmChatTurn(ChatGptRoles.assistant, pred.completion_text),
        LmChatTurn(ChatGptRoles.user, "What is its most famous landmark?"),
        LmChatTurn(ChatGptRoles.assistant, "The Eiffel Tower"),
        LmChatTurn(ChatGptRoles.user, "How tall is it?"),
    ])
