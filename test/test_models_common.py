import dataclasses
import math
import pickle
import threading
import time

import numpy as np
import pytest

from lmwrapper.batch_config import CompletionWindow
from lmwrapper.caching import cache_dir, clear_cache_dir
from lmwrapper.huggingface_wrapper.wrapper import get_huggingface_lm
from lmwrapper.openai_wrapper.wrapper import OpenAiModelNames, get_open_ai_lm
from lmwrapper.structs import LmPrompt

ALL_MODELS = [
    get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo_instruct),
    get_huggingface_lm("gpt2"),
    get_open_ai_lm(OpenAiModelNames.gpt_4o_mini),
]

ALL_MODELS_OLD = [
    *ALL_MODELS[:2],
    get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo),
]

COMPLETION_MODELS = ALL_MODELS[:-1]

CHAT_MODELS = [ALL_MODELS[2]]


ECHOABLE_MODELS = [
    # get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo_instruct),
    # Won't work with now that echo disabled
    get_huggingface_lm("gpt2"),
]


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_simple_pred(lm):
    out = lm.predict(
        LmPrompt(
            "Give a one word completion: 'Here is a story. Once upon a",
            max_tokens=1,
            cache=False,
        ),
    )
    assert out.completion_text.strip() == "time"


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_simple_pred_lp(lm):
    out = lm.predict(
        LmPrompt(
            "Give a one word completion: 'Here is a story. Once upon a",
            max_tokens=1,
            logprobs=1,
            cache=False,
            echo=False,
        ),
    )
    assert out.completion_text.strip() == "time"
    print(out)
    assert lm.remove_special_chars_from_tokens(out.completion_tokens) in (
        [" time"],
        ["time"],
    )
    assert len(out.completion_logprobs) == 1
    assert math.exp(out.completion_logprobs[0]) >= 0.85


@pytest.mark.parametrize("lm", ALL_MODELS)
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


@pytest.mark.parametrize("lm", ECHOABLE_MODELS)
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


@pytest.mark.parametrize("lm", ECHOABLE_MODELS)
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


@pytest.mark.parametrize("lm", ECHOABLE_MODELS)
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


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_many_gen(lm):
    val = lm.predict(
        LmPrompt(
            "Write a story about a pirate:",
            max_tokens=5,
            logprobs=1,
            cache=False,
        ),
    )
    assert len(val.completion_tokens) == 5


@pytest.mark.parametrize("lm", ALL_MODELS)
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


@pytest.mark.parametrize("lm", ALL_MODELS)
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


@pytest.mark.parametrize("lm", COMPLETION_MODELS)
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


@pytest.mark.parametrize("lm", COMPLETION_MODELS)
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


@pytest.mark.parametrize("lm", COMPLETION_MODELS)
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


@pytest.mark.parametrize("lm", COMPLETION_MODELS)
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


@pytest.mark.parametrize("lm", COMPLETION_MODELS)
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


@pytest.mark.parametrize("lm", COMPLETION_MODELS)
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


@pytest.mark.parametrize("lm", COMPLETION_MODELS)
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


@pytest.mark.parametrize("lm", COMPLETION_MODELS)
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


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_remove_prompt_from_cache(lm):
    prompt = LmPrompt(
        "Give a random base-64 guid:",
        max_tokens=100,
        temperature=2.0,
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


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_none_max_tokens(lm):
    prompt = LmPrompt(
        "Write a long and detailed story (multiple paragraphs) about a dog:",
        max_tokens=None,
        temperature=0.2,
        cache=False,
    )
    result = lm.predict(prompt)
    assert len(result.completion_tokens) == lm.default_tokens_generated


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


@pytest.mark.parametrize("lm", ALL_MODELS_OLD)
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


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_was_cached_marking(lm):
    prompt = LmPrompt(
        "Give a random base-64 guid:",
        max_tokens=10,
        temperature=2.0,
        cache=False,
    )
    r1 = lm.predict(prompt)
    assert not r1.was_cached
    r2 = lm.predict(prompt)
    assert not r2.was_cached
    prompt = LmPrompt(
        "Give a random base-64 guid:",
        max_tokens=100,
        temperature=2.0,
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


@pytest.mark.parametrize("lm", ALL_MODELS)
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
@pytest.mark.parametrize("lm", ALL_MODELS)
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


@pytest.mark.parametrize("lm", ALL_MODELS_OLD)
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


@pytest.mark.parametrize("lm", ECHOABLE_MODELS)
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


@pytest.mark.parametrize("lm", ALL_MODELS)
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


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_num_completions_two(lm):
    clear_cache_dir()
    prompt = LmPrompt(
        "Make up a random guid:",
        max_tokens=15,
        cache=False,
        temperature=1,
        logprobs=1,
        num_completions=2,
    )
    pred = lm.predict(prompt)
    assert len(pred) == 2
    assert isinstance(pred, list)
    assert 10 < len(pred[0].completion_tokens) <= 20
    assert 10 < len(pred[1].completion_tokens) <= 20
    assert pred[0].completion_text != pred[1].completion_text
    pred2 = lm.predict(prompt)
    assert pred[0].completion_text != pred2[0].completion_text
    assert pred[1].completion_text != pred2[1].completion_text


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_num_completions_one_list(lm):
    clear_cache_dir()
    prompt = LmPrompt(
        "Make up a random guid:",
        max_tokens=15,
        cache=False,
        temperature=1,
        logprobs=1,
        num_completions=1,
    )
    pred = lm.predict(prompt)
    assert isinstance(pred, list)
    assert len(pred) == 1
    assert 10 < len(pred[0].completion_tokens) <= 20
    pred2 = lm.predict(prompt)
    assert pred[0].completion_text != pred2[0].completion_text


@pytest.mark.parametrize("lm", ALL_MODELS)
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
        len(pred.completion_tokens) == num_actual_tokens
    ), f"got {len(pred.completion_tokens)} tokens"
    prompt_tokens = lm.tokenize(prompt.text)
    assert 9 < len(prompt_tokens) < 20
    total_tokens = len(prompt_tokens) + len(pred.completion_tokens)
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
        assert len(pred.completion_tokens) == num_actual_tokens
    print("Cache size", read_cache_size())
    assert read_cache_size() < (acceptable_bytes * num_runs) * 2


@pytest.mark.parametrize("lm", CHAT_MODELS)
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


@pytest.mark.parametrize("lm", ALL_MODELS)
def test_num_completions_two_cached(lm):
    clear_cache_dir()
    prompt = LmPrompt(
        "Make up a random guid:",
        max_tokens=20,
        cache=True,
        temperature=1.5,
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
