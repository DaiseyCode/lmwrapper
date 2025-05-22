import dataclasses
import os
from test.test_huggingface import BIG_MODELS, Models

import numpy as np
import pytest
import torch

from lmwrapper.huggingface_wrapper import get_huggingface_lm
from lmwrapper.internals import ModelInternalsRequest
from lmwrapper.structs import LmPrompt

IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

HAS_CUDA = torch.cuda.is_available()


@pytest.mark.parametrize(
    "model_name_layers_hidden",
    [
        (Models.GPT2, 12, 768),
        (Models.CodeGen_350M, 20, 64 * 16),
        (Models.CodeGen2_1B, 16, 2048),
        # ^ Important to run since it doesn't use the same attentions value
    ],
)
def test_get_internals_hidden_states(pytestconfig, model_name_layers_hidden):
    model_name, num_layers, hidden_size = model_name_layers_hidden
    is_run_slow = pytestconfig.getoption("--runslow")
    if IS_GITHUB_ACTIONS and model_name != Models.GPT2:
        pytest.skip(
            "We are only going to test small here. "
            "The other models are either large or use a similar attention pattern.",
        )
    if IS_GITHUB_ACTIONS and model_name in BIG_MODELS:
        pytest.skip("skip big models in github actions")
    model = get_huggingface_lm(model_name, trust_remote_code=True)
    num_gen_tokens = 2
    prompt = LmPrompt(
        text="The capital Germany is Berlin. The capital of France",
        max_tokens=num_gen_tokens,
        temperature=0,
        model_internals_request=ModelInternalsRequest(
            return_hidden_states=True,
        ),
    )
    prompt_tokens = model.tokenize(prompt.text)
    num_expected_tokens = len(prompt_tokens) + num_gen_tokens
    assert num_expected_tokens == 12
    pred = model.predict(prompt)

    def check_vals(pred, prompt):
        assert pred.internals.hidden_states is not None
        assert isinstance(pred.internals.hidden_states, tuple)
        assert (
            len(pred.internals.hidden_states) == 1 + num_layers
        )  # has embedding layer
        assert all(isinstance(x, np.ndarray) for x in pred.internals.hidden_states)
        assert all(
            x.shape == (num_expected_tokens, hidden_size)
            for x in pred.internals.hidden_states
        )
        assert pred.internals.attentions is None
        print(pred)

        # Try with a selection of indexes
        prompt = dataclasses.replace(
            prompt,
            model_internals_request=ModelInternalsRequest(
                return_hidden_states=True,
                hidden_layer_indexes=[0, 5, -1],
            ),
        )
        new_pred = model.predict(prompt)
        new_hidden = new_pred.internals.hidden_states
        assert len(new_hidden) == 3
        assert np.allclose(new_hidden[0], pred.internals.hidden_states[0])
        assert np.allclose(new_hidden[1], pred.internals.hidden_states[5])
        assert np.allclose(new_hidden[2], pred.internals.hidden_states[-1])

        # Try with fracs
        prompt = dataclasses.replace(
            prompt,
            model_internals_request=ModelInternalsRequest(
                return_hidden_states=True,
                hidden_layer_fractions=[0.0, 0.5, 1.0],
            ),
        )
        new_pred = model.predict(prompt)
        new_hidden = new_pred.internals.hidden_states
        assert len(new_hidden) == 3
        assert np.allclose(new_hidden[0], pred.internals.hidden_states[0])
        assert np.allclose(
            new_hidden[1],
            pred.internals.hidden_states[round((1 + num_layers) / 2)],
        )
        assert np.allclose(new_hidden[2], pred.internals.hidden_states[-1])

    check_vals(pred, prompt)

    ### Attentions
    # (we are packing a lot into one test to still parameterize but no reload model)
    prompt = dataclasses.replace(
        prompt,
        model_internals_request=ModelInternalsRequest(
            return_hidden_states=True,
            return_attentions=True,
        ),
    )
    pred = model.predict(prompt)
    assert pred.internals.attentions is not None
    attn_layer, attn_head, attn_seq, attn_seq2 = pred.internals.attentions.shape
    assert attn_layer == num_layers
    assert attn_seq == num_expected_tokens == attn_seq2
    assert pred.internals.attentions[0, 0, 0, 0] == 1.0
    assert pred.internals.attentions[0, 0, 0, 1] == 0.0

    assert pred.internals.has_a_bos

    pred_sub = model.predict(
        dataclasses.replace(
            prompt,
            model_internals_request=ModelInternalsRequest(
                return_hidden_states=True,
                return_attentions=True,
                hidden_layer_indexes=[0, 5, -1],
            ),
        ),
    )
    assert pred_sub.internals.attentions is not None
    attn_layer, attn_head, attn_seq, attn_seq2 = pred_sub.internals.attentions.shape
    assert attn_layer == 3
    assert attn_seq == num_expected_tokens == attn_seq2
    assert pred_sub.internals.attentions[0, 0, 0, 0] == 1.0
    assert pred_sub.internals.attentions[0, 0, 0, 1] == 0.0
    assert np.allclose(pred_sub.internals.attentions[0], pred.internals.attentions[0])
    assert np.allclose(pred_sub.internals.attentions[1], pred.internals.attentions[5])
    assert np.allclose(pred_sub.internals.attentions[2], pred.internals.attentions[-1])

    ### With logprobs
    prompt = LmPrompt(
        text="The capital Germany is Berlin. The capital of France",
        max_tokens=num_gen_tokens,
        temperature=0,
        echo=True,
        logprobs=1,
        model_internals_request=ModelInternalsRequest(
            return_hidden_states=True,
        ),
    )
    prompt_tokens = model.tokenize(prompt.text)
    num_expected_tokens = len(prompt_tokens) + num_gen_tokens
    assert num_expected_tokens == 12
    pred = model.predict(prompt)
    check_vals(pred, prompt)
    assert len(pred.full_logprobs) == num_expected_tokens
    assert len(pred.completion_logprobs) == num_gen_tokens

    prompt = LmPrompt(
        text="The capital Germany is Berlin. The capital of France",
        max_tokens=num_gen_tokens,
        temperature=0,
        echo=False,
        logprobs=1,
        model_internals_request=ModelInternalsRequest(
            return_hidden_states=True,
        ),
    )
    pred = model.predict(prompt)
    check_vals(pred, prompt)
    assert len(pred.completion_logprobs) == num_gen_tokens
