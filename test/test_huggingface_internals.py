from lmwrapper.huggingface_wrapper import get_huggingface_lm
import pytest
import numpy as np
from lmwrapper.structs import LmPrompt
from lmwrapper.interals import ModelInternalsRequest
from test.test_huggingface import Models, BIG_MODELS
import os

IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

@pytest.mark.parametrize(
    "model_name_layers_hidden", [
        (Models.GPT2, 1 + 12, 768),
        (Models.CodeGen_350M, 1 + 20, 64 * 16),
        (Models.CodeGen2_1B, 1 + 16, 2048),
        # ^ Important to run since it doesn't use the same attentions value
    ]
)
def test_get_internals_hidden_states(pytestconfig, model_name_layers_hidden):
    model_name, num_layers, hidden_size = model_name_layers_hidden
    is_run_slow = pytestconfig.getoption("--runslow")
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
    assert pred.internals.hidden_states is not None
    assert isinstance(pred.internals.hidden_states, tuple)
    assert len(pred.internals.hidden_states) == num_layers
    assert all(isinstance(x, np.ndarray) for x in pred.internals.hidden_states)
    assert all(x.shape == (
        num_expected_tokens, hidden_size)
        for x in pred.internals.hidden_states
    )
    assert pred.internals.attentions is None
    print(pred)