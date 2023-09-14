from lmwrapper.prompt_trimming import HfTokenTrimmer
from lmwrapper.util import StrEnum

import pytest
import torch
import numpy as np

from lmwrapper.huggingface_wrapper import Runtime, get_huggingface_lm
from lmwrapper.structs import LmPrompt


class Models(StrEnum):
    CodeT5plus_220M = "Salesforce/codet5p-220m"
    CodeT5plus_6B = "Salesforce/codet5p-6b"
    CodeGen2_1B = "Salesforce/codegen2-1B"
    CodeGen2_3_7B = "Salesforce/codegen2-3_7B"
    InstructCodeT5plus_16B = "Salesforce/instructcodet5p-16b"

    DistilGPT2 = "distilgpt2"
    GPT2 = "gpt2"


CUDA_UNAVAILABLE = not torch.cuda.is_available()
SMALL_GPU = CUDA_UNAVAILABLE or torch.cuda.mem_get_info()[0] < 17_179_869_184  # 16GB

SEQ2SEQ_MODELS = {Models.CodeT5plus_220M}
CAUSAL_MODELS = {Models.DistilGPT2, Models.GPT2, Models.CodeGen2_1B}
BIG_SEQ2SEQ_MODELS = {Models.CodeT5plus_6B, Models.InstructCodeT5plus_16B}
BIG_CAUSAL_MODELS = {Models.CodeGen2_3_7B}
BIG_MODELS = BIG_SEQ2SEQ_MODELS | BIG_CAUSAL_MODELS
ALL_MODELS = SEQ2SEQ_MODELS | CAUSAL_MODELS | BIG_MODELS


def test_trim_start():
    lm = get_huggingface_lm(Models.CodeGen2_1B, runtime=Runtime.PYTORCH)
    ltrimmer = HfTokenTrimmer(2, lm._tokenizer, start_from_left_side=True)

    lm.prompt_trimmer = ltrimmer
    prompt = LmPrompt(
        "def hello_world():\n   print('",
        max_tokens=1,
        cache=False,
        temperature=0,
    )
    lm._model.config.max_length = 3
    out = lm.predict(prompt)
    assert out._tokens[0] == "('"
    assert out._tokens[1] == "ob"


def test_logprobs_codegen2():
    lm = get_huggingface_lm(Models.CodeGen2_1B, runtime=Runtime.PYTORCH)
    prompt = LmPrompt(
        "def hello_world():\n   print('",
        max_tokens=15,
        cache=False,
        temperature=0,
    )
    outa = lm.predict(prompt)
    logprobs_a = np.array(outa.completion_logprobs)

    lm = get_huggingface_lm(Models.CodeGen2_1B, runtime=Runtime.PYTORCH)
    prompt = LmPrompt(
        "def hello_world():\n   print('",
        max_tokens=15,
        cache=False,
        temperature=0,
        echo=True
    )
    outb = lm.predict(prompt)
    logprobs_b = np.array(outb.completion_logprobs)

    assert np.allclose(logprobs_a, logprobs_b, atol=0.001, rtol=0.001)


def test_logprobs_stop_codegen2():
    lm = get_huggingface_lm(Models.CodeGen2_1B, runtime=Runtime.PYTORCH)
    prompt = LmPrompt(
        "place a newline here", max_tokens=5, cache=False, temperature=0, stop=["(o(o"]
    )

    out_a = lm.predict(prompt)
    logprobs_a = np.array(out_a.completion_logprobs)
    assert len(logprobs_a) == 1
    assert len(out_a.logprobs_dict) == 1
    assert "(o(o" not in out_a.completion_text

    assert out_a.logprobs_dict == [
        {
            "token": 78,
            "repr": "'o'",
            "logit": pytest.approx(-2.742025852203369, rel=0.001),
            "probability": pytest.approx(0.06443966925144196, rel=0.001),
        }
    ]

    lm = get_huggingface_lm(Models.CodeGen2_1B, runtime=Runtime.PYTORCH)
    prompt = LmPrompt(
        "place a newline here",
        max_tokens=5,
        cache=False,
        temperature=0,
        stop=["(o(o"],
        echo=True,
    )
    out_b = lm.predict(prompt)
    logprobs_b = np.array(out_b.completion_logprobs)
    assert "(o(o" not in out_b.completion_text

    #assert {
    #        "token": 78,
    #        "repr": "'o'",
    #        "logit": pytest.approx(-2.742025852203369, rel=0.001),
    #        "probability": pytest.approx(0.06443966925144196, rel=0.001),
    #    } in out_b.logprobs_dict # TODO: assert that the prompt is correct


    assert np.allclose(logprobs_a, logprobs_b, atol=0.001, rtol=0.001)


def test_stop_token_removal():
    prompt_str = """Please list the capitals of the following countries

1. Germany
2. USA
3. France
4. Mexico"""
    # Load model
    lm = get_huggingface_lm(Models.DistilGPT2, runtime=Runtime.PYTORCH)

    # Let's make sure we get a stop token in our prompt normally
    prompt = LmPrompt(
        prompt_str,
        max_tokens=15,
        cache=False,
        temperature=0,
    )
    out = lm.predict(prompt)
    assert "Italy" in out.completion_text
    assert out.logprobs_dict == [
        {
            "token": 198,
            "repr": "'\\n'",
            "logit": pytest.approx(-0.059126678854227066, rel=0.001),
            "probability": pytest.approx(0.9425873756408691, rel=0.001),
        },
        {
            "token": 20,
            "repr": "'5'",
            "logit": pytest.approx(-0.011661103926599026, rel=0.001),
            "probability": pytest.approx(0.9884065985679626, rel=0.001),
        },
        {
            "token": 13,
            "repr": "'.'",
            "logit": pytest.approx(-0.0012642494402825832, rel=0.001),
            "probability": pytest.approx(0.998736560344696, rel=0.001),
        },
        {
            "token": 4486,
            "repr": "' Germany'",
            "logit": pytest.approx(-2.1969234943389893, rel=0.001),
            "probability": pytest.approx(0.11114457249641418, rel=0.001),
        },
        {
            "token": 198,
            "repr": "'\\n'",
            "logit": pytest.approx(-0.054811663925647736, rel=0.001),
            "probability": pytest.approx(0.9466634392738342, rel=0.001),
        },
        {
            "token": 21,
            "repr": "'6'",
            "logit": pytest.approx(-0.025344248861074448, rel=0.001),
            "probability": pytest.approx(0.9749742150306702, rel=0.001),
        },
        {
            "token": 13,
            "repr": "'.'",
            "logit": pytest.approx(-0.0013435394503176212, rel=0.001),
            "probability": pytest.approx(0.9986573457717896, rel=0.001),
        },
        {
            "token": 8031,
            "repr": "' Italy'",
            "logit": pytest.approx(-2.757378101348877, rel=0.001),
            "probability": pytest.approx(0.0634579285979271, rel=0.001),
        },
        {
            "token": 198,
            "repr": "'\\n'",
            "logit": pytest.approx(-0.0332401879131794, rel=0.001),
            "probability": pytest.approx(0.9673061966896057, rel=0.001),
        },
        {
            "token": 22,
            "repr": "'7'",
            "logit": pytest.approx(-0.017078006640076637, rel=0.001),
            "probability": pytest.approx(0.983066976070404, rel=0.001),
        },
        {
            "token": 13,
            "repr": "'.'",
            "logit": pytest.approx(-0.001742750871926546, rel=0.001),
            "probability": pytest.approx(0.9982587695121765, rel=0.001),
        },
        {
            "token": 2869,
            "repr": "' Japan'",
            "logit": pytest.approx(-2.284379005432129, rel=0.001),
            "probability": pytest.approx(0.10183728486299515, rel=0.001),
        },
        {
            "token": 198,
            "repr": "'\\n'",
            "logit": pytest.approx(-0.040456708520650864, rel=0.001),
            "probability": pytest.approx(0.960350751876831, rel=0.001),
        },
        {
            "token": 23,
            "repr": "'8'",
            "logit": pytest.approx(-0.02017313987016678, rel=0.001),
            "probability": pytest.approx(0.9800289869308472, rel=0.001),
        },
        {
            "token": 13,
            "repr": "'.'",
            "logit": pytest.approx(-0.0018331881146878004, rel=0.001),
            "probability": pytest.approx(0.9981684684753418, rel=0.001),
        },
    ]

    prompt = LmPrompt(
        prompt_str, max_tokens=15, cache=False, temperature=0, stop=["Italy"]
    )
    out = lm.predict(prompt)
    assert "Italy" not in out.completion_text
    assert out.logprobs_dict == [
        {
            "token": 198,
            "repr": "'\\n'",
            "logit": pytest.approx(-0.059126678854227066, rel=0.001),
            "probability": pytest.approx(0.9425873756408691, rel=0.001),
        },
        {
            "token": 20,
            "repr": "'5'",
            "logit": pytest.approx(-0.011661103926599026, rel=0.001),
            "probability": pytest.approx(0.9884065985679626, rel=0.001),
        },
        {
            "token": 13,
            "repr": "'.'",
            "logit": pytest.approx(-0.0012642494402825832, rel=0.001),
            "probability": pytest.approx(0.998736560344696, rel=0.001),
        },
        {
            "token": 4486,
            "repr": "' Germany'",
            "logit": pytest.approx(-2.1969234943389893, rel=0.001),
            "probability": pytest.approx(0.11114457249641418, rel=0.001),
        },
        {
            "token": 198,
            "repr": "'\\n'",
            "logit": pytest.approx(-0.054811663925647736, rel=0.001),
            "probability": pytest.approx(0.9466634392738342, rel=0.001),
        },
        {
            "token": 21,
            "repr": "'6'",
            "logit": pytest.approx(-0.025344248861074448, rel=0.001),
            "probability": pytest.approx(0.9749742150306702, rel=0.001),
        },
        {
            "token": 13,
            "repr": "'.'",
            "logit": pytest.approx(-0.0013435394503176212, rel=0.001),
            "probability": pytest.approx(0.9986573457717896, rel=0.001),
        },
        {
            "token": 8031,
            "repr": "' Italy'",
            "logit": pytest.approx(-2.757378101348877, rel=0.001),
            "probability": pytest.approx(0.0634579285979271, rel=0.001),
        },
    ]

    prompt_str = """Repeat the following document \"Bob said 'I like to eat candy' and then dove into the pile of candy\""""

    # Let's make sure we get a stop token in our prompt normally
    prompt = LmPrompt(
        prompt_str,
        max_tokens=300,
        cache=False,
        temperature=0,
    )
    out = lm.predict(prompt)
    assert "I like to eat candy" in out.completion_text

    # Let's make sure we get a stop token in our prompt normally
    prompt = LmPrompt(
        prompt_str,
        max_tokens=300,
        cache=False,
        temperature=0,
        stop=["I like to eat candy"],
    )
    out = lm.predict(prompt)
    assert "I like to eat candy" not in out.completion_text


def test_stop_tokens():
    # Load model
    lm = get_huggingface_lm(Models.DistilGPT2, runtime=Runtime.PYTORCH)

    # Let's make sure we get a stop token in our prompt normally
    prompt = LmPrompt(
        "place a newline here",
        max_tokens=15,
        cache=False,
        temperature=0,
    )
    out = lm.predict(prompt)
    assert "\n\n" in out.completion_text

    # Now let's try with one character
    prompt = LmPrompt(
        "place a newline here",
        max_tokens=15,
        cache=False,
        temperature=0,
        stop=["\n"],
    )
    out = lm.predict(prompt)
    assert "\n\n" not in out.completion_text

    # Now with two
    prompt = LmPrompt(
        "place a newline here",
        max_tokens=15,
        cache=False,
        temperature=0,
        stop=["\n\n"],
    )
    out = lm.predict(prompt)
    assert "\n\n\n" not in out.completion_text

    # Now let's try with a sequence longer than the input
    prompt = LmPrompt(
        "place a newline here",
        max_tokens=15,
        cache=False,
        temperature=0,
        stop=["\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"],
    )
    out = lm.predict(prompt)
    assert "\n\n\n" in out.completion_text

    # Now let's try multiple
    prompt = LmPrompt(
        "place a newline here",
        max_tokens=15,
        cache=False,
        temperature=0,
        stop=["\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n", "blah", "\n"],
    )
    out = lm.predict(prompt)
    assert "\n\n\n" not in out.completion_text


def test_distilgpt2_pytorch_runtime():
    prompt = LmPrompt(
        "print('Hello world",
        max_tokens=15,
        cache=False,
        temperature=0,
    )
    lm = get_huggingface_lm(Models.DistilGPT2, runtime=Runtime.PYTORCH)
    out = lm.predict(prompt)
    assert out.completion_text


@pytest.mark.slow()
@pytest.mark.parametrize("lm", ALL_MODELS)
def test_all_pytorch_runtime(lm: str):
    if SMALL_GPU and lm in BIG_MODELS:
        pytest.skip(
            f"Skipped model '{lm}' as model too large for available GPU memory."
        )
    prompt = LmPrompt(
        "print('Hello world",
        max_tokens=15,
        cache=False,
        temperature=0,
    )
    lm = get_huggingface_lm(lm, runtime=Runtime.PYTORCH)
    out = lm.predict(prompt)
    assert out.completion_text


@pytest.mark.slow()
@pytest.mark.skip()  # ORT is not ready yet
@pytest.mark.parametrize("runtime", [Runtime.ORT_CPU, Runtime.ORT_CUDA])
@pytest.mark.parametrize("lm", ALL_MODELS)
def test_get_ort(runtime: Runtime, lm: str):
    prompt = LmPrompt(
        "print('Hello world",
        max_tokens=1,
        cache=False,
        temperature=0,
    )
    lm = get_huggingface_lm(lm, runtime=runtime)
    out = lm.predict(prompt)
    assert out.completion_text


@pytest.mark.slow()
@pytest.mark.skip()  # Better Transformer is not ready yet
@pytest.mark.parametrize("lm", [Models.DistilGPT2, Models.GPT2])
def test_get_better_transformer(lm):
    prompt = LmPrompt(
        "print('Hello world",
        max_tokens=1,
        cache=False,
        temperature=0,
    )
    lm = get_huggingface_lm(lm, runtime=Runtime.BETTER_TRANSFORMER)
    out = lm.predict(prompt)
    assert out.completion_text


@pytest.mark.slow()
@pytest.mark.skip()  # Better Transformer is not ready yet
def test_codegen2_predict_bt():
    lm = Models.CodeGen2_1B
    with pytest.raises(Exception) as e_info:
        get_huggingface_lm(lm, runtime=Runtime.BETTER_TRANSFORMER)
        assert str(e_info.value).startswith("WARNING BetterTransformer")


@pytest.mark.slow()
@pytest.mark.skip()  # TensorRT is not ready yet
@pytest.mark.parametrize("lm", CAUSAL_MODELS)
@pytest.mark.skipif(
    CUDA_UNAVAILABLE,
    reason="Cannot test ORT/ONNX CUDA runtime without CUDA",
)
def test_get_tensorrt(lm: str):
    get_huggingface_lm(lm, runtime=Runtime.ORT_TENSORRT)
