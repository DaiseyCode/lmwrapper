from lmwrapper.runtime import Runtime
from lmwrapper.prompt_trimming import HfTokenTrimmer
from lmwrapper.utils import StrEnum

import pytest
import torch
import numpy as np

from lmwrapper.huggingface_wrapper import get_huggingface_lm
from lmwrapper.structs import LmPrompt


class Models(StrEnum):
    CodeT5plus_220M = "Salesforce/codet5p-220m"
    CodeT5plus_6B = "Salesforce/codet5p-6b"
    CodeGen2_1B = "Salesforce/codegen2-1B"
    CodeGen2_3_7B = "Salesforce/codegen2-3_7B"
    InstructCodeT5plus_16B = "Salesforce/instructcodet5p-16b"
    CodeLLama_7B = "codellama/CodeLlama-7b-hf"
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


@pytest.mark.skip()
def test_code_llama():
    prompt = LmPrompt(
        "print('Hello world",
        max_tokens=15,
        cache=False,
        temperature=0,
    )
    lm = get_huggingface_lm(
        Models.CodeLLama_7B,
        runtime=Runtime.PYTORCH,
        trust_remote_code=True,
        precision=torch.float16,
    )
    out = lm.predict(prompt)
    assert out.completion_text


@pytest.mark.slow()
def test_trim_start():
    lm = get_huggingface_lm(
        Models.CodeGen2_1B, runtime=Runtime.PYTORCH, trust_remote_code=True
    )
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
    assert out._tokens[1] == "\\"


@pytest.mark.slow()
def test_logprobs_codegen2():
    lm = get_huggingface_lm(
        Models.CodeGen2_1B,
        allow_patch_model_forward=False,
        runtime=Runtime.PYTORCH,
        trust_remote_code=True,
    )
    prompt = LmPrompt(
        "def hello_world():\n   print('",
        max_tokens=15,
        cache=False,
        temperature=0,
    )
    outa = lm.predict(prompt)
    logprobs_a = np.array(outa.completion_logprobs)

    lm = get_huggingface_lm(
        Models.CodeGen2_1B,
        allow_patch_model_forward=True,
        runtime=Runtime.PYTORCH,
        trust_remote_code=True,
    )
    prompt = LmPrompt(
        "def hello_world():\n   print('",
        max_tokens=15,
        cache=False,
        temperature=0,
        echo=True,
    )
    outb = lm.predict(prompt)
    logprobs_b = np.array(outb.completion_logprobs)

    assert np.allclose(logprobs_a, logprobs_b, atol=0.001, rtol=0.001)


@pytest.mark.slow()
def test_stop_n_codet5():
    lm = get_huggingface_lm(Models.CodeT5plus_220M, runtime=Runtime.PYTORCH)
    no_logprobs_prompt = LmPrompt(
        text="def hello_world():",
        max_tokens=50,
        logprobs=0,
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        num_completions=1,
        cache=False,
        echo=False,
        add_bos_token=False,
    )
    no_logprobs_pred = lm.predict(no_logprobs_prompt)
    assert "\n" in no_logprobs_pred.completion_text
    assert no_logprobs_pred.completion_tokens[0] not in ["<s>", "<\s>"]
    assert len(no_logprobs_pred.completion_tokens) == 49

    no_logprobs_n_prompt = LmPrompt(
        text="def hello_world():\n",
        max_tokens=50,
        stop=["\n"],
        logprobs=0,
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        num_completions=1,
        cache=False,
        echo=False,
        add_bos_token=False,
    )
    no_logprobs_n_pred = lm.predict(no_logprobs_n_prompt)
    assert "\n" not in no_logprobs_n_pred.completion_text
    assert no_logprobs_n_pred.completion_tokens[0] not in ["<s>", "<\s>"]
    assert len(no_logprobs_n_pred.completion_tokens) == 5

    logprobs_prompt = LmPrompt(
        text="def hello_world():",
        max_tokens=50,
        logprobs=1,
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        num_completions=1,
        cache=False,
        echo=False,
        add_bos_token=False,
    )
    logprobs_pred = lm.predict(logprobs_prompt)
    assert "\n" in logprobs_pred.completion_text
    assert logprobs_pred.completion_tokens[0] not in ["<s>", "<\s>"]
    assert len(logprobs_pred.completion_tokens) == 49
    assert len(logprobs_pred.completion_logprobs) == len(
        logprobs_pred.completion_tokens
    )
    assert logprobs_pred.completion_logprobs[0] < 0.95

    logprobs_n_prompt = LmPrompt(
        text="def hello_world():",
        max_tokens=50,
        stop=["\n"],
        logprobs=1,
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        num_completions=1,
        cache=False,
        echo=False,
        add_bos_token=False,
    )
    logprobs_n_pred = lm.predict(logprobs_n_prompt)
    assert "\n" not in logprobs_n_pred.completion_text
    assert logprobs_n_pred.completion_tokens[0] not in ["<s>", "<\s>"]
    assert len(logprobs_n_pred.completion_tokens) == 2
    assert len(logprobs_n_pred.completion_logprobs) == len(
        logprobs_n_pred.completion_tokens
    )
    assert logprobs_n_pred.completion_logprobs[0] < 0.95


@pytest.mark.slow()
def test_stop_n_codegen2():
    lm = get_huggingface_lm(
        Models.CodeGen2_1B, runtime=Runtime.PYTORCH, trust_remote_code=True
    )
    prompt = LmPrompt(
        text="def hello_world():\n",
        max_tokens=500,
        stop=["\n"],
        logprobs=1,
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        num_completions=1,
        cache=False,
        echo=True,
        add_bos_token=True,
    )
    outa = lm.predict(prompt)
    # TODO: compare the first line of prompt a vs b
    prompt_n = LmPrompt(
        text='    def process_encoding(self, encoding: None | str = None) -> str:\n        """Process explicitly defined encoding or auto-detect it.\n\n        If encoding is explicitly defined, ensure it is a valid encoding the python\n        can deal with. If encoding is not specified, auto-detect it.\n\n        Raises unicodec.InvalidEncodingName if explicitly set encoding is invalid.\n        """\n',
        max_tokens=500,
        stop=["\n"],
        logprobs=1,
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        num_completions=1,
        cache=False,
        echo=False,
        add_bos_token=True,
    )
    outb = lm.predict(prompt_n)

    assert len(outb.completion_tokens) > 1


@pytest.mark.slow()
def test_logprobs_equal_stop_codegen2():
    lm = get_huggingface_lm(
        Models.CodeGen2_1B, runtime=Runtime.PYTORCH, trust_remote_code=True
    )
    stop = "    "
    prompt = LmPrompt(
        "place a newline here", max_tokens=5, cache=False, temperature=0, stop=[stop]
    )

    out_a = lm.predict(prompt)
    logprobs_a = np.array(out_a.completion_logprobs)
    assert len(logprobs_a) == 1
    assert len(out_a.logprobs_dict) == 1
    assert stop not in out_a.completion_text

    assert out_a.logprobs_dict == [
        {
            "token": 198,
            "repr": "'\\n'",
            "logit": pytest.approx(-1.363785982131958, rel=0.001),
            "probability": pytest.approx(0.25569090247154236, rel=0.001),
        }
    ]

    prompt = LmPrompt(
        "place a newline here",
        max_tokens=5,
        cache=False,
        temperature=0,
        stop=[stop],
        echo=True,
    )
    out_b = lm.predict(prompt)
    logprobs_b = np.array(out_b.completion_logprobs)
    assert stop not in out_b.completion_text
    assert np.allclose(logprobs_a, logprobs_b, atol=0.001, rtol=0.001)


@pytest.mark.slow()
def test_logprobs_echo_stop_codegen2():
    lm = get_huggingface_lm(
        Models.CodeGen2_1B, runtime=Runtime.PYTORCH, trust_remote_code=True
    )
    stop = "    "
    prompt = LmPrompt(
        "place a newline here",
        max_tokens=5,
        cache=False,
        temperature=0,
        stop=[stop],
        echo=True,
    )
    out_b = lm.predict(prompt)
    logprobs = np.array(out_b.completion_logprobs)
    assert stop not in out_b.completion_text
    assert len(logprobs) == len(out_b.completion_tokens)
    assert len(out_b.full_logprobs) == len(out_b.get_full_tokens())

    assert out_b.logprobs_dict[-1] == {
            "token": 198,
            "repr": "'\\n'",
            "logit": pytest.approx(-1.363785982131958, rel=0.001),
            "probability": pytest.approx(0.25569090247154236, rel=0.001),
        }


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
        add_bos_token=lm not in SEQ2SEQ_MODELS,
    )
    lm = get_huggingface_lm(lm, runtime=Runtime.PYTORCH, trust_remote_code=True)
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


def test_tokenizer():
    lm = get_huggingface_lm("gpt2")
    tokens = lm.tokenize("I like pie")
    assert tokens == ["I", "Ġlike", "Ġpie"]
