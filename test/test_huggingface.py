# test
import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from lmwrapper.huggingface_wrapper import get_huggingface_lm
from lmwrapper.HuggingfacePredictor import (
    _check_tokenizer_to_see_if_adds_bos,
    _expand_offsets_to_a_token_index_for_every_text_index,
    _get_token_offsets,
)
from lmwrapper.prompt_trimming import HfTokenTrimmer
from lmwrapper.runtime import Runtime
from lmwrapper.structs import LmPrompt
from lmwrapper.utils import StrEnum


class Models(StrEnum):
    CodeT5plus_220M = "Salesforce/codet5p-220m"
    CodeT5plus_6B = "Salesforce/codet5p-6b"
    CodeGen2_1B = "Salesforce/codegen2-1B"
    CodeGen2_3_7B = "Salesforce/codegen2-3_7B"
    InstructCodeT5plus_16B = "Salesforce/instructcodet5p-16b"
    CodeLLama_7B = "codellama/CodeLlama-7b-hf"
    CodeLLama_7B_Instruct = "codellama/CodeLlama-7b-Instruct-hf"
    DistilGPT2 = "distilgpt2"
    GPT2 = "gpt2"
    Mistral_7B = "mistralai/Mistral-7B-v0.1"


CUDA_UNAVAILABLE = not torch.cuda.is_available()
try:
    SMALL_GPU = (
        CUDA_UNAVAILABLE or torch.cuda.mem_get_info()[0] < 17_179_869_184
    )  # 16GB
except RuntimeError:
    SMALL_GPU = True

SEQ2SEQ_MODELS = {Models.CodeT5plus_220M}
CAUSAL_MODELS = {Models.DistilGPT2, Models.GPT2, Models.CodeGen2_1B}
BIG_SEQ2SEQ_MODELS = {Models.CodeT5plus_6B, Models.InstructCodeT5plus_16B}
BIG_CAUSAL_MODELS = {Models.CodeGen2_3_7B, Models.Mistral_7B}
BIG_MODELS = BIG_SEQ2SEQ_MODELS | BIG_CAUSAL_MODELS
ALL_MODELS = SEQ2SEQ_MODELS | CAUSAL_MODELS | BIG_MODELS


@pytest.mark.slow()
@pytest.mark.parametrize("model", [Models.CodeLLama_7B])
def test_code_llama_autoregressive(model):
    """7B and 13B *base* models can be used for text/code completion"""
    lm = get_huggingface_lm(
        model,
        runtime=Runtime.PYTORCH,
        trust_remote_code=True,
        precision=torch.float16,
    )

    prompt = LmPrompt(
        "def fibonacci(",
        max_tokens=3,
        cache=False,
        temperature=0,
        add_special_tokens=False,
        add_bos_token=False,
        logprobs=1,
    )

    out = lm.predict(prompt)
    assert out.completion_text == "n):\n"


@pytest.mark.slow()
@pytest.mark.parametrize("model", [Models.CodeLLama_7B, Models.CodeLLama_7B_Instruct])
def test_code_llama_infill(model):
    """7B and 13B base *and* instruct variants support infilling based on surrounding content"""
    lm = get_huggingface_lm(
        model,
        runtime=Runtime.PYTORCH,
        trust_remote_code=True,
        precision=torch.float16,
    )

    infill_prompt = '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
'''

    prompt = LmPrompt(
        infill_prompt,
        max_tokens=3,
        cache=False,
        temperature=0,
        add_special_tokens=False,
        add_bos_token=False,
        logprobs=1,
    )

    out = lm.predict(prompt)
    assert out.completion_text == "Remove non-"


@pytest.mark.slow()
@pytest.mark.parametrize("model", [Models.CodeLLama_7B_Instruct])
def test_code_llama_conversation(model):
    """Instruction fine-tuned models can be used in conversational interfaces"""
    lm = get_huggingface_lm(
        model,
        runtime=Runtime.PYTORCH,
        trust_remote_code=True,
        precision=torch.float16,
    )

    user = (
        "In Bash, how do I list all text files in the current directory (excluding"
        " subdirectories) that have been modified in the last month?"
    )

    instr_prompt1 = f"<s>[INST] {user.strip()} [/INST]"

    prompt = LmPrompt(
        instr_prompt1,
        max_tokens=3,
        cache=False,
        temperature=0,
        add_special_tokens=False,
        add_bos_token=False,
        logprobs=1,
    )

    out = lm.predict(prompt)
    assert out.completion_text == " You can"

    system = "Provide answers in JavaScript"
    user = (
        "Write a function that computes the set of sums of all contiguous sublists of a"
        " given list."
    )

    instr_prompt2 = f"<s>[INST] <<SYS>>\\n{system}\\n<</SYS>>\\n\\n{user}[/INST]"

    prompt = LmPrompt(
        instr_prompt2,
        max_tokens=3,
        cache=False,
        temperature=0,
        add_special_tokens=False,
        add_bos_token=False,
        logprobs=1,
    )

    out = lm.predict(prompt)
    assert out.completion_text == " ```\n"


@pytest.mark.slow()
def test_trim_start():
    lm = get_huggingface_lm(
        Models.CodeGen2_1B,
        runtime=Runtime.PYTORCH,
        trust_remote_code=True,
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
    # model = Models.CodeGen2_1B
    # model = Models.CodeGen2_3_7B
    model = "Salesforce/codegen2-16B"
    lm = get_huggingface_lm(
        model,
        # Models.CodeGen2_3_7B,
        # "Salesforce/codegen2-16B",
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
    del outa
    del lm
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    lm = get_huggingface_lm(
        model,
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
        logprobs=1,
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
    assert no_logprobs_pred.completion_tokens[0] not in ["<s>", r"<\s>"]
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
    assert no_logprobs_n_pred.completion_tokens[0] not in ["<s>", r"<\s>"]
    assert len(no_logprobs_n_pred.completion_tokens) == 6  # or 5?

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
    assert logprobs_pred.completion_tokens[0] not in ["<s>", r"<\s>"]
    assert len(logprobs_pred.completion_tokens) == 49
    assert len(logprobs_pred.completion_logprobs) == len(
        logprobs_pred.completion_tokens,
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
    assert logprobs_n_pred.completion_tokens[0] not in ["<s>", r"<\s>"]
    assert len(logprobs_n_pred.completion_tokens) == 2
    assert len(logprobs_n_pred.completion_logprobs) == len(
        logprobs_n_pred.completion_tokens,
    )
    assert logprobs_n_pred.completion_logprobs[0] < 0.95


@pytest.mark.slow()
def test_stop_n_codegen2():
    lm = get_huggingface_lm(
        Models.CodeGen2_1B,
        runtime=Runtime.PYTORCH,
        trust_remote_code=True,
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
        text=(
            "    def process_encoding(self, encoding: None | str = None) -> str:\n     "
            '   """Process explicitly defined encoding or auto-detect it.\n\n        If'
            " encoding is explicitly defined, ensure it is a valid encoding the"
            " python\n        can deal with. If encoding is not specified, auto-detect"
            " it.\n\n        Raises unicodec.InvalidEncodingName if explicitly set"
            ' encoding is invalid.\n        """\n'
        ),
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

    assert len(outa.completion_tokens) > 1
    assert len(outb.completion_tokens) > 1


@pytest.mark.slow()
def test_logprobs_equal_stop_codegen2():
    lm = get_huggingface_lm(
        Models.CodeGen2_1B,
        runtime=Runtime.PYTORCH,
        trust_remote_code=True,
    )
    stop = "    "
    prompt = LmPrompt(
        "place a newline here",
        max_tokens=5,
        cache=False,
        temperature=0,
        stop=[stop],
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
            "logit": pytest.approx(-1.363785982131958, abs=0.001),
            "probability": pytest.approx(0.25569090247154236, abs=0.001),
        },
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
        Models.CodeGen2_1B,
        runtime=Runtime.PYTORCH,
        trust_remote_code=True,
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
        "logit": pytest.approx(-1.363785982131958, abs=0.001),
        "probability": pytest.approx(0.25569090247154236, abs=0.001),
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
            "logit": pytest.approx(-0.059126678854227066, abs=0.001),
            "probability": pytest.approx(0.9425873756408691, abs=0.001),
        },
        {
            "token": 20,
            "repr": "'5'",
            "logit": pytest.approx(-0.011661103926599026, abs=0.001),
            "probability": pytest.approx(0.9884065985679626, abs=0.001),
        },
        {
            "token": 13,
            "repr": "'.'",
            "logit": pytest.approx(-0.0012642494402825832, abs=0.001),
            "probability": pytest.approx(0.998736560344696, abs=0.001),
        },
        {
            "token": 4486,
            "repr": "' Germany'",
            "logit": pytest.approx(-2.1969234943389893, abs=0.001),
            "probability": pytest.approx(0.11114457249641418, abs=0.001),
        },
        {
            "token": 198,
            "repr": "'\\n'",
            "logit": pytest.approx(-0.054811663925647736, abs=0.001),
            "probability": pytest.approx(0.9466634392738342, abs=0.001),
        },
        {
            "token": 21,
            "repr": "'6'",
            "logit": pytest.approx(-0.025344248861074448, abs=0.001),
            "probability": pytest.approx(0.9749742150306702, abs=0.001),
        },
        {
            "token": 13,
            "repr": "'.'",
            "logit": pytest.approx(-0.0013435394503176212, abs=0.001),
            "probability": pytest.approx(0.9986573457717896, abs=0.001),
        },
        {
            "token": 8031,
            "repr": "' Italy'",
            "logit": pytest.approx(-2.757378101348877, abs=0.001),
            "probability": pytest.approx(0.0634579285979271, abs=0.001),
        },
        {
            "token": 198,
            "repr": "'\\n'",
            "logit": pytest.approx(-0.0332401879131794, abs=0.001),
            "probability": pytest.approx(0.9673061966896057, abs=0.001),
        },
        {
            "token": 22,
            "repr": "'7'",
            "logit": pytest.approx(-0.017078006640076637, abs=0.001),
            "probability": pytest.approx(0.983066976070404, abs=0.001),
        },
        {
            "token": 13,
            "repr": "'.'",
            "logit": pytest.approx(-0.001742750871926546, abs=0.001),
            "probability": pytest.approx(0.9982587695121765, abs=0.001),
        },
        {
            "token": 2869,
            "repr": "' Japan'",
            "logit": pytest.approx(-2.284379005432129, abs=0.001),
            "probability": pytest.approx(0.10183728486299515, abs=0.001),
        },
        {
            "token": 198,
            "repr": "'\\n'",
            "logit": pytest.approx(-0.040456708520650864, abs=0.001),
            "probability": pytest.approx(0.960350751876831, abs=0.001),
        },
        {
            "token": 23,
            "repr": "'8'",
            "logit": pytest.approx(-0.02017313987016678, abs=0.001),
            "probability": pytest.approx(0.9800289869308472, abs=0.001),
        },
        {
            "token": 13,
            "repr": "'.'",
            "logit": pytest.approx(-0.0018331881146878004, abs=0.001),
            "probability": pytest.approx(0.9981684684753418, abs=0.001),
        },
    ]

    prompt = LmPrompt(
        prompt_str,
        max_tokens=15,
        cache=False,
        temperature=0,
        stop=["Italy"],
    )
    out = lm.predict(prompt)
    assert "Italy" not in out.completion_text
    assert out.logprobs_dict == [
        {
            "token": 198,
            "repr": "'\\n'",
            "logit": pytest.approx(-0.059126678854227066, abs=0.001),
            "probability": pytest.approx(0.9425873756408691, abs=0.001),
        },
        {
            "token": 20,
            "repr": "'5'",
            "logit": pytest.approx(-0.011661103926599026, abs=0.001),
            "probability": pytest.approx(0.9884065985679626, abs=0.001),
        },
        {
            "token": 13,
            "repr": "'.'",
            "logit": pytest.approx(-0.0012642494402825832, abs=0.001),
            "probability": pytest.approx(0.998736560344696, abs=0.001),
        },
        {
            "token": 4486,
            "repr": "' Germany'",
            "logit": pytest.approx(-2.1969234943389893, abs=0.001),
            "probability": pytest.approx(0.11114457249641418, abs=0.001),
        },
        {
            "token": 198,
            "repr": "'\\n'",
            "logit": pytest.approx(-0.054811663925647736, abs=0.001),
            "probability": pytest.approx(0.9466634392738342, abs=0.001),
        },
        {
            "token": 21,
            "repr": "'6'",
            "logit": pytest.approx(-0.025344248861074448, abs=0.001),
            "probability": pytest.approx(0.9749742150306702, abs=0.001),
        },
        {
            "token": 13,
            "repr": "'.'",
            "logit": pytest.approx(-0.0013435394503176212, abs=0.001),
            "probability": pytest.approx(0.9986573457717896, abs=0.001),
        },
        {
            "token": 8031,
            "repr": "' Italy'",
            "logit": pytest.approx(-2.757378101348877, abs=0.001),
            "probability": pytest.approx(0.0634579285979271, abs=0.001),
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


def test_degenerate_offsets():
    lm = get_huggingface_lm(Models.DistilGPT2)
    token_ids = [13, 198, 198]
    offsets = _get_token_offsets(lm._tokenizer, token_ids)
    assert offsets == [(0, 1), (1, 2), (2, 3)]


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
    assert "\n" not in out.completion_text
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
    assert "\n\n" not in out.completion_text
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
            f"Skipped model '{lm}' as model too large for available GPU memory.",
        )
    prompt = LmPrompt(
        "print('Hello world",
        max_tokens=15,
        cache=False,
        temperature=0,
        add_bos_token=lm not in SEQ2SEQ_MODELS | BIG_SEQ2SEQ_MODELS,
    )
    lm = get_huggingface_lm(lm, runtime=Runtime.PYTORCH, trust_remote_code=True)
    out = lm.predict(prompt)
    assert out.completion_text


@pytest.mark.slow()
@pytest.mark.skip(reason="ORT is not ready yet")
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
@pytest.mark.skip(reason="Better Transformer is not ready yet")
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
@pytest.mark.skip(reason="Better Transformer is not ready yet")
def test_codegen2_predict_bt():
    lm = Models.CodeGen2_1B
    with pytest.raises(Exception) as e_info:
        get_huggingface_lm(lm, runtime=Runtime.BETTER_TRANSFORMER)
        assert str(e_info.value).startswith("WARNING BetterTransformer")


@pytest.mark.slow()
@pytest.mark.skip(reason="TensorRT is not ready yet")
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


def test_max_length():
    lm = get_huggingface_lm("gpt2")
    assert lm.token_limit == 1024


@pytest.mark.skip()
def test_code_llama_stop():
    prompt = LmPrompt(
        'def double(x) -> int:\n    """Double the given number"""',
        max_tokens=40,
        stop=["\ndef", "\nclass", "\nprint", "\n#"],
        cache=False,
        temperature=0,
    )

    lm = get_huggingface_lm(
        Models.CodeLLama_7B,
        trust_remote_code=True,
        precision=torch.float16,
    )

    out = lm.predict(prompt)
    assert out.completion_text


def test_tokenizer_offsets_code_llama():
    model_name = Models.CodeLLama_7B
    # Get the huggingface tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    token_ids = [13, 1678, 736, 921, 334, 29871, 29906, 13, 13, 13, 1753]
    token_vals = [
        "\n",
        "   ",
        " return",
        " x",
        " *",
        " ",
        "2",
        "\n",
        "\n",
        "\n",
        "def",
    ]
    print([tokenizer.decode([t]) for t in token_ids])
    cum_len = np.cumsum([len(t) for t in token_vals])
    print(cum_len)
    expected_offsets = [0, *(cum_len[:-1])]
    print("Expected", expected_offsets)
    assert expected_offsets[:3] == [0, 1, 4]
    offsets = _get_token_offsets(tokenizer, token_ids)
    starts, ends = zip(*offsets)
    assert list(starts) == expected_offsets


def test_offsets_for_mistral():
    model_name = Models.Mistral_7B
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # token_ids = [2287,  2682,   618, 2287, 2682]
    # assert ' ' + tokenizer.decode(token_ids) == '    print("    print'
    token_ids = [2287, 2682, 618]
    offsets = _get_token_offsets(tokenizer, token_ids)
    assert offsets == [
        (0, len("▁▁▁")),
        (3, 3 + len("▁print")),
        (3 + len("▁print"), 3 + len("▁print") + len('("')),
    ]


def test_offsets_for_mistral2():
    model_name = Models.Mistral_7B
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    token_ids = [618, 2287, 2682, 618]
    offsets = _get_token_offsets(tokenizer, token_ids)
    assert offsets == [
        (0, len('("')),
        (2, 2 + len("▁▁▁")),
        (5, 5 + len("▁print")),
        (11, 11 + len('("')),
    ]


def test_offsets_for_mistral3():
    model_name = Models.Mistral_7B
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    token_ids = [2682, 618]
    offsets = _get_token_offsets(tokenizer, token_ids)
    assert offsets == [(0, 0 + len("▁print")), (6, 6 + len('("'))]


def test_offsets_for_removal_prompt():
    # get the tokenizer model
    lm = get_huggingface_lm(Models.DistilGPT2, runtime=Runtime.PYTORCH)
    tokenizer = lm._tokenizer
    seq = [4486, 198, 21, 13, 8031]
    print("dec\n" + tokenizer.decode(seq))
    text = " Germany\n6. Italy"
    print([tokenizer.decode([i]) for i in seq])
    offsets = _get_token_offsets(tokenizer, seq)
    first = 8
    assert offsets == [(0, first), (first, 9), (9, 10), (10, 11), (11, len(text))]
    assert text[first] == "\n"
    expanded = _expand_offsets_to_a_token_index_for_every_text_index(offsets)
    assert expanded == [
        *([0] * len(" Germany")),
        *([1] * len("\n")),
        *([2] * len("6")),
        *([3] * len(".")),
        *([4] * len(" Italy")),
    ]
    assert len(expanded) == len(text)


def test_token_expanding_weird_from_t5():
    expand = _expand_offsets_to_a_token_index_for_every_text_index(
        [(0, 1), (0, 1), (0, 1), (1, 6), (7, 13), (13, 14), (14, 15)],
    )
    assert expand == [
        0,
        *([3] * (6 - 1)),
        *([4] * (13 - 6)),
        *([5] * (14 - 13)),
        *([6] * (15 - 14)),
    ]


def test_degenerative_multiple():
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(Models.DistilGPT2, use_fast=True)
    tokens = [13, 198, 198, 198, 198]
    text = ".\n\n\n\n"
    assert tokenizer.decode(tokens) == text
    offsets = _get_token_offsets(tokenizer, tokens)
    assert offsets == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ]


def test_degenerative_multiple_2():
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(Models.DistilGPT2, use_fast=True)
    tokens = [13, 198, 198, 198, 198, 198]
    text = ".\n\n\n\n\n"
    assert tokenizer.decode(tokens) == text
    offsets = _get_token_offsets(tokenizer, tokens)
    assert offsets == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
    ]


# @pytest.mark.parametrize("lm", ALL_MODELS)
@pytest.mark.slow()
def test_hello_world_prompt():
    lm = Models.Mistral_7B
    if SMALL_GPU and lm in BIG_MODELS:
        pytest.skip(
            f"Skipped model '{lm}' as model too large for available GPU memory.",
        )
    lm = get_huggingface_lm(
        lm,
        runtime=Runtime.PYTORCH,
        trust_remote_code=True,
        precision=torch.float16,
    )
    hello_world_prompt = (
        "def hello():\n"
        '    """prints the string \'hello\'"""\n'
        '    print("hello")\n'
        "\n"
        "def hello_world():\n"
        '    """prints the string \'hello world\'"""\n'
    )
    resp = lm.predict(
        LmPrompt(
            hello_world_prompt,
            max_tokens=10,
            cache=False,
            # stop=["\n"],
            temperature=0,
        ),
    )
    assert (
        resp.completion_text.startswith("    print('hello world')")
        or resp.completion_text.startswith('    print("hello world")'),
    )

    # A version using stop. Breaks because the tokenization is wrong.
    resp = lm.predict(
        LmPrompt(
            hello_world_prompt,
            max_tokens=10,
            cache=False,
            stop=["\n"],
            temperature=0,
        ),
    )
    assert resp.completion_text in {
        "    print('hello world')",
        '    print("hello world")',
    }


def test_check_tokenizer_check():
    mistral_tokenizer = AutoTokenizer.from_pretrained(Models.Mistral_7B, use_fast=True)
    assert _check_tokenizer_to_see_if_adds_bos(mistral_tokenizer, True)
    gpt2_tokenizer = AutoTokenizer.from_pretrained(Models.DistilGPT2, use_fast=True)
    assert not _check_tokenizer_to_see_if_adds_bos(gpt2_tokenizer, True)
