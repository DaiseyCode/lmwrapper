from enum import StrEnum

import pytest
import torch

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
