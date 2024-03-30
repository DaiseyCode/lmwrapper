"""Attempts at chasing down potential memory leaks"""

import gc
from test.test_huggingface import Models

import pytest
import torch

from lmwrapper.huggingface_wrapper.wrapper import get_huggingface_lm
from lmwrapper.runtime import Runtime
from lmwrapper.structs import LmPrompt


@pytest.mark.skip("Not going to try with the profiler version stuff")
def test_cuda_memory_cleanup_no_pred():
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "./tensorboard",
            worker_name="worker1",
        ),
        with_stack=False,
        record_shapes=False,
        profile_memory=True,
    ):
        if not torch.cuda.is_available():
            pytest.skip("No CUDA available")
        gc.collect()
        torch.cuda.empty_cache()
        all_tensors = list(get_tensors())
        assert len(all_tensors) == 0
        assert torch.cuda.memory_allocated() == 0
        assert torch.cuda.memory_reserved() == 0
        available_gpus = [
            torch.cuda.device(i) for i in range(torch.cuda.device_count())
        ]
        print("Available gpus", available_gpus)
        lm = get_huggingface_lm(
            Models.DistilGPT2,
            runtime=Runtime.PYTORCH,
            # device="cuda:0",
        )
        assert str(lm._model.device) != "cpu"
        print(lm._model)
        assert torch.cuda.memory_allocated() > 0, "Before deling no mem"
        assert torch.cuda.memory_reserved() > 0, "Before deling no mem"
        print("Deleting model")
        del lm
        gc.collect()
        torch.cuda.empty_cache()
        assert len(all_tensors) == 0
        assert torch.cuda.memory_allocated() == 0
        assert torch.cuda.memory_reserved() == 0


@pytest.mark.skip("Huggingface/torch seems to hang on to some memory")
def test_cuda_memory_cleanup_pred_no_keep():
    # with torch.profiler.profile(
    #    activities=[
    #    torch.profiler.ProfilerActivity.CPU,
    #    torch.profiler.ProfilerActivity.CUDA],
    #    on_trace_ready=torch.profiler.tensorboard_trace_handler('./tensorboard', worker_name='worker1'),
    #    with_stack=False,
    #    record_shapes=False,
    #    profile_memory=True,
    # ) as profiler, torch.inference_mode():
    #    if not torch.cuda.is_available():
    #        pytest.skip("No CUDA available")
    #    gc.collect()
    #    torch.cuda.empty_cache()
    #    all_tensors = list(get_tensors())
    #    assert len(all_tensors) == 0
    #    assert torch.cuda.memory_allocated() == 0
    #    assert torch.cuda.memory_reserved() == 0
    #    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    #    print("Available gpus", available_gpus)
    #    model = AutoModelForCausalLM.from_pretrained("distilgpt2").to("cuda")
    #    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA available")
    gc.collect()
    torch.cuda.empty_cache()
    all_tensors = list(get_tensors())
    assert len(all_tensors) == 0
    assert torch.cuda.memory_allocated() == 0
    assert torch.cuda.memory_reserved() == 0
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    print("Available gpus", available_gpus)
    lm = get_huggingface_lm(
        "Salesforce/codegen2-16b",
        # Models.DistilGPT2,
        runtime=Runtime.PYTORCH,
        trust_remote_code=True,
        # device="cuda:0",
    )
    assert str(lm._model.device) != "cpu"
    print(lm._model)
    assert torch.cuda.memory_reserved() > 0, "Before deling no mem"
    print("pred no keep")
    for _i in range(50):
        lm.predict(LmPrompt(text="Hello world", logprobs=0, echo=False))
    print("After predict")
    assert torch.cuda.memory_reserved() > 0, "Before deling no mem"
    print("Deleting model")
    del lm
    gc.collect()
    torch.cuda.empty_cache()
    print("Tensors", list(get_tensors()))
    print("Memory", torch.cuda.memory_allocated())
    assert len(all_tensors) == 0
    assert torch.cuda.memory_allocated() == 0
    assert torch.cuda.memory_reserved() == 0


@pytest.mark.skip("Huggingface/torch seems to hang on to some memory")
def test_cuda_memory_cleanup_pred_keep():
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "./tensorboard",
            worker_name="worker1",
        ),
        with_stack=False,
        record_shapes=False,
        profile_memory=True,
    ):
        if not torch.cuda.is_available():
            pytest.skip("No CUDA available")
        gc.collect()
        torch.cuda.empty_cache()
        all_tensors = list(get_tensors())
        assert len(all_tensors) == 0
        assert torch.cuda.memory_allocated() == 0
        assert torch.cuda.memory_reserved() == 0
        available_gpus = [
            torch.cuda.device(i) for i in range(torch.cuda.device_count())
        ]
        print("Available gpus", available_gpus)
        lm = get_huggingface_lm(
            Models.DistilGPT2,
            runtime=Runtime.PYTORCH,
            # device="cuda:0",
        )
        assert str(lm._model.device) != "cpu"
        print(lm._model)
        assert torch.cuda.memory_reserved() > 0, "Before deling no mem"
        print("pred keep")
        pred = lm.predict("Hello world")
        print("After predict")
        assert torch.cuda.memory_reserved() > 0, "Before deling no mem"
        print("Deleting model")
        del lm
        gc.collect()
        torch.cuda.empty_cache()
        print("Tensors", list(get_tensors()))
        print("Memory", torch.cuda.memory_allocated())
        assert len(all_tensors) == 0
        assert torch.cuda.memory_allocated() == 0
        assert torch.cuda.memory_reserved() == 0
        print("pred")
        print(pred)


def get_tensors(gpu_only=True):
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda or not gpu_only:
                yield tensor
        except Exception:  # nosec B112 pylint: disable=broad-exception-caught
            continue
