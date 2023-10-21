from enum import Enum


class Runtime(Enum):
    """Enum to specify the runtime backend for model execution."""

    PYTORCH = 1
    """Use PyTorch as the backend runtime."""

    ACCELERATE = 2
    """Use Huggingface Accelerate as the backend runtime."""

    ORT_CUDA = 3
    """Use ONNX Runtime with CUDA as the backend runtime."""

    ORT_TENSORRT = 4
    """Use ONNX Runtime with TensorRT as the backend runtime."""

    ORT_CPU = 5
    """Use ONNX Runtime with CPU as the backend runtime."""

    BETTER_TRANSFORMER = 6
    """Use BetterTransformer as the backend runtime."""
