import os

_QUANT_CONFIG: dict | None = None
_MPS_ENABLED = os.getenv("MPS_ENABLED", "False").lower() in {"true", "1", "t"}
_ONNX_RUNTIME = os.getenv("ONNX_RUNTIME", "False").lower() in {"true", "1", "t"}
_QUANTIZATION_ENABLED = os.getenv("QUANTIZATION", "False").lower() in {"true", "1", "t"}
