from abc import ABC

from transformers import AutoModelForSeq2SeqLM, PreTrainedModel
from transformers.models.auto.modeling_auto import _BaseAutoModelClass


class HuggingFaceModelWrappingConfig(ABC):
    model_name: str
    model_class: type[PreTrainedModel | _BaseAutoModelClass]
    supports_bettertransformers: bool = True
    model_kwargs: dict
    unsupported_kwargs: list


class _BigModelMixin(HuggingFaceModelWrappingConfig):
    def __init__(self) -> None:
        self.model_kwargs |= {"low_cpu_mem_usage": True}


class _AutoModel(HuggingFaceModelWrappingConfig): ...


class _PreTrainedModel(HuggingFaceModelWrappingConfig):
    unsupported_kwargs = ["trust_remote_code"]
    """Trust remote code is only usable in AutoModel classes."""


class _AutoSeq2SeqModel(_AutoModel):
    model_class: AutoModelForSeq2SeqLM


class _CodegenFamily(_AutoModel):
    supports_bettertransformers = False
    model_kwargs = {
        "revision": "main",
        "use_cache": False,
    }


class InstructCodeT5P_16B(_CodegenFamily, _BigModelMixin):
    model_name = "Salesforce/instructcodet5p-16b"
