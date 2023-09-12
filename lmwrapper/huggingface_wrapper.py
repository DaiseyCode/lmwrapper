import inspect
import os
from enum import Enum
from functools import cached_property
from importlib.metadata import version as import_version
from pathlib import Path
import logging

from packaging import version
from lmwrapper.HuggingfacePrediction import HuggingfacePrediction
from lmwrapper._TokenStoppingCriteria import _TokenStoppingCriteria

from lmwrapper.abstract_predictor import LmPredictor
from lmwrapper.structs import LmPrediction, LmPrompt

import numpy as np

_QUANT_CONFIG = None
# TODO: Several models do not work on Apple MPS.
_MPS_ENABLED = os.getenv("MPS_ENABLED", "False").lower() in {"true", "1", "t"}
_ONNX_RUNTIME = os.getenv("ONNX_RUNTIME", "False").lower() in {"true", "1", "t"}

try:
    import torch

    assert version.parse(torch.__version__) >= version.parse("2.0")
except ImportError:
    msg = "Expect to work on torch. Please see https://pytorch.org/ for install info."
    raise ImportError(
        msg,
    )


try:
    import bitsandbytes

    assert version.parse(import_version("bitsandbytes")) >= version.parse(
        "0.41.1",
    )

    from transformers import BitsAndBytesConfig

    _QUANT_CONFIG = BitsAndBytesConfig(
        # load_in_8bit (bool, optional, defaults to False) — This flag is used to enable 8-bit quantization with LLM.int8().
        # load_in_4bit (bool, optional, defaults to False) — This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from bitsandbytes.
        # llm_int8_threshold (float, optional, defaults to 6) — This corresponds to the outlier threshold for outlier detection as described in LLM.int8() : 8-bit Matrix Multiplication for Transformers at Scale paper: https://arxiv.org/abs/2208.07339 Any hidden states value that is above this threshold will be considered an outlier and the operation on those values will be done in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but there are some exceptional systematic outliers that are very differently distributed for large models. These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6, but a lower threshold might be needed for more unstable models (small models, fine-tuning).
        # llm_int8_skip_modules (List[str], optional) — An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as Jukebox that has several heads in different places and not necessarily at the last position. For example for CausalLM models, the last lm_head is kept in its original dtype.
        # llm_int8_enable_fp32_cpu_offload (bool, optional, defaults to False) — This flag is used for advanced use cases and users that are aware of this feature. If you want to split your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use this flag. This is useful for offloading large models such as google/flan-t5-xxl. Note that the int8 operations will not be run on CPU.
        # llm_int8_has_fp16_weight (bool, optional, defaults to False) — This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not have to be converted back and forth for the backward pass.
        # bnb_4bit_compute_dtype (torch.dtype or str, optional, defaults to torch.float32) — This sets the computational type which might be different than the input time. For example, inputs might be fp32, but computation can be set to bf16 for speedups.
        # bnb_4bit_quant_type (str, {fp4, nf4}, defaults to fp4) — This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types which are specified by fp4 or nf4.
        # bnb_4bit_use_double_quant (bool, optional, defaults to False) — This flag is used for nested quantization where the quantization constants from the first quantization are quantized again.
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
    )
except ImportError:
    logging.warning(
        "8/4bit quantization is disabled as bitsandbytes could not be loaded.",
    )

try:
    import transformers

    assert version.parse(transformers.__version__) >= version.parse("4.31.0")

    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        GenerationConfig,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizerFast,
        T5ForConditionalGeneration,
        set_seed,
    )

    set_seed(42)
except ImportError:
    msg = (
        "You must install torch and transformers to use Huggingface models."
        " `pip install lmwrapper[huggingface]`. Please see https://pytorch.org/"
        " for install info."
    )
    raise ImportError(
        msg,
    )

if _ONNX_RUNTIME:
    try:
        from optimum import version as optimum_version

        assert version.parse(optimum_version.__version__) >= version.parse(
            "1.11.0",
        )

        import xformers

        assert version.parse(xformers.__version__) >= version.parse("0.0.20")

        import onnxruntime
        from optimum.bettertransformer import BetterTransformer
        from optimum.onnxruntime import (
            ORTModel,
            ORTModelForCausalLM,
            ORTModelForSeq2SeqLM,
            ORTOptimizer,
        )
        from optimum.onnxruntime.configuration import AutoOptimizationConfig

        assert version.parse(onnxruntime.__version__) >= version.parse("1.15.1")

        session_options = onnxruntime.SessionOptions()
        # session_options.log_severity_level = 0 TODO: set configurable log level
    except ImportError:
        msg = (
            "You must install Optimum, ONNX runtime, and Xformers to use"
            " accelerated Huggingface models. `pip install lmwrapper[ort-gpu]`"
        )
        raise ImportError(
            msg,
        )


class Runtime(Enum):
    PYTORCH = 1
    ACCELERATE = 2
    ORT_CUDA = 3
    ORT_TENSORRT = 4
    ORT_CPU = 5
    BETTER_TRANSFORMER = 6


class HuggingfacePredictor(LmPredictor):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        model: PreTrainedModel,
        device: torch.device,
        runtime: Runtime,
        patch_model_forward: bool = False,
    ):
        super().__init__()
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self.is_chat_model = False
        self.runtime = runtime
        self.patch_model_forward = patch_model_forward

    def _predict_maybe_cached(
        self,
        prompt: LmPrompt,
    ) -> LmPrediction | list[LmPrediction]:
        if not isinstance(prompt.text, str) and len(prompt.text) != 1:
            raise NotImplementedError(
                "Prompt batches other than size 1 are not supported."
            )

        if prompt.echo and not self.patch_model_forward:
            raise NotImplementedError(
                "Prompt echo is only supported with `patch_model_forward` = True."
            )

        if prompt.logprobs > 1:
            raise NotImplementedError(
                "Retrieving more than 1 logprob is not yet supported for HuggingFace models."
            )

        if prompt.logprobs and (prompt.temperature > 0 or prompt.top_p):
            logging.warning(
                "Logprobs may not be correct if temperature > 0 or top_p != 1.0"
            )

        if prompt.presence_penalty:
            raise NotImplementedError

        stopping_criteria = None
        if prompt.stop:
            stopping_criteria = [
                _TokenStoppingCriteria(
                    prompt.stop, decode=True, tokenizer=self._tokenizer
                )
            ]

        temperature = prompt.temperature
        if temperature == 0:
            temperature = None

        if prompt.text == "" and not prompt.add_bos_token:
            raise Exception(
                "Cannot do unconditional generation without `add_bos_token`."
            )

        if prompt.add_bos_token:
            assert self._tokenizer.bos_token
            prompt_text = self._tokenizer.bos_token + prompt.text
        else:
            prompt_text = prompt.text

        max_length = self._model.config.max_length
        model_requires_attention_mask = "attention_mask" in set(
            inspect.signature(self._model.forward).parameters.keys()
        )

        encoded_input = self._tokenizer(
            prompt_text,
            return_tensors="pt",
            return_attention_mask=model_requires_attention_mask,
        )

        if len(encoded_input) > max_length:
            raise ValueError("Prompt is too long for model. Please pass in a trimmer.")

        if self.runtime != Runtime.ACCELERATE:
            encoded_input = encoded_input.to(
                self._device,
            )  # Move to device

        # ONNX models themselves cannot be moved to a device
        # but their input tensors must be moved to GPU
        # Similarly, Accelerate takes care of moving tensors
        if self.runtime != Runtime.ACCELERATE and (
            not _ONNX_RUNTIME or not isinstance(self._model, ORTModel)
        ):
            self._model.to(self._device)  # Ensure model is on device

        need_log_prob = prompt.logprobs is not None and prompt.logprobs > 0

        # Some models do not have a pad token, default to 0
        if self._tokenizer.pad_token_id:
            pad_token_id = self._tokenizer.pad_token_id

        else:
            pad_token_id = 0
            logging.warning(
                "Tokenizer does not have a pad_token_id. Setting pad_token_id to 0. May cause unexpected behavior."
            )

        # Ref https://gist.github.com/kinoc/8a042d8c5683725aa8c372274c02ea2f
        gen_config = GenerationConfig(
            max_new_tokens=prompt.max_tokens,
            temperature=temperature,
            top_p=prompt.top_p,
            do_sample=prompt.temperature > 0,
            return_dict_in_generate=True,
            output_scores=need_log_prob,
            pad_token_id=pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            bos_token_id=self._tokenizer.bos_token_id,
        )

        if self.patch_model_forward:
            # We need a way of getting the raw logprobs of the whole sequence.
            #   The scores we get back are possibly already warped by the configuration
            #   https://github.com/huggingface/transformers/issues/17521#issue-1257881647
            #   Also, it does not return the input tokens. Existing hacks
            #   require calling the model again https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17
            # Instead we are going to patch the model forward to log calls
            old_forward = self._model.forward
            cached_logits = []

            def new_call(*args, **kwargs):
                nonlocal cached_logits
                val = old_forward(*args, **kwargs)
                cached_logits.append(val.logits)
                return val

            self._model.forward = new_call

        with torch.no_grad():
            generation_output = self._model.generate(
                **encoded_input,
                generation_config=gen_config,
                stopping_criteria=stopping_criteria,
            )

        if self.patch_model_forward:
            self._model.forward = old_forward

        output_sequence = generation_output.sequences[0]
        output_text = self._tokenizer.decode(output_sequence)

        # input_length is the length of the input prompt for decoder-only models,
        # like the GPT family, and 1 for encoder-decoder models, like BART or T5.
        input_length = (
            1
            if self._model.config.is_encoder_decoder
            else encoded_input.input_ids.shape[1]
        )

        generated_sequence = output_sequence[input_length:]
        generated_text = self._tokenizer.decode(generated_sequence)

        token_offsets = []
        token_offsets_full = []
        j = 0
        for i, token in enumerate(generated_sequence):
            token_len = len(self._tokenizer.decode(token))
            token_offsets.append(j)
            token_offsets_full.extend([i] * token_len)
            j += token_len

        stop_token_idx_output = None
        stop_token_idx_generated = None
        output_tokens = self._tokenizer.convert_ids_to_tokens(output_sequence)
        if prompt.stop:
            sorted_stop_sequences = sorted(prompt.stop, key=len, reverse=True)

            stop_idx = len(generated_sequence)
            for stop_sequence in sorted_stop_sequences:
                if stop_sequence in generated_text:
                    stop_idx = generated_text.index(stop_sequence)
                    generated_text = generated_text[:stop_idx]
                    stop_token_idx_generated = token_offsets_full[stop_idx]
                    if (
                        stop_token_idx_generated > 0  # ensure not first token
                        and token_offsets_full[
                            stop_idx - 1
                        ]  # compare previous token with current
                        == token_offsets_full[stop_idx]
                    ):
                        stop_token_idx_generated += (
                            1  # if they're equal, we include the current token
                        )
                    stop_token_idx_output = input_length + stop_token_idx_generated
                    output_sequence = output_sequence[:stop_token_idx_output]
                    generated_sequence = output_sequence[input_length:]
                    break

        # Calculate the logprobs if needed
        if need_log_prob:
            if self.patch_model_forward:
                all_logits = torch.cat(cached_logits, dim=1)
                assert all_logits.shape[0] == 1  # batch
                assert all_logits.shape[1] == len(output_tokens[1:])
                logprobs = _gather_logprobs_from_logits(
                    all_logits[0],
                    output_sequence[1:],
                )

                if not prompt.echo:
                    logprobs = logprobs[input_length - 1 :]
            else:
                logprobs = self._model.compute_transition_scores(
                    generation_output.sequences,
                    generation_output.scores,
                    normalize_logits=True,
                )[0, :stop_token_idx_generated]
            assert len(generated_sequence) == len(logprobs)

            # Create logprobs dict
            logprobs_dicts = []
            for tok, score in zip(generated_sequence, logprobs, strict=True):
                logprobs_dicts.append(
                    {
                        "token": int(tok.detach().cpu()),
                        "repr": repr(self._tokenizer.decode(tok)),
                        "logit": float(score.item()),
                        "probability": float(np.exp(score.detach().cpu())),
                    }
                )
        else:
            logprobs = None

        if prompt.max_tokens == 0:
            # Huggingface seems to default to one token always return an extra token
            output_tokens = output_tokens[:-1]
            logprobs = logprobs[:-1]
            generated_text = ""
            generation_output.sequences = generation_output.sequences[:, :-1]

        return HuggingfacePrediction(
            completion_text=generated_text,
            prompt=prompt,
            metad=generation_output,
            _prompt_encoding=encoded_input.to("cpu"),
            _tokens=output_tokens,
            _log_probs=logprobs.detach().cpu().numpy(),
            _logprobs_dict=logprobs_dicts,
        )

    def get_model_max_length(self) -> int:
        return int(self._model.config.max_length)

    @cached_property
    def space_char(self) -> str:
        # Try to discover the space char in the tokens
        tokens = self._tokenizer.tokenize("I went to")
        for tok in tokens:
            if "went" in tok:
                return tok.replace("went", "")
        return None

    def remove_special_chars_from_tokens(self, tokens: list[str]) -> list[str]:
        if self.space_char is None:
            return tokens
        return [tok.replace(self.space_char, " ") for tok in tokens]


def _gather_logprobs_from_logits(
    logits: torch.Tensor,
    selected_toks: torch.LongTensor,
):
    logprobs = torch.log_softmax(logits, dim=-1).detach()
    return torch.gather(logprobs, -1, selected_toks.unsqueeze(-1)).squeeze(-1)


def _get_accelerator() -> torch.device:
    if torch.cuda.is_available():
        if _QUANT_CONFIG:
            # If quantization is enabled and bits and bytes is not
            # compiled with CUDA, things don't work right
            assert bitsandbytes.COMPILED_WITH_CUDA
        return torch.device("cuda")

    if _MPS_ENABLED and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_huggingface_lm(
    model: str,
    runtime: Runtime = Runtime.PYTORCH,
    precision: torch.dtype = torch.float32,
    trust_remote_code: bool = False,
    patch_model_forward: bool = False,
) -> HuggingfacePredictor:
    if runtime != Runtime.PYTORCH:
        msg = (
            "Accelerated inference model support is still under"
            " development. Please use Runtime.PYTORCH until support matures."
        )
        raise Exception(
            msg,
        )

    _kwargs = {"trust_remote_code": trust_remote_code}

    model_class = AutoModelForCausalLM
    config_dict = PretrainedConfig.get_config_dict(model)

    has_remote_code = (
        "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
    )

    if not trust_remote_code and has_remote_code:
        msg = (
            "The model provided has remote code and likely will not work as"
            " expected. Please call with `trust_remote_code = True` If you have"
            " read and trust the code."
        )
        raise Exception(
            msg,
        )

    if "auto_map" in config_dict and "AutoModelForSeq2SeqLM" in config_dict["auto_map"]:
        model_class = AutoModelForSeq2SeqLM

    if model.startswith("Salesforce/codegen"):
        if runtime == Runtime.BETTER_TRANSFORMER:
            msg = (
                "WARNING BetterTransformer breaks CodeGen models with"
                " AutoClass. Please use a different model or runtime."
            )
            raise Exception(
                msg,
            )
        else:
            _kwargs |= {
                "revision": "main",
                "use_cache": False,
            }
    elif model.startswith("Salesforce/codet5") and not model.endswith("b"):
        model_class = T5ForConditionalGeneration

        # T5 class does not support this arg,
        # only autoclasses do
        _kwargs.pop("trust_remote_code", None)
    elif model.startswith("Salesforce/codet5p-") and model.endswith("b"):
        model_class = AutoModelForSeq2SeqLM
        _kwargs |= {
            "low_cpu_mem_usage": True,
        }
    elif model == "Salesforce/instructcodet5p-16b":
        model_class = AutoModelForSeq2SeqLM
        _kwargs = {
            "low_cpu_mem_usage": True,
        }

    return _initialize_hf_model(
        model,
        model_class,
        runtime=runtime,
        precision=precision,
        patch_model_forward=patch_model_forward,
        _kwargs=_kwargs,
    )


def get_ort_model(model: PreTrainedModel) -> "ORTModel":
    if model in {T5ForConditionalGeneration, AutoModelForSeq2SeqLM}:
        return ORTModelForSeq2SeqLM

    return ORTModelForCausalLM


def _initialize_hf_model(
    model_name: str,
    model_class: PreTrainedModel,
    runtime: Runtime = Runtime.PYTORCH,
    precision: torch.dtype | str = "auto",
    patch_model_forward: bool = False,
    _kwargs: dict = {},
) -> HuggingfacePredictor:
    torch_device = _get_accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not tokenizer.is_fast:
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

    if runtime == Runtime.PYTORCH:
        model = model_class.from_pretrained(
            model_name,
            torch_dtype=precision,
            **_kwargs,
        )
    elif runtime == Runtime.ACCELERATE:
        model = model_class.from_pretrained(
            model_name,
            torch_dtype=precision,
            device_map="auto",
            **_kwargs,
        )
    elif runtime == Runtime.ORT_CPU:
        if torch_device.type != "cpu":
            logging.warn(
                f"Specified torch device {torch_device} but ORT CPU runtime"
                " can only use CPU. Please specify device='cpu'.",
            )

        # ORT models do not support these flags
        _kwargs.pop("low_cpu_mem_usage", None)
        _kwargs.pop("device_map", None)

        save_dir = f"{model_name.replace('/','_')}_optimized_cpu_o3"

        if not Path(save_dir).exists():
            model = get_ort_model(model_class).from_pretrained(
                model_name,
                export=True,
                provider="CPUExecutionProvider",
                session_options=session_options,
                **_kwargs,
            )
            assert "CPUExecutionProvider" in model.providers
            optimizer = ORTOptimizer.from_pretrained(model)
            optimization_config = AutoOptimizationConfig.O3()
            optimizer.optimize(
                save_dir=save_dir,
                optimization_config=optimization_config,
            )
        model = get_ort_model(model_class).from_pretrained(
            save_dir,
            provider="CPUExecutionProvider",
        )
    elif runtime == Runtime.ORT_CUDA:
        if torch_device.type != "cuda":
            msg = (
                "Cannot run model on CUDA without CUDA. Please specify"
                " device='cuda'."
            )
            raise Exception(
                msg,
            )

        # ORT models do not support these flags
        _kwargs.pop("low_cpu_mem_usage", None)
        _kwargs.pop("device_map", None)

        save_dir = f"{model_name.replace('/','_')}_optimized_gpu_o3"

        if not Path(save_dir).exists():
            model = get_ort_model(model_class).from_pretrained(
                model_name,
                export=True,
                provider="CUDAExecutionProvider",
                session_options=session_options,
                **_kwargs,
            )
            assert "CUDAExecutionProvider" in model.providers
            optimizer = ORTOptimizer.from_pretrained(model)
            optimization_config = AutoOptimizationConfig.O3()
            optimizer.optimize(
                save_dir=save_dir,
                optimization_config=optimization_config,
            )
        model = get_ort_model(model_class).from_pretrained(
            save_dir,
            provider="CUDAExecutionProvider",
        )
    elif runtime == Runtime.ORT_TENSORRT:
        if torch_device.type != "cuda":
            msg = (
                "Cannot run model on CUDA without CUDA. Please specify"
                " device='cuda'."
            )
            raise Exception(
                msg,
            )

        provider_options = {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": (
                f"tmp/trt_cache_{model_name.replace('/','_')}_tensorrt"
            ),
        }

        # TensorRT models do not support these flags
        _kwargs.pop("low_cpu_mem_usage", None)
        _kwargs.pop("device_map", None)

        model = get_ort_model(model_class).from_pretrained(
            model_name,
            export=True,
            provider="TensorrtExecutionProvider",
            provider_options=provider_options,
            session_options=session_options,
            **_kwargs,
        )
        assert "TensorrtExecutionProvider" in model.providers
    elif runtime == Runtime.BETTER_TRANSFORMER:
        model = BetterTransformer.transform(
            model_class.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=precision,
                **_kwargs,
            ),
        )
    else:
        msg = "Invalid Runtime provided."
        raise Exception(msg)

    predictor = HuggingfacePredictor(
        tokenizer,
        model,
        device=torch_device,
        runtime=runtime,
        patch_model_forward=patch_model_forward,
    )

    if runtime == Runtime.ORT_TENSORRT:
        # Warm up TensorRT model once instantiated.
        logging.info("Warmimg up TensorRT model.")
        _warmup_model(predictor)
        logging.info("Warmup successful.")

    return predictor


def _warmup_model(predictor: HuggingfacePredictor):
    raise NotImplementedError("Model warmup is not support yet.")
    small_prompt = LmPrompt("!", cache=False, temperature=0)
    predictor.predict(small_prompt)

    long_prompt_str = "hello world" * predictor.get_model_max_length()
    long_prompt = LmPrompt(long_prompt_str, cache=False, temperature=0)
    predictor.predict(long_prompt)
