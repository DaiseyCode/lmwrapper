import logging
from importlib.metadata import version as import_version
from pathlib import Path
from typing import Literal

from packaging import version

from lmwrapper.env import _MPS_ENABLED, _ONNX_RUNTIME, _QUANTIZATION_ENABLED
from lmwrapper.huggingface_wrapper.predictor import HuggingFacePredictor
from lmwrapper.huggingface_wrapper.utilstorch import log_cuda_mem
from lmwrapper.prompt_trimming import PromptTrimmer
from lmwrapper.runtime import Runtime
from lmwrapper.structs import LmPrompt

try:
    import torch

    assert version.parse(torch.__version__) >= version.parse("2.0")
except ModuleNotFoundError:
    msg = ('`torch` package is not found. Note, you can '
           'install lmwrapper with `pip install "lmwrapper[hf]"` to install the '
           'required huggingface dependencies.')
    raise ModuleNotFoundError(
        msg,
    )
except ImportError:
    msg = "Expect to work on torch. Please see https://pytorch.org/ for install info."
    raise ImportError(
        msg,
    )

_QUANT_CONFIG = False

if _QUANTIZATION_ENABLED:
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

    assert version.parse(transformers.__version__) >= version.parse("4.33.2")

    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizerFast,
        T5ForConditionalGeneration,
        set_seed,
    )
    from transformers.models.auto.modeling_auto import _BaseAutoModelClass

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

# Flash Attention is a more efficient attention mechanism
# available for a limited number of models, notably LLaMa/CodeLLaMa!
_FLASH_ATTENTION_AVAILABLE = False
try:
    from transformers.utils.import_utils import is_flash_attn_available

    _FLASH_ATTENTION_AVAILABLE = is_flash_attn_available()
except ImportError:
    pass

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

# Types
AutoPretrainedModelType = type[_BaseAutoModelClass | PreTrainedModel]


def _get_accelerator() -> torch.device:
    """
    Returns the most suitable device (accelerator) for PyTorch operations.

    Returns
    -------
    torch.device
        The device determined to be the most suitable for PyTorch operations. One of 'cuda', 'mps', or 'cpu'.

    Raises
    ------
    AssertionError:
        If CUDA & quantization are enabled but `bitsandbytes` is not compiled with CUDA support.

    Notes
    -----
    * CUDA is prioritized if available.
    * MPS (Metal Performance Shaders) is used if `_MPS_ENABLED` is True and MPS backend is available. MacOS only.
    * If none of the above, CPU is used.

    Examples
    --------
    >>> device = _get_accelerator()
    >>> print(device)
    cuda # or mps or cpu

    """
    if torch.cuda.is_available():
        if _QUANTIZATION_ENABLED and _QUANT_CONFIG:
            # If quantization is enabled and bits and bytes is not
            # compiled with CUDA, things don't work right
            if not bitsandbytes.COMPILED_WITH_CUDA:
                raise Exception(
                    "Quantization was enabled but `bitsandbytes` is not compiled with"
                    " CUDA.",
                )
        return torch.device("cuda")

    if _MPS_ENABLED and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_huggingface_lm(
    model: str,
    runtime: Runtime = Runtime.PYTORCH,
    precision: torch.dtype = torch.float32,
    trust_remote_code: bool = False,
    allow_patch_model_forward: bool = True,
    prompt_trimmer: PromptTrimmer = None,
    device: torch.device | str = None,
    use_chat_mode: bool = None,
) -> HuggingFacePredictor:
    """
    Initialize and return a Hugging Face language model for prediction.

    Parameters
    ----------
    model : str
        The identifier of the pre-trained model to load from the Hugging Face Model Hub.

    runtime : Runtime, optional
        The backend to use for inference. Default is `Runtime.PYTORCH`.
        The only currently supported option is `Runtime.PYTORCH`.

    precision : torch.dtype, optional
        The floating-point precision of the model weights. Default is `torch.float32`.

    trust_remote_code : bool, optional
        Whether to trust or run the remote code from the loaded model. Default is False.

    allow_patch_model_forward : bool, optional
        Allows patching of the `forward` method of the model to compute logprobs. Default is True.
        This option is required logprobs are required for unconditional generation.

    prompt_trimmer : PromptTrimmer, optional
        An object that trims the input prompt to fit the model input size. Default is None.

    device : torch.device | str, optional
        The device on which to run the model. Defaults to the system's best available device.

    Returns
    -------
    HuggingFacePredictor
        An initialized Hugging Face model ready for prediction.

    Raises
    ------
    ValueError:
        If an empty string is provided for `device`
    NotImplementedError:
        If the specified `runtime` is not yet supported.

    Notes
    -----
    * The `trust_remote_code` option should only be enabled if you have verified and trust the remote code.
    * The function supports different types of models based on the `model` identifier and adjusts settings automatically.

    Examples
    --------
    >>> predictor = get_huggingface_lm("gpt-2")
    >>> predictor = get_huggingface_lm("gpt-2", precision=torch.float16, device="cuda:0")

    """
    if isinstance(device, str):
        if device.strip() == "":
            raise ValueError("Empty string provided for device.")
        else:
            device = torch.device(device)

    if runtime != Runtime.PYTORCH:
        msg = (
            "Accelerated inference model support is still under"
            " development. Please use Runtime.PYTORCH until support matures."
        )
        raise NotImplementedError(
            msg,
        )

    _kwargs = {"trust_remote_code": trust_remote_code}

    model_config: PretrainedConfig = AutoConfig.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
    )
    model_config_dict = model_config.to_dict()
    has_remote_code = (
        "auto_map" in model_config_dict
        and "AutoConfig" in model_config_dict["auto_map"]
    )
    has_vocab_size = "vocab_size" in model_config_dict
    has_decoder = "decoder" in model_config_dict
    has_decoder_vocab_size = (
        has_decoder and "vocab_size" in model_config_dict["decoder"]
    )

    # Addresses a bug in Transformers
    # Model transitions i.e. logprobs cannot be calculated if
    # the model config does not have a `vocab_size`
    # We check if the decoder has vocab size and update the config.
    if not has_vocab_size and has_decoder_vocab_size:
        model_config.vocab_size = model_config.decoder.vocab_size

    if has_remote_code and not trust_remote_code:
        msg = (
            "The model provided has remote code and likely will not work as"
            " expected. Please call with `trust_remote_code = True` If you have"
            " read and trust the code."
        )
        raise ValueError(
            msg,
        )

    model_class, _kwargs = _configure_model(model, model_config, runtime, _kwargs)

    return _initialize_hf_model(
        model_name=model,
        model_class=model_class,
        model_config=model_config,
        runtime=runtime,
        precision=precision,
        allow_patch_model_forward=allow_patch_model_forward,
        prompt_trimmer=prompt_trimmer,
        device=device,
        use_chat_mode=use_chat_mode,
        _kwargs=_kwargs,
    )


def _configure_model(
    model: str,
    model_config: PretrainedConfig,
    runtime: Runtime,
    _kwargs: dict,
) -> tuple[AutoPretrainedModelType, dict]:
    """
    Configure the Hugging Face model class and additional keyword arguments based on input parameters.

    Parameters
    ----------
    model : str
        Identifier of the pre-trained model to load from the Hugging Face Model Hub.

    model_config : PretrainedConfig
        The configuration object associated with the model.

    runtime : Runtime
        Backend runtime for inference. Only `Runtime.PYTORCH` and `Runtime.BETTER_TRANSFORMER` are considered.

    _kwargs : dict
        Additional keyword arguments to be modified and used in initializing the model.

    Returns
    -------
    tuple
        (model_class, updated_kwargs)
        - `model_class`: The model class to be used for initialization, either `AutoModelForCausalLM` or a variant.
        - `updated_kwargs`: Modified keyword arguments for model initialization.

    Raises
    ------
    Exception:
        If `Runtime.BETTER_TRANSFORMER` is selected for incompatible models.

    Notes
    -----
    * `_kwargs` can be modified within the function to add or remove keyword arguments for model initialization.
    * The function supports special configurations for Salesforce models.

    Examples
    --------
    >>> model_class, kwargs = _configure_model("gpt-2", config, Runtime.PYTORCH, {})
    >>> model_class, kwargs = _configure_model("Salesforce/codegen", config, Runtime.BETTER_TRANSFORMER, {})

    """
    model_class: AutoPretrainedModelType = AutoModelForCausalLM
    model_config_dict = model_config.to_dict()
    if ("auto_map" in model_config_dict) and (
        "AutoModelForSeq2SeqLM" in model_config_dict["auto_map"]
    ):
        model_class = AutoModelForSeq2SeqLM

    if model.startswith("Salesforce/codegen"):
        if runtime == Runtime.BETTER_TRANSFORMER:
            msg = (
                "WARNING BetterTransformer breaks CodeGen models with"
                " AutoClass. Please use a different model or runtime."
            )
            raise ValueError(
                msg,
            )

        _kwargs |= {
            "revision": "main",
            # "use_cache": False,
        }
    elif model.startswith("Salesforce/codet5") and not model.endswith("b"):
        model_class = T5ForConditionalGeneration

        # T5 class does not support this arg,
        # only autoclasses do
        _kwargs.pop("trust_remote_code", None)
    elif (
        model.startswith("Salesforce/codet5p-")
        and model.endswith("b")
        or model == "Salesforce/instructcodet5p-16b"
    ):
        model_class = AutoModelForSeq2SeqLM
        _kwargs |= {
            "low_cpu_mem_usage": True,
        }
    elif model.startswith("codellama/CodeLlama-"):
        _kwargs |= {
            "low_cpu_mem_usage": True,
            "use_flash_attention_2": (
                _FLASH_ATTENTION_AVAILABLE
            ),  # Use Flash Attention if available
        }

    return model_class, _kwargs


def get_ort_model(model: type[PreTrainedModel]) -> type["ORTModel"]:
    """
    Maps a given Hugging Face PreTrainedModel to its corresponding ONNX Runtime (ORT) model class.

    Parameters
    ----------
    model : PreTrainedModel
        Hugging Face model instance or class (e.g., T5ForConditionalGeneration, AutoModelForSeq2SeqLM).

    Returns
    -------
    ORTModel : str
        Corresponding ORT model class name as string (e.g., ORTModelForSeq2SeqLM, ORTModelForCausalLM).

    Notes
    -----
    * Currently supports mapping for `T5ForConditionalGeneration` and `AutoModelForSeq2SeqLM` to `ORTModelForSeq2SeqLM`.
    * Other models default to `ORTModelForCausalLM`.

    Examples
    --------
    >>> get_ort_model(T5ForConditionalGeneration)
    'ORTModelForSeq2SeqLM'
    >>> get_ort_model(AutoModelForCausalLM)
    'ORTModelForCausalLM'

    """
    if model in {T5ForConditionalGeneration, AutoModelForSeq2SeqLM}:
        return ORTModelForSeq2SeqLM

    return ORTModelForCausalLM


def _get_huggingface_predictor(
    tokenizer: PreTrainedTokenizerFast,
    model: PreTrainedModel,
    device: torch.device,
    runtime: Runtime = Runtime.PYTORCH,
    allow_patch_model_forward: bool = False,
    prompt_trimmer: PromptTrimmer | None = None,
    use_chat_mode: bool = None,
) -> HuggingFacePredictor:
    """
    Creates and returns a HuggingFacePredictor object configured with the specified parameters.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerFast
        Tokenizer instance responsible for converting text to tokens.

    model : PreTrainedModel
        Pre-trained Hugging Face model for predictions.

    device : torch.device
        Device on which the model and tokenizer will run. Can be a CPU or GPU device.

    runtime : Runtime, optional
        The runtime backend for the model. Default is PyTorch. Supports PyTorch, Accelerate, ONNX Runtime (ORT), etc.

    allow_patch_model_forward : bool, optional
        If True, allows the forward pass of the model to be patched. Default is False.

    use_chat_mode : bool, optional
        Whether to apply the chat template to the prompt and treat
        the input as chat turns. If None, the behavior is determined
        by whether the tokenizer has a chat template.

    prompt_trimmer : PromptTrimmer | None, optional
        An optional utility to trim prompts before feeding them to the model. None if no trimming is needed.
        This is somewhat messy right now since tokenizer also does some stuff
        with the prompt. This should probably be refactored.

    Returns
    -------
    HuggingFacePredictor
        Configured instance of HuggingFacePredictor.

    Examples
    --------
    >>> predictor = _get_huggingface_predictor(tokenizer, model, device=torch.device('cuda'))
    >>> type(predictor)
    <class 'HuggingFacePredictor'>

    """
    return HuggingFacePredictor(
        tokenizer,
        model,
        device=device,
        runtime=runtime,
        allow_patch_model_forward=allow_patch_model_forward,
        prompt_trimmer=prompt_trimmer,
        use_chat_mode=use_chat_mode,
    )


def _initialize_hf_model(
    model_name: str,
    model_class: AutoPretrainedModelType,
    model_config: PretrainedConfig,
    runtime: Runtime = Runtime.PYTORCH,
    precision: torch.dtype | Literal["auto"] = "auto",
    allow_patch_model_forward: bool = True,
    prompt_trimmer: PromptTrimmer = None,
    device: torch.device = None,
    use_chat_mode: bool = None,
    _kwargs: dict = {},
) -> HuggingFacePredictor:
    """
    Initialize a Hugging Face model for prediction based on various configurations.

    Parameters
    ----------
    model_name : str
        Name or identifier of the Hugging Face model to load.

    model_class : PreTrainedModel
        Class of the Hugging Face model, e.g., AutoModelForCausalLM.

    model_config : PretrainedConfig
        Configuration object for the model.

    runtime : Runtime, optional
        Backend runtime for the model. Default is Runtime.PYTORCH.

    precision : torch.dtype | "auto", optional
        Data type precision for the model. 'auto' by default.

    allow_patch_model_forward : bool, optional
        Allow patching model's forward method. Default is True.

    prompt_trimmer : PromptTrimmer, optional
        Instance of a prompt trimmer class. Default is None.
        This is somewhat messy right now since tokenizer also does some stuff
        with the prompt. This should probably be refactored.

    device : torch.device, optional
        Torch device to run the model on. Default is auto-detected.

    _kwargs : dict, optional
        Additional keyword arguments for model initialization.

    Returns
    -------
    HuggingFacePredictor
        Configured Huggingface Predictor instance.

    Raises
    ------
    ValueError:
        If invalid runtime or incompatible configurations are supplied.

    Notes
    -----
    * `_kwargs` can be modified within the function.
    * Function logs CUDA memory before and after model instantiation for PyTorch runtime.
    * Function may warm up models for TensorRT runtime.

    Examples
    --------
    >>> predictor = _initialize_hf_model('gpt-2', AutoModelForCausalLM, config)
    >>> predictor = _initialize_hf_model('Salesforce/codegen', AutoModelForSeq2SeqLM, config, runtime=Runtime.BETTER_TRANSFORMER)

    """
    torch_device = _get_accelerator() if device is None else device

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if not tokenizer.is_fast:
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

    if runtime == Runtime.PYTORCH:
        logging.debug("Before model instantiation")
        log_cuda_mem()
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=model_name,
            config=model_config,
            torch_dtype=precision,
            **_kwargs,
        )
        logging.debug("Post model instantiation")
        log_cuda_mem()
    elif runtime == Runtime.ACCELERATE:
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=model_name,
            config=model_config,
            torch_dtype=precision,
            device_map="auto",
            **_kwargs,
        )
    elif runtime == Runtime.ORT_CPU:
        if torch_device.type != "cpu":
            logging.warning(
                f"Specified torch device {torch_device} but ORT CPU runtime"
                " can only use CPU. Please specify device='cpu'.",
            )

        # ORT models do not support these flags
        _kwargs.pop("low_cpu_mem_usage", None)
        _kwargs.pop("device_map", None)

        save_dir = f"{model_name.replace('/','_')}_optimized_cpu_o3"

        if not Path(save_dir).exists():
            model = get_ort_model(model_class).from_pretrained(
                pretrained_model_name_or_path=model_name,
                config=model_config,
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
            msg = "Cannot run model on CUDA without CUDA. Please specify device='cuda'."
            raise ValueError(
                msg,
            )

        # ORT models do not support these flags
        _kwargs.pop("low_cpu_mem_usage", None)
        _kwargs.pop("device_map", None)

        save_dir = f"{model_name.replace('/','_')}_optimized_gpu_o3"

        if not Path(save_dir).exists():
            model = get_ort_model(model_class).from_pretrained(
                pretrained_model_name_or_path=model_name,
                config=model_config,
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
            msg = "Cannot run model on CUDA without CUDA. Please specify device='cuda'."
            raise ValueError(
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
            pretrained_model_name_or_path=model_name,
            config=model_config,
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
                pretrained_model_name_or_path=model_name,
                config=model_config,
                torch_dtype=precision,
                **_kwargs,
            ),
        )
    else:
        msg = "Invalid Runtime provided."
        raise ValueError(msg)

    # Some models do not have a pad token, default to 0
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = 0
        logging.warning(
            "Tokenizer does not have a pad_token_id. Setting pad_token_id to 0. May"
            " cause unexpected behavior.",
        )

    if runtime in {Runtime.PYTORCH, Runtime.BETTER_TRANSFORMER}:
        model.to(torch_device)  # Ensure model is on device

    predictor = _get_huggingface_predictor(
        tokenizer=tokenizer,
        model=model,
        device=torch_device,
        runtime=runtime,
        allow_patch_model_forward=allow_patch_model_forward,
        prompt_trimmer=prompt_trimmer,
        use_chat_mode=use_chat_mode,
    )

    if runtime == Runtime.ORT_TENSORRT:
        # Warm up TensorRT model once instantiated.
        logging.info("Warmimg up TensorRT model.")
        _warmup_model(predictor)
        logging.info("Warmup successful.")

    return predictor


def _warmup_model(predictor: HuggingFacePredictor):
    """
    Warms up a given Huggingface predictor model by running predictions.
    The purpose of this is primarily to build TensorRT kernels for various
    input sizes, as otherwise they would be built on the fly, causing
    significant delay.

    Parameters
    ----------
    predictor : HuggingFacePredictor
        Instance of a Huggingface predictor class.

    Notes
    -----
    * Performs a small prediction using a single '!' as prompt.
    * Verifies if token limit is respected by attempting a long token string.
    * Both predictions are done with cache disabled, temperature at 0 and max_tokens set to 1.

    Raises
    ------
    ValueError:
        If token limit is not respected.

    Examples
    --------
    >>> predictor = _initialize_hf_model('gpt-2', AutoModelForCausalLM, config)
    >>> _warmup_model(predictor)

    """
    raise NotImplementedError("Model warmup is not support yet.")
    small_prompt = LmPrompt("!", cache=False, temperature=0, max_tokens=1)
    predictor.predict(small_prompt)

    single_token = predictor.tokenize("Hello")[0]
    long_prompt_str = single_token * (predictor.token_limit - 1)
    if predictor.tokenize(long_prompt_str) != (predictor.token_limit - 1):
        raise ValueError("Prompt too long.")
    long_prompt = LmPrompt(long_prompt_str, cache=False, temperature=0, max_tokens=1)
    predictor.predict(long_prompt)
