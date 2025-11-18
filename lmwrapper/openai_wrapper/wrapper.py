import bisect
import dataclasses
import math
import random
import re
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import openai.types
import openai.types.chat
import tiktoken
from openai import OpenAI, RateLimitError
from openai.types.completion_choice import Logprobs

from lmwrapper.abstract_predictor import LmPredictor, LmReasoningStyle
from lmwrapper.batch_config import CompletionWindow
from lmwrapper.secrets_manager import (
    SecretEnvVar,
    SecretFile,
    SecretInterface,
    assert_is_a_secret,
)
from lmwrapper.structs import LmPrediction, LmPrompt

# Aliases for shorter lines
NEVER_THINK = LmReasoningStyle.NEVER_THINK
ALWAYS_THINK = LmReasoningStyle.ALWAYS_THINK
CONFIGURABLE_THINKING = LmReasoningStyle.CONFIGURABLE_THINKING

PRINT_ON_PREDICT = False

MAX_LOG_PROB_PARM = 5


class _ModelNamesMeta(type):
    def __iter__(cls):
        for attr in cls.__dict__:
            if isinstance(cls.__dict__[attr], OpenAiModelInfo):
                yield cls.__dict__[attr]


class OpenAiModelInfo(str):
    def __new__(
        cls,
        name: str,
        is_chat_model: bool,
        reasoning_style: LmReasoningStyle,
        token_limit_input: int,
    ):
        instance = super().__new__(cls, name)
        instance._is_chat_model = is_chat_model
        instance._reasoning_style = reasoning_style
        instance._token_limit_input = token_limit_input
        return instance

    @property
    def is_chat_model(self):
        return self._is_chat_model

    @property
    def reasoning_style(self):
        return self._reasoning_style

    @property
    def token_limit(self):
        return self._token_limit_input

    def __reduce__(self):
        return (
            self.__class__,
            (str(self), self._is_chat_model, self._reasoning_style, self._token_limit_input),
        )


class OpenAiModelNames(metaclass=_ModelNamesMeta):
    """
    Enum for available OpenAI models. Variable docstrings adapted from
    documentation on OpenAI's website at the time.
    """

    gpt_3_5_turbo = OpenAiModelInfo("gpt-3.5-turbo", True, NEVER_THINK, 4096)
    """Most capable GPT-3.5 model and optimized for chat at 1/10th the cost of text-davinci-003.
    Will be updated with our latest model iteration 2 weeks after it is released."""
    gpt_3_5_turbo_16k = OpenAiModelInfo("gpt-3.5-turbo-16k", True, NEVER_THINK, 16384)
    """Same capabilities as the standard gpt-3.5-turbo model but with 4 times the context."""
    gpt_3_5_turbo_instruct = OpenAiModelInfo(
        "gpt-3.5-turbo-instruct", False, NEVER_THINK, 4096
    )
    """A GPT-3.5 version but for completion"""
    code_davinci_002 = OpenAiModelInfo("code-davinci-002", False, NEVER_THINK, 4097)
    """Can do any language task with better quality, longer output, and consistent instruction-following
    than the curie, babbage, or ada models.
    Also supports some additional features such as inserting text."""
    gpt_4 = OpenAiModelInfo("gpt-4", True, NEVER_THINK, 8192)
    """More capable than any GPT-3.5 model, able to do more complex tasks, and optimized for chat.
    Will be updated with our latest model iteration 2 weeks after it is released."""
    gpt_4_32k = OpenAiModelInfo("gpt-4-32k", True, NEVER_THINK, 32768)
    """Same capabilities as the base gpt-4 mode but with 4x the context length.
    Will be updated with our latest model iteration."""
    gpt_4_turbo = OpenAiModelInfo("gpt-4-1106-preview", True, NEVER_THINK, 128_000)
    """GPT-4 model with improved instruction following, JSON mode,
    reproducible outputs, parallel function calling, and more.
    Returns a maximum of 4,096 output tokens. This preview model is
    not yet suited for production traffic.

    Note that we don't currently handle the differing input and output
    token limits (tracked #25).

    see: https://help.openai.com/en/articles/8555510-gpt-4-turbo
    """
    gpt_4o = OpenAiModelInfo("gpt-4o", True, NEVER_THINK, 128_000)
    """
    GPT-4o ("o" for "omni") is our most advanced model. 
    It is multimodal (accepting text or image inputs and outputting text), 
    and it has the same high intelligence as GPT-4 Turbo but 
    is much more efficient—it generates text 2x faster and is 50% cheaper. 
    Additionally, GPT-4o has the best vision and performance across 
    non-English languages of any of our models. GPT-4o is available in 
    the OpenAI API to paying customers.

    Point can change
    """
    gpt_4o_2024_11_20 = OpenAiModelInfo("gpt-4o-2024-11-20", True, NEVER_THINK, 128_000)
    gpt_4o_2024_05_13 = OpenAiModelInfo("gpt-4o-2024-05-13", True, NEVER_THINK, 128_000)
    gpt_4o_mini = OpenAiModelInfo("gpt-4o-mini", True, NEVER_THINK, 128_000)
    """Our affordable and intelligent small model for fast, lightweight tasks. 
    GPT-4o mini is cheaper and more capable than GPT-3.5 Turbo. """
    gpt_4o_mini_2024_07_18 = OpenAiModelInfo(
        "gpt-4o-mini-2024-07-18", True, NEVER_THINK, 128_000
    )

    o1 = OpenAiModelInfo("o1", True, ALWAYS_THINK, 200_000)
    """
    The o1 series of large language models are trained with reinforcement learning to perform complex reasoning. o1 models think before they answer, producing a long internal chain of thought before responding to the user.
    """
    o1_preview = OpenAiModelInfo("o1-preview", True, ALWAYS_THINK, 128_000)
    """o1-preview: reasoning model designed to solve hard problems across domains."""
    o1_preview_2024_09_12 = OpenAiModelInfo(
        "o1-preview-2024-09-12", True, ALWAYS_THINK, 128_000
    )
    """o1-mini: faster and cheaper reasoning model particularly good at coding, math, and science."""
    o1_mini = OpenAiModelInfo("o1-mini", True, ALWAYS_THINK, 128_000)
    o1_mini_2024_09_12 = OpenAiModelInfo("o1-mini-2024-09-12", True, ALWAYS_THINK, 128_000)

    o3_mini = OpenAiModelInfo("o3-mini", True, ALWAYS_THINK, 200_000)
    o4_mini = OpenAiModelInfo("o4-mini", True, ALWAYS_THINK, 200_000)
    o4_mini_2025_04_16 = OpenAiModelInfo("o4-mini-2025-04-16", True, ALWAYS_THINK, 200_000)

    o3 = OpenAiModelInfo("o3", True, ALWAYS_THINK, 200_000)

    gpt_4_1 = OpenAiModelInfo("gpt-4.1", True, NEVER_THINK, 1_047_576)
    gpt_4_1_mini = OpenAiModelInfo("gpt-4.1-mini", True, NEVER_THINK, 1_047_576)
    gpt_4_1_nano = OpenAiModelInfo("gpt-4.1-nano", True, NEVER_THINK, 1_047_576)

    gpt_5 = OpenAiModelInfo("gpt-5", True, ALWAYS_THINK, 400_000)
    gpt_5_mini = OpenAiModelInfo("gpt-5-mini", True, ALWAYS_THINK, 400_000)
    gpt_5_nano = OpenAiModelInfo("gpt-5-nano", True, ALWAYS_THINK, 400_000)

    gpt_5_1 = OpenAiModelInfo("gpt-5.1", True, CONFIGURABLE_THINKING, 400_000)

    @classmethod
    def name_to_info(cls, name: str) -> OpenAiModelInfo | None:
        if isinstance(name, OpenAiModelInfo):
            return name
        for info in cls:
            if info == name:
                return info
        return None


def get_open_ai_lm(
    model_name: str = OpenAiModelNames.gpt_4_1_mini,
    api_key_secret: SecretInterface | None = None,
    organization: str | None = None,
    cache_outputs_default: bool = False,
    retry_on_rate_limit: bool = False,
) -> "OpenAIPredictor":
    if api_key_secret is None:
        api_key_secret = SecretEnvVar("OPENAI_API_KEY")
        if not api_key_secret.is_readable():
            api_key_secret = SecretFile(Path("~/oai_key.txt").expanduser())
        if not api_key_secret.is_readable():
            msg = (
                "Cannot find an API key. By default the OPENAI_API_KEY environment"
                " variable is used if it is available. Otherwise it will read from a"
                " file at ~/oai_key.txt. Please place the key at one of the locations"
                " or pass in a SecretInterface (like SecretEnvVar or SecretFile object)"
                " to the api_key_secret argument.\nYou can get an API key from"
                " https://platform.openai.com/account/api-keys"
            )
            raise ValueError(
                msg,
            )
    assert_is_a_secret(api_key_secret)

    if not api_key_secret.is_readable():
        msg = "API key is not defined"
        raise ValueError(msg)

    if organization is None:
        org_secret = SecretEnvVar("OPENAI_ORGANIZATION")
        if org_secret.is_readable():
            organization = org_secret.get_secret().strip()

    client = OpenAI(
        api_key=api_key_secret.get_secret().strip(),
        organization=organization if organization is not None else None,
    )

    return OpenAIPredictor(
        api=client,
        engine_name=model_name,
        cache_outputs_default=cache_outputs_default,
        retry_on_rate_limit=retry_on_rate_limit,
    )


@dataclasses.dataclass
class OpenAiLmPrediction(LmPrediction):
    # tokenizer: tiktoken.Encoding = dataclasses.field(default=None, kw_only=False)

    def _get_completion_token_index(self):
        """
        If echoing the completion text might not start at the begining. Returns the
        index of the actually new tokens
        """
        if not self.prompt.echo:
            return 0
        # binary search for the first token after the prompt length
        prompt_len = len(self.prompt.text) - 1
        all_offsets: list[int] = self._all_toks_offsets()
        return bisect.bisect_right(all_offsets, prompt_len)

    def _all_toks(self):
        if not self.prompt.logprobs:
            msg = (
                "This property is only available when the prompt `logprobs` flag is set"
                " (openai endpoint only will return tokens when logprobs is set)"
            )
            raise ValueError(
                msg,
            )
        probably_chat = hasattr(self.metad.logprobs, "content")
        if not probably_chat:
            return self.metad.logprobs.tokens
        else:
            return [lp.token for lp in self.metad.logprobs.content]

    def _all_toks_offsets(self):
        return self.metad.logprobs.text_offset

    def _all_logprobs(self):
        if self.metad.logprobs is None:
            assert self.prompt.logprobs is None or self.prompt.logprobs == 0
            return None
        probably_chat = hasattr(self.metad.logprobs, "content")
        if not probably_chat:
            return self.metad.logprobs.token_logprobs
        else:
            return [lp.logprob for lp in self.metad.logprobs.content]

    @property
    def completion_tokens(self):
        return self._all_toks()[self._get_completion_token_index() :]

    @property
    def completion_token_offsets(self):
        return self._all_toks_offsets()[self._get_completion_token_index() :]

    @property
    def completion_logprobs(self):
        """Note that this will only be valid if set a logprob value in the prompt"""
        self._verify_logprobs()
        return self._all_logprobs()[self._get_completion_token_index() :]

    def _verify_echo(self):
        if not self.prompt.echo:
            msg = "This property is only available when the prompt `echo` flag is set"
            raise ValueError(
                msg,
            )

    @property
    def prompt_tokens(self):
        self._verify_echo()
        return self._all_toks()[: self._get_completion_token_index()]

    @property
    def prompt_token_offsets(self):
        self._verify_echo()
        return self._all_toks_offsets()[: self._get_completion_token_index()]

    @property
    def prompt_logprobs(self):
        self._verify_echo()
        self._verify_logprobs()
        return self._all_logprobs()[: self._get_completion_token_index()]

    @property
    def full_logprobs(self):
        return self.prompt_logprobs + self.completion_logprobs

    def get_full_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    def get_full_text(self):
        return self.prompt.text + self.completion_text

    @property
    def logprobs_dict(self):
        return [
            {
                "repr": repr(token),
                "probability": logprob,
            }
            for token, logprob in zip(
                self.completion_tokens,
                self.completion_logprobs,
                strict=True,
            )
        ]

    @property
    def top_token_logprobs(self) -> list[dict[str, float]]:
        """
        List of dictionaries of token:logprob for each completion.
        The API will always return the logprob of the sampled token,
        so there may be up to logprobs+1 elements in the response.
        """
        if self.metad.logprobs is None:
            msg = (
                "Response does not contain top_logprobs. Are you sure logprobs was set"
                f" > 0? Currently: {self.prompt.logprobs}"
            )
            raise ValueError(
                msg,
            )

        if isinstance(self.metad.logprobs, Logprobs):
            return self.metad.logprobs.top_logprobs

        top_logprobs = []
        for p in self.metad.logprobs.content:  # for each token
            odict = dict([(t.token, t.logprob) for t in p.top_logprobs])
            top_logprobs.append(odict)
        return top_logprobs


# class OpenAiLmChatPrediction(LmPrediction):
#    pass


class OpenAIPredictor(LmPredictor):
    _instantiation_hooks: list["OpenAiInstantiationHook"] = []

    def __init__(
        self,
        api: OpenAI,
        engine_name: str,
        chat_mode: bool | None = None,
        reasoning_style: LmReasoningStyle | None = None,
        cache_outputs_default: bool = False,
        retry_on_rate_limit: bool = False,
    ):
        for hook in self._instantiation_hooks:
            hook.before_init(
                self,
                api,
                engine_name,
                chat_mode,
                cache_outputs_default,
                retry_on_rate_limit,
            )
        super().__init__(cache_outputs_default)
        self._api = api
        self._engine_name = engine_name
        self._cache_outputs_default = cache_outputs_default
        self._retry_on_rate_limit = retry_on_rate_limit
        info = OpenAiModelNames.name_to_info(engine_name)
        self._chat_mode = info.is_chat_model if chat_mode is None else chat_mode
        self._reasoning_style = info.reasoning_style if reasoning_style is None else reasoning_style
        if self._chat_mode is None:
            msg = (
                "`chat_mode` is not provided as a parameter and cannot be inferred from"
                " engine name"
            )
            raise ValueError(
                msg,
            )
        self._token_limit = info.token_limit if info is not None else None
        self._tokenizer = None

    @classmethod
    def add_instantiation_hook(cls, hook: "OpenAiInstantiationHook"):
        """
        This method should likely not be used normally.
        It is intended add constraints on kinds of models that are
        instantiation to better control usage. An example usage checking
        keys are used correctly like that certain keys are used with particular
        models
        """
        cls._instantiation_hooks.append(hook)

    def find_prediction_class(self, prompt):
        # if self._chat_mode:
        #    return OpenAiLmChatPrediction
        return OpenAiLmPrediction

    @property
    def supports_token_operations(self) -> bool:
        return True

    @property
    def reasoning_style(self) -> LmReasoningStyle:
        return self._reasoning_style

    def _validate_prompt(self, prompt: LmPrompt, raise_on_invalid: bool = True) -> tuple[bool, LmPrompt | None]:
        is_valid = True
        modified_prompt = prompt
        if prompt.logprobs is not None and prompt.logprobs > MAX_LOG_PROB_PARM:
            message = (
                f"Openai limits logprobs to be <= {MAX_LOG_PROB_PARM}. Larger values"
                " might cause unexpected behavior if you later are dependingon more"
                " returns"
            )
            if raise_on_invalid:
                raise ValueError(message)
            else:
                warnings.warn(message)
                return False
        if self._reasoning_style == ALWAYS_THINK:
            # if prompt.max_tokens:
            #     message = f"o1 type models use `max_completion_tokens` instead of `max_tokens` but you have set max tokens to f{prompt.max_tokens}. Instead, `max_completion_tokens` will be used. `max_completion_tokens` is currently set to: {prompt.max_completion_tokens}"
            #     if raise_on_invalid:
            #         raise ValueError(message)
            #     else:
            #         warnings.warn(message)
            #         return False
            if prompt.logprobs is not None and prompt.logprobs > 0:
                message = f"logprobs is set to {prompt.logprobs} but o1 models do not support logprobs. This will be ignored and a value of `None` will be used instead."
                warnings.warn(message)
                modified_prompt = dataclasses.replace(modified_prompt, logprobs=None)
                is_valid = False
            if prompt.temperature != 1.0:
                message = f"temperature is set to {prompt.temperature} but o1 models do not support temperature."
                if raise_on_invalid:
                    raise ValueError(message)
                else:
                    message += " This will be ignored and a value of 1.0 will be used instead."
                    warnings.warn(message)
                    modified_prompt = dataclasses.replace(modified_prompt, temperature=1.0)
                    is_valid = False
            if prompt.stop:
                message = f"stop is set to {prompt.stop} but o1 models do not support stop tokens."
                if raise_on_invalid:
                    raise ValueError(message)
                else:
                    message += " This will be ignored."
                    warnings.warn(message)
                    modified_prompt = dataclasses.replace(modified_prompt, stop=None)
                    is_valid = False
        return is_valid, modified_prompt

    def model_name(self):
        return self._engine_name

    def get_model_cache_key(self):
        return "OpenAI::" + self._engine_name

    def list_engines(self):
        return self._api.Engine.list()

    @property
    def is_chat_model(self):
        return self._chat_mode

    @property
    def token_limit(self):
        return self._token_limit

    def _build_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = tiktoken.encoding_for_model(self._engine_name)

    def tokenize_ids(self, input_str: str) -> list[int]:
        self._build_tokenizer()
        return self._tokenizer.encode(input_str)

    def tokenize(self, input_str: str) -> list[str]:
        self._build_tokenizer()
        token_bytes: list[bytes] = self._tokenizer.decode_tokens_bytes(
            self.tokenize_ids(input_str),
        )
        return [tok.decode("utf-8") for tok in token_bytes]

    def estimate_tokens_in_prompt(self, prompt: LmPrompt) -> int:
        """
        Estimate the number of tokens in the prompt.
        This is not always an exact measure, as for the chat models there extra metadata provided.
        The documentation on ChatMl (https://github.com/openai/openai-python/blob/main/chatml.md)
        gives some details but is imprecise. We want to write this to ideally overestimate the
        number of tokens so that will conservatively not go over the limit.
        """
        self._build_tokenizer()
        if self._chat_mode:
            val = len(
                self._tokenizer.encode(
                    prompt.get_text_as_chat().to_default_string_prompt(),
                ),
            )
            val += (
                len(prompt.get_text_as_chat()) * 3
            )  # Extra buffer for each turn transition
            val += 2  # Extra setup tokens
        else:
            val = len(self._tokenizer.encode(prompt.get_text_as_string_default_form()))
        return val

    def _predict_maybe_cached(
        self,
        prompt: LmPrompt,
    ) -> LmPrediction | list[LmPrediction]:
        if PRINT_ON_PREDICT:
            print("RUN PREDICT ", prompt.text[: min(10, len(prompt.text))])

        def run_func():
            # Wait for rate limit
            LmPredictor._wait_ratelimit()
            args = prompt_to_openai_args_dict(
                prompt,
                self._engine_name,
                self._chat_mode,
                self._reasoning_style,
                default_tokens_generated=self.default_tokens_generated,
            )

            try:
                if not self._chat_mode:
                    return self._api.completions.create(**args)
                else:
                    return self._api.chat.completions.create(**args)
            except RateLimitError as e:
                print(e)
                return e

        def is_success_func(result):
            return not isinstance(result, RateLimitError)

        if self._retry_on_rate_limit:
            completion = attempt_with_exponential_backoff(
                run_func,
                is_success_func,
                backoff_time=parse_backoff_time,
            )
        else:
            completion = run_func()

        if not is_success_func(completion):
            raise completion

        preds = self.prediction_from_api_response(completion, prompt)
        # if prompt.num_completions is None:
        #    return preds[0]
        return preds

    def prediction_from_api_response(
        self,
        response: openai.types.Completion | openai.types.chat.ChatCompletion,
        prompt: LmPrompt,
    ) -> list[LmPrediction]:
        choices = response.choices

        def get_completion_text(text):
            if not prompt.echo:
                return text
            return text[len(prompt.text) :]

        def get_text_from_choice(choice):
            return choice.text if not self._chat_mode else choice.message.content

        return [
            OpenAiLmPrediction(
                get_completion_text(get_text_from_choice(choice)),
                prompt,
                choice,
                internals=None,
                # tokenizer=self._tokenizer
            )
            for choice in choices
        ]

    def predict_many(
        self,
        prompts: list[LmPrompt],
        completion_window: CompletionWindow,
    ) -> Iterable[LmPrediction | list[LmPrediction]]:
        self._validate_predict_many_prompts(prompts)
        if completion_window == CompletionWindow.BATCH_ANY:
            from lmwrapper.openai_wrapper import batching

            # ^ putting this here to prevent circular import.
            #   probably some more clever way...
            batch_manager = batching.OpenAiBatchManager(
                prompts=prompts,
                cache=self._disk_cache,
                maintain_order=True,
            )
            batch_manager.start_batch()
            yield from batch_manager.iter_results()
            return
        yield from super().predict_many(prompts, completion_window)

    def remove_special_chars_from_tokens(self, tokens: list[str]) -> list[str]:
        return tokens

    @property
    def default_tokens_generated(self) -> int | None:
        # For the completion models it used to 16.
        # https://platform.openai.com/docs/api-reference/completions/create#completions/create-max_tokens
        # However, now there does not appear to be a limit for the chat models
        # https://platform.openai.com/docs/api-reference/chat/create#chat-create-max_completion_tokens
        # We'll just switch over to no limit (or the total limit of the model)
        if not self.is_chat_model:
            return 16
        return None

    def __reduce__(self):
        return (
            get_open_ai_lm,  # top‑level factory
            (
                self._engine_name,  # model_name
                None, # Don't pickle the api key
                self._api.organization,  # serialize Org ID not sure if smart
                self._cache_outputs_default,
                self._retry_on_rate_limit,
            ),
        )


class OpenAiInstantiationHook(ABC):
    """
    Potentially used to add API controls on predictor instantiation.
    An example usecase is to make sure certain keys are only used with
    certain models.
    """

    def __init__(self):
        pass

    @abstractmethod
    def before_init(
        self,
        new_predictor: OpenAIPredictor,
        api,
        engine_name: str,
        chat_mode: bool,
        cache_outputs_default: bool,
        retry_on_rate_limit: bool,
    ):
        raise NotImplementedError


def attempt_with_exponential_backoff(
    call_func,
    is_success_func,
    backoff_time=None,
    backoff_cap=60,
):
    """
    Attempts to get a result from call_func. Uses is_success_func
    to determine if the result was a success or not. If not a success
    then will sleep for a random amount of time between 1 and 2^attempts
    """
    result = call_func()
    attempts = 1
    sleep_time = False
    while not is_success_func(result):
        if backoff_time:
            sleep_time = backoff_time(result)
        if not backoff_time or not sleep_time:
            sleep_time = random.randint(1, min(2**attempts, backoff_cap))
        print(f"Rate limit error. Sleeping for {sleep_time} seconds")
        time.sleep(sleep_time)
        result = call_func()
        attempts += 1
    return result


def parse_backoff_time(exception: RateLimitError) -> int | None:
    # Please try again in 3s.
    regex = r".*Please try again in (\d+(\.\d+)?)s\..*"
    matches = re.findall(regex, exception.message)
    if matches:
        return int(math.ceil(float(matches[0][0])))
    # Newer format
    # Please try again in 837ms
    regex = r".*Please try again in (\d+)ms.*"
    matches = re.findall(regex, exception.message)
    if matches:
        return int(max(int(matches[0]) / 1000, 1))
    print(f"Unable to parse backoff time. Message: {exception.message}")
    return None


def prompt_to_openai_args_dict(
    prompt: LmPrompt,
    engine_name: str,
    chat_model: bool,
    reasoning_style: LmReasoningStyle,
    default_tokens_generated: int | None = 20,
) -> dict[str, Any]:
    if prompt.max_tokens is not None:
        max_tokens = prompt.max_tokens
    else:
        max_tokens = default_tokens_generated
    if reasoning_style == ALWAYS_THINK:
        args = dict(
            messages=prompt.get_text_as_chat().as_dicts(),
            model=engine_name,
            max_completion_tokens=max_tokens,
            n=prompt.num_completions or 1,
            presence_penalty=prompt.presence_penalty,
            top_logprobs=None,
        )
        if prompt.top_p is not None:
            args["top_p"] = prompt.top_p
        return args
    # all o1 models are chat models
    elif chat_model:
        args = dict(
            messages=prompt.get_text_as_chat().as_dicts(),
            model=engine_name,
            logprobs=prompt.logprobs > 0,
            max_completion_tokens=max_tokens,
            n=prompt.num_completions or 1,
            presence_penalty=prompt.presence_penalty,
            stop=prompt.stop,
            temperature=prompt.temperature,
            # top_logprobs accepts ints 0 to 20, logprobs must be a boolean true
            top_logprobs=prompt.logprobs if prompt.logprobs > 0 else None,
        )
        if prompt.top_p is not None:
            args["top_p"] = prompt.top_p
        return args
    else:
        args = dict(
            model=engine_name,
            prompt=prompt.get_text_as_string_default_form(),
            max_tokens=max_tokens,
            stop=prompt.stop,
            stream=False,
            logprobs=prompt.logprobs,
            temperature=prompt.temperature,
            presence_penalty=prompt.presence_penalty,
            n=prompt.num_completions or 1,
            echo=prompt.echo,
        )
        if prompt.top_p is not None:
            args["top_p"] = prompt.top_p
        return args
