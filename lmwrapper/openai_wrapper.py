import random
import tiktoken
import time
import warnings
from pathlib import Path
from typing import Union, List, Optional, Iterable

import openai.error
from lmwrapper.abstract_predictor import LmPredictor
from lmwrapper.caching import get_disk_cache
from lmwrapper.secret_manage import SecretInterface, SecretFile, assert_is_a_secret, SecretEnvVar
from lmwrapper.structs import LmPrompt, LmPrediction
import bisect

from lmwrapper.util import StrEnum

cur_file = Path(__file__).parent.absolute()
diskcache = get_disk_cache()


PRINT_ON_PREDICT = True

MAX_LOG_PROB_PARM = 5


class OpenAiLmPrediction(LmPrediction):
    def _get_completion_token_index(self):
        """If echoing the completion text might not start at the begining. Returns the
        index of the actually new tokens"""
        if not self.prompt.echo:
            return 0
        # binary search for the first token after the prompt length
        prompt_len = len(self.prompt.text) - 1
        all_offsets: List[int] = self._all_toks_offsets()
        return bisect.bisect_right(all_offsets, prompt_len)

    def _all_toks(self):
        if not self.prompt.logprobs:
            raise ValueError("This property is only available when the prompt "
                             "`logprobs` flag is set (openai endpoint only will "
                             "return tokens when logprobs is set)")
        return self.metad['logprobs']['tokens']

    def _all_toks_offsets(self):
        return self.metad['logprobs']['text_offset']

    def _all_logprobs(self):
        if self.metad['logprobs'] is None:
            assert self.prompt.logprobs is None or self.prompt.logprobs == 0
            return None
        return self.metad['logprobs']['token_logprobs']

    @property
    def completion_tokens(self):
        return self._all_toks()[self._get_completion_token_index():]

    @property
    def completion_token_offsets(self):
        return self._all_toks_offsets()[self._get_completion_token_index():]

    @property
    def completion_logprobs(self):
        """Note that this will only be valid if set a logprob value in the prompt"""
        self._verify_logprobs()
        return self._all_logprobs()[self._get_completion_token_index():]

    def _verify_echo(self):
        if not self.prompt.echo:
            raise ValueError("This property is only available when the prompt `echo` flag is set")

    @property
    def prompt_tokens(self):
        self._verify_echo()
        return self._all_toks()[:self._get_completion_token_index()]

    @property
    def prompt_token_offsets(self):
        self._verify_echo()
        return self._all_toks_offsets()[:self._get_completion_token_index()]

    @property
    def prompt_logprobs(self):
        self._verify_echo()
        self._verify_logprobs()
        return self._all_logprobs()[:self._get_completion_token_index()]

    @property
    def full_logprobs(self):
        return self.prompt_logprobs + self.completion_logprobs

    def get_full_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    def get_full_text(self):
        return self.prompt.text + self.completion_text


class OpenAiLmChatPrediction(LmPrediction):
    pass


class OpenAIPredictor(LmPredictor):
    def __init__(
        self,
        api,
        engine_name: str,
        chat_mode: bool = None,
        cache_outputs_default: bool = False,
        retry_on_rate_limit: bool = False,
    ):
        super().__init__(cache_outputs_default)
        self._api = api
        self._engine_name = engine_name
        self._cache_outputs_default = cache_outputs_default
        self._retry_on_rate_limit = retry_on_rate_limit
        info = OpenAiModelNames.name_to_info(engine_name)
        self._chat_mode = (
            info.is_chat_model
            if chat_mode is None else chat_mode
        )
        if self._chat_mode is None:
            raise ValueError("`chat_mode` is not provided as a parameter and "
                             "cannot be inferred from engine name")
        self._token_limit = info.token_limit if info is not None else None
        self._tokenizer = None

    def _validate_prompt(self, prompt: LmPrompt, raise_on_invalid: bool = True) -> bool:
        if prompt.logprobs is not None and prompt.logprobs > MAX_LOG_PROB_PARM:
            warnings.warn(f"Openai limits logprobs to be <= {MAX_LOG_PROB_PARM}. "
                          f"Larger values might cause unexpected behavior if you later are depending"
                          f"on more returns")

    def model_name(self):
        return self._engine_name

    def _get_cache_key_metadata(self):
        return {
            "engine": self._engine_name,
        }

    def list_engines(self):
        return self._api.Engine.list()

    @property
    def is_chat_model(self):
        return self._chat_mode

    @property
    def token_limit(self):
        return self._token_limit

    def estimate_tokens_in_prompt(self, prompt: LmPrompt) -> int:
        """Estimate the number of tokens in the prompt. 
        This is not always an exact measure, as for the chat models there extra metadata provided.
        The documentation on ChatMl (https://github.com/openai/openai-python/blob/main/chatml.md)
        gives some details but is imprecise. We want to write this to ideally overestimate the
        number of tokens so that will conservatively not go over the limit."""
        if self._tokenizer is None:
            self._tokenizer = tiktoken.encoding_for_model(self._engine_name)
        if self._chat_mode:
            val = len(self._tokenizer.encode(prompt.get_text_as_chat().to_default_string_prompt()))
            val += len(prompt.get_text_as_chat()) * 3  # Extra buffer for each turn transition
            val += 2  # Extra setup tokens
        else:
            val = len(self._tokenizer.encode(prompt.get_text_as_string_default_form()))
        return val

    def _predict_maybe_cached(self, prompt: LmPrompt) -> Union[LmPrediction, List[LmPrediction]]:
        if PRINT_ON_PREDICT:
            print("RUN PREDICT ", prompt.text[:min(10, len(prompt.text))])

        def run_func():
            try:
                if not self._chat_mode:
                    return self._api.Completion.create(
                        engine=self._engine_name,
                        prompt=prompt.get_text_as_string_default_form(),
                        max_tokens=prompt.max_tokens,
                        stop=prompt.stop,
                        stream=False,
                        logprobs=prompt.logprobs,
                        temperature=prompt.temperature,
                        top_p=prompt.top_p,
                        presence_penalty=prompt.presence_penalty,
                        n=prompt.num_completions,
                        echo=prompt.echo,
                    )
                else:
                    return self._api.ChatCompletion.create(
                        model=self._engine_name,
                        messages=prompt.get_text_as_chat().as_dicts(),
                        temperature=prompt.temperature,
                        max_tokens=prompt.max_tokens,
                        stop=prompt.stop,
                        top_p=prompt.top_p,
                        n=prompt.num_completions,
                        presence_penalty=prompt.presence_penalty,
                    )
            except openai.error.RateLimitError as e:
                print(e)
                return e

        def is_success_func(result):
            return not isinstance(result, openai.error.RateLimitError)

        if self._retry_on_rate_limit:
            completion = attempt_with_exponential_backoff(run_func, is_success_func)
        else:
            completion = run_func()

        if not is_success_func(completion):
            raise completion

        choices = completion['choices']

        def get_completion_text(text):
            if not prompt.echo:
                return text
            return text[len(prompt.text):]

        def get_text_from_choice(choice):
            return choice['text'] if not self._chat_mode else choice['message']['content']

        out = [
            OpenAiLmPrediction(get_completion_text(get_text_from_choice(choice)), prompt, choice)
            for choice in choices
        ]
        if len(choices) == 1:
            return out[0]
        return out

    def remove_special_chars_from_tokens(self, tokens: list[str]) -> list[str]:
        return tokens


def attempt_with_exponential_backoff(
    call_func,
    is_success_func,
    backoff_cap=60,
):
    """Attempts to get a result from call_func. Uses is_success_func
    to determine if the result was a success or not. If not a success
    then will sleep for a random amount of time between 1 and 2^attempts"""
    result = call_func()
    attempts = 1
    while not is_success_func(result):
        sleep_time = random.randint(1, min(2**attempts, backoff_cap))
        print("Rate limit error. Sleeping for {} seconds".format(sleep_time))
        time.sleep(sleep_time)
        result = call_func()
        attempts += 1
    return result


def get_goose_lm(
    model_name: str = "gpt-neo-125m",
    api_key_secret: SecretInterface = None,
    retry_on_rate_limit: bool = False,
):
    if api_key_secret is None:
        api_key_secret = SecretFile(Path("~/goose_key.txt").expanduser())
    assert_is_a_secret(api_key_secret)
    import openai
    openai.api_key = api_key_secret.get_secret().strip()
    openai.api_base = "https://api.goose.ai/v1"
    return OpenAIPredictor(
        api=openai,
        engine_name=model_name,
        retry_on_rate_limit=retry_on_rate_limit,
    )


class OpenAiModelInfo(str):
    def __new__(cls, name: str, is_chat_model: bool, token_limit: int):
        instance = super().__new__(cls, name)
        instance._is_chat_model = is_chat_model
        instance._token_limit = token_limit
        return instance

    @property
    def is_chat_model(self):
        return self._is_chat_model

    @property
    def token_limit(self):
        return self._token_limit


class _ModelNamesMeta(type):
    def __iter__(cls):
        for attr in cls.__dict__:
            if isinstance(cls.__dict__[attr], OpenAiModelInfo):
                yield cls.__dict__[attr]


class OpenAiModelNames(metaclass=_ModelNamesMeta):
    text_ada_001 = OpenAiModelInfo("text-ada-001", False, 2049)
    """Capable of very simple tasks, usually the fastest model in the GPT-3 series, and lowest cost."""
    text_davinci_003 = OpenAiModelInfo("text-davinci-003", False, 4097)
    """Can do any language task with better quality, longer output, and consistent instruction-following
    than the curie, babbage, or ada models.
    Also supports some additional features such as inserting text."""
    gpt_3_5_turbo = OpenAiModelInfo("gpt-3.5-turbo", True, 4096)
    """	Most capable GPT-3.5 model and optimized for chat at 1/10th the cost of text-davinci-003. 
    Will be updated with our latest model iteration 2 weeks after it is released."""
    gpt_3_5_turbo_16k = OpenAiModelInfo("gpt-3.5-turbo-16k", True, 16384)
    """Same capabilities as the standard gpt-3.5-turbo model but with 4 times the context."""
    gpt_4 = OpenAiModelInfo("gpt-4", True, 8192)
    """More capable than any GPT-3.5 model, able to do more complex tasks, and optimized for chat. 
    Will be updated with our latest model iteration 2 weeks after it is released."""
    gpt_4_32k = OpenAiModelInfo("gpt-4-32k", True, 32768)
    """Same capabilities as the base gpt-4 mode but with 4x the context length. 
    Will be updated with our latest model iteration."""

    @classmethod
    def name_to_info(cls, name: str) -> Optional[OpenAiModelInfo]:
        if isinstance(name, OpenAiModelInfo):
            return name
        for info in cls:
            if info == name:
                return info
        return None


def get_open_ai_lm(
    model_name: str = OpenAiModelNames.text_ada_001,
    api_key_secret: SecretInterface = None,
    organization: str = None,
    cache_outputs_default: bool = False,
    retry_on_rate_limit: bool = False,
) -> OpenAIPredictor:
    if api_key_secret is None:
        api_key_secret = SecretEnvVar("OPENAI_API_KEY")
        if not api_key_secret.is_readable():
            api_key_secret = SecretFile(Path("~/oai_key.txt").expanduser())
        if not api_key_secret.is_readable():
            raise ValueError((
                "Cannot find an API key. "
                "By default the OPENAI_API_KEY environment variable is used if it is available. "
                "Otherwise it will read from a file at ~/oai_key.txt. "
                "Please place the key at one of the locations or pass in a SecretInterface "
                "(like SecretEnvVar or SecretFile object) to the api_key_secret argument."
                "\n"
                "You can get an API key from https://platform.openai.com/account/api-keys"
            ))
    assert_is_a_secret(api_key_secret)
    import openai
    if not api_key_secret.is_readable():
        raise ValueError("API key is not defined")
    openai.api_key = api_key_secret.get_secret().strip()
    if organization:
        openai.organization = organization
    return OpenAIPredictor(
        api=openai,
        engine_name=model_name,
        cache_outputs_default=cache_outputs_default,
        retry_on_rate_limit=retry_on_rate_limit,
    )
