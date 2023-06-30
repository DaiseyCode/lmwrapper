import random
import time
from pathlib import Path
from typing import Union, List

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

    def _verify_logprobs(self):
        if self.prompt.logprobs is None or self.prompt.logprobs == 0:
            raise ValueError("This property is not available unless the prompt logprobs is set")

    @property
    def completion_logprobs(self):
        """Note that this will only be valid if set a logprob value in the prompt"""
        all_logprobs = self._all_logprobs()
        self._verify_logprobs()
        return self._all_logprobs()[self._get_completion_token_index():]

    def _verify_echo(self):
        if not self.prompt.echo:
            raise ValueError("This property is not available when prompt is not echoed")

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
        return self._all_logprobs()[:self._get_completion_token_index()]

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
        self._chat_mode = (
            is_model_chat_model(engine_name)
            if chat_mode is None else chat_mode
        )
        if self._chat_mode is None:
            raise ValueError("Chat mode is not defined and cannot be inferred from engine name")

    def model_name(self):
        return self._engine_name

    def _get_cache_key_metadata(self):
        return {
            "engine": self._engine_name,
        }

    def list_engines(self):
        return self._api.Engine.list()

    def _predict_maybe_cached(self, prompt: LmPrompt) -> Union[LmPrediction, List[LmPrediction]]:
        if PRINT_ON_PREDICT:
            print("RUN PREDICT ", prompt.text[:min(10, len(prompt.text))])

        def run_func():
            try:
                if not self._chat_mode:
                    return self._api.Completion.create(
                        engine=self._engine_name,
                        prompt=prompt.text,
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


class OpenAiModelNames(StrEnum):
    text_ada_001 = "text-ada-001"
    """Capable of very simple tasks, usually the fastest model in the GPT-3 series, and lowest cost."""

    text_davinci_003 = "text-davinci-003"
    """Can do any language task with better quality, longer output, and consistent instruction-following 
    than the curie, babbage, or ada models. 
    Also supports some additional features such as inserting text."""

    gpt_3_5_turbo = "gpt-3.5-turbo"
    gpt_4 = "gpt-4"


def is_model_chat_model(model_name: str) -> bool:
    return {
        OpenAiModelNames.text_ada_001: False,
        OpenAiModelNames.text_davinci_003: False,
        OpenAiModelNames.gpt_3_5_turbo: True,

    }.get(model_name, None)


def get_open_ai_lm(
    model_name: str = OpenAiModelNames.text_ada_001,
    api_key_secret: SecretInterface = None,
    organization: str = None,
    retry_on_rate_limit: bool = False,
):
    if api_key_secret is None:
        api_key_secret = SecretEnvVar("OPENAI_API_KEY")
        if not api_key_secret.is_defined():
            api_key_secret = SecretFile(Path("~/oai_key.txt").expanduser())
    assert_is_a_secret(api_key_secret)
    import openai
    openai.api_key = api_key_secret.get_secret().strip()
    if organization:
        openai.organization = organization
    return OpenAIPredictor(
        api=openai,
        engine_name=model_name,
        retry_on_rate_limit=retry_on_rate_limit,
    )
