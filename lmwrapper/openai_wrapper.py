import random
from pathlib import Path
from typing import Union, List
from termcolor import colored
from lmwrapper.abstract_predictor import LmPredictor
from lmwrapper.caching import get_disk_cache
from lmwrapper.secret_manage import SecretInterface, SecretFile, assert_is_a_secret
from lmwrapper.structs import LmPrompt, LmPrediction
import bisect


cur_file = Path(__file__).parent.absolute()
diskcache = get_disk_cache()


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
        return self.metad['logprobs']['token_logprobs']

    @property
    def completion_tokens(self):
        return self._all_toks()[self._get_completion_token_index():]

    @property
    def completion_token_offsets(self):
        return self._all_toks_offsets()[self._get_completion_token_index():]

    @property
    def completion_logprobs(self):
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


class OpenAIPredictor(LmPredictor):
    def __init__(
        self,
        api,
        engine_name: str,
        cache_outputs_default: bool = False,
    ):
        super().__init__(cache_outputs_default)
        self._api = api
        self._engine_name = engine_name
        self._cache_outputs_default = cache_outputs_default

    def model_name(self):
        return self._engine_name

    def _get_cache_key_metadata(self):
        return {
            "engine": self._engine_name,
        }

    def list_engines(self):
        return self._api.Engine.list()

    def _predict_maybe_cached(self, prompt: LmPrompt) -> Union[LmPrediction, List[LmPrediction]]:
        print(self._engine_name)
        completion = self._api.Completion.create(
            engine=self._engine_name,
            prompt=prompt.text,
            max_tokens=prompt.max_toks,
            stop=prompt.stop,
            stream=False,
            logprobs=prompt.logprobs,
            temperature=prompt.temperature,
            top_p=prompt.top_p,
            presence_penalty=prompt.presence_penalty,
            n=prompt.num_completions,
            echo=prompt.echo,
        )
        choices = completion['choices']

        def get_completion_text(text):
            if not prompt.echo:
                return text
            return text[len(prompt.text):]
        out = [
            OpenAiLmPrediction(get_completion_text(choice['text']), prompt, choice)
            for choice in choices
        ]
        if len(choices) == 1:
            return out[0]
        return out


def get_goose_lm(
    model_name: str = "gpt-neo-125m",
    api_key_secret: SecretInterface = None,
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
    )


def get_open_ai_lm(
    model_name: str = "text-ada-001",
    api_key_secret: SecretInterface = None,
    organization: str = None,
):
    if api_key_secret is None:
        api_key_secret = SecretFile(Path("~/oai_key.txt").expanduser())
    assert_is_a_secret(api_key_secret)
    import openai
    openai.api_key = api_key_secret.get_secret().strip()
    if organization:
        openai.organization = organization
    return OpenAIPredictor(
        api=openai,
        engine_name=model_name,
    )
