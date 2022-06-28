import random
from pathlib import Path
from typing import Union
from termcolor import colored
from lmwrapper.abstract_predictor import LmPredictor
from lmwrapper.caching import get_disk_cache
from lmwrapper.secret_manage import SecretInterface, SecretFile, assert_is_a_secret
from lmwrapper.structs import LmPrompt, LmPrediction


cur_file = Path(__file__).parent.absolute()
diskcache = get_disk_cache()


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

    def _predict_maybe_cached(self, prompt: LmPrompt) -> LmPrediction:
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
        )
        return LmPrediction(completion.choices[0].text, completion)


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
