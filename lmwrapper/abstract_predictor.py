from abc import abstractmethod
from typing import Union, Dict

from joblib import Memory

from lmwrapper.caching import get_disk_cache
from lmwrapper.structs import LmPrompt, LmPrediction


disk_cache = get_disk_cache()


@disk_cache.cache(verbose=0, ignore=['func'])
def _predict_definately_cached(
    func,
    prompt: LmPrompt,
    extra_params: Dict,
):
    return func(prompt)


class LmPredictor:
    def __init__(
        self,
        cache_default: bool = False,
    ):
        self._cache_default = cache_default

    def predict(
        self,
        prompt: Union[str, LmPrompt],
    ) -> LmPrediction:
        prompt = self._cast_prompt(prompt)
        should_cache = self._cache_default if prompt.cache is None else prompt.cache
        if should_cache:
            return _predict_definately_cached(
                self._predict_maybe_cached, prompt, self._get_cache_key_metadata())
        else:
            return self._predict_maybe_cached(prompt)

    def _get_cache_key_metadata(self):
        return {'name': type(self).__name__}

    @abstractmethod
    def _predict_maybe_cached(
        self,
        prompt: LmPrompt,
    ) -> LmPrediction:
        pass

    def _cast_prompt(self, prompt: Union[str, LmPrompt]) -> LmPrompt:
        if isinstance(prompt, str):
            return LmPrompt(prompt, 100)
        return prompt

    def model_name(self):
        return self.__class__.__name__