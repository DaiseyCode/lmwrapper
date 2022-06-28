from abc import abstractmethod
from typing import Union, Dict

from joblib import Memory

from lmwrapper.caching import get_disk_cache
from lmwrapper.structs import LmPrompt, LmPrediction


class LmPredictor:
    def __init__(
        self,
        cache_default: bool = False,
        cache_object: Memory = None,
    ):
        if cache_object is None:
            cache_object = get_disk_cache()
        self._cache_default = cache_default
        self._cached_predict = cache_object.cache(
            func=self._predict_maybe_cached,
        )

    def _get_cache_key_metadata(self):
        """Used to potentially add extra info that defines how predictions are cached"""
        return {}

    def predict(
        self,
        prompt: Union[str, LmPrompt],
    ) -> LmPrediction:
        should_cache = self._cache_default if prompt.cache is None else prompt.cache
        if should_cache:
            return self._cached_predict(prompt, self._get_cache_key_metadata())
        else:
            return self._predict_maybe_cached(prompt)


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