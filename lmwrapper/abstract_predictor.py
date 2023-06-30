from abc import abstractmethod
from typing import Union, Dict, List
from lmwrapper.caching import get_disk_cache
from lmwrapper.structs import LmPrompt, LmPrediction


disk_cache = get_disk_cache()


@disk_cache.memoize(ignore=('func',))
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
            cache_key = (prompt, self._get_cache_key_metadata())
            if cache_key in disk_cache:
                return disk_cache.get(cache_key)
            val = self._predict_maybe_cached(prompt)
            disk_cache.set(cache_key, val)
            return val
        else:
            return self._predict_maybe_cached(prompt)

    def _get_cache_key_metadata(self):
        return {'name': type(self).__name__}

    @abstractmethod
    def _predict_maybe_cached(self, prompt: LmPrompt) -> Union[LmPrediction, List[LmPrediction]]:
        pass

    def _cast_prompt(self, prompt: Union[str, LmPrompt]) -> LmPrompt:
        if isinstance(prompt, str):
            return LmPrompt(prompt, 100)
        return prompt

    def model_name(self):
        return self.__class__.__name__
