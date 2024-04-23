from abc import abstractmethod
from sqlite3 import OperationalError

from ratemate import RateLimit

from lmwrapper.caching import get_disk_cache
from lmwrapper.structs import LmPrediction, LmPrompt


class LmPredictor:
    _rate_limit: RateLimit | None = None

    def __init__(
        self,
        cache_default: bool = False,
    ):
        self._cache_default = cache_default
        self._disk_cache = get_disk_cache()

    def predict(
        self,
        prompt: str | LmPrompt,
    ) -> LmPrediction:
        prompt = self._cast_prompt(prompt)
        should_cache = self._cache_default if prompt.cache is None else prompt.cache
        if should_cache and prompt.model_internals_request is not None:
            raise NotImplementedError(
                "Cannot yet cache predictions with model internals request",
            )
        self._validate_prompt(prompt, raise_on_invalid=True)
        if should_cache:
            cache_key = (prompt, self._get_cache_key_metadata())
            if cache_key in self._disk_cache:
                cache_copy = self._disk_cache.get(cache_key)
                if cache_copy:
                    cache_copy = cache_copy.mark_as_cached()
                return cache_copy
            val = self._predict_maybe_cached(prompt)
            try:
                self._disk_cache.set(cache_key, val)
            except OperationalError as e:
                print("Failed to cache", e)
            return val
        else:
            return self._predict_maybe_cached(prompt)

    def _cache_key_for_prompt(self, prompt):
        return (prompt, self._get_cache_key_metadata())

    def remove_prompt_from_cache(
        self,
        prompt: str | LmPrompt,
    ) -> bool:
        return self._disk_cache.delete(self._cache_key_for_prompt(prompt))

    def _validate_prompt(self, prompt: LmPrompt, raise_on_invalid: bool = True) -> bool:
        """Called on prediction to make sure the prompt is valid for the model"""
        return True

    @abstractmethod
    def _get_cache_key_metadata(self):
        return {"name": type(self).__name__}

    @abstractmethod
    def _predict_maybe_cached(
        self,
        prompt: LmPrompt,
    ) -> LmPrediction | list[LmPrediction]:
        pass

    def _cast_prompt(self, prompt: str | LmPrompt) -> LmPrompt:
        if isinstance(prompt, str):
            return LmPrompt(prompt, 100)
        return prompt

    def estimate_tokens_in_prompt(self, prompt: LmPrompt) -> int:
        raise NotImplementedError

    @property
    def token_limit(self):
        raise NotImplementedError

    def could_completion_go_over_token_limit(self, prompt: LmPrompt) -> bool:
        count = self.estimate_tokens_in_prompt(prompt)
        return (
            count + (prompt.max_tokens or self.default_tokens_generated)
        ) > self.token_limit

    def model_name(self):
        return self.__class__.__name__

    def remove_special_chars_from_tokens(self, tokens: list[str]) -> list[str]:
        """
        Certain tokenizers have special characters (such as a Ġ to represent a space).
        This method is to try to remove those and get it in a form that could be joined
        and represent the original text.
        """
        raise NotImplementedError

    def tokenize(self, input_str: str) -> list[str]:
        msg = "This predictor does not implement tokenization"
        raise NotImplementedError(msg)

    def configure_global_ratelimit(
        self,
        max_count=1,
        per_seconds=1,
        greedy=False,
    ) -> None:
        """
        Configure global ratelimiting, max tries per given seconds
        If greedy is set to true, requests will be made without time inbetween,
        followed by a long wait. Otherwise, requests are evenly spaced.
        """
        if max_count and per_seconds:
            LmPredictor._rate_limit = RateLimit(
                max_count=max_count,
                per=per_seconds,
                greedy=greedy,
            )
        else:
            LmPredictor._rate_limit = None

        return LmPredictor._rate_limit

    @classmethod
    def _wait_ratelimit(cls) -> float:
        if LmPredictor._rate_limit:
            return LmPredictor._rate_limit.wait()

        return 0.0

    @property
    @abstractmethod
    def is_chat_model(self) -> bool:
        raise NotImplementedError

    @property
    def default_tokens_generated(self) -> int:
        return self.token_limit // 16
