import dataclasses
from abc import abstractmethod
from collections.abc import Callable, Iterable
from sqlite3 import OperationalError

from ratemate import RateLimit

from lmwrapper.batch_config import CompletionWindow
from lmwrapper.sqlcache_struct import BatchPredictionPlaceholder
from lmwrapper.structs import LmPrediction, LmPrompt


class LmPredictor:
    _rate_limit: RateLimit | None = None

    def __init__(
        self,
        cache_default: bool = False,
    ):
        self._cache_default = cache_default
        # self._disk_cache = get_disk_cache()
        from lmwrapper.sqlcache import SqlBackedCache

        self._disk_cache = SqlBackedCache(self)

    def find_prediction_class(self, prompt):
        return LmPrediction

    def predict(
        self,
        prompt: str | LmPrompt,
    ) -> LmPrediction | list[LmPrediction]:
        prompt = self._cast_prompt(prompt)
        #if prompt.num_completions > 1:
        #    raise NotImplementedError(
        #        "Changes with with caching and batching means we need to test `num_completions > 1`."
        #    )
        should_cache = self._cache_default if prompt.cache is None else prompt.cache
        if should_cache and prompt.model_internals_request is not None:
            raise NotImplementedError(
                "Cannot yet cache predictions with model internals request",
            )
        self._validate_prompt(prompt, raise_on_invalid=True)
        if should_cache:
            cached_vals = self._read_cached_values(prompt)
            if len(cached_vals) >= prompt.num_completions:
                assert len(cached_vals) == prompt.num_completions
                if len(cached_vals) == 1:
                    return cached_vals[0]
                return cached_vals
            # There are some missing values. Let's predict for the missing ones.
            need_new_completions = prompt.num_completions - len(cached_vals)
            if need_new_completions != prompt.num_completions:
                new_prompt = dataclasses.replace(
                    prompt,
                    num_completions=prompt.num_completions - len(cached_vals)
                )
            else:
                new_prompt = prompt
            new_vals = self._predict_maybe_cached(new_prompt)
            # Add the new values we got
            if not isinstance(new_vals, list):
                new_vals = [new_vals]
            for val in new_vals:
                try:
                    # self._disk_cache.set(cache_key, val)
                    # TODO maybe figure out a way to bulk add
                    self._disk_cache.add_or_set(val)
                except OperationalError as e:
                    print("Failed to cache", e)
            vals = cached_vals + new_vals
            if len(vals) == 1:
                return vals[0]
            return vals
        else:
            return self._predict_maybe_cached(prompt)

    def _read_cached_values(self, prompt: LmPrompt) -> list[LmPrediction]:
        """Checks the cache for any matches of the prompt. Returns a list
        as if num_completions is >1 we might have multiple items"""
        cache_key = prompt
        try:
            cached_items = self._disk_cache.get(cache_key)
        except OperationalError as e:
            print("Failed to get from cache", e)
            cached_items = None
        if not cached_items:
            return []
        for i, item in enumerate(cached_items):
            if isinstance(item, BatchPredictionPlaceholder):
                raise NotImplementedError(
                    "We retrieved a non-finalized batched prediction from"
                    " the cache. This might be actually finished and we could"
                    " recover and check to see if it is done. However, this is"
                    " not yet implemented. For now, perhaps try to give this"
                    " prompt to predict_many to retrieve the batch data.",
                )
            cached_items[i] = item.mark_as_cached()
        return cached_items

    def predict_many(
        self,
        prompts: list[str | LmPrompt],
        completion_window: CompletionWindow,
    ) -> Iterable[LmPrediction | list[LmPrediction]]:
        for prompt in prompts:
            val = self.predict(prompt)
            yield val

    def remove_prompt_from_cache(
        self,
        prompt: str | LmPrompt,
    ) -> bool:
        return self._disk_cache.delete(prompt)

    def _validate_prompt(self, prompt: LmPrompt, raise_on_invalid: bool = True) -> bool:
        """Called on prediction to make sure the prompt is valid for the model"""
        return True

    @abstractmethod
    def get_model_cache_key(self):
        return type(self).__name__

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


def get_mock_predictor(
    predict_func: Callable[[LmPrompt], LmPrediction] = None,
    is_chat_model: bool = False,
):
    """Gets a mock predictor. By default returns whatever the prompt txt is"""

    class MockPredict(LmPredictor):
        def get_model_cache_key(self):
            return "mock_predictor"

        @property
        def is_chat_model(self) -> bool:
            return is_chat_model

        def _predict_maybe_cached(self, prompt):
            if predict_func is None:
                return LmPrediction(
                    prompt.get_text_as_string_default_form(),
                    prompt,
                    {},
                )
            return predict_func(prompt)

    return MockPredict()
