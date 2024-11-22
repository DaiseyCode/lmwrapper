import dataclasses
from abc import abstractmethod
from collections.abc import Callable, Iterable
from sqlite3 import OperationalError

from ratemate import RateLimit

from lmwrapper.batch_config import CompletionWindow
from lmwrapper.sqlcache_struct import BatchPredictionPlaceholder
from lmwrapper.structs import LM_CHAT_DIALOG_COERCIBLE_TYPES, LmPrediction, LmPrompt


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
        prompt: LmPrompt | str | LM_CHAT_DIALOG_COERCIBLE_TYPES,
    ) -> LmPrediction | list[LmPrediction]:
        prompt = self._cast_prompt(prompt)
        should_cache = self._cache_default if prompt.cache is None else prompt.cache
        if should_cache and prompt.model_internals_request is not None:
            raise NotImplementedError(
                "Cannot yet cache predictions with model internals request",
            )
        self._validate_prompt(prompt, raise_on_invalid=True)
        num_completions = prompt.num_completions or 1
        if should_cache:
            cached_vals = self._read_cached_values(prompt)
            if len(cached_vals) >= num_completions:
                assert len(cached_vals) == num_completions
                if prompt.num_completions is None:
                    assert len(cached_vals) == 1
                    return cached_vals[0]
                return cached_vals
            # There are some missing values. Let's predict for the missing ones.
            need_new_completions = num_completions - len(cached_vals)
            if need_new_completions != num_completions:
                new_prompt = dataclasses.replace(
                    prompt,
                    num_completions=num_completions - len(cached_vals),
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
        else:
            vals = self._predict_maybe_cached(prompt)
        if prompt.num_completions is None and isinstance(vals, list):
            assert len(vals) >= 1
            return vals[0]
        return vals

    def _read_cached_values(self, prompt: LmPrompt) -> list[LmPrediction]:
        """
        Checks the cache for any matches of the prompt. Returns a list
        as if num_completions is >1 we might have multiple items
        """
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
        prompts: list[LmPrompt],
        completion_window: CompletionWindow,
    ) -> Iterable[LmPrediction | list[LmPrediction]]:
        self._validate_predict_many_prompts(prompts)
        for prompt in prompts:
            val = self.predict(prompt)
            yield val

    def _validate_predict_many_prompts(self, prompts):
        if not isinstance(prompts, list):
            msg = (
                "prompts input to predict_many must be a list of LmPrompt objects. "
                "Got type: {type(prompts)}"
            )
            raise ValueError(msg)
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, LmPrompt):
                msg = (
                    f"prompts[{i}] must be a LmPrompt object. Got type: {type(prompt)}"
                )
                raise ValueError(msg)

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

    @property
    def supports_token_operations(self) -> bool:
        """Whether this lm exposes tokenizations"""
        return False

    @abstractmethod
    def _predict_maybe_cached(
        self,
        prompt: LmPrompt,
    ) -> list[LmPrediction]:
        pass

    def _cast_prompt(self, prompt: str | LmPrompt) -> LmPrompt:
        if isinstance(prompt, str):
            return LmPrompt(prompt, 100)
        if isinstance(prompt, list):
            if any(isinstance(e, LmPrompt) for e in prompt):
                msg = (
                    "The passed in prompt is a list that contains another prompt. This"
                    " is not allowed. If you would like to predict multiple prompts,"
                    " use the `predict_many` method."
                )
                raise ValueError(msg)
            if self.is_chat_model:
                return LmPrompt(prompt)
            else:
                msg = (
                    "Passing a list into `predict` is interpreted as a conversation"
                    f" with multiple turns. However, this LM ({self.model_name()}) is"
                    " not a chat model.\n\nIf you were instead intending to predict on"
                    " multiple prompts, use the `predict_many` method."
                )
                raise ValueError(msg)
        elif isinstance(prompt, LmPrompt):
            return prompt
        else:
            msg = (
                "The prompt input should be a `LmPrompt`, a string, or if a chat"
                " model, something coercible to a chat dialog. Got type:"
                f" {type(prompt)}"
            )
            raise ValueError(msg)

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
    def default_tokens_generated(self) -> int | None:
        return self.token_limit // 16

    @property
    def supports_prefilled_chat(self) -> bool:
        """Whether the predictor supports partially prefilled
        chat responses. This can be used to guide the model towards
        a specific format of response."""
        return False


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
