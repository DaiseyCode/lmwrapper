from lmwrapper.structs import LmPrediction


from dataclasses import dataclass
from typing import Any


@dataclass
class HuggingfacePrediction(LmPrediction):
    _prompt_encoding: Any
    _tokens: Any
    _log_probs: Any

    def __post_init__(self):
        assert len(self._prompt_encoding["input_ids"]) == 1
        self._num_prompt_tokens = len(self._prompt_encoding["input_ids"][0])
        if self.prompt.add_bos_token:
            self._num_prompt_tokens -= 1

    @property
    def completion_tokens(self) -> list[str]:
        return self._tokens[self._num_prompt_tokens :]

    @property
    def completion_logprobs(self) -> list[float]:
        self._verify_logprobs()
        return self._log_probs[self._num_prompt_tokens :]

    @property
    def prompt_tokens(self):
        return self._tokens[: self._num_prompt_tokens]

    @property
    def prompt_logprobs(self):
        return self._log_probs[: self._num_prompt_tokens]

    @property
    def full_logprobs(self):
        return self._log_probs

    def get_full_tokens(self):
        return self._tokens