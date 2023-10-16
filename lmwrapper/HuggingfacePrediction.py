from lmwrapper.structs import LmPrediction


from dataclasses import dataclass
from typing import Any


@dataclass
class HuggingfacePrediction(LmPrediction):
    _prompt_encoding: Any
    _tokens: Any
    _log_probs: Any
    _logprobs_dict: dict
    _num_prompt_tokens: int
    _completion_with_special_tok: str

    def __post_init__(self):
        assert len(self._prompt_encoding["input_ids"]) == 1
        assert self._num_prompt_tokens
        if self.prompt.add_bos_token:
            self._num_prompt_tokens -= 1

        if self.prompt.logprobs == 0:
            return

        if self.prompt.echo:
            assert len(self._tokens) == len(self._log_probs)
        else:
            assert len(self._tokens[self._num_prompt_tokens :]) == len(self._log_probs)

    @property
    def completion_tokens(self) -> list[str]:
        return self._tokens[self._num_prompt_tokens :]

    @property
    def completion_logprobs(self) -> list[float]:
        self._verify_logprobs()
        if self.prompt.echo:
            return self._log_probs[self._num_prompt_tokens :]
        else:
            return self._log_probs

    @property
    def prompt_tokens(self):
        return self._tokens[: self._num_prompt_tokens]

    @property
    def prompt_logprobs(self):
        if not self.prompt.echo:
            raise ValueError("This property is not available unless the prompt echo is set to True")
        self._verify_logprobs()
        return self._log_probs[: self._num_prompt_tokens]

    @property
    def full_logprobs(self):
        return self._log_probs

    def get_full_tokens(self):
        return self._tokens

    @property
    def logprobs_dict(self):
        return self._logprobs_dict
