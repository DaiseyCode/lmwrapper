import pickle
from dataclasses import dataclass
from typing import Any

from lmwrapper.structs import LmPrediction, LmPrompt


@dataclass
class HuggingFacePrediction(LmPrediction):
    _prompt_encoding: Any
    _tokens: Any
    _log_probs: Any
    _logprobs_dict: dict
    _num_prompt_tokens: int
    _completion_with_special_tok: str

    @classmethod
    def parse_from_cache(
        cls,
        completion_text: str,
        prompt: LmPrompt,
        metad_bytes: bytes,
        error_message: str,
    ) -> "HuggingFacePrediction":
        metad_and_params = pickle.loads(metad_bytes)
        return cls(
            prompt=prompt,
            completion_text=completion_text,
            **metad_and_params,
            error_message=error_message,
        )

    def serialize_metad_for_cache(self) -> bytes:
        assert "prompt" not in self.metad
        return pickle.dumps(
            {
                "metad": self.metad,
                "_prompt_encoding": self._prompt_encoding,
                "_tokens": self._tokens,
                "_log_probs": self._log_probs,
                "_logprobs_dict": self._logprobs_dict,
                "_num_prompt_tokens": self._num_prompt_tokens,
                "_completion_with_special_tok": self._completion_with_special_tok,
            },
        )

    def __post_init__(self):
        super().__post_init__()
        assert len(self._prompt_encoding["input_ids"]) == 1
        assert self._num_prompt_tokens

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
            msg = "This property is not available unless the prompt echo is set to True"
            raise ValueError(
                msg,
            )
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
