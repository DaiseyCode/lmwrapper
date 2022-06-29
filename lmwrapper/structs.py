from dataclasses import dataclass
import statistics
from typing import List, Any


@dataclass(frozen=True)
class LmPrompt:
    text: str
    max_toks: int
    stop: List[str] = None
    logprobs: int = None
    temperature: float = 1.0
    top_p: float = 0.9
    presence_penalty: float = 0.0
    num_completions: int = 1
    cache: bool = None  # Use the default of the predictor
    """Whether to attempt to cache the model output. This overrides any default
    settings of the model. This can be useful in saving computation but means 
    sampling might not work as expected."""
    echo: bool = False
    """Whether to echo back the original prompt. Also allows you to get the
    probability of the prompt under the model"""


@dataclass
class LmPrediction:
    completion_text: str
    prompt: LmPrompt
    metad: Any

    @property
    def completion_tokens(self):
        raise NotImplemented("This version of prediction does not support completion tokens")

    @property
    def completion_token_offsets(self):
        raise NotImplemented("This version of prediction does not support completion token offsets")

    @property
    def completion_logprobs(self):
        raise NotImplemented("This version of prediction does not support completion logprobs")

    @property
    def prompt_tokens(self):
        raise NotImplemented("This version of prediction does not support prompt tokens")

    @property
    def prompt_token_offsets(self):
        raise NotImplemented("This version of prediction does not support prompt token offsets")

    @property
    def prompt_logprobs(self):
        raise NotImplemented("This version of prediction does not support prompt logprobs")

    def get_full_text(self):
        return self.prompt.text + self.completion_text

    def completion_mean_logprob(self):
        return statistics.mean(self.completion_logprobs)
