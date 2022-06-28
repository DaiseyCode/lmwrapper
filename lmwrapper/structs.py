from dataclasses import dataclass
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
    cache: bool = None  # Use the default of the predictor


@dataclass
class LmPrediction:
    text: str
    metad: Any