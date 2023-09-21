from dataclasses import dataclass
import statistics
from typing import List, Any, Union, Tuple
from enum import Enum

from lmwrapper.util import StrEnum

LM_CHAT_DIALOG_COERCIBLE_TYPES = Union[
    str,
    List[Union["LmChatTurn", Tuple[str, str], dict, str]],
    "LmChatDialog",
]  # Defines a set of types that can be converted into a LmChatDialog

@dataclass(frozen=True)
class LmPrompt:
    text: Union[str, LM_CHAT_DIALOG_COERCIBLE_TYPES]
    max_tokens: int
    """The maximum number of tokens to generate in the completion."""
    stop: List[str] = None
    """Sequences where the model will stop generating further tokens.
    The returned text will not contain the stop sequence."""
    logprobs: int = 1
    """Include the log probabilities on the logprobs most likely tokens,
    as well the chosen tokens. For example, if logprobs is 5, the
    API will return a list of the 5 most likely tokens.
    The model will always return the logprob of the sampled token,
    so there may be up to logprobs+1 elements in the response.

    In the case of openai the maximum value for logprobs is 5.
    """
    temperature: float = 1.0
    """What sampling temperature to use, between 0 and 2.
    Higher values like 0.8 will make the output more random, while lower values
    like 0.2 will make it more focused and deterministic."""
    top_p: float = 1.0
    """An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with top_p probability mass. So 0.1 means
    only the tokens comprising the top 10% probability mass are considered.
    If set to float < 1, only the smallest set of most probable tokens with
    probabilities that add up to top_p or higher are kept for generation."""
    presence_penalty: float = 0.0
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on whether
    they appear in the text so far, increasing the model's likelihood
    to talk about new topics. This parameter is used to encourage the model to include a
    diverse range of tokens in the generated text. It is a value that is subtracted from
    the log-probability of a token each time it is generated. A higher presence_penalty
    value will result in the model being more likely to generate tokens that have not
    yet been included in the generated text."""
    frequency_penalty: float = 0.0
    """Number between -2.0 and 2.0. Positive values penalize new tokens based on their
    existing frequency in the text so far, decreasing the model's likelihood
    to repeat the same line verbatim. This parameter is used to discourage the model from
    repeating the same words or phrases too frequently within the generated text.
    It is a value that is added to the log-probability of a token each time it occurs in
    the generated text. A higher frequency_penalty value will result in the model being
    more conservative in its use of repeated tokens."""
    num_completions: int = 1
    """How many completions to generate for each prompt."""
    cache: bool = None  # Use the default of the predictor
    """Whether to attempt to cache the model output. This overrides any default
    settings of the model. This can be useful in saving computation but means
    sampling might not work as expected."""
    echo: bool = False
    """Whether to echo back the original prompt. Also allows you to get the
    probability of the prompt under the model"""
    add_bos_token: bool = True
    """Whether to add a bos (beginning-of-sentence) token at the beginning of the prompt.
    This allows for unconditional generation and allows for the first token to have
    a probability. This always happens in the openai endpoints (presumably), but
    could be controlled in other models (NOT IMPLEMENTED yet though)."""

    def __post_init__(self):
        if not isinstance(self.max_tokens, int):
            raise ValueError("The max_tokens parameter should be an int.")
        if self.stop is not None:
            if not isinstance(self.stop, list):
                raise ValueError(
                    "The stop parameter should be a list of strings on where to stop."
                )
            if not all(isinstance(x, str) for x in self.stop):
                raise ValueError(
                    "The stop parameter should be a list of strings on where to stop."
                )
        if isinstance(self.temperature, int):
            object.__setattr__(self, "temperature", float(self.temperature))
        if not isinstance(self.temperature, float):
            raise ValueError("The temperature parameter should be a float.")
        if self.temperature < 0.0:
            raise ValueError("The temperature parameter should be a positive float.")
        if not isinstance(self.top_p, float):
            raise ValueError("The top_p parameter should be a float.")
        if not isinstance(self.presence_penalty, float):
            raise ValueError("The presence_penalty parameter should be a float.")
        if not isinstance(self.num_completions, int):
            raise ValueError("The num_completions parameter should be an int.")
        if self.cache is not None and not isinstance(self.cache, bool):
            raise ValueError("The cache parameter should be a bool.")
        if self.logprobs is not None and not isinstance(self.logprobs, int):
            raise ValueError(
                "The logprob parameter should be int denoting number of probs return, or None."
            )

    def is_text_a_chat(self) -> bool:
        return isinstance(self.text, list)

    def get_text_as_chat(self) -> "LmChatDialog":
        return LmChatDialog(self.text)

    def get_text_as_string_default_form(self) -> str:
        """Will always return a string, even if it was originally a chat. It will use
        the default form of the chat specified in LmChatDialog.to_default_string_prompt()
        """
        if self.is_text_a_chat():
            return self.text.to_default_string_prompt()
        else:
            return self.text


class ChatGptRoles(StrEnum):
    system = "system"
    user = "user"
    assistant = "assistant"
    function = "function"


@dataclass
class LmChatTurn:
    role: str
    """The role of the messages author. For OpenAI: one of system, user, assistant, or function."""
    content: str
    """The contents of the message.
    content is required for all messages except assistant messages with function calls."""
    name: str = None
    """The name of the author of this message. name is required if role
    is function, and it should be the name of the function whose response is
    in the content. May contain a-z, A-Z, 0-9, and underscores,
    with a maximum length of 64 characters."""
    function_call: str = None
    """The name and arguments of a function that should be called, as generated by the model."""


class LmChatDialog(list[LmChatTurn]):
    def __init__(self, values: LM_CHAT_DIALOG_COERCIBLE_TYPES):
        out = []
        current_role = ChatGptRoles.user
        text_list = values if isinstance(values, list) else [values]
        for turn in text_list:
            observe_role = current_role
            match turn:
                case str(text):
                    out.append(LmChatTurn(role=current_role, content=text))
                case (str(role), str(content)):
                    out.append(LmChatTurn(role=role, content=content))
                    observe_role = role
                case dict(turn):
                    out.append(LmChatTurn(**turn))
                    observe_role = turn["role"]
                case LmChatTurn(role=observe_role, content=content):
                    out.append(turn)
                case _:
                    raise ValueError(
                        f"Invalid type for text: {type(turn)}. "
                        f"It should be a tuple of strings (role, content) or a LmChatTurn."
                    )
            current_role = (
                ChatGptRoles.user
                if observe_role == ChatGptRoles.assistant
                else ChatGptRoles.assistant
            )
        super().__init__(out)

    def as_dicts(self) -> List[dict]:
        return [
            {k: v for k, v in chat_turn.__dict__.items() if v is not None}
            for chat_turn in self
        ]

    def to_default_string_prompt(self) -> str:
        """A simple method of representing the dialogue as a string.
        Has the format:
        ```
        user: message
        assistant: message
        user: message
        ...
        ```
        Different models might not follow this default format though internally.
        """
        output = "\n".join(f"{turn.role}: {turn.content}" for turn in self)
        output += f"\n{ChatGptRoles.assistant}:"
        return output


@dataclass
class LmPrediction:
    completion_text: str
    prompt: LmPrompt
    metad: Any

    def _verify_logprobs(self):
        if self.prompt.logprobs is None or self.prompt.logprobs == 0:
            raise ValueError(
                "This property is not available unless the prompt logprobs is set"
            )

    @property
    def completion_tokens(self) -> List[str]:
        raise NotImplementedError(
            "This version of prediction does not support completion tokens"
        )

    @property
    def completion_token_offsets(self):
        raise NotImplementedError(
            "This version of prediction does not support completion token offsets"
        )

    @property
    def completion_logprobs(self) -> List[float]:
        raise NotImplementedError(
            "This version of prediction does not support completion logprobs"
        )

    @property
    def prompt_tokens(self):
        raise NotImplementedError(
            "This version of prediction does not support prompt tokens"
        )

    @property
    def prompt_token_offsets(self):
        raise NotImplementedError(
            "This version of prediction does not support prompt token offsets"
        )

    @property
    def prompt_logprobs(self):
        raise NotImplementedError(
            "This version of prediction does not support prompt logprobs"
        )

    def get_full_text(self):
        return self.prompt.text + self.completion_text

    def get_full_tokens(self):
        raise NotImplementedError(
            "This version of prediction does not support full tokens"
        )

    @property
    def full_logprobs(self):
        raise NotImplementedError(
            "This version of prediction does not support full logprobs"
        )

    def completion_mean_logprob(self):
        return statistics.mean(self.completion_logprobs)
