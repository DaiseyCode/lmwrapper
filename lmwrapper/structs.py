import dataclasses
import statistics
from dataclasses import dataclass
from typing import Any, Union

from lmwrapper.utils import StrEnum

LM_CHAT_DIALOG_COERCIBLE_TYPES = Union[
    str,
    list[Union["LmChatTurn", tuple[str, str], dict, str]],
    "LmChatDialog",
]  # Defines a set of types that can be converted into a LmChatDialog


@dataclass(frozen=True)
class LmPrompt:
    text: str | LM_CHAT_DIALOG_COERCIBLE_TYPES
    """The actual text of the prompt. If it is a LM_CHAT_DIALOG_COERCIBLE_TYPES
    which can become a LmChatDialog (such as a list of strings) it will be converted
    into a LmChatDialog."""
    max_tokens: int | None = None
    """The maximum number of tokens to generate in the completion. If `None`
    then the model downstream model will choose some default value. This value
    might be a function of the prompt input length, but this behaviour is not defined.
    This means it is possible that the default max might cause errors with long prompts.
    It recommended that you specify a limit yourself to have more predictable
    behaviour."""
    stop: list[str] = None
    """Sequences where the model will stop generating further tokens.
    The returned text will not contain the stop sequence. This sequence might span
    accross tokens and does not have to be an actual token in the vocabulary.
    For example could make a stop token of 'I like pie' even if that's not actually
    a token. Even though the completion text will not include the stop sequence,
    it is possible for the returned tokens to have it in cases where the stop sequence
    is a prefix of the next token.
    """
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
    cache: bool = None
    """Whether to attempt to cache the model output. This overrides any default
    settings of the model. This can be useful in saving computation but means
    sampling might not work as expected. When it set to `None` it will use the
    default of the predictor."""
    echo: bool = False
    """Whether to echo back the original prompt. Also allows you to get the
    probability of the prompt under the model"""
    add_bos_token: bool = True
    """Whether to add a bos (beginning-of-sentence) token at the beginning of the prompt.
    This allows for unconditional generation and allows for the first token to have
    a probability. This always happens in the openai endpoints (presumably), but
    could be controlled in other models."""
    add_special_tokens: bool = True
    """Whether or not to add special tokens when encoding the prompt."""

    # TODO: make a auto_reduce_max_tokens to reduce when might go over.

    def __post_init__(self):
        if self.max_tokens is not None and not isinstance(self.max_tokens, int):
            msg = "The max_tokens parameter should be an int."
            raise ValueError(msg)
        if self.stop is not None:
            if not isinstance(self.stop, list):
                msg = "The stop parameter should be a list of strings on where to stop."
                raise ValueError(
                    msg,
                )
            if not all(isinstance(x, str) for x in self.stop):
                msg = "The stop parameter should be a list of strings on where to stop."
                raise ValueError(
                    msg,
                )
        if isinstance(self.temperature, int):
            object.__setattr__(self, "temperature", float(self.temperature))
        if not isinstance(self.temperature, float):
            msg = "The temperature parameter should be a float."
            raise ValueError(msg)
        if self.temperature < 0.0:
            msg = "The temperature parameter should be a positive float."
            raise ValueError(msg)
        if not isinstance(self.top_p, float):
            msg = "The top_p parameter should be a float."
            raise ValueError(msg)
        if not isinstance(self.presence_penalty, float):
            msg = "The presence_penalty parameter should be a float."
            raise ValueError(msg)
        if not isinstance(self.num_completions, int):
            msg = "The num_completions parameter should be an int."
            raise ValueError(msg)
        if self.cache is not None and not isinstance(self.cache, bool):
            msg = "The cache parameter should be a bool."
            raise ValueError(msg)
        if self.logprobs is not None and not isinstance(self.logprobs, int):
            msg = (
                "The logprob parameter should be int denoting number of probs return,"
                " or None."
            )
            raise ValueError(
                msg,
            )

    def is_text_a_chat(self) -> bool:
        return isinstance(self.text, list)

    def get_text_as_chat(self) -> "LmChatDialog":
        return LmChatDialog(self.text)

    def get_text_as_string_default_form(self) -> str:
        """
        Will always return a string, even if it was originally a chat. It will use
        the default form of the chat specified in LmChatDialog.to_default_string_prompt()
        """
        if self.is_text_a_chat():
            return self.text.to_default_string_prompt()
        else:
            return self.text

    def as_dict(self) -> dict:
        """
        Serialize the prompt into a dictionary. Note this is not
        guaranteed to be the same as the JSON representation for use
        in an openai api call. This is just for serialization purposes.
        """
        out = dataclasses.asdict(self)
        if self.is_text_a_chat():
            out["text"] = self.get_text_as_chat().as_dicts()
        return out


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
                    msg = (
                        f"Invalid type for text: {type(turn)}. It should be a tuple of"
                        " strings (role, content) or a LmChatTurn."
                    )
                    raise ValueError(
                        msg,
                    )
            current_role = (
                ChatGptRoles.user
                if observe_role == ChatGptRoles.assistant
                else ChatGptRoles.assistant
            )
        super().__init__(out)

    def as_dicts(self) -> list[dict]:
        return [
            {k: str(v) for k, v in chat_turn.__dict__.items() if v is not None}
            for chat_turn in self
        ]

    def to_default_string_prompt(self) -> str:
        """
        A simple method of representing the dialogue as a string.
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

    def __post_init__(self):
        self._was_cached = False

    @property
    def was_cached(self) -> bool:
        return hasattr(self, "_was_cached") and self._was_cached

    def mark_as_cached(self) -> "LmPrediction":
        self._was_cached = True
        return self

    def _verify_logprobs(self):
        if self.prompt.logprobs is None or self.prompt.logprobs == 0:
            msg = "This property is not available unless the prompt logprobs is set"
            raise ValueError(
                msg,
            )

    @property
    def completion_tokens(self) -> list[str]:
        msg = "This version of prediction does not support completion tokens"
        raise NotImplementedError(
            msg,
        )

    @property
    def completion_token_offsets(self):
        msg = "This version of prediction does not support completion token offsets"
        raise NotImplementedError(
            msg,
        )

    @property
    def completion_logprobs(self) -> list[float]:
        msg = "This version of prediction does not support completion logprobs"
        raise NotImplementedError(
            msg,
        )

    @property
    def prompt_tokens(self):
        msg = "This version of prediction does not support prompt tokens"
        raise NotImplementedError(
            msg,
        )

    @property
    def prompt_token_offsets(self):
        msg = "This version of prediction does not support prompt token offsets"
        raise NotImplementedError(
            msg,
        )

    @property
    def prompt_logprobs(self):
        msg = "This version of prediction does not support prompt logprobs"
        raise NotImplementedError(
            msg,
        )

    def get_full_text(self):
        return self.prompt.text + self.completion_text

    def get_full_tokens(self):
        msg = "This version of prediction does not support full tokens"
        raise NotImplementedError(
            msg,
        )

    @property
    def full_logprobs(self):
        msg = "This version of prediction does not support full logprobs"
        raise NotImplementedError(
            msg,
        )

    def completion_mean_logprob(self):
        return statistics.mean(self.completion_logprobs)

    @property
    def top_token_logprobs(self) -> list[dict[str, float]]:
        msg = "This version of prediction does not support top token logprobs"
        raise NotImplementedError(
            msg,
        )

    def as_dict(
        self,
        pull_out_props: bool = True,
        include_metad: bool = False,
    ) -> dict[str, Any]:
        out = {
            "completion_text": self.completion_text,
            "prompt": self.prompt.as_dict(),
            "was_cached": self.was_cached,
        }
        if pull_out_props:
            try:
                out["prompt_tokens"] = self.prompt_tokens
            except Exception:
                pass
            try:
                out["completion_tokens"] = self.completion_tokens
            except Exception:
                pass
            try:
                out["prompt_logprobs"] = self.prompt_logprobs
            except Exception:
                pass
            try:
                out["completion_logprobs"] = self.completion_logprobs
            except Exception:
                pass
            try:
                out["full_logprobs"] = self.full_logprobs
            except Exception:
                pass
            try:
                out["top_token_logprobs"] = self.top_token_logprobs
            except Exception:
                pass
        if include_metad:
            out["metad"] = self.metad
        return out
