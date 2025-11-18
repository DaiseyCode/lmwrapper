import contextlib
import dataclasses
import json
import pickle
import statistics
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar, Union

from lmwrapper.internals import ModelInternalsResults
from lmwrapper.utils import StrEnum

# Define TypeVar for user metadata
T = TypeVar('T')

LM_CHAT_DIALOG_COERCIBLE_TYPES = Union[
    str,
    list[Union["LmChatTurn", tuple[str, str], dict, str]],
    "LmChatDialog",
]  # Defines a set of types that can be converted into a LmChatDialog


class StopMode(StrEnum):
    """Determines how to try to handle stop tokens"""

    WILL_NOT_CONTAIN = "WILL_NOT_CONTAIN"
    """This emulates the classic openai completion API. This makes
    sure the returned sequence will not contain the stop token.
    Even though the completion text will not include the stop sequence,
    it is possible for the returned tokens to have it in cases where the stop sequence
    is a prefix of the next token."""
    ONLY_STOP_GENERATING = "ONLY_STOP_GENERATING"
    """This is the style of the openai chat API. This stops generating once
    the sequence is seen. However, the output might contain the stop sequence."""
    AUTO = "AUTO"
    """This chooses a mode that is the default for the style of model. This
    means the behavior is somewhat undefined and unpredictable, but is more
    likely to work for more models."""


@dataclass(frozen=True, init=False)
class LmPrompt(Generic[T]):
    text: str | LM_CHAT_DIALOG_COERCIBLE_TYPES
    """The actual text of the prompt. If it is a LM_CHAT_DIALOG_COERCIBLE_TYPES
    which can become a LmChatDialog (such as a list of strings) it will be converted
    into a LmChatDialog."""
    max_tokens: int | None = None
    """The maximum number of tokens to generate in the completion. This might
    include hidden reasoning tokens. 
    If `None` then the downstream model will choose some default value. This value
    might be a function of the prompt input length, but this behaviour is not defined.
    This means it is possible that the default max might cause errors with long prompts.
    Specify a limit yourself to have more predictable behavior.

    Note that unlike in the OpenAI API, we do not separate out an arg
    for `max_completion_tokens`. We automatically map `max_tokens` to
    the appropriate arg for the model.
    """
    stop: list[str] = None
    """Sequences where the model will stop generating further tokens.
    The returned text will not contain the stop sequence. This sequence might span
    accross tokens and does not have to be an actual token in the vocabulary.
    For example could make a stop token of 'I like pie' even if that's not actually
    a token.
    """
    stop_mode: StopMode = StopMode.AUTO
    """Different models/providers handle stopping in different ways. You can
    either leave that as-is ("auto") or try to change the mode to try
    to emulate another mode (which may or may not work depending on
    the model. Models are expected to raise an error if an incompatible
    mode is given)."""
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
    top_p: float | None = None
    """An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with top_p probability mass. So 0.1 means
    only the tokens comprising the top 10% probability mass are considered.
    If set to float < 1, only the smallest set of most probable tokens with
    probabilities that add up to top_p or higher are kept for generation.
    If None, will be passed as None to the API (using model default)."""
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
    num_completions: int | None = None
    """How many completions to generate for each prompt. The default `None` will
    just return a single LmPrediction object. If it is an integer than a list
    of completion objects will be returned."""
    cache: bool = None
    """Whether to attempt to cache the model output. This overrides any default
    settings of the model. This can be useful in saving computation but means
    sampling might not work as expected. When it set to `None` it will use the
    default of the predictor."""
    echo: bool = False
    """Whether to echo back the original prompt. Also allows you to get the
    probability of the prompt under the model"""
    add_bos_token: bool = None
    """Whether to add a bos (beginning-of-sentence) token at the beginning of the prompt.
    Properly handling BOS tokens is important for several reasons
       1. Some models might be trained this way. Their learned algorithms might depend
          on the existence of the token, and we want to match the training setting.
       2. Having a BOS allows for unconditional generation (ie, no prompt)
       3. Having a BOS lets us have a probability even for the first token.
    There are three states this could be (None, True, False).
    By default (None), we will add a bos token if the tokenizer does not seem to
    add one by default, and it is not a seq2seq model.
    Set to True to always add a bos token.
    Set to False to never add a bos token (but the tokenizer might still add one).
    """
    add_special_tokens: bool = True
    """Whether or not to add special tokens when encoding the prompt."""
    model_internals_request: Optional["ModelInternalsRequest"] = None
    """Used to attempt to get hidden states and attentions from the model."""
    metadata: Optional[T] = None
    """Optional user-defined metadata that gets transferred to the resulting LmPrediction.
    This is not used for caching and can be any type. It's useful for tracking
    additional information with each prompt and prediction (e.g., ground truth labels,
    problem identifiers)."""

    # TODO: make a auto_reduce_max_tokens to reduce when might go over.

    def __init__(
        self,
        text: str | LM_CHAT_DIALOG_COERCIBLE_TYPES,
        max_tokens: int | None = None,
        stop: list[str] = None,
        stop_mode: StopMode = StopMode.AUTO,
        logprobs: int = 1,
        temperature: float = 1.0,
        top_p: float | None = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        num_completions: int | None = None,
        cache: bool = None,
        echo: bool = False,
        add_bos_token: bool = None,
        add_special_tokens: bool = True,
        model_internals_request: Optional["ModelInternalsRequest"] = None,
        metadata: Optional[T] = None,
        max_completion_tokens: int | None = None,
    ):
        # Convert text to chat dialog if it's a list
        if isinstance(text, list):
            text = LmChatDialog(text)

        # Validate max_tokens
        if max_tokens is not None and not isinstance(max_tokens, int):
            msg = "The max_tokens parameter should be an int or None."
            raise ValueError(msg)
        
        # Validate num_completions
        if num_completions is not None and (
            not isinstance(num_completions, int) or num_completions <= 0
        ):
            msg = (
                "The num_completions parameter should be an "
                "int greater than 0 or None (in which case a single "
                "non-list item will be returned)"
            )
            raise ValueError(msg)

        if max_completion_tokens is not None:
            msg = "max_completion_tokens is not supported in v0.17. Always max_tokens instead (regardless of model type)."
            raise ValueError(msg)
        
        # Validate stop
        if stop is not None:
            if not isinstance(stop, list):
                msg = "The stop parameter should be a list of strings on where to stop."
                raise ValueError(msg)
            if not all(isinstance(x, str) for x in stop):
                msg = "The stop parameter should be a list of strings on where to stop."
                raise ValueError(msg)
        
        # Convert temperature to float if it's an int
        if isinstance(temperature, int):
            temperature = float(temperature)
        
        # Validate temperature
        if not isinstance(temperature, float):
            msg = "The temperature parameter should be a float."
            raise ValueError(msg)
        if temperature < 0.0:
            msg = "The temperature parameter should be a positive float."
            raise ValueError(msg)
        
        # Validate top_p (allow None to pass through to API)
        if top_p is not None and not isinstance(top_p, float):
            msg = "The top_p parameter should be a float or None."
            raise ValueError(msg)
        
        # Validate presence_penalty
        if not isinstance(presence_penalty, float):
            msg = "The presence_penalty parameter should be a float."
            raise ValueError(msg)
        
        # Validate cache
        if cache is not None and not isinstance(cache, bool):
            msg = "The cache parameter should be a bool."
            raise ValueError(msg)
        
        # Validate logprobs
        if logprobs is not None and not isinstance(logprobs, int):
            msg = (
                "The logprob parameter should be int denoting number of probs return,"
                " or None."
            )
            raise ValueError(msg)
        
        # Validate stop_mode
        if stop_mode != StopMode.AUTO:
            raise NotImplementedError(
                "Only StopMode.AUTO is supported at this time as a temporary hack",
            )

        # Set all attributes using object.__setattr__ since the class is frozen
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "stop_mode", stop_mode)
        object.__setattr__(self, "logprobs", logprobs)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_p", top_p)
        object.__setattr__(self, "presence_penalty", presence_penalty)
        object.__setattr__(self, "frequency_penalty", frequency_penalty)
        object.__setattr__(self, "num_completions", num_completions)
        object.__setattr__(self, "cache", cache)
        object.__setattr__(self, "echo", echo)
        object.__setattr__(self, "add_bos_token", add_bos_token)
        object.__setattr__(self, "add_special_tokens", add_special_tokens)
        object.__setattr__(self, "model_internals_request", model_internals_request)
        object.__setattr__(self, "metadata", metadata)

    def is_deterministic_sampling(self) -> bool:
        return (self.temperature < 1e-4) or (self.top_p is not None and self.top_p < 1e-4)

    def replace(self, **kwargs) -> "LmPrompt":
        """Returns a new prompt with the given parameters replaced."""
        return dataclasses.replace(self, **kwargs)

    def is_text_a_chat(self) -> bool:
        return isinstance(self.text, LmChatDialog)

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

    def dict_serialize(self, include_metadata: bool = False) -> dict:
        """
        Serialize the prompt into a json-compatible dictionary. Note this is not
        guaranteed to be the same as the JSON representation for use
        in an openai api call. This is just for serialization purposes.
        
        Args:
            include_metadata: Whether to include metadata in serialization.
                Default is False since metadata should not be part of the cache key.
        """
        out = {
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "logprobs": self.logprobs,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "num_completions": self.num_completions,
            "cache": self.cache,
            "echo": self.echo,
            "add_bos_token": self.add_bos_token,
            "add_special_tokens": self.add_special_tokens,
        }
        if self.is_text_a_chat():
            out["text"] = self.get_text_as_chat().as_dicts()
        else:
            out["text"] = self.text
            
        if include_metadata and self.metadata is not None:
            try:
                # Test if it's JSON serializable
                json.dumps(self.metadata)
                out["metadata"] = self.metadata
            except (TypeError, ValueError):
                # If metadata isn't JSON serializable, leave it out
                pass
                
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
            if observe_role != ChatGptRoles.system:
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
class LmPrediction(Generic[T]):
    completion_text: str | None
    """The new text generated. It might be None if errors"""
    prompt: LmPrompt[T]
    metad: Any
    internals: ModelInternalsResults | None = field(default=None, kw_only=True)
    error_message: str | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.error_message is not None:
            if not isinstance(self.error_message, str):
                msg = "The error_message parameter should be a string."
                raise ValueError(msg)

    @property
    def has_errors(self):
        return self.error_message is not None

    @classmethod
    def parse_from_cache(
        cls,
        completion_text: str,
        prompt: LmPrompt,
        metad_bytes: bytes,
        error_message: str | None,
    ):
        return cls(
            completion_text=completion_text,
            prompt=prompt,
            metad=pickle.loads(metad_bytes),
            error_message=error_message,
        )

    def serialize_metad_for_cache(self) -> bytes:
        return pickle.dumps(self.metad)

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
    def usage_output_tokens(self):
        return len(self.completion_tokens)

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
        if self.prompt.logprobs == 1:
            # If we only have a single logprob, then we can adapt the
            #   logprob property into a list of dictionaries. This allows
            #   partial support for implementations that support only that.
            return [
                {token: float(logprob)}
                for token, logprob in zip(
                    self.completion_tokens,
                    self.completion_logprobs,
                    strict=True,
                )
            ]
        msg = "This version of prediction does not support top token logprobs"
        raise NotImplementedError(
            msg,
        )

    def make_reply_prompt(
        self,
        new_turns: LM_CHAT_DIALOG_COERCIBLE_TYPES,
        **override_prompt_params,
    ) -> LmPrompt:
        """
        Continue the dialog with the given input.
        This will return a new prompt that can be used to continue the dialog.
        The new prompt will have the same prompt as the original prompt, but with the
        new input added to the end.
        
        Args:
            new_turns: The new dialog turns to add to the conversation.
            **override_prompt_params: Optional parameters to override in the new prompt
                (e.g., temperature=0.5, max_tokens=100).
        """
        # Get the existing dialog (handles both string and chat cases)
        existing_dialog = list(self.prompt.get_text_as_chat())
        
        # Add the completion as an assistant turn if there is one
        if self.completion_text is not None:
            existing_dialog.append(LmChatTurn(role=ChatGptRoles.assistant, content=self.completion_text))
        
        # Create new dialog with existing dialog + new turns
        extended_dialog_content = existing_dialog + list(LmChatDialog(new_turns))
        new_dialog = LmChatDialog(extended_dialog_content)
        
        # Create a new prompt with the extended dialog and any parameter overrides
        return self.prompt.replace(text=new_dialog, **override_prompt_params)

    def dict_serialize(
        self,
        pull_out_props: bool = True,
        include_metad: bool = False,
        include_metadata: bool = False,
    ) -> dict[str, Any]:
        out = {
            "completion_text": self.completion_text,
            "prompt": self.prompt.dict_serialize(include_metadata=include_metadata),
            "was_cached": self.was_cached,
            "error_message": self.error_message,
        }
        if pull_out_props:
            with contextlib.suppress(Exception):
                out["prompt_tokens"] = self.prompt_tokens

            with contextlib.suppress(Exception):
                out["completion_tokens"] = self.completion_tokens

            with contextlib.suppress(Exception):
                out["prompt_logprobs"] = self.prompt_logprobs

            with contextlib.suppress(Exception):
                out["completion_logprobs"] = self.completion_logprobs

            with contextlib.suppress(Exception):
                out["full_logprobs"] = self.full_logprobs

            with contextlib.suppress(Exception):
                out["top_token_logprobs"] = self.top_token_logprobs

        if include_metad:
            out["metad"] = self.metad
        return out
