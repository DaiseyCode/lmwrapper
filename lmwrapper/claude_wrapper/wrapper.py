import pickle
from typing import Any, Iterable
import warnings
from anthropic.types import Usage
from lmwrapper.abstract_predictor import LmPredictor
from dataclasses import dataclass, field
from pprint import pprint
import anthropic
from anthropic.types.text_block import TextBlock
from pathlib import Path
from lmwrapper.abstract_predictor import (
    LmPredictor, LmPrompt, LmPrediction
)
from lmwrapper.batch_config import CompletionWindow
from lmwrapper.secrets_manager import SecretEnvVar, SecretFile, SecretInterface, assert_is_a_secret


@dataclass
class ClaudePrediction(LmPrediction):
    def __init__(
        self,
        completion_text: str,
        prompt: LmPrompt,
        error_message: str | None = None,
        usage: Usage | None = None,
    ):
        super().__init__(
            completion_text=completion_text,
            prompt=prompt,
            metad={
                "usage": usage.model_dump(),
            },
            error_message=error_message,
        )


    @classmethod
    def parse_from_cache(
        cls,
        completion_text: str,
        prompt: LmPrompt,
        metad_bytes: bytes,
        error_message: str | None,
    ):
        metad = pickle.loads(metad_bytes)
        print(metad)
        usage = Usage(**metad["usage"])
        return cls(
            completion_text=completion_text,
            prompt=prompt,
            error_message=error_message,
            usage=usage,
        )

    def serialize_metad_for_cache(self) -> bytes:
        return pickle.dumps(self.metad)

    @property
    def usage_output_tokens(self):
        return self.metad["usage"]['output_tokens']


class ClaudePredictor(LmPredictor):
    def __init__(
        self,
        api: anthropic.Anthropic,
        model: str,
        cache_outputs_default: bool = False,
    ):
        super().__init__(cache_outputs_default)
        self._api = api
        self._model = model

    def find_prediction_class(self, prompt):
        return ClaudePrediction

    @property
    def supports_token_operations(self) -> bool:
        return False

    def model_name(self):
        return self._model

    def get_model_cache_key(self):
        return self.model_name()

    def is_chat_model(self) -> bool:
        return True

    @property
    def token_limit(self):
        return ClaudeModelNames.name_to_info(self._model).token_limit

    def predict_many(
        self,
        prompts: list[LmPrompt],
        completion_window: CompletionWindow,
    ) -> Iterable[LmPrediction | list[LmPrediction]]:
        if completion_window == CompletionWindow.BATCH_ANY:
            warnings.warn("Batching for claude not yet implemented")
        yield from super().predict_many(prompts, completion_window)

    def _predict_maybe_cached(
        self,
        prompt: LmPrompt,
    ) -> ClaudePrediction | list[ClaudePrediction]:
        messages = prompt.get_text_as_chat().as_dicts()
        # Claude treats system message not as a message but as a separate field
        messages, system_message = pull_system_message_out_of_messages(messages)
        predictions = []
        for _ in range(prompt.num_completions or 1):
            # Claude doesn't actually support num_completions, so we just loop
            response = dict(
                model=self._model,
                max_tokens=(
                    prompt.max_tokens if prompt.max_tokens is not None
                    else self.default_tokens_generated
                ),
                temperature=prompt.temperature,
                messages=messages,
                top_p=prompt.top_p,
            )
            if system_message is not None:
                response["system"] = system_message['content']
            response = self._api.messages.create(**response)
            # Create predictions for each choice
            assert len(response.content) == 1
            content = response.content[0]
            assert isinstance(content, TextBlock)
            text = content.text
            pprint(response)
            predictions.append(
                ClaudePrediction(
                    completion_text=text,
                    prompt=prompt,
                    usage=response.usage,
                )
            )

        return predictions[0] if prompt.num_completions is None else predictions

    @property
    def supports_prefilled_chat(self) -> bool:
        return True


def pull_system_message_out_of_messages(messages):
    new_messages = []
    system_message = None
    for message in messages:
        if message['role'] == "system":
            if system_message is not None:
                raise ValueError("More than one system message found")
            system_message = message
        else:
            new_messages.append(message)
    return new_messages, system_message


class _ModelNamesMeta(type):
    def __iter__(cls):
        for attr in cls.__dict__:
            if isinstance(cls.__dict__[attr], ClaudeModelInfo):
                yield cls.__dict__[attr]


class ClaudeModelInfo(str):
    def __new__(
        cls, 
        name: str, 
        token_limit: int
    ):
        instance = super().__new__(cls, name)
        instance._token_limit = token_limit
        return instance

    @property
    def token_limit(self):
        return self._token_limit


class ClaudeModelNames(metaclass=_ModelNamesMeta):
    claude_3_5_sonnet = ClaudeModelInfo(
        "claude-3-5-sonnet-20241022", 8196)
    claude_3_5_haiku = ClaudeModelInfo(
        "claude-3-5-haiku-20241022", 8196)
    claude_4_sonnet = ClaudeModelInfo(
        "claude-sonnet-4-20250514", 200_000)
    claude_4_opus = ClaudeModelInfo(
        "claude-opus-4-20250514", 200_000)

    @classmethod
    def name_to_info(cls, name: str) -> ClaudeModelInfo | None:
        if isinstance(name, ClaudeModelInfo):
            return name
        for info in cls:
            if info == name:
                return info
        return None


def get_claude_lm(
    model_name: str = ClaudeModelNames.claude_4_sonnet,
    api_key_secret: SecretInterface | None = None,
    cache_outputs_default: bool = False,
) -> ClaudePredictor:
    if api_key_secret is None:
        api_key_secret = SecretEnvVar("ANTHROPIC_API_KEY")
        if not api_key_secret.is_readable():
            api_key_secret = SecretFile(Path("~/anthropic_key.txt").expanduser())
        if not api_key_secret.is_readable():
            msg = (
                "Cannot find an API key. By default the ANTHROPIC_API_KEY environment"
                " variable is used if it is available. Otherwise it will read from a"
                " file at ~/anthropic_key.txt. Please place the key at one of the locations"
                " or pass in a SecretInterface (like SecretEnvVar or SecretFile object)"
                " to the api_key_secret argument.\nYou can get an API key from"
                " https://console.anthropic.com/settings/keys"
            )
            raise ValueError(
                msg,
            )
    assert_is_a_secret(api_key_secret)

    if not api_key_secret.is_readable():
        msg = "API key is not defined"
        raise ValueError(msg)

    client = anthropic.Client(
        api_key=api_key_secret.get_secret().strip(),
    )

    return ClaudePredictor(
        api=client,
        model=model_name,
        cache_outputs_default=cache_outputs_default,
    )


if __name__ == "__main__":
    print("hello world")
    lm = get_claude_lm(model_name=ClaudeModelNames.claude_3_5_haiku)
    print(lm.predict("What is 2+2?"))
    print(lm.predict("Define 'anthropology' in one short sentence"))
