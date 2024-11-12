from lmwrapper.abstract_predictor import LmPredictor
from pprint import pprint
import anthropic
from anthropic.types.text_block import TextBlock
from pathlib import Path
from lmwrapper.abstract_predictor import (
    LmPredictor, LmPrompt, LmPrediction
)
from lmwrapper.secrets_manager import SecretEnvVar, SecretFile, SecretInterface, assert_is_a_secret


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
        return ClaudePredictor

    def model_name(self):
        return self._model

    def _predict_maybe_cached(
        self,
        prompt: LmPrompt,
    ) -> LmPrediction | list[LmPrediction]:
        messages = prompt.get_text_as_chat().as_dicts()
        
        try:
            response = self._api.messages.create(
                model=self._model,
                max_tokens=prompt.max_tokens,
                temperature=prompt.temperature,
                messages=messages,
                #system=prompt.system_prompt or "",
            )
            
            # Create predictions for each choice
            assert len(response.content) == 1
            content = response.content[0]
            assert isinstance(content, TextBlock)
            print(type(content))
            text = content.text
            predictions = [
                LmPrediction(
                    completion_text=text,
                    prompt=prompt,
                    metad=response,
                    internals=None
                )
            ]
            
            return predictions[0] if prompt.num_completions is None else predictions
            
        except Exception as e:
            # Re-raise with context
            raise RuntimeError(f"Claude API call failed: {str(e)}") from e


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

    @classmethod
    def name_to_info(cls, name: str) -> ClaudeModelInfo | None:
        if isinstance(name, ClaudeModelInfo):
            return name
        for info in cls:
            if info == name:
                return info
        return None


def get_claude_lm(
    model_name: str = ClaudeModelNames.claude_3_5_haiku,
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

    key = Path("~/anthropic_key.txt").expanduser().read_text().strip()
    client = anthropic.Anthropic(api_key=key)

    pred = ClaudePredictor(
        api=client,
        model=ClaudeModelNames.claude_3_5_haiku,
        cache_outputs_default=False,
    )
    print(pred.predict("What is 2+2?"))
    exit()

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=0,
        system="",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Write an example nvim plugin in python. Explain some of the important considerations when writing them",
                    }
                ]
           },
        ],
    )
    pprint(message)
    exit()
    print(message.content)

