# LmWrapper

Provides a wrapper around OpenAI API and Hugging Face Language models, focusing
on being a clean and user-friendly interface. Because every input
and output is object-oriented (rather than just JSON dictionaries with string
keys and values), your IDE can help you with things like argument and
property names and catch certain bugs statically. Additionally, it allows
you to switch inbetween OpenAI endpoints and local models with minimal changes.

## Installation

For usage with just OpenAI models:

```bash
pip install "lmwrapper @ git+https://github.com/DNGros/lmwrapper.git"
```

For usage with HuggingFace models as well:

```bash
pip install "lmwrapper[hf] @ git+https://github.com/DNGros/lmwrapper.git"
```

For development dependencies:

```bash
pip install "lmwrapper[dev] @ git+https://github.com/DNGros/lmwrapper.git"
```

Additionally, if you wish to use the latest version of `transformers` from GitHub:

```bash
pip install "lmwrapper[hf-dev] @ git+https://github.com/DNGros/lmwrapper.git"
```

If you prefer using `conda`/`mamba` to manage your environments, you may edit the `environment.yml` file to your liking & setup and create a new environment based on it:

```bash
mamba env create -f environment.yml
```

Please note that this method is for development and not supported.

## Example usage

### Completion models

```python
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from lmwrapper.structs import LmPrompt

lm = get_open_ai_lm(
    model_name=OpenAiModelNames.text_ada_001,  # Or use OpenAiModelNames.gpt_3_5_turbo_instruct
                                               # for the most capable completion model.
    api_key_secret=None, # By default this will read from the OPENAI_API_KEY environment variable.
                         # If that isn't set, it will try the file ~/oai_key.txt
                         # You need to place the key in one of these places,
                         # or pass in a different location. You can get an API
                         # key at (https://platform.openai.com/account/api-keys)
)

prediction = lm.predict(
    LmPrompt(  # A LmPrompt object lets your IDE hint on args
        "Once upon a",
        max_tokens=10,
    )
)
print(prediction.completion_text)
# " time, there were three of us." - Example. This will change with each sample.
```

### Chat models

```python
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from lmwrapper.structs import LmPrompt, LmChatTurn
lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo)

# Single user utterance
pred = lm.predict("What is 2+2?")
print(pred.completion_text)  # "2+2 is equal to 4."

# Conversation alternating between `user` and `assistant`.
pred = lm.predict(LmPrompt(
    [
        "What is 2+2?",  # user turn
        "4",             # assistant turn
        "What is 5+3?"   # user turn
        "8",             # assistant turn
        "What is 4+4?"   # user turn
        # We use few-shot turns to encourage the answer to be our desired format.
        #   If you don't give example turns you might get something like
        #   "The answer is 8." instead of just "8".
    ],
    max_tokens=10,
))
print(pred.completion_text) # "8"

# If you want things like the system message, you can use LmChatTurn objects
pred = lm.predict(LmPrompt(
    text=[
        LmChatTurn(role="system", content="You always answer like a pirate"),
        LmChatTurn(role="user", content="How does bitcoin work?"),
    ],
    max_tokens=25,
    temperature=0,
))
print(pred.completion_text)
# "Arr, me matey! Bitcoin be a digital currency that be workin' on a technology called blockchain..."
```

### Hugging Face models

Causal LM models on Hugging Face models can be used interchangeably with the
OpenAI models.

```python
from lmwrapper.huggingface_wrapper import get_huggingface_lm
from lmwrapper.structs import LmPrompt
lm = get_huggingface_lm("gpt2")  # The smallest 124M parameter model

prediction = lm.predict(LmPrompt(
    "The capital of Germany is Berlin. The capital of France is",
    max_tokens=1,
    temperature=0,
))
print(prediction.completion_text)
assert prediction.completion_text == " Paris"
```

## Features

`lmwrapper` provides several features missing from the OpenAI API.

### Caching

Add `caching = True` in the prompt to cache the output to disk. Any
subsequent calls with this prompt will return the same value. Note that
this might be unexpected behavior if your temperature is non-zero. (You
will always sample the same output on reruns.)

### Retries on rate limit

An OpenAIPredictor can be configured to read rate limit errors and wait the appropriate
amount of seconds in the error before retrying.

```python
from lmwrapper.openai_wrapper import *
lm = get_open_ai_lm(OpenAiModelNames.text_ada_001, retry_on_rate_limit=True)
```

## Other features

### Built-in token counting

```python
from lmwrapper.openai_wrapper import *
lm = get_open_ai_lm(OpenAiModelNames.text_ada_001)
assert lm.estimate_tokens_in_prompt(
    LmPrompt("My name is Spingldorph", max_tokens=10)) == 7
assert not lm.could_completion_go_over_token_limit(LmPrompt(
    "My name is Spingldorph", max_tokens=1000))
```

## TODOs

If you are interested in one of these particular features or something else
please make a Github Issue.

- [X] Openai completion
- [X] Openai chat
- [X] Huggingface interface
- [X] Huggingface device checking on PyTorch
- [X] Move cache to be per project
- [ ] sort through usage of quantized models
- [ ] async / streaming
- [ ] Redesign cache to make it easier to manage
- [ ] Additional Huggingface runtimes (TensorRT, BetterTransformers, etc)
- [ ] Anthropic interface
- [ ] Cost estimating (so can estimate cost of a prompt before running / track total cost)
