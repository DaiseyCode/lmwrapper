This provides a wrapper around OpenAI API and Huggingface Language models focusing
on being a clean and user-friendly interface. Because every input 
and output is object-oriented (rather than just JSON dictionaries with string
keys and values), your IDE can help you with things like argument and
property names and catch certain bugs statically. Additionally, it allows
you to switch inbetween openai endpoints and local models with minimal changes.


# Installation

```bash
pip install git+https://github.com/DNGros/lmwrapper.git
```

# Example usage

## Completion models

```python
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from lmwrapper.structs import LmPrompt

lm = get_open_ai_lm(
    model_name=OpenAiModelNames.text_ada_001,
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

## Chat models

```python
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from lmwrapper.structs import LmPrompt, LmChatTurn
lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo)

# Single user utterance
pred = lm.predict("What is 2+2?")
print(pred.completion_text)  # "2+2 is equal to 4."

# Conversation alternating between `user` and `system`.
pred = lm.predict(LmPrompt(
    [
        "What is 2+2?",  # user turn
        "4",             # system turn
        "What is 5+3?"   # user turn
        "8",             # system turn
        "What is 4+4?"   # user turn
        # Because we have the fewshot examples, we might expect one number
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


## Huggingface models

Causal LM models on huggingface models can be used interchangeably with the
OpenAI models. Note it is still a todo to make sure devices and GPUs are used
appropriately.

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

# Features

`lmwrapper` provides several features missing from the OpenAI API.

## Caching
Add `caching = True` in the prompt to cache the output to disk. Any
subsequent calls with this prompt will return the same value. Note that
this might be unexpected behavior if your temperature is non-zero. (You
will always sample the same output on reruns.)

## Retries on rate limit
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
- [X] Openai completion
- [X] Openai chat
- [X] Huggingface interface
- [ ] Proper GPU handling with huggingface
- [ ] sort through usage of quantized models
- [ ] Improved caching (per-project cache and committable)
- [ ] Anthropic interface
- [ ] Cost estimating
