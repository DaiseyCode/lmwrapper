This provides a wrapper around the OpenAI API focusing
on being a clean and user-friendly interface. Because every input 
and output is object-oriented (rather than just JSON dictionaries with string
keys and values), your IDE can help you with things like argument and
property names and catch certain bugs statically.

The goal is also to unify the interface for both openai models and huggingface
models, but that is still a work in progress.

# Installation

```bash
pip install git+https://github.com/DNGros/lmwrapper.git
```

# Example usage

## Completion models

```python
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from lmwrapper.structs import LmPrompt

lm = get_open_ai_lm(  # Returns a LmPredictor object
    model_name=OpenAiModelNames.text_ada_001,
    api_key_secret=None, # By default this will read from the OPENAI_API_KEY environment variable.
                         # If that isn't set, it will try the file ~/oai_key.txt
                         # You need to place the key one of these places, 
                         # or pass in a different location. You can get an API 
                         # key at (https://platform.openai.com/account/api-keys)
)

prediction = lm.predict(
    LmPrompt(  # A LmPrompt object lets your IDE write args
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
    max_tokens=30,
    temperature=0,
))
print(pred.completion_text) 
# "Arr, me matey! Bitcoin be a digital currency that be workin' on a technology called blockchain..."
```

# Features

## Caching
Add `caching = True` in the prompt to cache the output to disk. Any
subsequent calls with this prompt will return the same value. Note that
this might be unexpected behavior if your temperature is non-zero. (You
will always sample the same output on reruns.)