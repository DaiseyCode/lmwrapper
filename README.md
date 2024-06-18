<!--
Provides a wrapper around OpenAI API and Hugging Face Language models, focusing
on being a clean and user-friendly interface. Because every input
and output is object-oriented (rather than just JSON dictionaries with string
keys and values), your IDE can help you with things like argument and
property names and catch certain bugs statically. Additionally, it allows
you to switch inbetween OpenAI endpoints and local models with minimal changes.

Because every input
and output is object-oriented (rather than just JSON dictionaries with string
keys and values), your IDE can help you with things like argument and
property names and catch certain bugs statically. Additionally, it allows
you to switch inbetween OpenAI endpoints and local models with minimal changes.
-->

`lmwrapper` provides a wrapper around OpenAI API and Hugging Face Language models, focusing
on being a clean, object-oriented, and user-friendly interface. It has two main goals:

A) Make it easier to use the OpenAI API

B) Make it easy to reuse your code for other language models with minimal changes.

Some key features currently include local caching of responses, and super simple
use of the OpenAI batching API which can save 50% on costs.

`lmwrapper` is lightweight and can serve as a flexible stand-in for the OpenAI API.

## Installation

For usage with just OpenAI models:

```bash
pip install lmwrapper
```

For usage with HuggingFace models as well:

```bash
pip install 'lmwrapper[hf]'
```

For development dependencies:

```bash
pip install 'lmwrapper[dev]'
```

<!---
If you prefer using `conda`/`mamba` to manage your environments, you may edit the `environment.yml` file to your liking & setup and create a new environment based on it:

```bash
mamba env create -f environment.yml
```

Please note that this method is for development and not supported.
-->

## Example usage

### Completion models

```python
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from lmwrapper.structs import LmPrompt

lm = get_open_ai_lm(
    model_name=OpenAiModelNames.gpt_3_5_turbo_instruct,
    api_key_secret=None,  # By default, this will read from the OPENAI_API_KEY environment variable.
    # If that isn't set, it will try the file ~/oai_key.txt
    # You need to place the key in one of these places,
    # or pass in a different location. You can get an API
    # key at (https://platform.openai.com/account/api-keys)
)

prediction = lm.predict(
    LmPrompt(  # A LmPrompt object lets your IDE hint on args
        "Once upon a",
        max_tokens=10,
        temperature=1, # Set this to 0 for deterministic completions
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
        "4",  # assistant turn
        "What is 5+3?"  # user turn
        "8",  # assistant turn
        "What is 4+4?"  # user turn
        # We use few-shot turns to encourage the answer to be our desired format.
        #   If you don't give example turns you might get something like
        #   "The answer is 8." instead of just "8".
    ],
    max_tokens=10,
))
print(pred.completion_text)  # "8"

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

## Caching

Add `caching = True` in the prompt to cache the output to disk. Any
subsequent calls with this prompt will return the same value. Note that
this might be unexpected behavior if your temperature is non-zero. (You
will always sample the same output on reruns.)


## OpenAI Batching

The OpenAI [batching API](https://platform.openai.com/docs/guides/batch) has a 50% reduced cost when willing to accept a 24-hour turnaround. This makes it good for processing datasets or other non-interactive tasks (which is the main target for `lmwrapper` currently).

`lmwrapper` takes care of managing the batch files so that it's as easy 
as the normal API.

<!-- no skip test -->
```python
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from lmwrapper.structs import LmPrompt
from lmwrapper.batch_config import CompletionWindow

def load_dataset() -> list:
    """Load some toy task"""
    return ["France", "United States", "China"]

def make_prompts(data) -> list[LmPrompt]:
    """Make some toy prompts for our data"""
    return [
        LmPrompt(
            f"What is the capital of {country}? Answer with just the city name.",
            max_tokens=10,
            temperature=0,
            cache=True,
        ) 
        for country in data
    ]

data = load_dataset()
prompts = make_prompts(data)
lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo)
predictions = lm.predict_many(
    prompts,
    completion_window=CompletionWindow.BATCH_ANY 
    #                 ^ swap out for CompletionWindow.ASAP
    #                   to complete as soon as possible via
    #                   the non-batching API at a higher cost.
) # The batch is submitted here

for ex, pred in zip(data, predictions):  # Will wait for the batch to complete
    print(f"Country: {ex} --- Capital: {pred.completion_text}")
    if ex == "France": assert pred.completion_text == "Paris" 
    # ...
```

The above code could technically take up to 24hrs to complete. However,
OpenAI seems to complete these quicker (for example, these three prompts in ~1 minute or less). In a large batch, you don't have to keep the process running for hours. Thanks to `lmwrapper` cacheing it will automatically load or pick back up waiting on the
existing batch.

The `lmwrapper` cache lets you also intermix cached and uncached examples.

<!-- skip test -->
```python
# ... above code

def load_dataset_more_data() -> list:
    """Load some toy task"""
    return ["Mexico", "Canada"]

data = load_data() + load_dataset_more_data()
prompts = make_prompts(data)
# If we submit the new data, only the new data will get
# submitted to the batch. The already completed prompts will
# be loaded near-instantly from the local cache.
predictions = list(lm.predict_many(
    prompts,
    completion_window=CompletionWindow.BATCH_ANY
))
```

This feature is mostly designed for the OpenAI cost savings. You could swap out the model for HuggingFace and the same code
will still work. However, internally it is like a loop over the prompts.
Eventually in `lmwrapper` we want to do more complex batching if
memory is available.

#### Caveats / Implementation needs
This feature is still somewhat experimental. There are a few known
things to sort out:

- [ ] Automatically splitting up batches when have >50,000 prompts (limit from OpenAI)
- [ ] Automatically splitting up batch when exceeding 100MB prompts limit
- [ ] Recovering / splitting up batches when hitting your token Batch Queue Limit (see [docs on limits](https://platform.openai.com/docs/guides/rate-limits/usage-tiers))
- [ ] Handling of failed prompts / batches
- [ ] Fancy batching of HF
- [ ] Concurrent batching when in ASAP mode
 
Feel free to open an issue to discuss one of these or something else.

### Retries on rate limit

```python
from lmwrapper.openai_wrapper import *

lm = get_open_ai_lm(
    OpenAiModelNames.gpt_3_5_turbo_instruct,
    retry_on_rate_limit=True
)
```

## Other features

### Built-in token counting

```python
from lmwrapper.openai_wrapper import *
from lmwrapper.structs import LmPrompt

lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo_instruct)
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
- [X] Redesign cache to make it easier to manage
- [X] OpenAI batching interface (experimental)
- [ ] Anthropic interface
- [ ] Multimodal/images in super easy format (like automatically process pil, opencv, etc)
- [ ] sort through usage of quantized models
- [ ] Cost estimating (so can estimate cost of a prompt before running / track total cost)
- [ ] Additional Huggingface runtimes (TensorRT, BetterTransformers, etc)
- [ ] async / streaming (not a top priority for non-interactive research use cases)