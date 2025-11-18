`lmwrapper` provides a wrapper around OpenAI API and Hugging Face Language models, focusing
on being a clean, object-oriented, and user-friendly interface. It has three main goals:

A) Make it easier to use the OpenAI API.

B) Make it easier to reuse your code for other language models with minimal changes.

C) Simplifying and support research-focused use cases (like help when running on datasets, and allow research that requires accessing model internals of local models).

Some key features currently include local disk caching of responses, and super simple
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

For usage with Claude models from Anthropic:
```bash
pip install 'lmwrapper[anthropic]'
```

For development dependencies:

```bash
pip install 'lmwrapper[dev]'
```

The above args can be combined. For example:
```bash
pip install 'lmwrapper[hf,anthropic]'
```

<!---
If you prefer using `conda`/`mamba` to manage your environments, you may edit the `environment.yml` file to your liking & setup and create a new environment based on it:

```bash
mamba env create -f environment.yml
```

Please note that this method is for development and not supported.
-->

## Example usage

<!---
### Basic Completion and Prompting

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
-->

### Chat

```python
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from lmwrapper.structs import LmPrompt, LmChatTurn

lm = get_open_ai_lm(
    model_name=OpenAiModelNames.gpt_4_1_nano,
    api_key_secret=None,  # By default, this will read from the OPENAI_API_KEY environment variable.
    # If that isn't set, it will try the file ~/oai_key.txt
    # You need to place the key in one of these places,
    # or pass in a different location. You can get an API
    # key at (https://platform.openai.com/account/api-keys)
)

# Single user utterance
pred = lm.predict("What is 2+2?")
print(pred.completion_text)  # "2 + 2 equals 4."


# Use a LmPrompt to have more control of the parameters
pred = lm.predict(LmPrompt(
    "What is 2+6?",
    max_tokens=10,
    temperature=0, # Set this to 0 for deterministic completions. 
                   # Note, GPT-5 does not support temperature.
))
print(pred.completion_text)  # "2 + 6 equals 8."

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
        #   "4 + 4 equals 8." instead of just "8" as desired.
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

## Caching

Add `cache = True` in the prompt to cache the output to disk. Any
subsequent calls with this prompt will return the same value. Note that
this might be unexpected behavior if your temperature is non-zero. (You
will always sample the same output on reruns).

```python
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames
from lmwrapper.structs import LmPrompt

lm = get_open_ai_lm(OpenAiModelNames.gpt_4_1_nano)

prompt = LmPrompt(
  "Briefly describe the city of Paris", 
  cache=True,
  temperature=1,
  max_tokens=25,
)
first_prediction = lm.predict(prompt)
print(first_prediction.completion_text) 
# ... eg, "Paris is a city of romance and art, renowned for its iconic landmarks, vibrant culture, and rich history."

# The response to this prompt is now saved to the disk.
# You could rerun this script and you would load from cache near-instantly.
# This can simplify running experimentation and data processing scripts
# where you are running a dataset through a model and doing analysis.
# You then only have to actually query the model once.
repredict = lm.predict(prompt)
print(repredict.completion_text)
assert first_prediction.completion_text == repredict.completion_text
lm.remove_prompt_from_cache(prompt)
pred_after_clear = lm.predict(prompt)
assert pred_after_clear.completion_text != first_prediction.completion_text
```


## OpenAI Batching

The OpenAI [batching API](https://platform.openai.com/docs/guides/batch) has a 50% reduced cost when willing to accept a 24-hour turnaround. This makes it good for processing datasets or other non-interactive tasks (which is the main target for `lmwrapper` currently).

`lmwrapper` takes care of managing the batch files and other details so that it's as easy 
as the normal API.

<!-- skip gh-action -->
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
            metadata={"country": country}
        ) 
        for country in data
    ]

data = load_dataset()
prompts = make_prompts(data)
lm = get_open_ai_lm(OpenAiModelNames.gpt_4_1_nano)
predictions = lm.predict_many(
    prompts,
    completion_window=CompletionWindow.BATCH_ANY 
    #                 ^ swap out for CompletionWindow.ASAP
    #                   to complete as soon as possible via
    #                   the non-batching API at a higher cost.
) # The batch is submitted here

for pred in predictions:  # Will wait for the batch to complete
    country = pred.prompt.metadata['country']
    print(f"Country: {country} --- Capital: {pred.completion_text}")
    if country == "France": assert pred.completion_text == "Paris" 
    # ...
```

The above code could technically take up to 24hrs to complete. However,
OpenAI seems to complete these quicker (for example, these three prompts in ~1 minute or less). In a large batch, you don't have to keep the process running for hours. Thanks to `lmwrapper` cacheing it will automatically load or pick back up waiting on the
existing batch when the script is reran.

The `lmwrapper` cache lets you also intermix cached and uncached examples.

<!-- skip test -->
```python
# ... above code

def load_more_data() -> list:
    """Load some toy task"""
    return ["Mexico", "Canada"]

data = load_data() + load_more_data()
prompts = make_prompts(data)
# If we submit the five prompts, only the two new prompts will be
# submitted to the batch. The already completed prompts will
# be loaded near-instantly from the local cache.
predictions = list(lm.predict_many(
    prompts,
    completion_window=CompletionWindow.BATCH_ANY
))
```

`lmwrapper` is designed to automatically manage the batching of thousands or millions of prompts. 
If needed, it will automatically split up prompts into sub-batches and will manage
issues around rate limits.

This feature is mostly designed for the OpenAI cost savings. You could swap out the model for HuggingFace and the same code
will still work. However, internally it is like a loop over the prompts.
Eventually in `lmwrapper` we want to do more complex batching if
GPU/CPU/accelerator memory is available.

#### Caveats / Implementation needs
This feature is still somewhat experimental. It likely works in typical
usecases, but there are few known things 
to sort out / TODOs:

- [X] Retry batch API connection errors
- [X] Automatically splitting up batches when have >50,000 prompts (limit from OpenAI) 
- [X] Recovering / splitting up batches when hitting your token Batch Queue Limit (see [docs on limits](https://platform.openai.com/docs/guides/rate-limits/usage-tiers))
- [X] Handle canceled batches during current run (use the [web interface](https://platform.openai.com/batches) to cancel)
- [X] Handle/recover canceled batches outside of current run
- [X] Handle if openai batch expires unfinished in 24hrs (though not actually tested or observed this)
- [X] Automatically splitting up batch when exceeding 100MB prompts limit
- [X] Handling of failed prompts (like when have too many tokens). Use LmPrediction.has_errors and LmPrediction.error_message to check for an error on a response.
- [X] Handle when there are duplicate prompts in batch submission
- [ ] Claude batching
- [ ] Handle when a given prompt has `num_completions>1`
- [ ] Automatically clean up API files after done (right now end up with a lot of file in [storage](https://platform.openai.com/storage/files). There isn't an obvious cost for these batch files, but this might change and it would be better to clean them up.)
- [ ] Fancy batching of HF
- [ ] Configuration setting for whether process killing (like with ctrl+c) should attempt to kill the batch. Right now it intentially persists in background. Maybe takes adding a predictions.cancel_all() prop and user can make their own context.
- [ ] Concurrent batching when in ASAP mode
- [ ] A "scheduler" paradigm trading off batch, flex, and ASAP processing to hit some timing target
- [ ] Test on free-tier accounts. It is not clear what the tiny request limit counts

Please open an issue if you want to discuss one of these or something else.

Note, in the progress bars in PyCharm can be bit cleaner if you enable 
[terminal emulation](https://stackoverflow.com/a/64727188) in your run configuration.

## Hugging Face models

Local Causal LM models on Hugging Face models can be used interchangeably with the
OpenAI models.

Note: The universe of Huggingface models is diverse and inconsistent. Some (especially the non-completion ones) might require special prompt formatting to work as expected. Some models might not work at all.

```python
from lmwrapper.huggingface_wrapper import get_huggingface_lm
from lmwrapper.structs import LmPrompt

# Download a small model for demo
lm = get_huggingface_lm("gpt2") # 124M parameters

prediction = lm.predict(LmPrompt(
    "The capital of Germany is Berlin. The capital of France is",
    max_tokens=1,
    temperature=0,
))
print(prediction.completion_text)
assert prediction.completion_text == " Paris"
```
<!-- Model internals -->

Additionally, with HuggingFace models `lmwrapper` provides an interface for
accessing the model internal states.

## Claude
```python
from lmwrapper.claude_wrapper import (
    get_claude_lm, ClaudeModelNames
)
lm = get_claude_lm(ClaudeModelNames.claude_3_5_haiku)
prediction = lm.predict("Define 'anthropology' in one short sentence") 
print(prediction.completion_text) # Anthropology is the scientific study of human cultures, societies, behaviors, and...
```
Note Anthropic does not expose any information about the tokenization for Claude.

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
- [X] Redesign cache away from generic `diskcache` to make it easier to manage as a sqlite db
- [X] Smart caching when num_completions > 1 (reusing prior completions)
- [X] OpenAI batching interface (experimental)
- [X] Anthropic interface (basic)
    - [X] Claude system messages
- [X] Use the huggingface chat templates for chat models if available
- [X] Be able to add metadata to a prompt
- [ ] Automatic cache eviction to limit count or disk size (right now have to run a SQL query to delete entries before a certain time or matching your criteria)
- [ ] Multimodal/images in super easy format (like automatically process pil, opencv, etc)
- [ ] sort through usage of quantized models
- [ ] Cost estimation of a prompt before running / "observability" monitoring of total cost
- [ ] Additional Huggingface runtimes (TensorRT, BetterTransformers, etc)
- [ ] async / streaming (not a top priority for non-interactive research use cases)
- [ ] some lightweight utilities to help with tool use
