import os
import time
from collections.abc import Iterable

import tiktoken

from lmwrapper.batch_config import CompletionWindow
from lmwrapper.caching import clear_cache_dir
from lmwrapper.openai_wrapper import OpenAiModelNames, get_open_ai_lm
from lmwrapper.openai_wrapper.batching import OpenAiBatchManager
from lmwrapper.sqlcache import SqlBackedCache
from lmwrapper.structs import LmPrompt

IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def simple():
    clear_cache_dir()

    def load_dataset() -> list:
        """Load some toy task"""
        return ["France", "Japan", "China", "UK"]

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
    predictions = list(
        lm.predict_many(prompts, completion_window=CompletionWindow.BATCH_ANY),
    )
    assert len(predictions) == len(data)
    for ex, pred in zip(
        data,
        predictions,
        strict=False,
    ):  # Will wait for the batch to complete
        print(f"Country: {ex} --- Capital: {pred.completion_text}")
        assert {
            "France": "Paris",
            "Japan": "Tokyo",
            "China": "Beijing",
            "UK": "London",
        }[ex] == pred.completion_text


def get_unique_texts(n: int, model_name: str) -> Iterable[str]:
    tokenizer = tiktoken.encoding_for_model(model_name)
    special_token_ids = set(
        [
            tokenizer.encode_single_token(token)
            for token in tokenizer.special_tokens_set
        ],
    )
    available_tokens = list(set(range(tokenizer.n_vocab - 1000)) - special_token_ids)
    # two digits will work
    for val in range(n):
        remainder = val % len(available_tokens)
        second_digit = val // len(available_tokens)
        seq = [tokenizer.decode([available_tokens[remainder]])]
        if second_digit > 0:
            seq = [tokenizer.decode([available_tokens[second_digit]])] + seq
        yield " ".join(seq)


def over50k():
    clear_cache_dir()
    model_name = OpenAiModelNames.gpt_4o_mini
    lm = get_open_ai_lm(model_name)
    cache = SqlBackedCache(lm=lm)
    batching_manager = OpenAiBatchManager(
        [
            LmPrompt(text, cache=True, max_tokens=1)
            for text in get_unique_texts(60000, model_name)
        ],
        cache=cache,
    )
    batching_manager.start_batch()
    print(batching_manager)
    for i, pred in enumerate(batching_manager.iter_results()):
        pass


def bigarthmatic():
    # clear_cache_dir()
    model_name = OpenAiModelNames.gpt_3_5_turbo
    lm = get_open_ai_lm(model_name)
    cache = SqlBackedCache(lm=lm)

    def prompt_text_for_num(num):
        return f"Answer with just the number. What is {num} plus 10?"

    num_prompts = 500
    batching_manager = OpenAiBatchManager(
        [
            LmPrompt(
                prompt_text_for_num(number),
                cache=True,
                max_tokens=5,
                temperature=0,
            )
            for number in range(num_prompts)
        ],
        cache=cache,
        max_prompts_per_batch=400,
    )
    batching_manager.start_batch()
    print(batching_manager)
    fails = 0
    samples = 0
    for i, pred in enumerate(batching_manager.iter_results()):
        samples += 1
        assert pred.prompt.text == prompt_text_for_num(i)
        if int(pred.completion_text.strip()) != i + 10:
            fails += 1
            print(f"Fail: {i} + 10 = {pred.completion_text}")
    assert fails == 0
    assert samples == num_prompts
    print("Done. All arithmetic correct.")


def token_queue_limit_try():
    clear_cache_dir()
    model_name = OpenAiModelNames.gpt_3_5_turbo
    lm = get_open_ai_lm(model_name)
    cache = SqlBackedCache(lm=lm)

    batching_manager = OpenAiBatchManager(
        [
            LmPrompt(" a" * 100 + str(unique), cache=True, max_tokens=1, temperature=0)
            for unique in range(1000)
        ],
        cache=cache,
        max_prompts_per_batch=10000,
    )
    batching_manager.start_batch()
    #
    print("sleep")
    time.sleep(2)
    batching_manager = OpenAiBatchManager(
        [
            LmPrompt(
                " a" * 100 + str(unique) + "t2",
                cache=True,
                max_tokens=1,
                temperature=0,
            )
            for unique in range(3500)
        ],
        cache=cache,
        max_prompts_per_batch=10000,
    )
    # This should over the limit
    batching_manager.start_batch()
    print(list(batching_manager.iter_results()))
