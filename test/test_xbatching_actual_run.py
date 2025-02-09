import os

import pytest

from lmwrapper.caching import clear_cache_dir
from lmwrapper.openai_wrapper import OpenAiModelNames, get_open_ai_lm
from lmwrapper.openai_wrapper.batching import OpenAiBatchManager
from lmwrapper.sqlcache import SqlBackedCache
from lmwrapper.structs import LmPrompt

IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(
    IS_GITHUB_ACTIONS or True, # taking too long right now
    reason="Batches could take a long time so skip in CI",
)
def test_split_up_prompt_with_arithmetic():
    clear_cache_dir()
    model_name = OpenAiModelNames.gpt_4o_mini
    lm = get_open_ai_lm(model_name)
    cache = SqlBackedCache(lm=lm)

    def prompt_text_for_num(num):
        return f"Answer with just the number. What is {num} plus 10?"

    num_prompts = 10
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
        max_prompts_per_batch=4,
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


@pytest.mark.skipif(
    IS_GITHUB_ACTIONS or True, # taking too long right now
    reason="Batches could take a long time so skip in CI",
)
def test_failed_prompt():
    clear_cache_dir()
    model_name = OpenAiModelNames.gpt_3_5_turbo
    lm = get_open_ai_lm(model_name)
    cache = SqlBackedCache(lm=lm)
    batching_manager = OpenAiBatchManager(
        [
            LmPrompt(
                "a",
                cache=True,
                max_tokens=30_000,  # output too big
                temperature=0,
            ),
            LmPrompt(
                "a",
                cache=True,
                max_tokens=30_000,  # output too big
                temperature=1000,  # Bad temp
            ),
            LmPrompt(  # A good prompt
                "Output only the next letter in the sequence: a b c d",
                cache=True,
                max_tokens=1,
                temperature=0,
            ),
        ],
        cache=cache,
    )
    batching_manager.start_batch()
    results = list(batching_manager.iter_results())
    assert len(results) == 3
    assert results[0].has_errors
    assert results[1].has_errors
    assert not results[2].has_errors
    assert results[2].completion_text.strip() == "e"


def test_why_fail():
    assert True


if __name__ == "__main__":
    if not IS_GITHUB_ACTIONS:
        # simple()
        # print(list(get_unique_texts(int(1e6), OpenAiModelNames.gpt_3_5_turbo)))
        # over50k()
        # clear_cache_dir()
        # bigarthmatic()
        # token_queue_limit_try()
        # failed_prompt()
        pass
