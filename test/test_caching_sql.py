from lmwrapper import sqlcache
from lmwrapper.caching import clear_cache_dir
from lmwrapper.openai_wrapper import get_open_ai_lm
from lmwrapper.sqlcache import (
    add_or_set_prediction_to_cache,
    get_from_cache,
    prompt_to_text_hash,
)
from lmwrapper.structs import LmPrompt


def test_add_cache():
    clear_cache_dir()
    lm = get_open_ai_lm()
    pred = lm.predict("Once upon a time")
    add_or_set_prediction_to_cache(pred, lm.model_name())


def test_get_cache():
    clear_cache_dir()
    lm = get_open_ai_lm()
    model_key = lm.model_name()
    prompt = LmPrompt("Once upon a time", cache=True)
    assert get_from_cache(prompt, lm) is None
    pred = lm.predict(prompt)
    add_or_set_prediction_to_cache(pred, model_key)
    rets = get_from_cache(prompt, lm)
    assert rets is not None
    assert len(rets) == 1
    ret = rets[0]
    assert isinstance(ret, pred.__class__), f"{ret.__class__} != {pred.__class__}"
    assert ret.completion_text == pred.completion_text
    assert pred == ret
    prompt2 = LmPrompt("Once upon a time", cache=True)
    assert get_from_cache(prompt2, lm) is not None
    assert get_from_cache(LmPrompt("blah", cache=True), lm) is None


def test_text_hash():
    clear_cache_dir()
    prompt = LmPrompt("hello world")
    hash = prompt_to_text_hash(prompt)
    assert hash == prompt_to_text_hash(prompt)
    assert hash != prompt_to_text_hash(LmPrompt("hello world!"))
    assert len(hash) == sqlcache._text_hash_len
    assert len(prompt_to_text_hash(LmPrompt(""))) == sqlcache._text_hash_len
    assert len(prompt_to_text_hash(LmPrompt("d" * 500))) == sqlcache._text_hash_len
    assert len(prompt_to_text_hash(LmPrompt(["foo", "bar"]))) == sqlcache._text_hash_len
    print(hash)
