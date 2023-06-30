from lmwrapper.caching import clear_cache_dir
from lmwrapper.openai_wrapper import get_goose_lm, get_open_ai_lm, OpenAiModelNames
from lmwrapper.structs import LmPrompt, LmChatDialog


def play_with_probs():
    #clear_cache_dir()
    lm = get_open_ai_lm()
    out = lm.predict(
        LmPrompt(
            "Once upon",
            max_tokens=2,
            logprobs=10,
            cache=True,
            num_completions=1,
            echo=False
        ))
    print(out)
    print(out._get_completion_token_index())
    print(out.completion_tokens)
    print(out.completion_token_offsets)
    print(out.completion_logprobs)
    print(out.prompt_tokens)


def test_simple_pred():
    lm = get_open_ai_lm()
    out = lm.predict(
        LmPrompt(
            "Once upon a",
            max_tokens=1,
            logprobs=10,
            cache=True,
            num_completions=1,
            echo=False
        ))
    assert out.completion_text.strip() == "time"
    print(out)


def test_simple_chat_mode():
    lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo)
    out = lm.predict(LmPrompt(
        "What is 2+2? Answer with just one number.",
        max_tokens=1,
        num_completions=1,
        cache=False,
    ))
    assert out.completion_text.strip() == "4"


def test_simple_chat_mode_multiturn():
    lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo)
    prompt = [
        "What is 2+2? Answer with just one number.",
        "4",
        "What is 3+2?"
    ]
    assert LmChatDialog(prompt).as_dicts() == [
        {'role': 'user', 'content': 'What is 2+2? Answer with just one number.'},
        {'role': 'system', 'content': '4'},
        {'role': 'user', 'content': 'What is 3+2?'}
    ]
    out = lm.predict(LmPrompt(
        prompt,
        max_tokens=1,
        num_completions=1,
        cache=False,
    ))
    assert out.completion_text.strip() == "5"


def main():
    play_with_probs()
    exit()
    lm = get_open_ai_lm()
    clear_cache_dir()
    text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_tokens=1, logprobs=10, cache=True
        ))
    print(text.completion_text)
    lm = get_open_ai_lm()
    new_text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_tokens=1, logprobs=10, cache=True
        ))
    print(new_text.completion_text)
    assert text.completion_text == new_text.completion_text
    exit()
    lm = get_open_ai_lm()
    text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_tokens=1, logprobs=10,
        ))
    #print(type(completion_text))
    print(text.completion_text)


if __name__ == "__main__":
    main()
