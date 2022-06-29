from lmwrapper.caching import clear_cache_dir
from lmwrapper.openai_wrapper import get_goose_lm, get_open_ai_lm
from lmwrapper.structs import LmPrompt


def play_with_probs():
    #clear_cache_dir()
    lm = get_open_ai_lm()
    out = lm.predict(
        LmPrompt(
            "Once upon",
            max_toks=2,
            logprobs=10,
            cache=True,
            num_completions=1,
            echo=False
        ))
    print(out)
    print(out._get_completion_token_index())
    print(out.completion_tokens)
    print(out.completion_token_offsets)
    print(out.prompt_tokens)


def main():
    play_with_probs()
    exit()
    lm = get_open_ai_lm()
    clear_cache_dir()
    text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_toks=1, logprobs=10, cache=True
        ))
    print(text.completion_text)
    lm = get_open_ai_lm()
    new_text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_toks=1, logprobs=10, cache=True
        ))
    print(new_text.completion_text)
    assert text.completion_text == new_text.completion_text
    exit()
    lm = get_open_ai_lm()
    text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_toks=1, logprobs=10,
        ))
    #print(type(completion_text))
    print(text.completion_text)


if __name__ == "__main__":
    main()
