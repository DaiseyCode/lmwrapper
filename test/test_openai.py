from lmwrapper.caching import clear_cache_dir
from lmwrapper.openai_wrapper import get_goose_lm, get_open_ai_lm
from lmwrapper.structs import LmPrompt


def main():
    lm = get_open_ai_lm()
    clear_cache_dir()
    text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_toks=1, logprobs=10, cache=True
        ))
    print(text.text)
    lm = get_open_ai_lm()
    new_text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_toks=1, logprobs=10, cache=True
        ))
    print(new_text.text)
    assert text.text == new_text.text
    exit()
    lm = get_open_ai_lm()
    text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_toks=1, logprobs=10,
        ))
    #print(type(text))
    print(text.text)


if __name__ == "__main__":
    main()
