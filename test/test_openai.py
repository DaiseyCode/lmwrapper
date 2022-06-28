from lmwrapper.openai_wrapper import get_goose_lm, get_open_ai_lm
from lmwrapper.structs import LmPrompt


def main():
    lm = get_goose_lm()
    text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_toks=1, logprobs=10,
        ))
    print(text.text)
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
