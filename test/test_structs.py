from lmwrapper.structs import LmPrompt


def test_prompt_to_dict_conversion():
    prompt = LmPrompt(
        text="Hello world",
        max_tokens=10,
        stop=["world"],
        logprobs=1,
        temperature=1.0,
        cache=False,
    )
    prompt_dict = prompt.as_dict()
    expected = {
        "text": "Hello world",
        "max_tokens": 10,
        "stop": ["world"],
        "logprobs": 1,
        "temperature": 1.0,
        "cache": False,
    }
    assert all(key in prompt_dict for key in expected)
    assert all(prompt_dict[key] == expected[key] for key in expected)


def test_prompt_to_dict_conversion_chat():
    prompt = LmPrompt(
        text=[
            "Hello world",
            "How are you?",
        ],
        max_tokens=10,
        temperature=1.0,
        cache=False,
    )
    prompt_dict = prompt.as_dict()
    expected = {
        "text": [
            {
                "role": "user",
                "content": "Hello world",
            },
            {
                "role": "assistant",
                "content": "How are you?",
            },
        ],
        "max_tokens": 10,
        "logprobs": 1,
        "temperature": 1.0,
        "cache": False,
        "stop": None,
    }
    print(prompt_dict)
    assert all(key in prompt_dict for key in expected)
    assert all(prompt_dict[key] == expected[key] for key in expected)
