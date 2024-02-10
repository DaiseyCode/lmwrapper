from lmwrapper.structs import ChatGptRoles, LmChatTurn, LmPrompt


def test_prompt_to_dict_conversion():
    prompt = LmPrompt(
        text="Hello world",
        max_tokens=10,
        stop=["world"],
        logprobs=1,
        temperature=1.0,
        cache=False,
    )
    prompt_dict = prompt.dict_serialize()
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
    prompt_dict = prompt.dict_serialize()
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


def test_chat_serialize_2():
    # Copied from a failing serializaiton?
    prompt = LmPrompt(
        text=[
            LmChatTurn(
                role=ChatGptRoles.user,
                content="Complete the following method...",
                name=None,
                function_call=None,
            ),
        ],
        max_tokens=300,
        stop=None,
        logprobs=0,
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        num_completions=1,
        cache=True,
        echo=False,
        add_bos_token=True,
        add_special_tokens=True,
    )
    prompt.dict_serialize()
