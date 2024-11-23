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
    for num_completions in (None, 1):
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


def test_sys_message():
    prompt = LmPrompt(
        [
            LmChatTurn(ChatGptRoles.system, "You are a calculator assistant. Please only respond with the number answer and nothing else."),
            "What is 2+2?",
        ],
    )
    chat = prompt.get_text_as_chat()
    assert chat[0].role == ChatGptRoles.system
    assert chat[0].content == "You are a calculator assistant. Please only respond with the number answer and nothing else."
    assert chat[1].role == ChatGptRoles.user
    assert chat[1].content == "What is 2+2?"


def test_chat_roles():
    prompt = LmPrompt(
        [
            "What is 2+2?",
            "4",
            "What is 5+3?",
            "8",
            "What is 1+4?",
        ],
    )
    chat = prompt.get_text_as_chat()
    assert chat[0].role == ChatGptRoles.user
    assert chat[1].role == ChatGptRoles.assistant
    assert chat[2].role == ChatGptRoles.user
    assert chat[3].role == ChatGptRoles.assistant
    assert chat[4].role == ChatGptRoles.user
