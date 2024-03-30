# TODO (DNGros): Integrate this in for use
import ast

import requests

from lmwrapper.openai_wrapper.openai_wrapper import OpenAiModelNames, get_open_ai_lm


def get_github_file_var(url, var_name):
    # Get the file content
    response = requests.get(url)
    assert response.status_code == 200, f"Error: {response.content}"

    # Parse the python code
    parsed_python_file = ast.parse(response.text)

    # Search for the variable in the file's nodes
    for node in parsed_python_file.body:
        if isinstance(node, ast.Assign):  # if the statement is an assignment
            for target in node.targets:
                if target.id == var_name:  # if the variable name matches
                    return ast.literal_eval(
                        node.value,
                    )  # return the value of the variable

    return None  # return None if the variable is not found


def get_cost_dict_from_langchain() -> dict[str, float]:
    url = "https://raw.githubusercontent.com/hwchase17/langchain/master/langchain/callbacks/openai_info.py"
    var = "MODEL_COST_PER_1K_TOKENS"
    return get_github_file_var(url, var)


def get_all_max_tokens():
    for model_name in OpenAiModelNames:
        print("model_name = ", model_name)
        get_open_ai_lm(model_name)


if __name__ == "__main__":
    # print(f'MODEL_COST_PER_1K_TOKENS = {get_cost_dict_from_langchain()}')
    get_all_max_tokens()
