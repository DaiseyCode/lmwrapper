from lmwrapper.openai_wrapper import prompt_to_openai_args_dict, OpenAiModelNames, OpenAiModelInfo, \
    OpenAIPredictor, get_open_ai_lm
from lmwrapper.sqlcache import prompt_to_sample_hash_text, SqlBackedCache
import json
from lmwrapper.structs import LmPrompt
import io
import time


class OpenAiBatchManager:
    def __init__(
        self,
        prompts: list[LmPrompt],
        cache: SqlBackedCache,
        maintain_order: bool = True,
    ):
        self._validate_prompts_input(prompts)
        self._awaiting_marker = object()
        self._output = [self._awaiting_marker] * len(prompts)
        self._num_yielded = 0
        self._cache = cache
        self._maintain_order = maintain_order

    def _organize_prompts(self, prompts):
        # Loop through and figure out the already completed ones
        for prompts in prompts:
            pass

    def _validate_prompts_input(self, prompts):
        if not prompts:
            raise ValueError("No prompts provided")
        if not isinstance(prompts, list):
            raise ValueError("Prompts must be a list")
        if not all(isinstance(prompt, LmPrompt) for prompt in prompts):
            raise ValueError("All prompts must be LmPrompt instances")
        if not all(prompt.cache for prompt in prompts):
            raise ValueError(
                "All prompts must have cache set to True to currently use batching manager"
            )


def main():
    lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo)
    prompts = [
        _prompt_to_arg_dict_for_batch(prompt, lm)
        for prompt in [
            LmPrompt("hello world"),
            #LmPrompt("goodbye world"),
        ]
    ]
    # Convert prompts to JSONL string
    jsonl_str = "\n".join(json.dumps(prompt) for prompt in prompts)
    # Convert JSONL string to bytes
    jsonl_bytes = io.BytesIO(jsonl_str.encode('utf-8'))
    batch_input_file = lm._api.files.create(
        file=jsonl_bytes,
        purpose="batch",
    )
    print(batch_input_file)
    batch_data = lm._api.batches.create(
        input_file_id=batch_input_file.id,
        endpoint=(
            "/v1/chat/completions"
            if lm.is_chat_model
            else "/v1/completions"
        ),
        completion_window="24h",
    )
    print(batch_data)
    time.sleep(1)
    for _ in range(10000):
        retrieve_data = lm._api.batches.retrieve(batch_data.id)
        print(retrieve_data)
        if retrieve_data.status not in (
            "validating",
            "in_progress",
            "finalizing",
        ):
            break
        time.sleep(3)
    content = lm._api.files.content(retrieve_data.output_file_id)
    print(content)
    # Decode the content into a string
    content_str = content.content.decode('utf-8')
    print(content_str)


def _prompt_to_arg_dict_for_batch(
    prompt: LmPrompt,
    lm: OpenAIPredictor,
):
    args = prompt_to_openai_args_dict(
        prompt,
        engine_name=lm.model_name(),
        chat_model=lm.is_chat_model,
    )
    request = {
        "body": args,
        "method": "POST",
        "url": (
            "/v1/chat/completions"
            if lm.is_chat_model
            else "/v1/completions"
        ),
        # TODO multiple responses
        "custom_id": prompt_to_sample_hash_text(prompt, lm.get_model_cache_key()),
    }
    return request


if __name__ == "__main__":
    main()