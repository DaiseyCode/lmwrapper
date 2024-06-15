"""
An interface for using the openai batch API.
This offers cost saving by accepting larger latencies.

It is designed closely around the openai API, but might be
abstracted later.
"""
import openai.types
import tqdm
import uuid
import openai.types.chat
from lmwrapper.openai_wrapper import prompt_to_openai_args_dict, OpenAiModelNames, OpenAiModelInfo, \
    OpenAIPredictor, get_open_ai_lm
from lmwrapper.sqlcache import prompt_to_sample_hash_text, SqlBackedCache, BatchRow
from lmwrapper.sqlcache_struct import BatchPredictionPlaceholder
import json
from lmwrapper.structs import LmPrompt, LmPrediction
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
        self._prompts = prompts
        self._started = False
        self._lm: OpenAIPredictor = cache.lm
        self._batch_id_to_pbar = {}

    def start_batch(self):
        need_prompts = self._organize_prompts(self._prompts)
        if need_prompts:
            self._send_batch(need_prompts)
        self._started = True

    def _organize_prompts(self, prompts) -> list[tuple[int, LmPrompt]]:
        # Loop through and figure out the already completed ones
        needed = []
        for i, prompt in enumerate(prompts):
            value = self._cache.get(prompt)
            print("Lookup", i, "value", value)
            if value is None:
                needed.append((i, prompt))
            elif isinstance(value, BatchPredictionPlaceholder):
                self._output[i] = value
            else:
                value.mark_as_cached()
                self._output[i] = value
        return needed

    def _send_batch(self, prompts: list[tuple[int, LmPrompt]]):
        jsonl = "\n".join(
            json.dumps(
                _prompt_to_arg_dict_for_batch(prompt, self._lm, str(index))
            )
            for index, prompt in prompts
        )
        ids, just_prompts = zip(*prompts)
        batch_input_file = _put_batch_in_file(jsonl, self._lm)
        batch_data = _make_batch(batch_input_file, self._lm)
        batch_row = BatchRow(
            batch_id=uuid.uuid4().hex,
            user_batch_name="",
            api_id=batch_data.id,
            api_category="openai",
            status=batch_data.status,
            waiting_for_a_result=True,
            created_at=batch_data.created_at,
            total_inputs=batch_data.request_counts.total,
            api_json_data=json.dumps(batch_data.dict()),
        )
        place_holders = self._cache.put_batch_placeholders(
            batch_row, just_prompts
        )
        for (index, prompt), place_holder in zip(prompts, place_holders):
            self._output[index] = place_holder
        print(batch_row)

    def _poll_completion(
        self,
        target: BatchPredictionPlaceholder
    ):
        print("start poll")
        start_time = time.time()
        pbar = self._pbar_for_targer(target)
        while (time_waited := time.time() - start_time) < 60 * 60 * 24:
            retrieve_data: openai.types.Batch = self._cache.lm._api.batches.retrieve(
                target.api_id
            )
            print("retrieved_data", retrieve_data)
            waiting_for_results = retrieve_data.status in (
                "validating",
                "in_progress",
                "finalizing",
            )
            self._cache.update_batch_row(
                retrieve_data.id,
                retrieve_data.status,
                waiting_for_a_result=waiting_for_results,
            )
            pbar.update(
                (retrieve_data.request_counts.completed + retrieve_data.request_counts.failed) - pbar.n
            )
            # Update description with failure count
            desc = f"Waiting for batch {target.api_id} completion."
            if retrieve_data.request_counts.failed:
                desc += f" ({retrieve_data.request_counts.failed} failed)"
            pbar.set_description(desc)
            if not waiting_for_results:
                self._update_cache_rows(retrieve_data)
                break
            if time_waited < 60:
                sleep_time = 1
            else:
                sleep_time = 5
            time.sleep(sleep_time)

    def _update_cache_rows(
        self,
        batch_data: openai.types.Batch,
    ):
        content = self._cache.lm._api.files.content(batch_data.output_file_id)
        content_str = content.content.decode('utf-8')
        print("Get content")
        print(content_str)
        for line in content_str.split("\n"):
            if not line:
                continue
            data = json.loads(line)
            custom_id = data["custom_id"]
            response = data["response"]
            prompt = self._prompts[int(custom_id)]
            body = response['body']
            if self._lm.is_chat_model:
                body = openai.types.chat.ChatCompletion.parse_obj(body)
            else:
                body = openai.types.Completion.parse_obj(body)
            pred = self._lm.prediction_from_api_response(body, prompt)
            assert len(pred) == 1
            pred = pred[0]
            self._output[int(custom_id)] = pred
            self._cache.add_or_set(pred)


    def _pbar_for_targer(self, target: BatchPredictionPlaceholder):
        batch_id = target.api_id
        if batch_id not in self._batch_id_to_pbar:
            print("Pbar total", target.batch_total_inputs)
            pbar = tqdm.tqdm(
                total=target.batch_total_inputs,
                desc=f"Waiting for batch {batch_id} completion"
            )
            self._batch_id_to_pbar[batch_id] = pbar
        return self._batch_id_to_pbar[batch_id]

    def iter_results(self):
        num_yielded = 0
        while num_yielded < len(self._output):
            result = self._output[num_yielded]
            if result is self._awaiting_marker:
                raise RuntimeError
            if isinstance(result, BatchPredictionPlaceholder):
                self._poll_completion(result)
                result = self._output[num_yielded]
                if isinstance(result, BatchPredictionPlaceholder):
                    raise RuntimeError
            assert isinstance(result, LmPrediction)
            num_yielded += 1
            yield result

    def _validate_prompts_input(self, prompts):
        if not prompts:
            raise ValueError("No prompts provided")
        if not isinstance(prompts, list):
            raise ValueError("Prompts must be a list")
        if not all(isinstance(prompt, LmPrompt) for prompt in prompts):
            raise ValueError("All prompts must be LmPrompt instances")
        if not all(prompt.cache for prompt in prompts):
            raise ValueError(
                "All prompts must have caching enabled with `LmPrompt(cache=True)` currently use batching manager"
            )


def _put_batch_in_file(
    jsonl_str: str,
    lm: OpenAIPredictor,
) -> openai.types.FileObject:
    jsonl_bytes = io.BytesIO(jsonl_str.encode('utf-8'))
    batch_input_file = lm._api.files.create(
        file=jsonl_bytes,
        purpose="batch",
    )
    return batch_input_file


def _make_batch(
    batch_input_file: openai.types.FileObject,
    lm: OpenAIPredictor
) -> openai.types.Batch:
    batch_data = lm._api.batches.create(
        input_file_id=batch_input_file.id,
        endpoint=(
            "/v1/chat/completions"
            if lm.is_chat_model
            else "/v1/completions"
        ),
        completion_window="24h",
    )
    return batch_data


def _prompt_to_arg_dict_for_batch(
    prompt: LmPrompt,
    lm: OpenAIPredictor,
    custom_id: str = None,
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
    if custom_id is not None:
        request["custom_id"] = custom_id
    return request




def main():
    lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo)
    prompts = [
        LmPrompt("hello world", cache=True),
        LmPrompt("hello world! I come", cache=True),
    ]
    batch_manager = OpenAiBatchManager(prompts, lm._disk_cache)
    batch_manager.start_batch()
    for result in batch_manager.iter_results():
        print("result", result)
    exit()
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


if __name__ == "__main__":
    main()