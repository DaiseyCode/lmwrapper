"""
An interface for using the openai batch API.
This offers cost saving by accepting larger latencies.

It is designed closely around the openai API, but might be
abstracted later.
"""

import io
import json
import time
import uuid
from typing import TypeVar, Callable

import openai.types
import openai.types.chat
import tqdm

from lmwrapper.caching import clear_cache_dir
from lmwrapper.openai_wrapper import (
    OpenAiModelNames,
    OpenAIPredictor,
    get_open_ai_lm,
    prompt_to_openai_args_dict,
)
from lmwrapper.sqlcache import BatchRow, SqlBackedCache, prompt_to_text_and_sample_hash
from lmwrapper.sqlcache_struct import BatchPredictionPlaceholder
from lmwrapper.structs import LmPrediction, LmPrompt
from lmwrapper.utils import retry_func_on_exception

T_C = TypeVar("T_C")
T_R = TypeVar("T_R")


_retry_func_on_connect_error = retry_func_on_exception(
    exception=openai.APIConnectionError,
    max_retries=8,
    linear_backoff_factor=3,
    exponential_backoff_factor=2,
    extra_message="(If you get this repeated, you might want to check your network connection)"
)


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
        self._max_batch_size = 50_000  # Actual limit is 50,000
        self._max_input_file_size = 100e6  # 100MB
        #if len(prompts) > self._max_batch_size:
        #    raise ValueError(
        #        f"Batch size of {len(prompts)} is too large. Max is"
        #        f" {self._max_batch_size}. This is a temporary restriction. The goal of"
        #        " this API is to automatically create sub-batches for you to handle"
        #        " this.",
        #    )

    def start_batch(self):
        need_prompts = self._organize_prompts(self._prompts)
        if need_prompts:
            self._send_batch(need_prompts)
        self._started = True

    def _organize_prompts(self, prompts) -> list[tuple[int, LmPrompt]]:
        # Loop through and figure out the already completed ones
        needed = []
        already_added_hash_to_batch_pred = {}
        for i, prompt in enumerate(prompts):
            prompt_hash = prompt_to_text_and_sample_hash(
                prompt, self._lm.get_model_cache_key())
            # We might have already added this prompt to needed
            if prompt_hash in already_added_hash_to_batch_pred:
                self._output[i] = already_added_hash_to_batch_pred[prompt_hash]
            value = self._cache.get(prompt)
            if value is None:
                needed.append((i, prompt))
            elif isinstance(value, BatchPredictionPlaceholder):
                self._output[i] = value
            else:
                value.mark_as_cached()
                self._output[i] = value
        return needed

    def _send_batch(self, prompts: list[tuple[int, LmPrompt]]):
        if not prompts:
            return
        jsonl = "\n".join(
            json.dumps(_prompt_to_arg_dict_for_batch(prompt, self._lm, str(index)))
            for index, prompt in prompts
        )
        ids, just_prompts = zip(*prompts, strict=False)
        batch_input_file = _put_batch_in_file(jsonl, self._lm)
        batch_data = _retry_func_on_connect_error(_make_batch)(batch_input_file, self._lm)
        batch_row = BatchRow(
            batch_id=uuid.uuid4().hex,
            user_batch_name="",
            api_id=batch_data.id,
            api_category="openai",
            status=batch_data.status,
            waiting_for_a_result=True,
            created_at=batch_data.created_at,
            total_inputs=len(prompts),
            api_json_data=json.dumps(batch_data.dict()),
        )
        place_holders = self._cache.put_batch_placeholders(batch_row, just_prompts)
        for (index, prompt), place_holder in zip(prompts, place_holders, strict=False):
            self._output[index] = place_holder

    def _poll_completion(self, target: BatchPredictionPlaceholder):
        start_time = time.time()
        pbar = self._pbar_for_targer(target)
        while (time_waited := time.time() - start_time) < 60 * 60 * 24:
            retrieve_data: openai.types.Batch = _retry_func_on_connect_error(
                self._cache.lm._api.batches.retrieve
            )(
                target.api_id,
            )
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
                (
                    retrieve_data.request_counts.completed
                    + retrieve_data.request_counts.failed
                )
                - pbar.n,
            )
            # Update description with failure count
            desc = f"Waiting for batch `{target.api_id}` completion"
            if retrieve_data.request_counts.failed:
                raise RuntimeError("Batch failed. This needs to be handled")
                desc += f" ({retrieve_data.request_counts.failed} failed)"
            pbar.set_description(desc)
            if not waiting_for_results:
                self._update_cache_rows(retrieve_data)
                break
            if time_waited < 60:
                sleep_time = 1
            elif time_waited < 60 * 60:
                sleep_time = 5
            else:
                sleep_time = 20
            time.sleep(sleep_time)

    def _update_cache_rows(
        self,
        batch_data: openai.types.Batch,
    ):
        content = _retry_func_on_connect_error(
            self._cache.lm._api.files.content)(batch_data.output_file_id)
        content_str = content.content.decode("utf-8")
        for line in content_str.split("\n"):
            if not line:
                continue
            data = json.loads(line)
            custom_id = data["custom_id"]
            response = data["response"]
            prompt = self._prompts[int(custom_id)]
            body = response["body"]
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
            pbar = tqdm.tqdm(
                total=target.batch_total_inputs,
                desc=f"Waiting for batch {batch_id} completion",
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
                "All prompts must have caching enabled with `LmPrompt(cache=True)`"
                " currently use batching manager",
            )


def _put_batch_in_file(
    jsonl_str: str,
    lm: OpenAIPredictor,
) -> openai.types.FileObject:
    jsonl_bytes = io.BytesIO(jsonl_str.encode("utf-8"))
    batch_input_file = _retry_func_on_connect_error(lm._api.files.create)(
        file=jsonl_bytes,
        purpose="batch",
    )
    return batch_input_file


def _make_batch(
    batch_input_file: openai.types.FileObject,
    lm: OpenAIPredictor,
) -> openai.types.Batch:
    batch_data = _retry_func_on_connect_error(lm._api.batches.create)(
        input_file_id=batch_input_file.id,
        endpoint=("/v1/chat/completions" if lm.is_chat_model else "/v1/completions"),
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
        "url": "/v1/chat/completions" if lm.is_chat_model else "/v1/completions",
        # TODO multiple responses
        "custom_id": prompt_to_text_and_sample_hash(prompt, lm.get_model_cache_key()),
    }
    if custom_id is not None:
        request["custom_id"] = custom_id
    return request


def main():
    lm = get_open_ai_lm(OpenAiModelNames.gpt_3_5_turbo)
    clear_cache_dir()
    prompts = [
        LmPrompt("hello world", cache=True),
        LmPrompt("hello world! I come", cache=True),
        LmPrompt("goodbye", cache=True),
        LmPrompt("Woah", cache=True),
    ]
    batch_manager = OpenAiBatchManager(prompts, lm._disk_cache)
    batch_manager.start_batch()
    print("Sleep")
    time.sleep(2)
    for result in batch_manager.iter_results():
        print("result", result)


if __name__ == "__main__":
    main()
