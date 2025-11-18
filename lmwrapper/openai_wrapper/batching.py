"""
An interface for using the openai batch API.
This offers cost saving by accepting larger latencies.

It is designed closely around the openai API, but might be
abstracted later.
"""

import io
import json
import re
import sys
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

import openai.types
import openai.types.chat
import tqdm
from openai import OpenAI

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

_retry_func_on_connect_error = retry_func_on_exception(
    exception=openai.APIConnectionError,
    max_retries=8,
    linear_backoff_factor=3,
    exponential_backoff_factor=2,
    extra_message=(
        "(If you get this repeated, you might want to check your network connection)"
    ),
)


class OpenAiBatchManager:
    def __init__(
        self,
        prompts: list[LmPrompt],
        cache: SqlBackedCache,
        maintain_order: bool = True,
        max_prompts_per_batch: int = 20_000,
    ):
        self._validate_prompts_input(prompts)
        self._awaiting_marker = object()
        self._output = [self._awaiting_marker] * len(prompts)
        self._num_yielded = 0
        self._cache = cache
        self._maintain_order = maintain_order
        self._prompts = prompts
        _cache_key = cache.lm.get_model_cache_key()
        self._index_to_hash = [
            prompt_to_text_and_sample_hash(p, _cache_key) for i, p in enumerate(prompts)
        ]
        self._prompt_hashes_to_index = {}
        for i, h in enumerate(self._index_to_hash):
            if h not in self._prompt_hashes_to_index:
                self._prompt_hashes_to_index[h] = []
            self._prompt_hashes_to_index[h].append(i)
        self._started = False
        self._lm: OpenAIPredictor = cache.lm
        self._batch_id_to_pbar = {}
        self._max_prompts_batch_size = max_prompts_per_batch
        if max_prompts_per_batch > 50_000:
            raise ValueError(
                "Due to API limits max prompts per batch cannot be more than 50,000",
            )
        self._max_input_file_size = 80e6  # Actual 100MB

        self._batches_to_submit: list[_BatchToMonitor] = []
        self._in_progress_api_id_to_monitor: dict[str, _BatchToMonitor] = {}

        self._last_checked_in_batch_set = None

    def start_batch(self):
        self._batches_to_submit.extend(self._organize_prompts(self._prompts))
        self._try_to_empty_needed_submissions()
        self._started = True

    def _check_if_change_in_listed_batches(self) -> bool:
        all_in_progress_batches = list(list_all_in_progress_batches(self._lm._api))
        all_in_progress_ids = {b.id for b in all_in_progress_batches}
        unmanaged_ids = (
            all_in_progress_ids
            - set(self._in_progress_api_id_to_monitor.keys())
            - {b.api_id for b in self._batches_to_submit}
        )
        is_value_new = unmanaged_ids != self._last_checked_in_batch_set
        self._last_checked_in_batch_set = unmanaged_ids
        return is_value_new

    def _try_to_empty_needed_submissions(self, max_to_submit: int = None):
        if not self._batches_to_submit:
            return
        num_sent = 0
        while self._batches_to_submit and num_sent < (max_to_submit or 1000):
            batch = self._batches_to_submit.pop()
            self._send_batch(batch)
            num_sent += 1

    def _organize_prompts(self, prompts) -> list["_BatchToMonitor"]:
        # Loop through and figure out the already completed ones
        needed = []
        # api_id_to_batch = {}
        for i, prompt in enumerate(prompts):
            if (prompt.num_completions or 1) != 1:
                raise NotImplementedError
            value = self._cache.get(prompt)
            if isinstance(value, list):
                if len(value) != 1:
                    raise RuntimeError
                value = value[0]

            if value is None:
                needed.append((i, prompt))
            elif isinstance(value, BatchPredictionPlaceholder):
                if value.api_id not in self._in_progress_api_id_to_monitor:
                    self._in_progress_api_id_to_monitor[value.api_id] = _BatchToMonitor(
                        api_id=value.api_id,
                        submitted=True,
                        prompts=[prompt],
                        fresh_in_this_manager=False,
                    )
                elif value.api_id in self._in_progress_api_id_to_monitor:
                    existing = self._in_progress_api_id_to_monitor[value.api_id]
                    existing.prompts.append(prompt)
                self._output[i] = value
            elif isinstance(value, LmPrediction):
                value.mark_as_cached()
                self._output[i] = value
            else:
                raise ValueError("Unexpected value", value)
        if not needed:
            return []
        batch = _BatchToMonitor(
            api_id=None,
            submitted=False,
            prompts=[p for i, p in needed],
            fresh_in_this_manager=True,
        )
        return self._split_batch_to_known_requirements(batch)

    def _split_batch_to_known_requirements(
        self,
        batch: "_BatchToMonitor",
        token_limit: int = None,
    ) -> list["_BatchToMonitor"]:
        if len(batch.prompts) > self._max_prompts_batch_size:
            a, b = batch.split()
            return self._split_batch_to_known_requirements(
                a,
            ) + self._split_batch_to_known_requirements(b)
        if token_limit is not None:
            tokens = batch.calculate_prompt_tokens(self._lm)
            if tokens > token_limit:
                if len(batch.prompts) == 1:
                    raise RuntimeError(
                        "Single prompt in batch is larger than available token limit. "
                        "(Ideally, this should just fail this prompt, but for now "
                        "it is a error for the whole batch).",
                    )
                a, b = batch.split()
                return self._split_batch_to_known_requirements(
                    a,
                    token_limit,
                ) + self._split_batch_to_known_requirements(b, token_limit)
        return [batch]

    def _send_batch(self, batch: "_BatchToMonitor"):
        if not batch or not batch.prompts:
            return
        lines = []
        custom_ids = set()
        # Deduplicate prompts at batch level to avoid sending identical requests
        for prompt in batch.prompts:
            custom_id = prompt_to_text_and_sample_hash(
                prompt,
                self._lm.get_model_cache_key(),
            )
            if custom_id in custom_ids:
                # Skip duplicate prompts - result will be distributed to all positions
                continue
            l = json.dumps(_prompt_to_arg_dict_for_batch(prompt, self._lm, custom_id))
            custom_ids.add(custom_id)
            lines.append(l)
        jsonl = "\n".join(lines)
        try:
            batch_input_file = _put_batch_in_file(jsonl, self._lm)
        except InvalidBatchFile as e:
            if len(batch.prompts) == 1:
                raise e
            self._batches_to_submit.extend(batch.split())
            return
        batch_data = _retry_func_on_connect_error(_make_batch)(
            batch_input_file,
            self._lm,
        )
        self._log(
            "Started OpenAI batch"
            f" (https://platform.openai.com/batches/{batch_data.id})",
        )
        batch.submitted = True
        batch.api_id = batch_data.id
        self._in_progress_api_id_to_monitor[batch.api_id] = batch
        batch_row = BatchRow(
            batch_id=uuid.uuid4().hex,
            user_batch_name="",
            api_id=batch_data.id,
            api_category="openai",
            status=batch_data.status,
            waiting_for_a_result=True,
            created_at=batch_data.created_at,
            total_inputs=len(batch.prompts),
            api_json_data=json.dumps(batch_data.model_dump()),
        )
        place_holders = self._cache.put_batch_placeholders(batch_row, batch.prompts)
        for place_holder in place_holders:
            indices = self._prompt_hashes_to_index[place_holder.text_and_sample_hash]
            for index in indices:
                self._output[index] = place_holder

    def _poll_completion(self, target: BatchPredictionPlaceholder | object):
        if not isinstance(target, BatchPredictionPlaceholder):
            assert target is self._awaiting_marker, target
            target_api_id = None
        else:
            if target.api_id not in self._in_progress_api_id_to_monitor:
                raise ValueError("Batch not in progress")
            target_api_id = target.api_id
        start_time = time.time()
        while (time_waited := time.time() - start_time) < 60 * 60 * 24:
            # We might end up in a situation where there are other batches
            #   that we are not manageing that we need to wait on. Check for those.
            if len(self._in_progress_api_id_to_monitor) == 0:
                if len(self._batches_to_submit) == 0:
                    return
                was_none = self._last_checked_in_batch_set is None
                change_in_num_in_progress = self._check_if_change_in_listed_batches()
                if change_in_num_in_progress and was_none:
                    assert self._last_checked_in_batch_set is not None  # for mypy
                    msg = (
                        "Waiting for one of theses unmanaged batches to finish (not"
                        " submitted by this BatchManager)"
                    )
                    for bid in self._last_checked_in_batch_set:
                        msg += f"\n-- https://platform.openai.com/batches/{bid}"
                    self._log(msg)
                if change_in_num_in_progress and not (  # See if we can clear
                    # Only count a change from None if a new change to empty
                    was_none
                    and len(self._last_checked_in_batch_set) > 0
                ):
                    self._try_to_empty_needed_submissions()
                else:
                    if len(self._last_checked_in_batch_set) == 0:
                        raise RuntimeError("No running batches")
                    self._poll_sleep(time_waited)
            else:
                self._last_checked_in_batch_set = None
            # Normal listening of batches in progress
            for batch in list(self._in_progress_api_id_to_monitor.values()):
                pbar = self._pbar_for_targer(batch.api_id, batch.total)
                retrieve_data: openai.types.Batch = _retry_func_on_connect_error(
                    self._cache.lm._api.batches.retrieve,
                )(
                    batch.api_id,
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
                desc = f"Batch `{batch.api_id}` status: {retrieve_data.status}"
                if retrieve_data.request_counts.failed:
                    desc += f" ({retrieve_data.request_counts.failed} failed)"
                    # self._cancel_all_batches()
                    # raise RuntimeError(
                    #    "Batch has failed prompts. This needs to be handled",
                    # )
                pbar.set_description(desc)
                if self._handle_batch_if_failed(
                    batch,
                    retrieve_data,
                ) or self._handle_if_batch_canceled(batch, retrieve_data):
                    pbar.close()
                    break
                if not waiting_for_results:
                    self._update_cache_rows_from_output(batch, retrieve_data)
                    self._update_cache_rows_from_errors(batch, retrieve_data)
                    del self._in_progress_api_id_to_monitor[batch.api_id]
                    self._handle_if_batch_expired(batch, retrieve_data)
                    pbar.close()
                    self._try_to_empty_needed_submissions()
                    if target_api_id == batch.api_id:
                        return
                    break
                self._poll_sleep(time_waited)

    def _handle_if_batch_expired(
        self,
        batch_monitor: "_BatchToMonitor",
        batch_data: openai.types.Batch,
    ) -> bool:
        if batch_data.status != "expired":
            return False
        self._log(f"{batch_monitor.api_id} expired. We will resubmit other prompts")
        needed_prompts = []
        for prompt in batch_monitor.prompts:
            phash = prompt_to_text_and_sample_hash(
                prompt,
                self._lm.get_model_cache_key(),
            )
            if phash not in self._prompt_hashes_to_index:
                continue
            indices = self._prompt_hashes_to_index[phash]
            # Check if any of the duplicate positions still need results
            if any(
                self._output[index] is self._awaiting_marker or isinstance(
                    self._output[index],
                    BatchPredictionPlaceholder,
                )
                for index in indices
            ):
                needed_prompts.append(prompt)
        new_monitor = _BatchToMonitor(
            api_id=None,
            submitted=False,
            prompts=needed_prompts,
            fresh_in_this_manager=True,
        )
        self._batches_to_submit.extend(new_monitor.split())
        return True

    def _log(self, message: str):
        print("lmwrapper:: " + message, file=sys.stderr)

    def _resubmit_batch(self, batch_monitor):
        # Filter the prompts to ones we might need
        old_prompt_hashes = [
            prompt_to_text_and_sample_hash(p, self._lm.get_model_cache_key())
            for p in batch_monitor.prompts
        ]
        new_prompts = [
            p
            for p, h in zip(batch_monitor.prompts, old_prompt_hashes, strict=False)
            if h in self._prompt_hashes_to_index
        ]
        self._batches_to_submit.append(
            _BatchToMonitor(
                api_id=None,
                submitted=False,
                prompts=new_prompts,
                fresh_in_this_manager=True,
            ),
        )
        self._try_to_empty_needed_submissions()

    def _handle_if_batch_canceled(
        self,
        batch_monitor: "_BatchToMonitor",
        batch_data: openai.types.Batch,
    ) -> bool:
        if batch_data.status not in (
            "cancelling",
            "canceled",
        ):
            return False
        self._log(f"Batch {batch_monitor.api_id} was cancelled")
        self._remove_in_progress_batch(batch_monitor)
        if not batch_monitor.fresh_in_this_manager:
            self._log(
                "Found canceled batch not submitted by this BatchManager. "
                "We will try to resubmit the prompts.",
            )
            self._resubmit_batch(batch_monitor)
            return True
        if len(self._in_progress_api_id_to_monitor) > 1:
            self._log("Attempting to cancel other batches")
            self._cancel_all_batches()
        raise RuntimeError("Batch was cancelled.")

    def _poll_sleep(self, time_already_waited: float):
        if time_already_waited < 30:
            sleep_time = 1
        elif time_already_waited < 60 * 2:
            sleep_time = 5
        elif time_already_waited < 60 * 60:
            sleep_time = 10
        else:
            sleep_time = 30
        if len(self._in_progress_api_id_to_monitor) == 0:
            sleep_time = 0.5
        else:
            sleep_time /= min(len(self._in_progress_api_id_to_monitor), 4)
        time.sleep(sleep_time)

    def _cancel_batch(self, batch: "_BatchToMonitor"):
        if batch.manager_canceled:
            return
        try:
            self._lm._api.batches.cancel(batch.api_id)
        except openai.ConflictError:
            # Might already be done
            pass
        self._remove_in_progress_batch(batch)
        batch.manager_canceled = True

    def _cancel_all_batches(self):
        for batch in list(self._in_progress_api_id_to_monitor.values()):
            self._cancel_batch(batch)

    def _remove_in_progress_batch(self, batch: "_BatchToMonitor"):
        if batch.api_id in list(self._in_progress_api_id_to_monitor):
            del self._in_progress_api_id_to_monitor[batch.api_id]
        for prompt in batch.prompts:
            phash = prompt_to_text_and_sample_hash(
                prompt,
                self._lm.get_model_cache_key(),
            )
            indices = self._prompt_hashes_to_index[phash]
            for index in indices:
                self._output[index] = self._awaiting_marker
        for prompt in batch.prompts:
            self._cache.delete(prompt)

    def _handle_batch_if_failed(
        self,
        batch_monitor: "_BatchToMonitor",
        batch_data: openai.types.Batch,
    ) -> bool:
        if batch_data.status != "failed":
            return False
        self._remove_in_progress_batch(batch_monitor)
        for error in batch_data.errors.data:
            if error.code == "token_limit_exceeded":
                self._log("Token queue limit exceeded.")
                limit = _extract_limit_from_error_message(error.message)
                if limit is not None:
                    self._log(f"Organization limit: {limit}")
                split_batches = self._split_batch_to_known_requirements(
                    batch_monitor,
                    token_limit=int(limit * 0.95),  # give a bit of buffer
                )
                if len(split_batches) == 1:
                    self._log("Retrying later.")
                    self._batches_to_submit.extend(split_batches)
                else:
                    self._log(
                        f"Splitting into {len(split_batches)} new smaller batches.",
                    )
                    self._batches_to_submit.extend(split_batches)
                    self._try_to_empty_needed_submissions(1)
                return True
            else:
                self._log(f"Error in batch {batch_monitor.api_id}")
                self._log(error)
        raise RuntimeError("Batch failed. This needs to be handled.")

    def _update_cache_rows_from_output(
        self,
        batch_monitor: "_BatchToMonitor",
        batch_data: openai.types.Batch,
    ):
        if batch_data.output_file_id is None:
            if batch_data.request_counts.completed > 0:
                raise RuntimeError(
                    "Batch has completed some prompts but has no output file. "
                    "This is unexpected.",
                )
            if batch_data.request_counts.failed > 0:
                self._cancel_all_batches()
                raise NotImplementedError(
                    "All prompts in batch failed. This is not currently handled",
                )

        content = _retry_func_on_connect_error(self._cache.lm._api.files.content)(
            batch_data.output_file_id,
        )
        content_str = content.content.decode("utf-8")
        for line in content_str.split("\n"):
            if not line:
                continue
            data = json.loads(line)
            custom_id = data["custom_id"]
            response = data["response"]
            if custom_id not in self._prompt_hashes_to_index:
                if not batch_monitor.fresh_in_this_manager:
                    # This might happen if we are reading from an old batch
                    #  that we didn't start that contains some other prompts.
                    continue
                raise RuntimeError(
                    "Custom id not found in a batch we started",
                    custom_id,
                )
            out_indices = self._prompt_hashes_to_index[custom_id]
            # Use first index to get the prompt (all duplicates have same prompt)
            prompt = self._prompts[out_indices[0]]
            body = response["body"]
            if self._lm.is_chat_model:
                body = openai.types.chat.ChatCompletion.parse_obj(body)
            else:
                body = openai.types.Completion.parse_obj(body)
            pred = self._lm.prediction_from_api_response(body, prompt)
            assert len(pred) == 1
            pred = pred[0]
            # Set the same prediction for all duplicate prompt positions
            for out_index in out_indices:
                self._output[out_index] = pred
            self._cache.add_or_set(pred)

    def _update_cache_rows_from_errors(
        self,
        batch_monitor: "_BatchToMonitor",
        batch_data: openai.types.Batch,
    ):
        if batch_data.error_file_id is None:
            if batch_data.request_counts.failed > 0:
                raise RuntimeError(
                    "Batch has failed prompts but has no error file. "
                    "This is unexpected.",
                )
            return
        content = _retry_func_on_connect_error(self._cache.lm._api.files.content)(
            batch_data.error_file_id,
        )
        content_str = content.content.decode("utf-8")
        for line in content_str.split("\n"):
            if not line:
                continue
            data = json.loads(line)
            body = data["response"]["body"]
            error = openai.types.ErrorObject.parse_obj(body["error"])
            custom_id = data["custom_id"]
            if custom_id not in self._prompt_hashes_to_index:
                if not batch_monitor.fresh_in_this_manager:
                    continue
                raise RuntimeError(
                    "Custom id not found in a batch we started",
                    custom_id,
                )
            out_indices = self._prompt_hashes_to_index[custom_id]
            # Use first index to get the prompt (all duplicates have same prompt)
            prompt = self._prompts[out_indices[0]]
            pred = LmPrediction(
                completion_text=None,
                prompt=prompt,
                metad=None,
                error_message=json.dumps(body["error"]),
            )
            # Set the same error prediction for all duplicate prompt positions
            for out_index in out_indices:
                self._output[out_index] = pred
            self._cache.add_or_set(pred)

    def _pbar_for_targer(self, api_id: str, total: int):
        if api_id not in self._batch_id_to_pbar:
            pbar = tqdm.tqdm(
                total=total,
                leave=True,
                position=len(self._batch_id_to_pbar),
            )
            self._batch_id_to_pbar[api_id] = pbar
        return self._batch_id_to_pbar[api_id]

    def iter_results(self) -> Iterable[LmPrediction]:
        num_yielded = 0
        while num_yielded < len(self._output):
            result = self._output[num_yielded]
            if (
                isinstance(result, BatchPredictionPlaceholder)
                or result is self._awaiting_marker
            ):
                try:
                    self._poll_completion(result)
                except RuntimeError as e:
                    self._cancel_all_batches()
                    raise e
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
                " currently to use batching manager",
            )
        if not all((prompt.num_completions or 1) == 1 for prompt in prompts):
            raise NotImplementedError(
                "Only num_completions of 1 (or None) is currently supported in batches"
                " (need to be fixed)",
            )


class InvalidBatchFile(ValueError):
    pass


def _extract_limit_from_error_message(message):
    pattern = r"Limit: ([\d,]+)"
    match = re.search(pattern, message)

    if match:
        # Extract the matched number and remove commas
        number_str = match.group(1).replace(",", "")
        # Convert to integer
        return int(number_str)
    else:
        return None


def _put_batch_in_file(
    jsonl_str: str,
    lm: OpenAIPredictor,
) -> openai.types.FileObject:
    jsonl_bytes = io.BytesIO(jsonl_str.encode("utf-8"))
    if len(jsonl_bytes.getvalue()) > 80e6:
        raise InvalidBatchFile(
            "Batch file is too large. Max is 100MB."
            " We still need to implement automatic splitting for this",
        )
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
        reasoning_style=lm.reasoning_style,
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


def get_all_batch_data(api: OpenAI) -> Iterable[openai.types.Batch]:
    after = None
    has_more = True
    while has_more:
        args = {
            "limit": 100,
        }
        if after:
            args["after"] = after
        val = _retry_func_on_connect_error(api.batches.list)(**args)
        yield from val.data
        has_more = val.has_more and val.data
        after = val.last_id


def list_all_in_progress_batches(api: OpenAI) -> Iterable[openai.types.Batch]:
    for batch in get_all_batch_data(api):
        if batch.status in ("validating", "in_progress", "finalizing", "cancelling"):
            yield batch


@dataclass
class _BatchToMonitor:
    api_id: str | None
    submitted: bool
    prompts: list[LmPrompt]
    """Mapping the prompt values to where the output is written.
    We return a list in case multiple identical prompts are combined
    together. The first index is the output index. The second is the
    starting slice index in the returned choices. We can calculate
    the slice by looking back at the original prompts and seeing
    how many completions were requested.
    """
    fresh_in_this_manager: bool
    """If this batch was new for this manager vs loaded from some prior
    run."""
    manager_canceled: bool = False

    def calculate_prompt_tokens(self, lm: OpenAIPredictor):
        return sum(
            len(lm.tokenize(p.get_text_as_string_default_form())) for p in self.prompts
        )

    @property
    def total(self):
        return len(self.prompts)

    def split(self) -> tuple["_BatchToMonitor", "_BatchToMonitor"]:
        """Splits the batch in half"""
        if len(self.prompts) == 1:
            raise RuntimeError("Cannot split a batch with only one prompt")
        mid = len(self.prompts) // 2
        return (
            _BatchToMonitor(
                api_id=self.api_id,
                submitted=self.submitted,
                prompts=self.prompts[:mid],
                fresh_in_this_manager=self.fresh_in_this_manager,
            ),
            _BatchToMonitor(
                api_id=self.api_id,
                submitted=self.submitted,
                prompts=self.prompts[mid:],
                fresh_in_this_manager=self.fresh_in_this_manager,
            ),
        )


def main():
    lm = get_open_ai_lm(OpenAiModelNames.gpt_4o_mini)
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
