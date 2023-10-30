import inspect
import logging
from functools import cached_property
from typing import TYPE_CHECKING

import torch
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerFast
from transformers.utils.generic import TensorType

from lmwrapper._TokenStoppingCriteria import _TokenStoppingCriteria
from lmwrapper.abstract_predictor import LmPredictor
from lmwrapper.HuggingfacePrediction import HuggingfacePrediction
from lmwrapper.prompt_trimming import PromptTrimmer
from lmwrapper.runtime import Runtime
from lmwrapper.structs import LmPrediction, LmPrompt
from lmwrapper.utils import log_cuda_mem

if TYPE_CHECKING:
    from transformers.generation.utils import GenerateOutput


class HuggingfacePredictor(LmPredictor):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        model: PreTrainedModel,
        device: torch.device,
        runtime: Runtime,
        allow_patch_model_forward: bool,
        prompt_trimmer: PromptTrimmer,
    ):
        super().__init__()
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._is_chat_model = False
        self.runtime = runtime
        self.allow_patch_model_forward = allow_patch_model_forward
        self.prompt_trimmer = prompt_trimmer

    def _get_cache_key_metadata(self):
        return {
            "model": "HuggingFacePredictor",
            "name_or_path": self._model.name_or_path,
        }

    def _predict_hf(
        self,
        prompt: LmPrompt,
    ) -> LmPrediction | list[LmPrediction]:
        if not isinstance(prompt.text, str) and len(prompt.text) != 1:
            msg = "Prompt batches other than size 1 are not supported."
            raise NotImplementedError(
                msg,
            )

        if prompt.echo and not self.allow_patch_model_forward:
            msg = (
                "Prompt echo is only supported with `allow_patch_model_forward` = True."
            )
            raise NotImplementedError(
                msg,
            )

        patch_model_forward = False
        if self.allow_patch_model_forward:
            patch_model_forward = prompt.echo

        if prompt.logprobs > 1:
            msg = (
                "Retrieving more than 1 logprob is not yet supported for HuggingFace"
                " models."
            )
            raise NotImplementedError(
                msg,
            )

        if prompt.logprobs and prompt.top_p != 1.0:
            logging.warning("Logprobs may not be correct if top_p != 1.0")

        if prompt.presence_penalty != 0.0:
            msg = "Presence penalty not implemented"
            raise NotImplementedError(msg)

        if prompt.text == "" and not prompt.add_bos_token:
            msg = "Cannot do unconditional generation without `add_bos_token`."
            raise Exception(
                msg,
            )

        is_encoder_decoder = self._model.config.is_encoder_decoder

        if is_encoder_decoder and prompt.add_bos_token:
            msg = "Encoder/decoder models should not have bos tokens added manually."
            raise Exception(
                msg,
            )

        if prompt.add_bos_token:
            assert self._tokenizer.bos_token
            prompt_text = self._tokenizer.bos_token + prompt.text
        else:
            prompt_text = prompt.text

        max_length = self._model.config.max_length
        model_parameters = set(inspect.signature(self._model.forward).parameters.keys())
        model_requires_attention_mask = "attention_mask" in model_parameters

        if self.prompt_trimmer:
            prompt_text = self.prompt_trimmer.trim_text(prompt_text)

        encoded_input = self._tokenizer(
            prompt_text,
            return_tensors="pt",
            return_attention_mask=model_requires_attention_mask,
            add_special_tokens=prompt.add_special_tokens,
        )

        if len(encoded_input.input_ids) > max_length:
            if self.prompt_trimmer:
                msg = (
                    "Prompt is too long for model. Please check that the provided"
                    " trimmer is configured correctly."
                )
                raise ValueError(
                    msg,
                )
            else:
                msg = "Prompt is too long for model. Please pass in a trimmer."
                raise ValueError(
                    msg,
                )

        if is_encoder_decoder:
            encoded_input["decoder_input_ids"] = encoded_input["input_ids"].clone()

        logging.debug("Pre moving encoded tokens")
        log_cuda_mem()
        if self.runtime != Runtime.ACCELERATE:
            encoded_input = encoded_input.to(
                self._device,
            )  # Move to device
        logging.debug("Post moving encoded tokens")
        log_cuda_mem()
        # ONNX models themselves cannot be moved to a device
        # but their input tensors must be moved to GPU
        # Similarly, Accelerate takes care of moving tensors
        logging.debug("Pre model moving")
        log_cuda_mem()

        if self.runtime in {Runtime.PYTORCH, Runtime.BETTER_TRANSFORMER}:
            self._model.to(self._device)  # Ensure model is on device

        logging.debug("Post model moving")
        log_cuda_mem()
        need_log_prob = prompt.logprobs is not None and prompt.logprobs > 0

        do_sample = prompt.temperature > 0
        num_beams = 1
        penalty_alpha = 0.0
        top_k = 50
        num_beam_groups = 1
        optional_generation_kwargs = {
            "temperature": prompt.temperature,
            "do_sample": do_sample,
        }

        if self._tokenizer.pad_token_id is not None:
            optional_generation_kwargs["pad_token_id"] = self._tokenizer.pad_token_id

        if self._tokenizer.eos_token_id is not None:
            optional_generation_kwargs["eos_token_id"] = self._tokenizer.eos_token_id

        if self._tokenizer.bos_token_id is not None:
            optional_generation_kwargs["bos_token_id"] = self._tokenizer.bos_token_id

        # Temperature cannot be set if do_sample is False
        # do_sample is False if prompt.temperature == 0
        # Otherwise you get the following error from HuggingFace:
        # UserWarning: `do_sample` is set to `False`.
        # However, `temperature` is set to `0.0` -- this flag is only used in
        # sample-based generation modes. You should set `do_sample=True` or unset
        # `temperature`. This was detected when initializing the generation config
        # instance, which means the corresponding file may hold incorrect
        # parameterization and should be fixed.
        if not do_sample:  # i.e. prompt.temperature == 0.0:
            optional_generation_kwargs.pop("temperature", None)

        if num_beams == 1 and do_sample is False:
            logging.info("Decoding strategy: greedy decoding")
        elif penalty_alpha > 0.0 and top_k > 1:
            logging.info("Decoding strategy: contrastive search")
        elif num_beams == 1 and do_sample is True:
            logging.info("Decoding strategy: multinomial sampling")
        elif num_beams > 1 and do_sample is False:
            logging.info("Decoding strategy: beam-search decoding")
        elif num_beams > 1 and do_sample is True:
            logging.info("Decoding strategy: beam-search multinomial sampling")
        elif num_beams > 1 and num_beam_groups > 1:
            logging.info("Decoding strategy: diverse beam-search")
        else:
            logging.info("Unable to predict decoding strategy!")

        # Ref https://gist.github.com/kinoc/8a042d8c5683725aa8c372274c02ea2f
        gen_config = GenerationConfig(
            max_new_tokens=(
                self.default_tokens_generated
                if prompt.max_tokens is None
                else prompt.max_tokens
            ),
            return_dict_in_generate=True,
            output_scores=need_log_prob,
            **optional_generation_kwargs,
        )

        if patch_model_forward:
            # We need a way of getting the raw logprobs of the whole sequence.
            #   The scores we get back are possibly already warped by the configuration
            #   https://github.com/huggingface/transformers/issues/17521#issue-1257881647
            #   Also, it does not return the input tokens. Existing hacks
            #   require calling the model again https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17
            # Instead we are going to patch the model forward to log calls
            old_forward = self._model.forward
            cached_logits = torch.zeros(0)

            if model_requires_attention_mask:

                def new_call(attention_mask, *args, **kwargs):
                    nonlocal cached_logits
                    val = old_forward(attention_mask=attention_mask, *args, **kwargs)
                    cached_logits = val.logits
                    return val

            else:

                def new_call(*args, **kwargs):
                    nonlocal cached_logits
                    val = old_forward(*args, **kwargs)
                    cached_logits = val.logits
                    return val

            self._model.forward = new_call

        logging.debug("Pre generate")
        log_cuda_mem()

        stopping_criteria = None
        # input_length is the length of the input prompt for decoder-only models,
        # like the GPT family, and 1 ?? for encoder-decoder models, like BART or T5.
        # we add 2 to consider the </s> at the end of the prompt and the first <s> as input
        input_length = (
            encoded_input.input_ids.shape[1] + 2
            if is_encoder_decoder
            else encoded_input.input_ids.shape[1]
        )
        if prompt.stop:
            stopping_criteria = [
                _TokenStoppingCriteria(
                    prompt.stop,
                    decode=True,
                    tokenizer=self._tokenizer,
                    input_length=input_length,
                ),
            ]

        with torch.no_grad():
            generation_output: GenerateOutput = self._model.generate(
                **encoded_input,
                generation_config=gen_config,
                stopping_criteria=stopping_criteria,
            )
        logging.info("Generation output type:" + str(type(generation_output)))
        logging.debug("Post generate")
        log_cuda_mem()

        if patch_model_forward:
            self._model.forward = old_forward

        model_output_sequence = (
            generation_output.sequences[0].detach().cpu()
        )  # we will not mutate this one

        output_sequence = model_output_sequence.clone()  # we will mutate this one

        generated_sequence = model_output_sequence[input_length:]

        stop_token_idx_output = None
        stop_token_idx_generated = None

        generated_text = self._tokenizer.decode(generated_sequence)
        clean_generated_text = self._tokenizer.decode(
            generated_sequence,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        if prompt.stop:
            token_offsets = _get_token_offsets(self._tokenizer, generated_sequence)
            token_offsets_full = []
            for i in range(1, len(token_offsets)):
                token_offsets_full.extend(
                    [i - 1] * (token_offsets[i] - token_offsets[i - 1]),
                )
            sorted_stop_sequences = sorted(prompt.stop, key=len, reverse=True)

            for stop_sequence in sorted_stop_sequences:
                if stop_sequence in generated_text:
                    stop_idx = generated_text.index(stop_sequence)
                    generated_text = generated_text[:stop_idx]

                    clean_stop_idx = clean_generated_text.index(stop_sequence)
                    clean_generated_text = clean_generated_text[:clean_stop_idx]

                    stop_token_idx_generated = token_offsets_full[stop_idx]
                    if (
                        stop_token_idx_generated > 0  # ensure not first token
                        and token_offsets_full[
                            stop_idx - 1
                        ]  # compare previous token with current
                        == token_offsets_full[stop_idx]
                    ):
                        stop_token_idx_generated += (
                            1  # if they're equal, we include the current token
                        )
                    stop_token_idx_output = input_length + stop_token_idx_generated
                    output_sequence = model_output_sequence[:stop_token_idx_output]
                    generated_sequence = output_sequence[input_length:]
                    break

        # Use .decode as convert_ids_to_tokens leaves artifacts:
        # e.g.: convert: 'Ġprocess'
        # decode: ' process'
        # Original: self._tokenizer.convert_ids_to_tokens(output_sequence)
        output_tokens = [self._tokenizer.decode(t) for t in output_sequence]
        if len(output_tokens) != len(output_sequence):
            msg = "Output token length did not match output sequence length!"
            raise Exception(msg)

        if prompt.add_bos_token:
            output_tokens = output_tokens[1:]
            output_sequence = output_sequence[1:]

        logprobs_dicts = []
        # Calculate the logprobs if needed
        if need_log_prob:
            if patch_model_forward:
                assert prompt.echo
                assert not is_encoder_decoder

                assert cached_logits.shape[0] == 1  # batch
                assert cached_logits.shape[1] == len(model_output_sequence[1:])
                output_logprobs = _gather_logprobs_from_logits(
                    cached_logits[0],
                    model_output_sequence[1:],
                )

                logprobs = output_logprobs.detach().cpu()

                # Free memory
                del output_logprobs
                del cached_logits

                assert len(model_output_sequence[1:]) == len(logprobs)
                if stop_token_idx_output and stop_token_idx_output > 0:
                    logprobs = logprobs[: stop_token_idx_output - 1]

                assert len(output_sequence) == len(logprobs)
            else:
                output_logprobs = self._model.compute_transition_scores(
                    generation_output.sequences,
                    generation_output.scores,
                    normalize_logits=True,
                )[0]

                logprobs = output_logprobs.detach().cpu()
                # Free memory
                del output_logprobs

                if is_encoder_decoder:
                    # we need to chop off the <s> first token
                    # as its probability will throw off uncertainty estimates
                    if stop_token_idx_generated:
                        # if a stop token is defined, we need to step one further due to the <s>
                        # TODO: we can clean this up with better input_length logic
                        logprobs = logprobs[1 : stop_token_idx_generated + 1]
                    else:
                        logprobs = logprobs[1:]
                else:
                    logprobs = logprobs[:stop_token_idx_generated]
                assert len(generated_sequence) == len(logprobs)

            token_sequence = output_sequence if prompt.echo else generated_sequence
            token_sequence = token_sequence.detach().cpu()
            probabilities = logprobs.exp()

            assert len(token_sequence) == len(logprobs)
            assert len(probabilities) == len(token_sequence)

            # Create logprobs dict
            for token, score, probability in zip(
                token_sequence,
                logprobs,
                probabilities,
                strict=True,
            ):
                logprobs_dicts.append(
                    {
                        "token": int(token),
                        "repr": repr(self._tokenizer.decode(token)),
                        "logit": float(score),
                        "probability": float(probability),
                    },
                )
        else:
            logprobs = None

        if prompt.max_tokens == 0:
            # Huggingface seems to default to one token always return an extra token
            output_tokens = output_tokens[:-1]
            logprobs = logprobs[:-1]
            generated_text = ""
            generation_output.sequences = generation_output.sequences[:, :-1]
            clean_generated_text = ""

        logging.debug("Pre del statements")
        log_cuda_mem()

        np_logprobs = logprobs.numpy() if logprobs is not None else None
        np_encoded_input = (
            encoded_input.to("cpu").convert_to_tensors(TensorType.NUMPY).copy()
        )

        # generation_output needs to be mapped to .detach().cpu().numpy() for all tensors
        updated_output = {}

        def numpy_tuple(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            if isinstance(value, tuple):
                return tuple([numpy_tuple(v) for v in value])
            return None

        for key, value in generation_output.items():
            updated_output[key] = numpy_tuple(value)

        del generation_output
        del logprobs
        del encoded_input

        logging.debug("Post del statements")
        log_cuda_mem()

        return HuggingfacePrediction(
            completion_text=clean_generated_text,
            prompt=prompt,
            metad=updated_output,
            _completion_with_special_tok=generated_text,
            _num_prompt_tokens=int(input_length),
            _prompt_encoding=np_encoded_input,
            _tokens=output_tokens,
            _log_probs=np_logprobs,
            _logprobs_dict=logprobs_dicts,
        )

    def _predict_maybe_cached(
        self,
        prompt: LmPrompt,
    ) -> LmPrediction | list[LmPrediction]:
        if torch.cuda.is_available():
            pre_memory = torch.cuda.memory_allocated()

        prediction = self._predict_hf(prompt)

        if torch.cuda.is_available():
            post_memory = torch.cuda.memory_allocated()
            if (post_memory - pre_memory) > 31_457_280:  # 30mb delta
                logging.warning("Possible memory leak detected in model prediction.")

        return prediction

    @cached_property
    def space_char(self) -> str:
        # Try to discover the space char in the tokens
        tokens = self._tokenizer.tokenize("I went to")
        for tok in tokens:
            if "went" in tok:
                return tok.replace("went", "")
        return None

    def remove_special_chars_from_tokens(self, tokens: list[str]) -> list[str]:
        if self.space_char is None:
            return tokens
        return [tok.replace(self.space_char, " ") for tok in tokens]

    def tokenize(self, text: str) -> list[str]:
        return self._tokenizer.tokenize(text)

    @property
    def token_limit(self):
        return self._model.config.max_length

    def estimate_tokens_in_prompt(self, prompt: LmPrompt) -> int:
        if prompt.is_text_a_chat():
            raise NotImplementedError
        return len(self.tokenize(prompt.text))

    @property
    def is_chat_model(self):
        return self._is_chat_model


def _gather_logprobs_from_logits(
    logits: torch.Tensor,
    selected_toks: torch.LongTensor,
):
    logprobs = torch.log_softmax(logits, dim=-1).detach()
    return torch.gather(logprobs, -1, selected_toks.unsqueeze(-1)).squeeze(-1)


def _verify_concatenable(generated_tokens: list[str], generated_text: str):
    raise_exception = False
    if "".join(generated_tokens) == generated_text:
        msg = (
            "Tokens do not appear to be concatenatable. This likely means the tokenizer"
            " is doing some extra processing or we need to better handle special"
            " tokens."
        )
        if raise_exception:
            raise NotImplementedError(msg)
        else:
            logging.warning(msg)


def _get_token_offsets(
    tokenizer: PreTrainedTokenizerFast,
    token_ids: torch.Tensor | list[int],
) -> list[int]:
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    if not isinstance(token_ids, list):
        raise TypeError("token_ids must be a list or tensor")

    if not tokenizer.is_fast:
        raise ValueError(
            "This requires a fast tokenizer so that way we can "
            "use the return_offsets_mapping option",
        )

    new_tokenize = tokenizer(
        tokenizer.decode(token_ids),
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    new_token_ids = new_tokenize["input_ids"]
    offset_mapping = new_tokenize["offset_mapping"]

    if new_token_ids != token_ids:
        # Do a bit hacky things to at least try to align them
        if len(new_token_ids) - 1 == len(token_ids) and new_token_ids[1:] == token_ids:
            # for some reason seems to be an extra start of token or something
            new_token_ids = new_token_ids[1:]
            offset_mapping = offset_mapping[1:]
        else:
            msg = (
                "Token IDs do not match without an easy fix\n"
                f"Original: {token_ids}\n"
                f"New: {new_token_ids}"
            )
            raise AssertionError(msg)

    starts, _ = list(zip(*offset_mapping))

    if new_token_ids != token_ids:
        msg = f"Token IDs do not match\nOriginal: {token_ids}\nNew: {new_token_ids}"
        raise AssertionError(msg)

    return list(starts)
