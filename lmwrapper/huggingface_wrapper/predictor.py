import inspect
import logging
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerFast
from transformers.utils.generic import TensorType

from lmwrapper._TokenStoppingCriteria import _TokenStoppingCriteria
from lmwrapper.abstract_predictor import LmPredictor, LmReasoningStyle
from lmwrapper.compatibility import check_transformers_compatibility
from lmwrapper.huggingface_wrapper.prediction import HuggingFacePrediction
from lmwrapper.huggingface_wrapper.utilstorch import log_cuda_mem
from lmwrapper.internals import ModelInternalsRequest, ModelInternalsResults
from lmwrapper.prompt_trimming import PromptTrimmer
from lmwrapper.runtime import Runtime
from lmwrapper.structs import LmPrediction, LmPrompt, ChatGptRoles

if TYPE_CHECKING:
    from transformers.generation.utils import GenerateOutput


class HuggingFacePredictor(LmPredictor):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        model: PreTrainedModel,
        device: torch.device,
        runtime: Runtime,
        allow_patch_model_forward: bool,
        prompt_trimmer: PromptTrimmer,
        use_chat_mode: bool | None = None,
    ):
        super().__init__()
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self.runtime = runtime
        self.allow_patch_model_forward = allow_patch_model_forward
        self.prompt_trimmer = prompt_trimmer
        self._tokenizer_already_adds_bos = {}
        if use_chat_mode is None:
            use_chat_mode = self._tokenizer.chat_template is not None
        self._use_chat_mode = use_chat_mode

    def get_model_cache_key(self):
        return self._model.name_or_path

    @property
    def supports_token_operations(self) -> bool:
        return True

    @property
    def reasoning_style(self) -> LmReasoningStyle:
        return LmReasoningStyle.NEVER_THINK

    def _verify_initial_prompt(self, prompt: LmPrompt):
        if prompt.is_text_a_chat() and not self._use_chat_mode:
            msg = (
                "The model is not configured for chat mode, but the prompt is a chat"
                " prompt."
            )
            raise ValueError(msg)

        if prompt.echo and not self.allow_patch_model_forward:
            msg = (
                "Prompt echo is only supported with `allow_patch_model_forward` = True."
            )
            raise NotImplementedError(
                msg,
            )

        if prompt.logprobs is not None and prompt.logprobs > 1:
            msg = (
                "Retrieving more than 1 logprob is not yet supported for HuggingFace"
                " models."
            )
            raise NotImplementedError(
                msg,
            )
        if prompt.logprobs and prompt.top_p is not None and prompt.top_p != 1.0:
            logging.warning("Logprobs may not be correct if top_p != 1.0")

        if prompt.presence_penalty != 0.0:
            msg = "Presence penalty not implemented"
            raise NotImplementedError(msg)

    def _will_add_and_have_bos(self, prompt: LmPrompt):
        will_add_bos = prompt.add_bos_token or (
            prompt.add_bos_token is None
            and self._tokenizer.bos_token is not None
            and not self.is_encoder_decoder
            and not self._does_this_tokenizer_seem_add_a_bos(prompt.add_special_tokens)
        )
        will_have_bos = will_add_bos or self._does_this_tokenizer_seem_add_a_bos(
            prompt.add_special_tokens,
        )
        if prompt.text == "" and not will_have_bos:
            msg = "Cannot do unconditional generation without `add_bos_token`."
            raise Exception(
                msg,
            )

        if self.is_encoder_decoder and will_add_bos:
            msg = "Encoder/decoder models should not have bos tokens added manually."
            raise Exception(
                msg,
            )
        return will_add_bos, will_have_bos

    @property
    def is_encoder_decoder(self) -> bool:
        return self._model.config.is_encoder_decoder

    @property
    def supports_prefilled_chat(self) -> bool:
        return self.is_chat_model

    def _optional_args_for_internals(self, prompt: LmPrompt):
        args = {}
        if prompt.model_internals_request is None:
            return args
        if prompt.model_internals_request.return_hidden_states:
            args["output_hidden_states"] = True
        if prompt.model_internals_request.return_attentions:
            args["output_attentions"] = True
        return args

    def _predict_hf(
        self,
        prompt: LmPrompt,
    ) -> list[LmPrediction]:
        check_transformers_compatibility()
        self._verify_initial_prompt(prompt)

        patch_model_forward = False
        if self.allow_patch_model_forward:
            patch_model_forward = prompt.echo

        will_add_bos, will_have_bos = self._will_add_and_have_bos(prompt)

        if self._use_chat_mode:
            chat = prompt.get_text_as_chat()
            is_prefill = chat[-1].role == ChatGptRoles.assistant
            prompt_text = self._tokenizer.apply_chat_template(
                chat.as_dicts(),
                tokenize=False,
                add_generation_prompt=None if is_prefill else True,
                continue_final_message=is_prefill,
            )
        else:
            prompt_text = prompt.text

        if will_add_bos:
            assert self._tokenizer.bos_token
            prompt_text = self._tokenizer.bos_token + prompt_text

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

        if len(encoded_input.input_ids) > self.token_limit:
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

        if self.is_encoder_decoder:
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
            if self._model.device != self._device:
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

        optional_generation_kwargs |= self._optional_args_for_internals(prompt)

        # Ref https://gist.github.com/kinoc/8a042d8c5683725aa8c372274c02ea2f
        gen_config = GenerationConfig(
            max_new_tokens=max(
                (
                    self.default_tokens_generated
                    if prompt.max_tokens is None
                    else prompt.max_tokens
                ),
                # Transformers v4.38+ throws an exception when
                # max_new_tokens is set to 0 due to a change in #28621.
                # As a hack fix we can set it to 1 and trim later.
                1,
            ),
            return_dict_in_generate=True,
            output_scores=need_log_prob,
            return_legacy_cache=True,
            **optional_generation_kwargs,
        )
        outputs = []
        for gen_num in range(prompt.num_completions or 1):
            outputs.append(
                self._do_actual_generate(
                    prompt,
                    encoded_input,
                    gen_config,
                    patch_model_forward,
                    will_have_bos,
                    model_requires_attention_mask,
                    need_log_prob,
                ),
            )
        del encoded_input

        # if prompt.num_completions is None:
        #    return outputs[0]
        return outputs

    def _do_actual_generate(
        self,
        prompt,
        encoded_input,
        gen_config,
        patch_model_forward,
        will_have_bos,
        model_requires_attention_mask,
        need_log_prob,
    ):
        if patch_model_forward:
            # We need a way of getting the raw logprobs of the whole sequence.
            #   The scores we get back are possibly already warped by the configuration
            #   https://github.com/huggingface/transformers/issues/17521#issue-1257881647
            #   Also, it does not return the input tokens. Existing hacks
            #   require calling the model again https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17
            # Instead we are going to patch the model forward to log calls
            old_forward = self._model.forward
            cached_logits = []

            def new_call(attention_mask, *args, **kwargs):
                nonlocal cached_logits
                if model_requires_attention_mask:
                    kwargs["attention_mask"] = attention_mask
                val = old_forward(*args, **kwargs)
                cached_logits.append(val.logits.detach())
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
            if self.is_encoder_decoder
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
                return_legacy_cache=True,
            )
        #logging.info("Generation output type:" + str(type(generation_output)))
        logging.debug("Post generate")
        log_cuda_mem()

        if patch_model_forward:
            self._model.forward = old_forward

        model_output_sequence = generation_output.sequences[
            0
        ].detach()  # we will not mutate this one

        output_sequence = model_output_sequence.clone()  # we will mutate this one

        generated_sequence = model_output_sequence[input_length:]

        stop_token_idx_output = None
        stop_token_idx_generated = None

        # Get the generated text (which might have special tokens and unclean spaces)
        # and the clean_generated_text (which skips special tokens and cleans up spaces)
        generated_text, clean_generated_text = _figure_out_generated_text(
            self._tokenizer,
            generated_sequence,
        )

        if prompt.stop:
            token_offsets = _get_token_offsets(self._tokenizer, generated_sequence)
            token_offsets_full = _expand_offsets_to_a_token_index_for_every_text_index(
                token_offsets,
            )
            if len(token_offsets_full) != len(generated_text):
                raise RuntimeError(
                    "Unexpected token offsets length\nGenerated text:"
                    f" '{generated_text}'\nToken offsets: {token_offsets}\nTokens:"
                    f" {self._tokenizer.convert_ids_to_tokens(generated_sequence)}\nToken"
                    f" offsets full: {token_offsets_full}\nToken ids:"
                    f" {generated_sequence}\nSpecial ids:"
                    f" {self._tokenizer.all_special_ids}\nlen(token_offsets_full):"
                    f" {len(token_offsets_full)}\nlen(generated_text):"
                    f" {len(generated_text)}\n",
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
        # output_tokens = [self._tokenizer.decode(t) for t in output_sequence]
        output_tokens = self.remove_special_chars_from_tokens(
            self._tokenizer.convert_ids_to_tokens(output_sequence),
        )
        if len(output_tokens) != len(output_sequence):
            msg = "Output token length did not match output sequence length!"
            raise Exception(msg)

        if will_have_bos:
            output_tokens = output_tokens[1:]
            output_sequence = output_sequence[1:]

        logprobs_dicts = []
        # Calculate the logprobs if needed
        logprobs = None
        if need_log_prob:
            if patch_model_forward:
                assert prompt.echo
                assert not self.is_encoder_decoder

                assert cached_logits[0].shape[0] == 1  # batch
                if len(cached_logits) > 1 and cached_logits[-1].shape[1] == 1:
                    combined_logits = torch.cat(cached_logits, dim=1)
                else:
                    combined_logits = cached_logits[-1]
                if combined_logits.shape[1] != len(model_output_sequence[1:]):
                    msg = (
                        "Logits shape does not match the output sequence length\n"
                        f"Logits shape: {cached_logits.shape}\n"
                        f"Output sequence length: {len(model_output_sequence[1:])}"
                    )
                    raise RuntimeError(msg)
                output_logprobs = _gather_logprobs_from_logits(
                    combined_logits[0],
                    model_output_sequence[1:],
                )

                logprobs = output_logprobs.detach().cpu()

                # Free memory
                del output_logprobs
                del cached_logits
                del combined_logits

                assert len(model_output_sequence[1:]) == len(logprobs)
                if stop_token_idx_output and stop_token_idx_output > 0:
                    logprobs = logprobs[: stop_token_idx_output - 1]

                if not will_have_bos:
                    # There's no BOS, so we won't have logprob for the first
                    # token. We will just fill it with NaN
                    nan_tensor = torch.tensor([float("nan")], device=logprobs.device)
                    logprobs = torch.cat([nan_tensor, logprobs], dim=0)

                assert len(output_sequence) == len(logprobs), f"{len(output_sequence)} {len(logprobs)}\n{output_sequence}\n{self._tokenizer.decode(output_sequence)}\n{self._tokenizer.decode(generated_sequence)}"
            else:
                output_logprobs = self._model.compute_transition_scores(
                    generation_output.sequences,
                    generation_output.scores,
                    normalize_logits=True,
                )[0]

                logprobs = output_logprobs.detach().cpu()
                # Free memory
                del output_logprobs

                if self.is_encoder_decoder:
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

        if prompt.max_tokens == 0:
            # We will have set max tokens to 1 to avoid bug. Now trim back to 0
            output_tokens = output_tokens[:-1]
            logprobs = logprobs[:-1]
            generated_text = ""
            generation_output.sequences = generation_output.sequences[:, :-1]
            clean_generated_text = ""

        logging.debug("Pre del statements")
        log_cuda_mem()

        np_logprobs = logprobs.numpy() if logprobs is not None else None
        np_encoded_input = (
            encoded_input.copy().to("cpu").convert_to_tensors(TensorType.NUMPY)
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

        internals = self._parse_model_internals_results(
            prompt,
            generation_output,
            will_have_bos,
            output_tokens,
        )

        del generation_output
        del logprobs

        logging.debug("Post del statements")
        log_cuda_mem()

        return HuggingFacePrediction(
            completion_text=clean_generated_text,
            prompt=prompt,
            # metad=updated_output,
            # Passing the updated_output is huge and has stuff like the whole logit
            # array. I don't think we need that.
            metad={},
            internals=internals,
            _completion_with_special_tok=generated_text,
            _num_prompt_tokens=(int(input_length) - (1 if will_have_bos else 0)),
            _prompt_encoding=np_encoded_input,
            _tokens=output_tokens,
            _log_probs=np_logprobs,
            _logprobs_dict=logprobs_dicts,
        )

    def find_prediction_class(self, prompt):
        return HuggingFacePrediction

    def _parse_model_internals_results(
        self,
        prompt: LmPrompt,
        generation_output,
        will_have_bos,
        output_tokens,
    ):
        if prompt.model_internals_request is None:
            return None
        request: ModelInternalsRequest = prompt.model_internals_request
        internal_args = {}
        if request.return_hidden_states:
            hidden_states = self._get_hidden_states_combined(
                generation_output.hidden_states,
                output_tokens,
            )
            hidden_states = request.select_layer_sequence(hidden_states)
            internal_args["hidden_states"] = hidden_states
        if request.return_attentions:
            attentions = self._get_attentions_combined(
                generation_output.attentions,
                output_tokens,
            )
            attentions = request.select_layer_sequence(attentions)
            internal_args["attentions"] = attentions
        internal_args["has_a_bos"] = will_have_bos

        return ModelInternalsResults(**internal_args)

    def _get_attentions_combined(
        self,
        attentions,
        output_tokens,
    ):
        """
        Attentions is a tuple
        (token_output: tuple, layers: tuple, (batch, head, seq, seq_prompt): tensor).
        We want to transform a single to (layers, head, seq, seq_prompt): ndarray
        with the future tokens appropriately masked out.

        Note: that this precludes some more weird schemes such as a different
        number of heads per layer or some weird sparse attentions where each
        token_output has different dimensions. We will have to potentially
        handle those latter.
        """
        if len(output_tokens) == 0:
            return None
        last_token_attentions = attentions[-1]
        seems_to_be_token_by_token = last_token_attentions[1].shape[2] == 1
        if seems_to_be_token_by_token and len(attentions) > 1:
            # We have something like
            # ( for each token gen
            #   (layers, (batch, head, seq_prompt, seq_prompt)),
            #   (layers, (batch, head, 1, seq_prompt + 1)),
            #   (layers, (batch, head, 1, seq_prompt + 2)),
            #   ...
            # )
            # We need to combine these and fill any of the added tokens with zeros
            num_layers = len(attentions[0])
            num_heads = attentions[0][1].shape[1]
            total_seq_length = sum(
                att[1].shape[2] for att in attentions
            )  # Total sequence length including prompt
            seq_prompt_length = attentions[0][1].shape[
                3
            ]  # Sequence length of the prompt

            # Initialize a tensor to hold the combined attentions for all layers
            combined_attentions = torch.zeros(
                (num_layers, num_heads, total_seq_length, total_seq_length),
            )

            # Fill in the combined attentions tensor
            seq_offset = 0
            for tok_gen_i, tok_gen_atten in enumerate(attentions):
                for layer_idx, att_tensor in enumerate(tok_gen_atten):
                    batch, head, seq_length_in_this_generation, seq_attend = (
                        att_tensor.shape
                    )
                    assert batch == 1
                    assert head == num_heads
                    combined_attentions[
                        layer_idx,
                        :,
                        seq_offset : seq_offset + seq_length_in_this_generation,
                        :seq_attend,
                    ] = att_tensor[0, :, :, :]

                seq_offset += seq_length_in_this_generation

            return combined_attentions.cpu().numpy()
        else:
            # The last token should have everything
            # So last token should be (layers [tuple], (batch, head, seq, seq_prompt) [tensor])
            # We want to be (layers, head, seq, seq_prompt)
            layers = [layer.squeeze(0).cpu().numpy() for layer in last_token_attentions]
            # combine the layers
            return np.stack(layers, axis=0)

    def _get_hidden_states_combined(
        self,
        hidden_states,
        output_tokens,
    ) -> tuple[np.ndarray, ...] | None:
        """
        Different models seem to treat the hidden states differently.
        Try to parse that out and return (layers: tuple, (seq, hidden): ndarray)
        """
        if len(output_tokens) == 0:
            return None
        last_token_states = hidden_states[-1]
        seems_to_be_token_by_token = last_token_states[1].shape[1] == 1
        if seems_to_be_token_by_token and len(hidden_states) > 1:
            #  So right now it is (tokens: tuple, layers: tuple, (batch, seq, hidden): tensor)
            #     where the first seq includes the prompt, and the rest is seq==1 for each token
            #  We want to stack the layers to get (layers: tuple, (seq, hidden): ndarray)
            batch_size = last_token_states[1].shape[0]
            if batch_size != 1:
                raise NotImplementedError("Batch size > 1 not implemented")
            num_layers = len(hidden_states[0])
            layers_tokens_array = [[] for _ in range(num_layers)]
            for token in hidden_states:
                for i, layer in enumerate(range(num_layers)):
                    layers_tokens_array[i].append(token[layer].squeeze(0))
            layers = [
                torch.cat(layer, dim=0).cpu().numpy() for layer in layers_tokens_array
            ]
            return tuple(layers)
        else:
            # The last token should have everything
            # So last token should be (layers [tuple], (batch, seq, hidden) [tensor])
            # We want to be (layers [tuple], (seq, hidden) [ndarray])
            layers = [layer.squeeze(0).cpu().numpy() for layer in last_token_states]
            return tuple(layers)

    def _predict_maybe_cached(
        self,
        prompt: LmPrompt,
    ) -> LmPrediction | list[LmPrediction]:
        if torch.cuda.is_available():
            pre_memory = torch.cuda.memory_allocated()

        with torch.inference_mode():
            prediction = self._predict_hf(prompt)

        if torch.cuda.is_available():
            post_memory = torch.cuda.memory_allocated()
            if (post_memory - pre_memory) > 31_457_280:  # 30mb delta
                logging.warning("Possible memory leak detected in model prediction.")

        return prediction

    def _does_this_tokenizer_seem_add_a_bos(self, add_special_tokens) -> bool:
        if self._tokenizer_already_adds_bos.get(add_special_tokens, None) is not None:
            return self._tokenizer_already_adds_bos[add_special_tokens]
        self._tokenizer_already_adds_bos[add_special_tokens] = (
            _check_tokenizer_to_see_if_adds_bos(
                self._tokenizer,
                add_special_tokens,
            )
        )
        return self._tokenizer_already_adds_bos[add_special_tokens]

    @cached_property
    def space_char(self) -> str | None:
        # Try to discover the space char in the tokens
        tokens = self._tokenizer.tokenize("I went to")
        for tok in tokens:
            if "went" in tok:
                return tok.replace("went", "")
        return None

    @cached_property
    def newline_char(self) -> str | None:
        for attempt in ("I\nI", "a\na"):
            tokens = self._tokenizer.tokenize(attempt)
            if len(tokens) != 3:
                continue
            return tokens[1]
        return None

    def remove_special_chars_from_tokens(self, tokens: list[str]) -> list[str]:
        if self.space_char is not None and self.space_char != " ":
            tokens = [tok.replace(self.space_char, " ") for tok in tokens]
        if self.newline_char is not None and self.newline_char != "\n":
            tokens = [tok.replace(self.newline_char, "\n") for tok in tokens]
        return tokens

    def tokenize(self, text: str) -> list[str]:
        return self._tokenizer.tokenize(text)

    @property
    def token_limit(self):
        """
        Returns the max tokens of this model. For encoder-decoder models
        it returns jus the encoder limit (this behaviour should probably be
        refined)
        """
        possible_attr_names = [
            "max_length",
            "max_position_embeddings",
            "n_positions",
        ]
        vals = [
            getattr(self._model.config, attr_name, None)
            for attr_name in possible_attr_names
        ]
        if hasattr(self._model.config, "encoder"):
            vals.extend(
                [
                    getattr(self._model.config.encoder, attr_name, None)
                    for attr_name in possible_attr_names
                ],
            )
        if all(val is None for val in vals):
            msg = "Unknown max length"
            raise ValueError(msg)
        limit = max(val for val in vals if val is not None)
        if limit < 100:
            msg = "Unexpectedly low token limit"
            raise ValueError(msg)
        return limit

    def estimate_tokens_in_prompt(self, prompt: LmPrompt) -> int:
        if prompt.is_text_a_chat():
            raise NotImplementedError
        return len(self.tokenize(prompt.text))

    @property
    def is_chat_model(self):
        return self._use_chat_mode


def _check_tokenizer_to_see_if_adds_bos(
    tokenizer: PreTrainedTokenizerFast,
    add_special_tokens: bool,
) -> bool:
    # Use a little test tokenization to see if a bos token is added
    tokens = tokenizer.tokenize("Test prompt", add_special_tokens=add_special_tokens)
    return len(tokens) > 1 and tokens[0] == tokenizer.bos_token


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


def _figure_out_generated_text(
    tokenizer: PreTrainedTokenizerFast,
    generated_sequence: list | torch.Tensor,
):
    # Some tokenizers (notably mistral) has buggy behaviour when
    # not having a first token. We try to work around this by adding
    # a bos token
    if isinstance(generated_sequence, torch.Tensor):
        assert len(generated_sequence.shape) == 1
    else:
        assert isinstance(generated_sequence, list)
    have_mod = False
    if tokenizer.bos_token and generated_sequence[0] != tokenizer.bos_token_id:
        if isinstance(generated_sequence, list):
            mod_gen_seq = [tokenizer.bos_token_id, *generated_sequence]
        else:
            mod_gen_seq = torch.cat(
                [
                    torch.tensor(
                        [tokenizer.bos_token_id],
                        device=generated_sequence.device,
                    ),
                    generated_sequence,
                ],
                dim=0,
            )
        have_mod = True
    else:
        mod_gen_seq = generated_sequence
    generated_text = tokenizer.decode(
        mod_gen_seq,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    clean_generated_text = tokenizer.decode(
        mod_gen_seq,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    if have_mod:
        if generated_text.startswith(tokenizer.bos_token):
            generated_text = generated_text[len(tokenizer.bos_token) :]
        if clean_generated_text.startswith(tokenizer.bos_token):
            clean_generated_text = clean_generated_text[len(tokenizer.bos_token) :]
    generated_text_has_one_more_leading_space = (
        len(generated_text) - len(generated_text.lstrip()) - 1
    ) == (len(clean_generated_text) - len(clean_generated_text.lstrip()))
    if (
        _tokenizer_removes_prefix_space_on_detok(tokenizer)
        and generated_text_has_one_more_leading_space
    ):
        clean_generated_text = " " + clean_generated_text
    return generated_text, clean_generated_text


_tokenizer_adds_prefix_space_cache = {}


def _tokenizer_removes_prefix_space_on_detok(
    tokenizer: PreTrainedTokenizerFast,
) -> bool | None:
    if tokenizer.name_or_path in _tokenizer_adds_prefix_space_cache:
        return _tokenizer_adds_prefix_space_cache[tokenizer.name_or_path]

    def figure_out_val():
        tokens = tokenizer.tokenize("the", add_special_tokens=False)
        if len(tokens) != 1:
            return None
        tokens *= 2
        tokens = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens))
        return tokens == "the the"

    _tokenizer_adds_prefix_space_cache[tokenizer.name_or_path] = figure_out_val()
    return _tokenizer_adds_prefix_space_cache[tokenizer.name_or_path]


def _get_token_offsets(
    tokenizer: PreTrainedTokenizerFast,
    token_ids: torch.Tensor | list[int],
) -> list[tuple[int, int]]:
    """Starts and ends of tokens"""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    if not isinstance(token_ids, list):
        msg = "token_ids must be a list or tensor"
        raise TypeError(msg)

    if not tokenizer.is_fast:
        msg = (
            "This requires a fast tokenizer so that way we can use the"
            " return_offsets_mapping option"
        )
        raise ValueError(
            msg,
        )

    add_fake_bos = (
        _tokenizer_removes_prefix_space_on_detok(tokenizer) and tokenizer.bos_token
    )
    fake_bos = "😂"  # We can't use a real bos because codellama/mistral
    # will still try to add a space to the start of the first
    # real token. Instead we are going to the 😂 emoji (it
    # is the most common emoji, so likely in a tokenizer,
    # but is also unlikely to be merged with anything, so unlikely
    # to screw up future tokens. (would prefer a sadder emoji tbh))
    # NOTE (actually maybe a be a way to get to this to work without this,
    # it seems to work)
    re_decoded = tokenizer.decode(
        [tokenizer.bos_token_id, *token_ids],
        clean_up_tokenization_spaces=False,
        skip_special_tokens=False,
    )
    re_decoded = re_decoded[len(tokenizer.bos_token) :]
    new_tokenize = tokenizer(
        fake_bos + re_decoded if add_fake_bos else re_decoded,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    new_token_ids = new_tokenize["input_ids"]
    offset_mapping = new_tokenize["offset_mapping"]

    if add_fake_bos:
        fake_tokenizer_len = len(tokenizer.tokenize(fake_bos, add_special_tokens=False))
        new_token_ids = new_token_ids[fake_tokenizer_len:]
        _, first_offset = offset_mapping[fake_tokenizer_len - 1]
        offset_mapping = [
            (left - first_offset, right - first_offset)
            for left, right in offset_mapping[fake_tokenizer_len:]
        ]

    if new_token_ids != token_ids:
        # Do a bit hacky things to at least try to align them
        if len(new_token_ids) - 1 == len(token_ids) and new_token_ids[1:] == token_ids:
            # for some reason seems to be an extra start of token or something
            new_token_ids = new_token_ids[1:]
            offset_mapping = offset_mapping[1:]
        else:
            offset_mapping = _attempt_to_fix_degenerate_merges(
                token_ids,
                new_token_ids,
                offset_mapping,
                tokenizer.convert_ids_to_tokens(token_ids),
                tokenizer.convert_ids_to_tokens(new_token_ids),
            )
            new_token_ids = token_ids
            assert len(new_token_ids) == len(offset_mapping)

    # starts, ends = list(zip(*offset_mapping))

    if new_token_ids != token_ids:
        msg = f"Token IDs do not match\nOriginal: {token_ids}\nNew: {new_token_ids}"
        raise AssertionError(msg)

    return offset_mapping


def _attempt_to_fix_degenerate_merges(
    output_tokens: Sequence[int],
    new_tokenization: Sequence[int],
    new_tokenization_offsets: Sequence[tuple[int, int]],
    output_token_strs: Sequence[str],
    new_tokenization_strs: Sequence[str],
) -> Sequence[tuple[int, int]]:
    """
    Sometimes a model might output what I'm calling 'degenerate merges'.
    Here subtokens are outputted when a different token exists that is a merged
    version of the subtokens. This is a problem because the way we get
    the tokenize offsets relies on detokenizing the output tokens and then
    retokenizing with the 'return_offsets_mapping' option set.

    This function tries to fix this by returning a new
    tokenization and new offsets with things unmerged to match the model.
    I don't think we are going to try to handle all the edge cases, but hopefully
    a more common.
    """
    if len(output_tokens) == len(new_tokenization):
        if output_tokens == new_tokenization:
            return new_tokenization, new_tokenization_offsets
    if len(output_tokens) < len(new_tokenization):
        msg = (
            "Cannot fix solutions when there are more new tokens than output tokens."
            " Expect cases where output has more because of extra unmerged tokens."
        )
        raise ValueError(
            msg,
        )
    output_idx = 0
    new_idx = 0
    output_offsets = []
    while output_idx < len(output_tokens):
        if output_tokens[output_idx] == new_tokenization[new_idx]:
            output_offsets.append(new_tokenization_offsets[new_idx])
            output_idx += 1
            new_idx += 1
        else:
            # Check to see if output tokens should be merged
            output_tokens_idxes_for_this_new_token = []
            this_new_token = new_tokenization_strs[new_idx]
            combo_of_output_tokens = ""
            while (
                this_new_token.startswith(combo_of_output_tokens)
                and output_idx < len(output_tokens)
                and len(combo_of_output_tokens) < len(this_new_token)
            ):
                combo_of_output_tokens += output_token_strs[output_idx]
                output_tokens_idxes_for_this_new_token.append(output_idx)
                output_idx += 1
            if combo_of_output_tokens == this_new_token:
                # Split up the offsets
                start_offset, _ = new_tokenization_offsets[new_idx]
                for split_idx in output_tokens_idxes_for_this_new_token:
                    output_token_str_for_this_split = output_token_strs[split_idx]
                    output_offsets.append(
                        (
                            start_offset,
                            start_offset + len(output_token_str_for_this_split),
                        ),
                    )
                    start_offset += len(output_token_str_for_this_split)
            else:
                msg = (
                    "Attempted to fix degenerate merges but failed\n"
                    f"Original: {output_tokens}\n"
                    f"New: {new_tokenization}\n"
                    f"Original tokens: {output_token_strs}\n"
                    f"New tokens: {new_tokenization_strs}\n"
                )
                raise ValueError(msg)
            new_idx += 1
    return output_offsets


def _expand_offsets_to_a_token_index_for_every_text_index(
    token_offsets: list[tuple[int, int]],
) -> list[int]:
    if len(token_offsets) == 0:
        return []
    token_offsets = _merge_equivalent_consecutive_spans(token_offsets)
    token_offsets_full = []
    for i, (start, end) in enumerate(token_offsets):
        last_start, last_end = token_offsets[i - 1] if i > 0 else (0, start)
        token_offsets_full.extend([i] * (end - last_end))
    return token_offsets_full


def _merge_equivalent_consecutive_spans(
    spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Merges spans that are equivalent and consecutive.
    Only the first instance of the span is kept with the rest
    will be converted to a span with the same start end
    """
    if len(spans) == 0:
        return []
    if len(spans) == 1:
        return spans
    merged_spans = [spans[0]]
    last_span = spans[0]
    for span in spans[1:]:
        if span == last_span:
            merged_spans.append((span[1], span[1]))
        else:
            merged_spans.append(span)
            last_span = span
    return merged_spans
