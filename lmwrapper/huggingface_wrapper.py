from dataclasses import dataclass
from functools import cached_property
from typing import Union, List, Any

from sympy.polys.polyoptions import Auto
from transformers.models.auto.tokenization_auto import PreTrainedTokenizerFast

from lmwrapper.abstract_predictor import LmPredictor
from lmwrapper.structs import LmPrompt, LmPrediction


try:
    from transformers import TextGenerationPipeline, pipeline, GenerationConfig, AutoTokenizer, AutoModel, \
    AutoModelForCausalLM
except ImportError:
    raise ImportError("You must install transformers to use Huggingface models. "
                      "`pip install transformers` and torch")

try:
    import torch
except ImportError:
    raise ImportError("Expect to work on torch. Please see https://pytorch.org/ for install info")

@dataclass
class HuggingfacePrediction(LmPrediction):
    _prompt_encoding: Any
    _tokens: Any
    _log_probs: Any

    def __post_init__(self):
        assert len(self._prompt_encoding["input_ids"]) == 1
        self._num_prompt_tokens = len(self._prompt_encoding["input_ids"][0])
        if self.prompt.add_bos_token:
            self._num_prompt_tokens -= 1

    @property
    def completion_tokens(self) -> List[str]:
        return self._tokens[self._num_prompt_tokens:]

    @property
    def completion_logprobs(self) -> List[float]:
        self._verify_logprobs()
        return self._log_probs[self._num_prompt_tokens:]

    @property
    def prompt_tokens(self):
        return self._tokens[:self._num_prompt_tokens]

    @property
    def prompt_logprobs(self):
        return self._log_probs[:self._num_prompt_tokens]

    @property
    def full_logprobs(self):
        return self._log_probs

    def get_full_tokens(self):
        return self._tokens


class HuggingfacePredictor(LmPredictor):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        model: Any,
    ):
        super().__init__()
        self._tokenizer = tokenizer
        self._model = model

    def _predict_maybe_cached(self, prompt: LmPrompt) -> Union[LmPrediction, List[LmPrediction]]:
        if prompt.stop:
            raise NotImplementedError
        if prompt.presence_penalty:
            raise NotImplementedError
        temperature = prompt.temperature
        if temperature == 0:
            temperature = 1e-9
        assert self._tokenizer.bos_token
        encoded_input = self._tokenizer(
            self._tokenizer.bos_token + prompt.text,
            return_tensors='pt'
        )
        #output = self._model(**encoded_input)
        #text = self._tokenizer.decode(output[0])
        # Ref https://gist.github.com/kinoc/8a042d8c5683725aa8c372274c02ea2f
        gen_config = GenerationConfig(
            temperature=temperature,
            top_p=prompt.top_p,
            do_sample=prompt.temperature > 0,
        )
        need_log_prob = prompt.logprobs is not None and prompt.logprobs > 0

        # We need a way of getting the raw logprobs of the whole sequence.
        #   The scores we get back are possibly already warped by the configuration
        #   https://github.com/huggingface/transformers/issues/17521#issue-1257881647
        #   Also, it does not return the input tokens. Existing hacks
        #   require calling the model again https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17
        # Instead we are going to patch the model forward to log calls

        old_forward = self._model.forward
        cached_returns = None

        def new_call(*args, **kwargs):
            nonlocal cached_returns
            cached_returns = old_forward(*args, **kwargs)
            return cached_returns
        self._model.forward = new_call

        with torch.no_grad():
            generation_output = self._model.generate(
                input_ids=encoded_input['input_ids'],
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_scores=need_log_prob,
                max_new_tokens=prompt.max_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                bos_token_id=self._tokenizer.bos_token_id,
            )

        self._model.forward = old_forward

        s = generation_output.sequences[0]
        text = self._tokenizer.decode(s[len(encoded_input['input_ids'][0]):])
        tokens = self._tokenizer.convert_ids_to_tokens(s)
        # strip the bos token
        tokens = tokens[1:]
        # Calculate the logprobs if needed
        if need_log_prob:
            assert cached_returns.logits.shape[0] == 1  # batch
            assert cached_returns.logits.shape[1] == len(tokens)
            logprobs = _gather_logprobs_from_logits(
                cached_returns.logits[0], s[1:],
            )
            assert len(logprobs) == len(tokens)
        else:
            logprobs = None

        return HuggingfacePrediction(
            completion_text=text,
            prompt=prompt,
            metad=generation_output,
            _prompt_encoding=encoded_input,
            _tokens=tokens,
            _log_probs=logprobs
        )

    @cached_property
    def space_char(self) -> str:
        # Try to discover the space char in the tokens
        tokens = self._tokenizer.tokenize("I went to")
        for tok in tokens:
            if "went" in tok:
                val = tok.replace("went", "")
                return val
        return None

    def remove_special_chars_from_tokens(self, tokens: list[str]) -> list[str]:
        if self.space_char is None:
            return tokens
        return [tok.replace(self.space_char, " ") for tok in tokens]

def _gather_logprobs_from_logits(
    logits: torch.Tensor,
    selected_toks: torch.LongTensor,
):
    logprobs = torch.log_softmax(logits, dim=-1).detach()
    gen_probs = torch.gather(logprobs, -1, selected_toks.unsqueeze(-1)).squeeze(-1)
    return gen_probs


def get_huggingface_lm(
    model_name: str
) -> HuggingfacePredictor:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return HuggingfacePredictor(tokenizer, model)