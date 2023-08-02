from dataclasses import dataclass
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

    def __post_init__(self):
        assert len(self._prompt_encoding["input_ids"]) == 1
        self._num_prompt_tokens = len(self._prompt_encoding["input_ids"][0])

    @property
    def completion_tokens(self) -> List[str]:
        return self._tokens[self._num_prompt_tokens:]


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
        if prompt.logprobs is not None:
            raise NotImplementedError
        if prompt.presence_penalty:
            raise NotImplementedError
        temperature = prompt.temperature
        if temperature == 0:
            temperature = 1e-9
        encoded_input = self._tokenizer(prompt.text, return_tensors='pt')
        #output = self._model(**encoded_input)
        #text = self._tokenizer.decode(output[0])
        # Ref https://gist.github.com/kinoc/8a042d8c5683725aa8c372274c02ea2f
        gen_config = GenerationConfig(
            temperature=temperature,
            top_p=prompt.top_p,
        )
        with torch.no_grad():
            generation_output = self._model.generate(
                input_ids=encoded_input['input_ids'],
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=prompt.max_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        s = generation_output.sequences[0]
        text = self._tokenizer.decode(s[len(encoded_input['input_ids'][0]):])
        tokens = self._tokenizer.convert_ids_to_tokens(s)
        #output = self._pipeline(
        #    prompt.get_text_as_string_default_form(),
        #    max_length=prompt.max_tokens,
        #    return_text=False,
        #    generation_config=gen_config,
        #    return_dict_in_generate=True,
        #)
        return HuggingfacePrediction(
            completion_text=text,
            prompt=prompt,
            metad=generation_output,
            _prompt_encoding=encoded_input,
            _tokens=tokens,
        )



def get_huggingface_lm(
    model_name: str
) -> HuggingfacePredictor:
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    return HuggingfacePredictor(tokenizer, model)