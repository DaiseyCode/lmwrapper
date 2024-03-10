import copy
from abc import abstractmethod

import tiktoken
from transformers import PreTrainedTokenizer


class PromptTrimmer:
    @abstractmethod
    def trim_text(self, text: str | list[str]) -> str:
        """
        Trims text to meet certain length (or other) requirements.
        By default trims from the left (i.e. from the beginning of the text).
        If a list is supplied then will trim never splitting within the list elements.
        """


class CharPromptTrimmer(PromptTrimmer):
    def __init__(self, char_limit):
        self.char_limit = char_limit

    def trim_text_line_level(self, text: str) -> str:
        lines = [line + "\n" for line in text.split("\n")]
        lines[-1] = lines[-1][:-1]
        return self.trim_text(lines)

    def trim_text(self, text: str | list[str]) -> str:
        if isinstance(text, str):
            return text[-min(self.char_limit, len(text)) :]
        # Returns as many of the list elements as possible while staying under
        #    the character limit
        added_len = 0
        i = len(text) - 1
        while i >= 0:
            added_len += len(text[i])
            if added_len > self.char_limit:
                break
            i -= 1
        return "".join(text[i + 1 :])


class TrimmingTokenizer:
    @abstractmethod
    def tokenize(self, text: str) -> list[str]: ...


class HfTrimmingTokenizer(TrimmingTokenizer):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self._tokenizer = tokenizer

    def tokenize(self, text: str) -> list[str]:
        return self._tokenizer.convert_ids_to_tokens(
            self._tokenizer(text),
        )


class TikTokenTrimmingTokenizer(TrimmingTokenizer):
    def __init__(self, encoding: tiktoken.Encoding):
        self._encoding = encoding

    def tokenize(self, text: str) -> list[str]:
        return [
            self._encoding.decode(token)
            for token in self._encoding.encode(text, allowed_special="all")
        ]


class GenericTokenTrimmer(PromptTrimmer):
    def __init__(
        self,
        token_limit: int,
        tokenizer: TrimmingTokenizer,
        start_from_left_side=True,
    ):
        self.token_limit = token_limit
        self.tokenizer = tokenizer
        self.start_from_left_side = start_from_left_side

    def trim_text(self, text: str | list[str]) -> str:
        if isinstance(text, list):
            raise NotImplementedError

        tokenized = self.tokenizer.tokenize(text)

        offset = None

        if self.start_from_left_side:
            offset = self.token_limit
        elif len(tokenized) > self.token_limit:
            offset = -self.token_limit

        tokenized = tokenized[:offset]

        return "".join(tokenized)


class HfTokenTrimmer(PromptTrimmer):
    def __init__(
        self,
        token_limit: int,
        tokenizer: PreTrainedTokenizer,
        start_from_left_side=True,
    ):
        self.token_limit = token_limit
        self.tokenizer = tokenizer
        self.start_from_left_side = start_from_left_side

    def trim_text(self, text: str | list[str]) -> str:
        if isinstance(text, list):
            raise NotImplementedError

        original_truncation = copy.copy(self.tokenizer.truncation_side)
        self.tokenizer.truncation_side = (
            "left" if self.start_from_left_side else "right"
        )

        trimmed = self.tokenizer.decode(
            self.tokenizer(
                text,
                truncation=True,
                max_length=self.token_limit,
            ).input_ids,
        )

        self.tokenizer.truncation_side = original_truncation

        return trimmed
