from abc import abstractmethod
from typing import Optional, List, Union
import numpy as np


class PromptTrimmer():
    @abstractmethod
    def trim_text(self, text: Union[str, List[str]]) -> str:
        """
        Trims text to meet certain length (or other) requirements.
        By default trims from the left (i.e. from the beginning of the text).
        If a list is supplied then will trim never splitting within the list elements.
        """
        pass


class CharPromptTrimmer(PromptTrimmer):
    def __init__(self, char_limit):
        self.char_limit = char_limit

    def trim_text_line_level(self, text: str) -> str:
        lines = [l + "\n" for l in text.split("\n")]
        lines[-1] = lines[-1][:-1]
        return self.trim_text(lines)

    def trim_text(self, text: Union[str, List[str]]) -> str:
        if isinstance(text, str):
            return text[-min(self.char_limit, len(text)):]
        # Returns as many of the list elements as possible while staying under
        #    the character limit
        added_len = 0
        i = len(text) - 1
        while i >= 0:
            added_len += len(text[i])
            if added_len > self.char_limit:
                break
            i -= 1
        return "".join(text[i+1:])
