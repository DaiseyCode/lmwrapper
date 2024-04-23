from dataclasses import dataclass
import math
from typing import Sequence, Any

import numpy as np
import numpy.typing


@dataclass
class ModelInternalsRequest:
    return_hidden_states: bool = False
    """Whether to return the hidden states of the model."""
    return_attentions: bool = False
    """Whether to return the self attentions of the model."""
    hidden_layer_indexes: Sequence[int] = None
    """The indexes of the hidden layers to return. If None, then all hidden layers.
    If return_hidden_states is False, then this parameter is ignored.
    Note, depending on the model, first layer might be the input embeddings, 
    and the last layer might be the output embeddings. 
    
    Note that some models might have different ordering for layers (like OPT?). We still
    need to figure out how to handle that.
    """
    hidden_layer_fractions: Sequence[float] = None
    """The relative position of the hidden layers to return. 
    If None, then all hidden layers. For example [0.5, 1.0] would return the middle
    layer and the last layer."""

    def __post_init__(self):
        getting_some_internals = self.return_hidden_states or self.return_attentions
        if self.hidden_layer_indexes is not None:
            if not all(isinstance(x, int) for x in self.hidden_layer_indexes):
                msg = "The hidden_layer_indexes parameter should be a list of ints."
                raise ValueError(msg)
        if self.hidden_layer_fractions is not None:
            if not all(isinstance(x, float) for x in self.hidden_layer_fractions):
                msg = "The hidden_layer_fractions parameter should be a list of floats."
                raise ValueError(msg)
            if not all(0.0 <= x <= 1.0 for x in self.hidden_layer_fractions):
                msg = "The hidden_layer_fractions parameter should be a list of floats between 0 and 1."
                raise ValueError(msg)
        if self.hidden_layer_indexes is not None and self.hidden_layer_fractions is not None:
            msg = "Cannot specify both hidden_layer_indexes and hidden_layer_fractions."
            raise ValueError(msg)

    def select_layer_sequence(
        self,
        layerwise_sequence: Sequence[Any],
    ):
        """Given some sequence with something per layer (like hidden states), selects
        the desired ones for a give"""
        if self.hidden_layer_indexes is not None:
            return [layerwise_sequence[i] for i in self.hidden_layer_indexes]
        if self.hidden_layer_fractions is not None:
            num_layers = len(layerwise_sequence)
            selected_layers = [
                layerwise_sequence[round((num_layers - 1) * f)]
                for f in self.hidden_layer_fractions
            ]
            return selected_layers
        return layerwise_sequence


@dataclass
class ModelInternalsResults:
    hidden_states: tuple[np.typing.NDArray, ...] = None
    attentions: np.typing.NDArray = None
