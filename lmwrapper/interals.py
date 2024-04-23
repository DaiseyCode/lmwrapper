from dataclasses import dataclass

import numpy as np
import numpy.typing


@dataclass
class ModelInternalsRequest:
    return_hidden_states: bool = False
    """Whether to return the hidden states of the model."""
    return_attentions: bool = False
    """Whether to return the self attentions of the model."""
    hidden_layer_indexes: list[int] = None
    """The indexes of the hidden layers to return. If None, then all hidden layers.
    If return_hidden_states is False, then this parameter is ignored.
    Note, depending on the model, first layer might be the input embeddings, 
    and the last layer might be the output embeddings.
    """
    hidden_layer_fractions: list[float] = None
    """The relative position of the hidden layers to return. 
    If None, then all hidden layers. For example [0.5, 1.0] would return the middle
    layer and the last layer"""

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


@dataclass
class ModelInternalsResults:
    hidden_states: tuple[np.typing.NDArray, ...] = None
    attentions: np.typing.NDArray = None
