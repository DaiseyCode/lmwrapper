from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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
        if self.hidden_layer_indexes is not None:
            if not all(isinstance(x, int) for x in self.hidden_layer_indexes):
                msg = "The hidden_layer_indexes parameter should be a list of ints."
                raise ValueError(msg)
        if self.hidden_layer_fractions is not None:
            if not all(isinstance(x, float) for x in self.hidden_layer_fractions):
                msg = "The hidden_layer_fractions parameter should be a list of floats."
                raise ValueError(msg)
            if not all(0.0 <= x <= 1.0 for x in self.hidden_layer_fractions):
                msg = (
                    "The hidden_layer_fractions parameter should be a list of floats"
                    " between 0 and 1."
                )
                raise ValueError(msg)
        if (
            self.hidden_layer_indexes is not None
            and self.hidden_layer_fractions is not None
        ):
            msg = "Cannot specify both hidden_layer_indexes and hidden_layer_fractions."
            raise ValueError(msg)

    def get_effective_selected_layers_idxs(self, num_layers) -> Sequence[int]:
        if self.hidden_layer_indexes is not None:
            return self.hidden_layer_indexes
        if self.hidden_layer_fractions is not None:
            selected_layers = [
                round((num_layers - 1) * f) for f in self.hidden_layer_fractions
            ]
            return selected_layers
        return tuple(range(num_layers))

    def select_layer_sequence(
        self,
        layerwise_sequence: Sequence[Any] | np.ndarray,
    ):
        """
        Given some sequence with something per layer (like hidden states), selects
        the desired ones for a give
        """
        if (
            self.hidden_layer_indexes is not None
            or self.hidden_layer_fractions is not None
        ):
            if isinstance(layerwise_sequence, np.ndarray):
                return layerwise_sequence[
                    self.get_effective_selected_layers_idxs(layerwise_sequence.shape[0])
                ]
            else:
                return [
                    layerwise_sequence[i]
                    for i in self.get_effective_selected_layers_idxs(
                        len(layerwise_sequence),
                    )
                ]
        return layerwise_sequence


@dataclass
class ModelInternalsResults:
    hidden_states: tuple[np.typing.NDArray, ...] = None
    """A tuple of (seq, hidden) numpy arrays for each hidden layer.
    Some models will have an embedding layer as the first layer."""
    attentions: np.typing.NDArray = None
    """(layer, head, seq_gen, seq_prompt) numpy array of attentions."""
    has_a_bos: bool = True
