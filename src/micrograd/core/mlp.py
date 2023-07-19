"""MLP module implementation."""
from typing import List

from .layer import Layer
from .typing import Numeric
from .value import Value


class MLP:
    """Multi-layer perceptron class."""

    def __init__(self, nin: int, nouts: List[int]) -> None:
        self.layers = [
            Layer(nin=nin, nout=nout, label=f"Layer{i}") for i, nout in enumerate(nouts)
        ]

    def __call__(self, x: List[Numeric]) -> List[Value]:
        x_val = [Value(el, label=f"Input{i}") for i, el in enumerate(x)]
        for layer in self.layers:
            x_val = layer(x_val)
        return x_val

    def parameters(self) -> List[Value]:
        return [param for layer in self.layers for param in layer.parameters()]
