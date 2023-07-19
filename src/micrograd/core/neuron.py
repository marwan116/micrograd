"""Neuron module class implementation."""
from typing import List

import numpy as np

from .module import Module
from .typing import ValueLikeList
from .value import Value


class Neuron(Module):
    """Neuron class."""

    def __init__(self, nin: int, label: str) -> None:
        self.w = [
            Value(np.random.uniform(-1, 1), label=f"{label}__Weight{i}")
            for i in range(nin)
        ]
        self.b = Value(np.random.uniform(-1, 1), label=f"{label}__Bias")

    def __call__(self, xs: ValueLikeList) -> Value:
        """Perform dot product and activation."""
        out = sum([w * x for w, x in zip(self.w, xs)], self.b) # type: ignore
        return out.tanh()

    def parameters(self) -> List[Value]:
        """Return parameters."""
        return self.w + [self.b]
