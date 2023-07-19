"""Layer module implementation."""
from typing import List

from .module import Module
from .neuron import Neuron
from .typing import ValueLikeList
from .value import Value


class Layer(Module):
    """Layer class."""

    def __init__(self, nin: int, nout: int, label: str) -> None:
        self.neurons = [Neuron(nin, label=f"{label}__Neuron{i}") for i in range(nout)]

    def __call__(self, x: ValueLikeList) -> List[Value]:
        return [neuron(x) for neuron in self.neurons]

    def parameters(self) -> List[Value]:
        return [param for neuron in self.neurons for param in neuron.parameters()]
