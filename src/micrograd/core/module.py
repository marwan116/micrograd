"""Module base class for all neural network modules."""
from abc import ABC, abstractmethod
from typing import List

from .value import Value


class Module(ABC):
    """Module base class for all neural network modules."""

    @abstractmethod
    def parameters(self) -> List[Value]:
        pass

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0
