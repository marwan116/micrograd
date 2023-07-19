"""Scalar value with autograd support."""
from threading import Lock
from typing import Callable, List, Optional, Set, Tuple, cast

import numpy as np
from graphviz import Digraph
from varname import varname

from .typing import Numeric, ValueLike


class Value:
    """Scalar value with autograd support."""

    instance_count = 0
    lock = Lock()

    def __init__(self, data: Numeric, label: Optional[str] = None) -> None:
        self.val_count = type(self).instance_count

        with self.lock:
            type(self).instance_count += 1

        self.data = data
        self._children = None
        self._op = None
        self._grad = 0.0
        self._grad_fn = None

        if label is not None:
            self.label = label
        else:
            try:
                self.label = varname()
            except Exception:
                self.label = "val"

    @property
    def node_id(self) -> str:
        return f"{id(self):x}"

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        if value == "val":
            value = f"{value}_{self.val_count}"
        self._label = value

    @property
    def children(self) -> Optional[Tuple["Value", ...]]:
        return self._children

    @children.setter
    def children(self, value: Tuple["Value", ...]) -> None:
        if not isinstance(value, tuple):
            raise TypeError(f"expected tuple but got {type(value)}")
        if self._children is not None:
            raise ValueError("children is already set")
        self._children = value

    @property
    def supported_ops(self) -> Set[str]:
        return {"+", "*", "**", "tanh", "exp"}

    @property
    def op(self) -> Optional[str]:
        return self._op

    @op.setter
    def op(self, value: str) -> None:
        if value not in self.supported_ops:
            raise ValueError(f"expected {self.supported_ops} but got {value}")
        if self._op is not None:
            raise ValueError("op is already set")
        self._op = value

    @property
    def grad(self) -> Numeric:
        return self._grad

    @grad.setter
    def grad(self, value: Numeric) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"expected int or float but got {type(value)}")
        self._grad = value

    @property
    def grad_fn(self) -> Callable[[], None]:
        if self._grad_fn is None:
            raise ValueError("_grad_fn is not set")
        return cast(Callable[[], None], self._grad_fn)

    @grad_fn.setter
    def grad_fn(self, value: Callable[[], None]) -> None:
        if not callable(value):
            raise TypeError(f"expected callable but got {type(value)}")
        self._grad_fn = value

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"label={self.label}, data={self.data}, grad={self.grad})"
        )

    def __add__(self, other: ValueLike) -> "Value":
        other_val = Value(other) if not isinstance(other, Value) else other

        out = Value(self.data + other_val.data)
        out.children = (self, other_val)
        out.op = "+"

        def grad_fn() -> None:
            self.grad += out.grad * 1
            other_val.grad += out.grad * 1

        out.grad_fn = grad_fn
        return out

    def __radd__(self, other: ValueLike) -> "Value":
        return self.__add__(other)

    def __mul__(self, other: ValueLike) -> "Value":
        other_val = Value(other) if not isinstance(other, Value) else other
        out = Value(self.data * other_val.data)
        out.children = (self, other_val)
        out.op = "*"

        def grad_fn() -> None:
            self.grad += out.grad * other_val.data
            other_val.grad += out.grad * self.data

        out.grad_fn = grad_fn

        return out

    def __rmul__(self, other: ValueLike) -> "Value":
        return self.__mul__(other)

    def __neg__(self) -> "Value":
        return self.__mul__(-1)

    def __sub__(self, other: ValueLike) -> "Value":
        return self.__add__(-other)

    def __rsub__(self, other: ValueLike) -> "Value":
        return (-self).__add__(other)

    def __pow__(self, other: Numeric) -> "Value":
        if not isinstance(other, (int, float)):
            raise TypeError(f"expected int or float but got {type(other)}")

        out = Value(self.data**other)
        out.children = (self,)
        out.op = "**"

        def grad_fn() -> None:
            # dy/dx (y = x^n) -> n * x^(n-1)
            self.grad += out.grad * other * self.data ** (other - 1)

        out.grad_fn = grad_fn
        return out

    def __truediv__(self, other: ValueLike) -> "Value":
        return self.__mul__(other**-1)

    def __rtruediv__(self, other: ValueLike) -> "Value":
        return (self**-1).__mul__(other)

    def exp(self) -> "Value":
        out = Value(np.exp(self.data))
        out.children = (self,)
        out.op = "exp"

        def grad_fn() -> None:
            # dy/dx (y = e^x) -> e^x
            self.grad += out.grad * out.data

        out.grad_fn = grad_fn
        return out

    def tanh(self) -> "Value":
        out = Value(np.tanh(self.data))
        out.children = (self,)
        out.op = "tanh"

        def grad_fn() -> None:
            self.grad += out.grad * (1 - out.data**2)

        out.grad_fn = grad_fn
        return out

    def _build_node(self, graph: Digraph) -> None:
        return graph.node(  # type: ignore
            name=self.node_id,
            label=f"{self.label} | data={self.data:.4f} | grad={self.grad:.4f}",
            shape="record",
        )

    def _update_graph(
        self, graph: Digraph, visited: Optional[Set["Value"]] = None
    ) -> None:
        if visited is None:
            visited = set()

        if self in visited:
            return

        visited.add(self)

        if self.children is None:
            return

        for child in self.children:
            child._build_node(graph)
            graph.edge(tail_name=self.node_id, head_name=child.node_id, label=self.op)
            child._update_graph(graph, visited)

    def visualize(self) -> Digraph:
        graph = Digraph()
        self._build_node(graph)
        self._update_graph(graph)
        graph.graph_attr["rankdir"] = "LR"
        return graph

    def backward(self) -> None:
        self.grad = 1
        for child in topo_sort(self):
            if child.children is not None:
                child.grad_fn()


def topo_sort(
    node: Value,
    visited: Optional[Set[Value]] = None,
    stack: Optional[List[Value]] = None,
    ascending: bool = False,
) -> List[Value]:
    if visited is None:
        visited = set()
    if stack is None:
        stack = []

    visited.add(node)
    if node.children is not None:
        for child in node.children:
            if child not in visited:
                topo_sort(child, visited, stack)
    stack.append(node)
    return list(reversed(stack)) if not ascending else stack
