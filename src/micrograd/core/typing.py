from typing import List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .value import Value

ValueLike = Union[float, int, "Value"]
ValueLikeList = Union[List[float], List[int], List["Value"]]
Numeric = Union[float, int]
