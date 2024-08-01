from functools import wraps
from typing import Any, Callable

from ..cell_range import CellRange


def vectorize_cells(func: Callable) -> Callable:
    """@private"""

    def call_fn(*args: Any, **kwargs: Any) -> Any:
        for idx, a in enumerate(args):
            if isinstance(a, CellRange):
                return CellRange(
                    [
                        call_fn(*(args[:idx] + (elem,) + args[idx + 1 :]), **kwargs)
                        for elem in a
                    ]
                )

        for k, v in kwargs.items():
            if isinstance(v, CellRange):
                return CellRange([call_fn(*args, **{**kwargs, k: elem}) for elem in v])
        return func(*args, **kwargs)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return call_fn(*args, **kwargs)

    return wrapper
