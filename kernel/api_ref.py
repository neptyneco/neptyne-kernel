from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import pandas as pd

from .cell_address import Address, Range
from .neptyne_protocol import (
    CellAttribute,
)

Int = int | np.integer
IntOrSlice = Int | slice


def slice_or_int_to_range(key: IntOrSlice, start: int, end: int) -> tuple[int, int]:
    def check_bounds(val: int) -> int:
        if val < start and start > 0:
            raise IndexError(f"Index {val} out of bounds for range {start} to {end}")
        if val > end > 0:
            raise IndexError(f"Index {val} out of bounds for range {start} to {end}")
        return val

    def adjust(v: Int) -> int:
        return v + (end if v < 0 else start)

    if isinstance(key, slice):
        start_new = adjust(key.start) if key.start is not None else start

        if key.stop is not None:
            end = adjust(key.stop)

        return check_bounds(start_new), check_bounds(end - 1)

    adjusted = check_bounds(adjust(key))
    return adjusted, adjusted


def shape(obj: Any, recurse: bool = True) -> tuple[int, int]:
    """Return the height, width of obj as if set into a dash object."""
    if isinstance(obj, dict):
        return len(obj), 2
    if isinstance(obj, str | type) or not hasattr(obj, "__len__"):
        return 1, 1
    if len(obj) == 0:
        return 0, 0
    if isinstance(obj, np.ndarray | np.generic):
        return obj.shape  # type: ignore
    if isinstance(obj, pd.DataFrame):
        return obj.shape[0] + 1, obj.shape[1] + 1

    if not recurse:
        return len(obj), 1

    max_row_size = 1
    for row in obj:
        sub_shape = shape(row, recurse=False)
        max_row_size = max(max_row_size, sub_shape[0])

    return len(obj), max_row_size


def assure_compatible_shape(
    value: Any, target_shape: tuple[int, int]
) -> tuple[Any, tuple[int, int]]:
    """Make sure that value fits the target shape.

    Returns:
        (value, value_shape) if the shape fits
        (value transposed, value_transposed_shape) if value is one dimensional but in the wrong direction
    Raises:
        ValueError when not match can be found
    """
    value_shape = shape(value)
    if value_shape != target_shape:
        if (
            value_shape[1] == 1
            and target_shape[0] == 1
            and value_shape[0] == target_shape[1]
        ):
            return [value], (value_shape[1], 1)
        else:
            raise ValueError(
                f"Assigning ranges with different shapes {value_shape} vs {target_shape}"
            )
    return value, value_shape


class ApiRef(ABC):
    range: Range

    def __init__(self, ref: Address | Range):
        if isinstance(ref, Address):
            ref = Range.from_address(ref)
        self.range = ref

    @abstractmethod
    def set_attributes(self, attributes: dict[CellAttribute, str]) -> None: ...

    @abstractmethod
    def get_attribute(
        self,
        attribute: str,
        modifier: Callable[[str], str] | None = None,
        default_value: int | str | None = None,
    ) -> list[list[str | None]] | list[str | None] | str | None: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def row_entry(self, idx: int) -> Any: ...

    @abstractmethod
    def current_max_col_row(self) -> tuple[int, int]: ...

    def setitem(
        self, key: tuple[IntOrSlice, IntOrSlice] | IntOrSlice, value: Any
    ) -> None:
        row: IntOrSlice
        col: IntOrSlice
        if isinstance(key, IntOrSlice):  # type: ignore
            if self.range.min_row == self.range.max_row:
                row = 0
                col = key  # type: ignore
                is_slice_row = False
                is_slice_col = isinstance(col, slice)
            else:
                row = key  # type: ignore
                col = 0
                is_slice_row = isinstance(row, slice)
                is_slice_col = False
        else:
            row, col = key  # type: ignore
            is_slice_col = isinstance(col, slice)
            is_slice_row = isinstance(row, slice)

        def slice_to_row_or_col_and_size(
            is_slice_row_col: bool,
            row_col: IntOrSlice,
            range_min_row_col: int,
            range_max_row_col: int,
        ) -> tuple[Int, int]:
            if is_slice_row_col:
                min_val, max_val = slice_or_int_to_range(
                    row_col, range_min_row_col, range_max_row_col + 1
                )
                return min_val, max_val - min_val + 1
            if not isinstance(row_col, Int):  # type: ignore
                raise IndexError(row_col)
            return row_col, 1  # type: ignore

        if is_slice_col or is_slice_row:
            col, col_size = slice_to_row_or_col_and_size(
                is_slice_col, col, self.range.min_col, self.range.max_col
            )
            row, row_size = slice_to_row_or_col_and_size(
                is_slice_row, row, self.range.min_row, self.range.max_row
            )
            value, (h, w) = assure_compatible_shape(value, (row_size, col_size))
        else:
            if not isinstance(col, Int):  # type: ignore
                raise IndexError(col)

            if not isinstance(row, Int):  # type: ignore
                raise IndexError(row)

            h, w = shape(value)
            col, _ = slice_or_int_to_range(
                col, self.range.min_col, self.range.max_col + 1
            )
            row, _ = slice_or_int_to_range(
                row, self.range.min_row, self.range.max_row + 1
            )

        if (col + w - 1 > self.range.max_col >= 0) or (
            row + h - 1 > self.range.max_row >= 0
        ):
            raise IndexError(f"Trying to write outside of cellrange ({key})")

        assert isinstance(col, np.integer | int)
        assert isinstance(row, np.integer | int)

        self._write_value(int(col), int(row), value)

    @abstractmethod
    def getitem(self, key: IntOrSlice) -> Any: ...

    @abstractmethod
    def _write_value(self, col: int, row: int, value: Any) -> None: ...

    def xy(self) -> tuple[int, int]:
        return self.range.min_col, self.range.min_row
