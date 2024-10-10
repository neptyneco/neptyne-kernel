import copyreg
import itertools
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import datetime
from numbers import Number
from operator import (
    __add__,
    __eq__,
    __floordiv__,
    __ge__,
    __gt__,
    __le__,
    __lt__,
    __mul__,
    __ne__,
    __sub__,
    __truediv__,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Optional,
    TypeVar,
)

import numpy as np
import pandas as pd

from .api_ref import ApiRef, Int, IntOrSlice, shape, slice_or_int_to_range
from .cell_address import Address, Range
from .cell_api import CellApiMixin
from .neptyne_protocol import (
    CellAttribute,
    Dimension,
    NumberFormat,
    SheetTransform,
)
from .primitives import Empty, NeptyneFloat, NeptyneInt, unproxy_val
from .transformation import Transformation, is_insert_delete_unbounded

if TYPE_CHECKING:
    from geopandas import GeoDataFrame

    from .dash import Dash
    from .dash_ref import DashRef
    from .formulas.helpers import SimpleCellValue
    from .gsheets_api import GSheetRef


T = TypeVar("T")


def unproxy_for_dataframe(val: T) -> T | float | datetime:
    from .dash_ref import DashRef

    if isinstance(val, Empty):
        return np.nan
    if (
        isinstance(val, NeptyneInt | NeptyneFloat)
        and val.ref
        and isinstance(val.ref, DashRef)
    ):
        address = val.ref.range.origin()
        meta = val.ref.dash.cell_meta.get(address)
        if meta:
            num_format = meta.attributes.get(CellAttribute.NUMBER_FORMAT.value)
            if num_format and num_format.startswith(NumberFormat.DATE.value):
                return val.to_datetime()
    return unproxy_val(val)


class CellRange(Sequence):
    two_dimensional: bool
    """Whether or not the cell range is 2D. Single rows and columns are represented as 1D cell range."""
    _values: "DashRef | list"

    def __new__(  # type: ignore
        cls, init_value: "list | DashRef | CellRange", *args, **kwargs
    ) -> "CellRange":
        from .dash_ref import DashRef
        from .gsheets_api import GSheetRef

        if cls is not __class__:  # type: ignore
            # Subclass, create an instance
            return super().__new__(cls)
        if isinstance(init_value, list | CellRangeList):
            return CellRangeList(init_value)
        if isinstance(init_value, DashRef | CellRangeRef):
            return CellRangeRef(init_value, *args, **kwargs)
        if isinstance(init_value, GSheetRef):
            return CellRangeGSheet(init_value, *args, **kwargs)
        raise TypeError(
            "Invalid init variable type: expected list, DashRef or CellRange"
        )

    def __init__(self, init_value: "list | DashRef | CellRange | GSheetRef"):
        """@private"""
        assert isinstance(init_value, CellRange)
        self._values = init_value._values
        self.two_dimensional = init_value.two_dimensional

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        has_numeric = False
        has_str = False

        def check_types(x: Any) -> Any:
            nonlocal has_numeric, has_str
            if isinstance(x, str):
                has_str = True
            elif isinstance(x, Number):
                has_numeric = True
            if isinstance(x, Empty):
                return np.nan
            return x

        if len(self.shape) == 1:
            data = [check_types(x) for x in self]
        else:
            data = [[check_types(x) for x in row] for row in self]
        # If we have mixed str and numeric, numpy wants to create a 'str' array. If we don't have
        # that, let it decide which type to create.
        return np.array(data, dtype=object if has_str and has_numeric else None)

    @property
    def ref(self) -> ApiRef | None:
        """@private"""
        return None

    @property
    def dash(self) -> "Dash":
        """@private"""
        from .dash_ref import DashRef

        if self.ref and isinstance(self.ref, DashRef):
            return self.ref.dash
        raise NotImplementedError("This operation is not possible")

    def __repr__(self) -> str:
        lst = [([el for el in x] if isinstance(x, CellRange) else x) for x in self]
        return f"CellRange:{lst}"

    def _row_key(self, row_idx: int) -> tuple[int, int] | int:
        return (row_idx, 0) if self.two_dimensional else row_idx

    def _validate_insert_delete_index(self, index: int, dimension: Dimension) -> None:
        assert self.ref is not None
        if not isinstance(index, int):
            raise TypeError("Index must be an integer at least 0")
        if index < 0:
            raise ValueError("Index must be at least 0")
        if (
            dimension == Dimension.ROW
            and self.ref.range.max_row != -1
            and index > (self.ref.range.max_row - self.ref.range.min_row)
        ):
            raise IndexError("Index must be less than the number of rows")
        if (
            dimension == Dimension.COL
            and self.ref.range.max_col != -1
            and index > (self.ref.range.max_col - self.ref.range.min_col)
        ):
            raise IndexError("Index must be less than the number of cols")

    def _validate_and_transform_insert_params(
        self,
        dimension: Dimension,
        index: int,
        amount: int | None,
        data: Any | None,
    ) -> tuple[int, Any | None]:
        """Returns a resolved amount and data, or raises an exception if the params are invalid."""
        self._validate_insert_delete_index(index, dimension)
        assert self.ref is not None

        if amount is not None and amount < 1:
            raise ValueError("Amount must be at least 0")

        default_amount = amount or 1
        if data is None:
            return default_amount, None

        # Check for attributes that indicate emptiness or zero size
        if hasattr(data, "empty") and data.empty:
            return default_amount, None
        if hasattr(data, "size") and data.size == 0:
            return default_amount, None
        if hasattr(data, "shape") and len(data.shape) == 0:
            return default_amount, None

        n_rows, n_cols = shape(data)
        if dimension == Dimension.ROW and n_cols == 1 and 1 < n_rows != amount:
            data = [data]
            n_rows, n_cols = n_cols, n_rows

        h, w = self.ref.range.shape()
        if dimension == Dimension.ROW and n_cols > w and w != -1:
            raise ValueError(f"Data has {n_cols} columns but the range has {w} columns")
        if dimension == Dimension.COL and n_rows > h and h != -1:
            raise ValueError(f"Data has {n_rows} rows but the range has {h} rows")

        if amount:
            if dimension == Dimension.ROW and amount != n_rows:
                raise ValueError(
                    f"Amount ({amount}) must match the number of rows in the data ({n_rows})"
                )
            if dimension == Dimension.COL and amount != n_cols:
                raise ValueError(
                    f"Amount ({amount}) must match the number of cols in the data ({n_cols})"
                )
        else:
            amount = n_rows if dimension == Dimension.ROW else n_cols

        return amount, data

    def _validate_delete_params(
        self, dimension: Dimension, index: int, amount: int
    ) -> None:
        assert self.ref is not None
        self._validate_insert_delete_index(index, dimension)

        if amount < 1:
            raise ValueError("Amount must be at least 1")

        if dimension == Dimension.ROW and self.ref.range.max_row != -1:
            avail_to_delete = (
                self.ref.range.max_row - self.ref.range.min_row + index + 1
            )
            if amount > avail_to_delete:
                raise ValueError(
                    f"Cannot delete {amount} rows starting at index {index} because there are only {avail_to_delete} rows available"
                )

        if dimension == Dimension.COL and self.ref.range.max_col != -1:
            avail_to_delete = (
                self.ref.range.max_col - self.ref.range.min_col + index + 1
            )
            if amount > avail_to_delete:
                raise ValueError(
                    f"Cannot delete {amount} cols starting at index {index} because there are only {avail_to_delete} cols available"
                )

    def _insert_delete_row_column(
        self,
        dimension: Dimension,
        transform: SheetTransform,
        index: int,
        amount: int,
        data: Any = None,
    ) -> "CellRange|None":
        """Simpler implementation for non-dash backed cell ranges."""
        assert self.ref is not None

        if dimension != Dimension.ROW:
            raise ValueError(
                "Insert and delete operations are currently only supported on rows."
            )
        if transform == SheetTransform.DELETE:
            to_shift = self[index + amount :]
            empty_row_index = self._get_next_empty_row_index_or_none()
            if empty_row_index is None:
                last_filled_index = self.ref.range.max_row
            else:
                last_filled_index = empty_row_index - 1
            self[index, 0] = [*to_shift]
            self[last_filled_index - amount + 1 :].clear()
            self.ref.range = self.ref.range.extended(0, -amount)
            return None
        elif transform == SheetTransform.INSERT_BEFORE:
            stop = (
                self.ref.range.max_row
                if dimension == Dimension.ROW
                else self.ref.range.max_col
            )
            to_shift = self[index : stop - amount + 1]

            self[index + amount, 0] = to_shift
            self[index : index + amount].clear()
            self[index, 0] = data

            return (
                self[index : (index + amount)]
                if dimension == Dimension.ROW
                else self[:, index : (index + amount)]
            )

    def is_empty(self) -> bool:
        """Returns True if all the cells in the cell range are empty."""
        if self.two_dimensional:
            return all(all(self._is_empty(v) for v in row) for row in self)
        return all(self._is_empty(v) for v in self)

    def insert_row(
        self, index: int, data: Any = None, amount: int | None = None
    ) -> "CellRange":
        """Insert a row at the specified **index**.\n
        If **data** is provided, **data** will be spilled into the new row(s).\n
        If **amount** is provided, **amount** rows are inserted\n
        Returns a new cell range representing the inserted row(s).\n
        Data supports a wide variety of types such as lists, dictionaries and dataframes using standard Neptyne spilling.
        An error will be thrown if the data does not fit the dimensions of the cell range.\n\n
        All other insert/append/delete operations accept data and amount parameters in the same way.\n\n
        Example usage:\n\n
        ```A1:B4.insert_row(1, data=[[1, 2], [3, 4]], amount=2)```
        """
        assert self.ref is not None

        amount, data = self._validate_and_transform_insert_params(
            Dimension.ROW, index, amount, data
        )
        new_rows = self._insert_delete_row_column(
            Dimension.ROW, SheetTransform.INSERT_BEFORE, index, amount, data
        )
        assert isinstance(new_rows, CellRange)
        return new_rows

    def insert_column(
        self, index: int, data: Any = None, amount: int | None = None
    ) -> "CellRange":
        """Insert columns at the specified **index**.\n\n
        Works the same way as insert_row, but for columns.
        Data is still spilled the same way spreading multi-dimensional arrays across rows into the new column(s)\n\n"""
        amount, data = self._validate_and_transform_insert_params(
            Dimension.COL, index, amount, data
        )
        new_columns = self._insert_delete_row_column(
            Dimension.COL, SheetTransform.INSERT_BEFORE, index, amount, data
        )
        assert isinstance(new_columns, CellRange)
        return new_columns

    def delete_row(self, index: int, amount: int = 1) -> None:
        """Delete 1 row starting at the specified **index**.\n\n
        If **amount** is provided, **amount** rows are deleted.\n\n"""
        self._validate_delete_params(Dimension.ROW, index, amount)

        self._insert_delete_row_column(
            Dimension.ROW, SheetTransform.DELETE, index, amount, None
        )

    def delete_column(self, index: int, amount: int = 1) -> None:
        """Delete 1 column starting at the specified **index**.\n\n
        If **amount** is provided, **amount** columns are deleted.\n\n"""
        self._validate_delete_params(Dimension.COL, index, amount)

        self._insert_delete_row_column(
            Dimension.COL, SheetTransform.DELETE, index, amount, None
        )

    def _is_empty(self, v: Any) -> bool:
        none_is_empty = isinstance(self, CellRangeGSheet)
        return isinstance(v, Empty) or (none_is_empty and v is None)

    def _get_iterable(self) -> Iterable:
        return (
            itertools.count()
            if self.ref and self.ref.range.max_row == -1
            else range(len(self))
        )

    def _get_next_empty_row_index_or_none(self) -> int | None:
        if not len(self):
            return 0  # Empty infinite row
        assert self.ref is not None
        current_max = -1
        for row_idx in range(len(self)):
            existing = self[row_idx]
            if self.two_dimensional:
                if any(not self._is_empty(v) for v in existing):
                    current_max = row_idx
            elif not self._is_empty(existing):
                current_max = row_idx

        if self.ref.range.max_row != -1 and current_max >= (
            self.ref.range.max_row - self.ref.range.min_row
        ):
            return None

        return current_max + 1

    def _get_next_empty_col_index_or_none(self) -> int | None:
        if not len(self):
            return 0  # Empty infinite column
        current_max = -1

        def check_row(row: Any) -> bool:
            """Returns true if the row is full"""
            assert self.ref is not None
            nonlocal current_max
            if not isinstance(row, CellRange):
                row = [row]
            last_item = -1
            for index, item in enumerate(row):
                if not self._is_empty(item):
                    last_item = index
            assert self.ref is not None
            if self.ref.range.max_col != -1 and last_item >= (
                self.ref.range.max_col - self.ref.range.min_col
            ):
                return True
            current_max = max(current_max, last_item)
            return False

        if self.two_dimensional:
            for row_idx in self._get_iterable():
                row = self[row_idx]
                if check_row(row):
                    return None
        else:
            if check_row(self):
                return None

        return current_max + 1

    def append_row(self, data: Any = None, amount: int | None = None) -> "CellRange":
        """Append a row to the cell range.\n\n
        There is no index argument as the row is appended to the end of the cell range.\n
        If **data** is provided, **data** will be spilled into the new row(s).\n
        If **amount** is provided, **amount** rows are appended\n
        Returns a new cell range representing the inserted row(s).\n"""
        row_index = self._get_next_empty_row_index_or_none()
        if row_index is None:
            raise ValueError("No empty row found")
        return self.insert_row(row_index, data, amount)

    def append_column(self, data: Any = None, amount: int | None = None) -> "CellRange":
        """Append column(s) to the cell range.\n\n
        Works the same way as append_row, but for columns.\n"""
        col_index = self._get_next_empty_col_index_or_none()
        if col_index is None:
            raise ValueError("No empty column found")
        return self.insert_column(col_index, data, amount)

    def sort_rows(
        self,
        by_column: int | Sequence[int] | None = None,
        reverse: bool = False,
        key: Callable | None = None,
    ) -> None:
        """Sorts the rows in the cell range in place.\n\n
        If **by_column** is provided, the rows will be sorted by the values in the specified column(s).\n
        If **reverse** is True, the rows will be sorted in descending order.\n
        If **key** is provided, it will be used to extract a comparison key from each row.\n\n
        **by_column** and **key** are mutually exclusive.\n\n
        """
        assert self.ref is not None
        if key and by_column:
            raise ValueError("Cannot specify both key and by_column")

        if by_column is None and key is None:
            by_column = [*range(self.ref.range.max_col - self.ref.range.min_col + 1)]

        if by_column is not None:
            if isinstance(by_column, int):
                by_column = [by_column]

            for column in by_column:
                if column < 0 or column >= (
                    self.ref.range.max_col - self.ref.range.min_col + 1
                ):
                    raise ValueError(f"Column index: {column} out of range")

            def splicer(row: Sequence[Any]) -> Any:
                return [getattr(row[col], "value", row[col]) for col in by_column]

            key = splicer

        self[0] = sorted(self, key=key, reverse=reverse)

    def __bool__(self) -> bool:
        raise ValueError(
            "The truth value of a CellRange is ambiguous."
            " Use .any() or .all() or convert to a list with .to_list()"
        )

    def any(self) -> bool:
        """Returns true if any the values in the cell range are 'truthy'"""
        if self.two_dimensional:
            return any(any(row) for row in self)
        return any(self)

    def all(self) -> bool:
        """Returns true if all the values in the cell range are 'truthy'"""
        if self.two_dimensional:
            return all(all(row) for row in self)
        return all(self)

    def to_list(self) -> list:
        """Converts the cell range to a list. If the cell range is 2D, it will be a list of lists."""
        if self.two_dimensional:
            return [[*row] for row in self]
        return [*self]

    def boolean_mask_operator(self, other: Any, op: Callable) -> "CellRangeList":
        """Applies a boolean mask operator between the cell range and a scalar 'other'. The operator is applied element-wise.\n\n
        This function is used to implement the comparison operators (<, >, <=, >=, !=, ==) which can all be applied to cell ranges against a scalar.\n\n
        For example:\n\n
        `A1:A4 < 4`\n
        will return a cell range with the same shape as A1:A4, where each cell is True if the value in A1:A4 is less than 4, and False otherwise.\n\n
        """
        return CellRangeList([op(value, other) for value in self])

    def __lt__(self, other: "SimpleCellValue"):  # type: ignore
        return self.boolean_mask_operator(other, __lt__)

    def __gt__(self, other: "SimpleCellValue"):  # type: ignore
        return self.boolean_mask_operator(other, __gt__)

    def __le__(self, other: "SimpleCellValue"):  # type: ignore
        return self.boolean_mask_operator(other, __le__)

    def __ge__(self, other: "SimpleCellValue"):  # type: ignore
        return self.boolean_mask_operator(other, __ge__)

    def __ne__(self, other: "SimpleCellValue"):  # type: ignore
        return self.boolean_mask_operator(other, __ne__)

    def __eq__(self, other: Any):  # type: ignore
        if not isinstance(other, Iterable) or isinstance(other, str):
            return self.boolean_mask_operator(other, __eq__)
        if len(self) != len(other):  # type: ignore
            return False
        return CellRange([s == o for s, o in zip(self, other)])

    def __contains__(self, item: Any) -> bool:
        if self.two_dimensional:
            return any(item in row for row in self)
        return any(item == v for v in self)

    # Pairwise operators:
    def apply_operator(
        self, other: Any, op: Callable, reverse: bool = False
    ) -> "CellRangeList":
        """Applies a binary operator to the cell range and a scalar 'other' of the same shape. The operator is applied element-wise.\n\n
        This function is used to implement the binary operators (+, -, *, /, //) which can all be applied between cell ranges or lists other collections of the same size.\n\n
        For example:\n\n
        `A1:B2 + A3:B4`\n
        will return a cell range maintaining the 2x2 shape of A1:B2 and A3:B4, where each cell is the sum of the corresponding cells in A1:B2 and A3:B4.\n\n
        """
        if isinstance(other, str) or not hasattr(other, "__iter__"):
            other = [other] * len(self)
        src = zip(other, self) if reverse else zip(self, other)
        return CellRangeList([op(v1, v2) for v1, v2 in src])

    def __add__(self, other: Any) -> "CellRangeList":
        return self.apply_operator(other, __add__)

    def __radd__(self, other: Any) -> "CellRangeList":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "CellRangeList":
        return self.apply_operator(other, __sub__)

    def __rsub__(self, other: Any) -> "CellRangeList":
        return self.apply_operator(other, __sub__, reverse=True)

    def __mul__(self, other: Any) -> "CellRangeList":
        return self.apply_operator(other, __mul__)

    def __rmul__(self, other: Any) -> "CellRangeList":
        return self.__mul__(other)

    def __floordiv__(self, other: Any) -> "CellRangeList":
        return self.apply_operator(other, __floordiv__)

    def __rfloordiv__(self, other: Any) -> "CellRangeList":
        return self.apply_operator(other, __floordiv__, reverse=True)

    def __truediv__(self, other: Any) -> "CellRangeList":
        return self.apply_operator(other, __truediv__)

    def __rtruediv__(self, other: Any) -> "CellRangeList":
        return self.apply_operator(other, __truediv__, reverse=True)

    def map(self, to_apply: Callable) -> "CellRangeList":
        """Applies a function to each cell in the cell range. The function should take a single argument and return a single value.\n\n"""
        if isinstance(self[0], CellRange):
            return CellRangeList([x.map(to_apply) for x in self])
        return CellRangeList([to_apply(x) for x in self])

    def __json__(self) -> list:
        return [*self]

    def to_dataframe(
        self, header: bool = True, dtype: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Converts the cell range to a pandas DataFrame. The first row will be used as the column names. Subsequent rows will be the data."""
        from .dash_ref import DashRef

        if isinstance(self.ref, DashRef) and not self.ref.range.is_fully_bounded():
            return CellRange(self.ref.resolve()).to_dataframe(header=header)
        self_shape = self.shape
        start = self._get_first_row_number(header)

        def column(data: Any, name: str | None = None) -> pd.Series:
            col_dtype = dtype.get(name) if dtype and name in dtype else None
            return pd.Series(
                [unproxy_for_dataframe(x) for x in data],
                name=name,
                dtype=col_dtype,
                index=pd.RangeIndex(start=start, stop=start + len(data)),
            )

        if len(self_shape) == 1:
            if header:
                return column(self[1:], self[0]).to_frame()
            return column(self).to_frame()

        columns = [col for col in zip(*self)]
        if header:
            series = [column(col[1:], col[0]) for col in columns]
        else:
            series = [column(col) for col in columns]

        return pd.concat(series, axis=1)

    def to_pandas(self) -> pd.DataFrame:
        """A wrapper around to_dataframe which emulates arrow's to_pandas method."""
        return self.to_dataframe()

    def pivot_table(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Returns a pivot table from the cell range. This function is a wrapper around pandas' pivot_table function."""
        return self.to_dataframe().pivot_table(*args, **kwargs)

    def to_geodataframe(
        self, header: bool = True, crs: str = "epsg:4326"
    ) -> "GeoDataFrame":
        """@private"""
        try:
            import geopandas as gpd
            from shapely.geometry.base import BaseGeometry
        except ImportError:
            raise ImportError(
                "Geopandas is not installed. Please install it using 'pip install geopandas'"
            )
        df = self.to_dataframe(header=header)
        geom_col = next(
            (
                col
                for col in df.columns
                if isinstance(df[col].dropna().iloc[0], BaseGeometry)
            ),
            None,
        )
        if geom_col:
            gdf = gpd.GeoDataFrame(
                df.drop(columns=[geom_col]), geometry=df[geom_col], crs=crs
            )
            return gdf
        else:
            raise ValueError("No geometry column found")

    @property
    def shape(self) -> tuple[int, int] | tuple[int]:
        """Returns the shape of the cell range. If the cell range is 2D, it will return a tuple of (rows, cols). If the cell range is 1D, it will return a tuple of (length,)."""
        raise NotImplementedError

    @property
    def range(self) -> Range:
        """Returns the range of the cell range."""
        raise NotImplementedError

    def __iter__(self) -> Iterator:
        raise NotImplementedError

    def _check_valid_dict_size(self) -> None:
        assert self.ref is not None
        h, w = self.ref.range.shape()
        if 0 <= w <= 1:
            raise ValueError("At least 2 columns required for dictionary access")

    def __getitem__(self, key: tuple | list | IntOrSlice | str) -> Any:
        if isinstance(key, IntOrSlice):  # type: ignore
            return self._get_item_by_slice_or_int(key)  # type: ignore

        if isinstance(key, tuple):
            return self._get_item_by_tuple(key)

        if isinstance(key, list):
            return self._copy_items_by_list(key)

        if isinstance(key, str):
            self._check_valid_dict_size()

            for row in self:
                if key == row[0]:
                    return row[1]
            raise KeyError(key)
        raise IndexError(key)

    def __delitem__(self, key: IntOrSlice | str) -> None:
        if isinstance(key, int):
            return self.delete_row(key)

        if isinstance(key, slice):
            return self.delete_row(key.start, key.stop - key.start)

        if isinstance(key, str):
            self._check_valid_dict_size()

            for i, row in enumerate(self):
                if key == row[0]:
                    return self.delete_row(i)

        raise KeyError(key)

    def __setitem__(
        self, key: tuple[IntOrSlice, IntOrSlice] | IntOrSlice, value: Any
    ) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def _maybe_get_item_by_unpacked_tuple(self, key: tuple) -> tuple[bool, Any]:
        if len(key) == 1:
            return True, self[key[0]]
        if isinstance(key[0], Int):  # type: ignore
            return True, self[key[0]][key[1]]
        return False, None

    def _copy_items_by_list(self, key: list) -> "CellRangeList":
        return CellRangeList([self[idx] for idx in key])

    def _copy_items_by_tuple(self, key: tuple) -> "CellRangeList":
        return CellRangeList([row[key[1]] for row in self[key[0]]])

    def _get_first_row_number(self, header: bool = True) -> int:
        raise NotImplementedError

    def _get_item_by_slice_or_int(self, key: IntOrSlice) -> Any:
        raise NotImplementedError

    def _get_item_by_tuple(self, key: tuple) -> Any:
        raise NotImplementedError


class CellRangeRef(CellRange, CellApiMixin):
    _values: "DashRef"

    def __init__(self, init_value: "DashRef | CellRange", *args, **kwargs):  # type: ignore
        if isinstance(init_value, CellRange):
            super().__init__(init_value)
        else:
            self._values = init_value
            self.two_dimensional = init_value.range.dimensions() > 1

    def __len__(self) -> int:
        max_col, max_row = self.ref.current_max_col_row()
        if max_col < 0 or max_row < 0:
            return 0
        w = max_col - self.ref.range.min_col + 1
        h_1 = max_row - self.ref.range.min_row
        if not h_1:
            return max(0, w)
        return max(0, h_1 + 1)

    def _get_item_by_slice_or_int(self, key: IntOrSlice) -> Any:
        if not self.two_dimensional and isinstance(key, int | np.integer) and key >= 0:
            return self.ref.row_entry(int(key))
        res = self.ref.getitem(key)

        from .dash_ref import DashRef
        from .gsheets_api import GSheetRef

        if isinstance(res, DashRef):
            return CellRangeRef(res)
        if isinstance(res, GSheetRef):
            return CellRangeGSheet(res)

        return res

    def _get_item_by_tuple(self, key: tuple) -> Any:
        success, item = self._maybe_get_item_by_unpacked_tuple(key)
        if success:
            return item
        if isinstance(key[0], slice) and isinstance(key[1], IntOrSlice):  # type: ignore
            return self._get_subrange_ref(key)
        return self._copy_items_by_tuple(key)

    def _insert_delete_row_column(
        self,
        dimension: Dimension,
        transform: SheetTransform,
        index: int,
        amount: int,
        data: Any = None,
    ) -> Optional["CellRange"]:
        assert self.ref is not None
        assert self.dash is not None

        global_index = (
            index + self.ref.range.min_row
            if dimension == Dimension.ROW
            else index + self.ref.range.min_col
        )

        self.dash.add_delete_cells_internal(
            Transformation(
                dimension=dimension,
                operation=transform,
                index=global_index,
                amount=amount,
                sheet_id=self.ref.range.sheet,
                boundary=self.ref.range if self.ref.range.is_bounded() else None,
            ),
            [],
        )

        transform_amount = (
            amount if transform == SheetTransform.INSERT_BEFORE else -amount
        )

        self.ref.range = (
            self.ref.range.extended(0, transform_amount)
            if dimension == Dimension.ROW
            else self.ref.range.extended(transform_amount, 0)
        )

        if data:
            coord = (index, 0) if dimension == Dimension.ROW else (0, index)

            self[coord] = data
            self.dash.flush_side_effects()

        if transform == SheetTransform.INSERT_BEFORE:
            return (
                self[index : (index + amount), :]
                if dimension == Dimension.ROW
                else self[:, index : (index + amount)]
            )

    def __setitem__(
        self, key: tuple[IntOrSlice, IntOrSlice] | IntOrSlice | str, value: Any
    ) -> None:
        if isinstance(key, str):
            self._check_valid_dict_size()

            for i, row in enumerate(self):
                if key == row[0]:
                    row[1] = [[value]]
                    return

            self.append_row([[key, value]])
            return

        if not isinstance(key, tuple | Int):
            raise IndexError(key)
        self.ref.setitem(key, value)

    def __iter__(self) -> Iterator:
        from .gsheets_api import GSheetRef

        if isinstance(self.ref, GSheetRef):
            # self.ref can be a GSheetRef in case of sheets. That's broken there but hard to fix.
            # We can however cheaply create the right cell range here and let that do the work.
            gsheet_range = CellRangeGSheet(self.ref)
            yield from gsheet_range
            return

        # if we have -1 in the range, loop through it even if it is one dimensional:
        max_col, max_row = self.ref.current_max_col_row()
        if max_row == -1 or max_col == -1:
            return

        r = replace(self.ref.range, max_col=max_col, max_row=max_row)
        h, w = r.shape()
        if h == 1 and self.ref.range.max_row != -1:
            for col in range(r.min_col, max_col + 1):
                yield self.dash[col, r.min_row, r.sheet]
        else:
            for row in range(r.min_row, max_row + 1):
                if w == 1 and self.ref.range.max_col != -1:
                    yield self.dash[r.min_col, row, r.sheet]
                else:
                    yield self.dash[replace(r, min_row=row, max_row=row)]

    def _get_first_row_number(self, header: bool = True) -> int:
        return (self.ref.range.min_row + 1) if header else self.ref.range.min_row

    @property
    def ref(self) -> ApiRef:  # type: ignore
        return self._values

    @property
    def shape(self) -> tuple[int, int] | tuple[int]:
        """Tuple of range dimensions"""
        h, w = self.ref.range.shape()
        if w == 1:
            return (h,)
        if h == 1:
            return (w,)
        return h, w

    @property
    def range(self) -> Range:
        """Returns a Range object representing the bounds of the cell range."""
        return self.ref.range

    def _get_subrange_ref(self, key: tuple) -> "CellRangeRef":
        r = self.ref.range
        min_row, max_row = slice_or_int_to_range(key[0], r.min_row, r.max_row + 1)
        min_col, max_col = slice_or_int_to_range(key[1], r.min_col, r.max_col + 1)

        from .dash_ref import DashRef

        return CellRangeRef(
            DashRef(
                self.dash,
                Range(min_col, max_col, min_row, max_row, r.sheet),
            )
        )


class CellRangeList(CellRange):
    _values: list

    def __init__(self, init_value: list | CellRange):
        if isinstance(init_value, list):
            self.two_dimensional = len(init_value) > 0 and (
                not isinstance(init_value[0], str | Empty)
                and hasattr(init_value[0], "__iter__")
            )
            self._values = (
                [CellRangeList([*row]) for row in init_value]
                if self.two_dimensional
                else init_value
            )

        else:
            super().__init__(init_value)

    def __len__(self) -> int:
        return len(self._values)

    def __setitem__(
        self, key: tuple[IntOrSlice, IntOrSlice] | IntOrSlice, value: Any
    ) -> None:
        raise ReferenceError("Can't assign item")

    def __iter__(self) -> Iterator:
        yield from self._values

    def _get_first_row_number(self, header: bool = True) -> int:
        return 0

    @property
    def shape(self) -> tuple[int, int] | tuple[int]:
        """Tuple of range dimensions"""
        return (len(self), len(self[0])) if self.two_dimensional else (len(self),)

    def _get_item_by_slice_or_int(self, key: IntOrSlice) -> Any:
        return self._values[key]

    def _get_item_by_tuple(self, key: tuple) -> Any:
        success, item = self._maybe_get_item_by_unpacked_tuple(key)
        return item if success else self._copy_items_by_tuple(key)


class CellRangeGSheet(CellRangeList, CellApiMixin):
    def __init__(self, init_value: "GSheetRef | CellRange", *args: Any, **kwargs: Any):
        if isinstance(init_value, CellRange):
            super().__init__(init_value)
        else:
            self.gsheet_ref = init_value
            super().__init__(init_value.values)
            self.two_dimensional = init_value.range.dimensions() > 1

    @property
    def ref(self) -> ApiRef:  # type: ignore
        return self.gsheet_ref

    @property
    def range(self) -> Range:
        return self.ref.range

    def __iter__(self) -> Iterator:
        if self.two_dimensional:
            for row_idx, row in enumerate(self._values):
                min_row = self.ref.range.min_row + row_idx
                yield CellRangeGSheet(
                    self.gsheet_ref.with_range(
                        replace(
                            self.gsheet_ref.range, min_row=min_row, max_row=min_row
                        ),
                        [*row],
                    )
                )
        else:
            for idx, val in enumerate(self._values):
                horizontal = self.range.min_row == self.range.max_row
                address = Address(
                    self.range.min_col + (idx if horizontal else 0),
                    self.range.min_row + (idx if not horizontal else 0),
                    0,
                )
                yield self.gsheet_ref.with_value(address, val)

    def __setitem__(
        self, key: tuple[IntOrSlice, IntOrSlice] | IntOrSlice | str, value: Any
    ) -> None:
        if isinstance(key, str):
            self._check_valid_dict_size()

            for i, row in enumerate(self):
                if key == row[0]:
                    row[1] = [[value]]
                    return

            raise KeyError(key)

        if not isinstance(key, tuple | Int):
            raise IndexError(key)
        self.ref.setitem(key, value)

    def _get_item_by_slice_or_int(self, key: IntOrSlice) -> Any:
        if not self.two_dimensional and isinstance(key, int | np.integer):
            return self.ref.row_entry(int(key))
        res = self.ref.getitem(key)
        return CellRangeGSheet(res)

    def _insert_delete_row_column(
        self,
        dimension: Dimension,
        transform: SheetTransform,
        index: int,
        amount: int,
        data: Any = None,
    ) -> Optional["CellRange"]:
        if is_insert_delete_unbounded(dimension, self.ref.range):
            self.ref.insert_delete_sheet_row_col(  # type: ignore
                dimension, transform, index, amount, data
            )
            self[index, 0] = data
            if transform == SheetTransform.INSERT_BEFORE:
                return (
                    self[index : (index + amount), :]
                    if dimension == Dimension.ROW
                    else self[:, index : (index + amount)]
                )
            else:
                return None
        from .gsheets_api import insert_delete_row_column

        return insert_delete_row_column(
            self.gsheet_ref.service,
            self.gsheet_ref.spreadsheet_id,
            self.gsheet_ref.range,
            dimension,
            transform,
            index,
            amount,
            data,
            self.gsheet_ref.sheet_prefix,
        )


def pickle_cell_range(val: CellRange) -> tuple:
    return type(val), ([*val],)


copyreg.pickle(CellRange, pickle_cell_range)  # type: ignore
