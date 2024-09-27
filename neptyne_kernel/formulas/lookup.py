from bisect import bisect_left, bisect_right
from collections import Counter
from inspect import stack
from itertools import compress
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Iterable, Optional

from ..cell_address import Address
from ..cell_range import CellRange
from ..spreadsheet_error import (
    CALC_ERROR,
    NA_ERROR,
    REF_ERROR,
    VALUE_ERROR,
    SpreadsheetError,
)
from .boolean import FALSE, TRUE, BooleanValue
from .helpers import CellValue, SimpleCellValue, _flatten_range, search_wildcard

__all__ = [
    "ADDRESS",
    "AREAS",
    "CHOOSE",
    "CHOOSECOLS",
    "CHOOSEROWS",
    "COLUMN",
    "COLUMNS",
    "DROP",
    "EXPAND",
    "FILTER",
    "HLOOKUP",
    "HSTACK",
    "INDEX",
    "INDIRECT",
    "LOOKUP",
    "MATCH",
    "OFFSET",
    "ROW",
    "ROWS",
    "SORT",
    "SORTBY",
    "TAKE",
    "TOCOL",
    "TOROW",
    "TRANSPOSE",
    "UNIQUE",
    "VLOOKUP",
    "VSTACK",
    "WRAPCOLS",
    "WRAPROWS",
    "XLOOKUP",
    "XMATCH",
]

_IGNORE_CODES = {
    0: lambda x: list(x),
    1: lambda x: [val for val in x if val is not None],
    2: lambda x: [val for val in x if not isinstance(val, SpreadsheetError)],
    3: lambda x: [
        val for val in x if val is not None and not isinstance(val, SpreadsheetError)
    ],
}


def _is_row(cell_range: CellRange) -> bool:
    """Check if CellRange instance is row or col"""
    if cell_range.two_dimensional:
        if cell_range.shape[0] == 1:
            return True
        return False
    if hasattr(cell_range.ref, "range"):
        rng = cell_range.ref.range
        return rng.max_col - rng.min_col > 0
    return False


_MAX_COL_IDX = 18278
_ALPHABET_LEN = 26

if TYPE_CHECKING:
    from ..dash_ref import DashRef


def _idx2letter(col_idx: int) -> str:
    """Convert a column number into a column letter (3 -> 'C')"""
    # these indicies corrospond to A -> ZZZ and include all allowed
    # columns
    if not 1 <= col_idx <= _MAX_COL_IDX:
        raise ValueError(f"Invalid column index {col_idx}")
    letters = []
    while col_idx > 0:
        col_idx, remainder = divmod(col_idx, _ALPHABET_LEN)
        # check for exact division and borrow if needed
        if remainder == 0:
            remainder = _ALPHABET_LEN
            col_idx -= 1
        letters.append(chr(remainder + 64))
    return "".join(reversed(letters))


_STRING_COL_CACHE = {}
for i in range(1, _MAX_COL_IDX + 1):
    col = _idx2letter(i)
    _STRING_COL_CACHE[i] = col


def _get_column_letter(idx: int) -> str:
    """Convert a column index into a column letter
    (3 -> 'C')
    """
    try:
        return _STRING_COL_CACHE[idx]
    except KeyError:
        raise ValueError(f"Invalid column index {idx}")


def ADDRESS(
    row_num: int,
    column_num: int,
    abs_num: int = 1,
    a1: BooleanValue = TRUE,
    sheet_text: str = "",
) -> str:
    """Returns a reference as text to a single cell in a worksheet"""
    if not 1 <= abs_num <= 4:
        return VALUE_ERROR
    col = _get_column_letter(column_num) if a1 else str(column_num)
    row = str(row_num)
    abs_row = abs_num in [1, 2]
    abs_col = abs_num in [1, 3]
    if abs_row:
        row = f"${row_num}" if a1 else str(row_num)
    elif not a1:
        row = f"[{row}]"
    if abs_col:
        if a1:
            col = f"${col}"
    elif not a1:
        col = f"[{col}]"
    cell = f"{col}{row}" if a1 else f"R{row}C{col}"
    return f"'{sheet_text}'!{cell}" if sheet_text else cell


def AREAS(reference: CellValue | tuple) -> int:
    """Returns the number of areas in a reference"""
    if isinstance(reference, tuple):
        return len(reference)
    return 1


def CHOOSE(index_num: int, value1: CellValue, *values: tuple[CellValue]) -> CellValue:
    """Chooses a value from a list of values"""
    vals = [value1, *values]
    if not 1 <= index_num <= len(vals):
        return VALUE_ERROR
    return vals[index_num - 1]


def _positive_index(rowcol, length) -> int:
    """Returns positive index of row or column for a given range length"""
    return (length - abs(rowcol)) if rowcol < 0 else rowcol - 1


def CHOOSECOLS(array: CellRange, col_num1: int, *col_nums: tuple[int]) -> CellRange:
    """Returns the specified columns from an array"""
    cols = [col_num1, *col_nums]
    result = []

    if array.two_dimensional:
        for row in array:
            new_row = []
            for c in cols:
                c = _positive_index(c, array.shape[1])
                if not 0 <= c <= array.shape[1] - 1:
                    return VALUE_ERROR
                if len(cols) == 1:
                    result.append(row[c])
                else:
                    new_row.append(row[c])
            if len(cols) > 1:
                result.append(new_row)

    else:
        for c in cols:
            c = _positive_index(c, len(array))
            if c != 0:
                return VALUE_ERROR
            if len(cols) == 1:
                return CellRange(array)
            result.append(array)

    return CellRange(result)


def CHOOSEROWS(array: CellRange, row_num1: int, *row_nums: tuple[int]) -> CellRange:
    """Returns the specified rows from an array"""
    rows = [row_num1, *row_nums]
    result = []
    for row in rows:
        row = _positive_index(row, array.shape[0])
        if not 0 <= row < array.shape[0]:
            return VALUE_ERROR
        result.append(array[row])
    return result


def COLUMN(reference: Optional["DashRef"] = None) -> int | CellRange:
    """Returns the column number of a reference"""
    if reference is None:
        ip = get_ipython()  # type: ignore # noqa: F821
        return ip.parent_header["content"]["toRun"][0]["cellId"][0] + 1
    else:
        rng = reference.ref.range
        if isinstance(reference, CellRange):
            return CellRange([list(range(rng.min_col + 1, rng.max_col + 2))])
        else:
            return rng.min_col + 1


def COLUMNS(array: CellRange) -> int:
    """Returns the number of columns in a reference"""
    return array.shape[1] if array.two_dimensional else len(array)


def DROP(array: CellRange, rows: int | None, columns: int | None = None) -> CellRange:
    """Excludes a specified number of rows or columns from the start or end of an array"""
    if rows is None:
        rows = 0
    else:
        if rows >= array.shape[0]:
            return VALUE_ERROR
    if columns is None:
        columns = 0
    else:
        if array.two_dimensional and columns >= array.shape[1]:
            return VALUE_ERROR
    if array.two_dimensional:
        result = array[
            slice(rows, None) if rows >= 0 else slice(rows),
            slice(columns, None) if columns >= 0 else slice(columns),
        ]
    elif columns > 0:
        result = []
    else:
        result = array[rows:] if rows >= 0 else array[:rows]
    return result if len(result) else VALUE_ERROR


def EXPAND(
    array: CellRange,
    rows: int,
    columns: int | None = None,
    pad_with: SimpleCellValue | None = None,
):
    """Expands or pads an array to specified row and column dimensions"""
    num_rows = len(array)
    if rows < num_rows:
        return VALUE_ERROR
    new_range = []
    if columns is not None:
        if array.two_dimensional and columns < array.shape[1]:
            return VALUE_ERROR
        col_diff = columns - (array.shape[1] if array.two_dimensional else 1)
        for r in array:
            head = r._values if array.two_dimensional else [r]
            new_range.append(head + [pad_with] * col_diff)
    new_range += [[pad_with] * columns] * (rows - num_rows)
    return CellRange(new_range)


def FILTER(
    array: CellRange, include: CellRange, if_empty: SimpleCellValue | None = None
) -> CellRange:
    """Filters a range of data based on criteria you define"""
    if len(array) != len(include):
        return VALUE_ERROR
    result = list(compress(array, include))
    if not result:
        return CALC_ERROR if if_empty is None else if_empty
    return CellRange(result)


def _lookup(
    lookup_value: SimpleCellValue,
    source_vector: CellRange,
    result_vector: CellRange,
    range_lookup: BooleanValue = TRUE,
):
    for i, val in enumerate(source_vector):
        if range_lookup:
            try:
                if val <= lookup_value and (
                    i == len(source_vector) - 1 or source_vector[i + 1] > lookup_value
                ):
                    return result_vector[i]
            except TypeError:
                pass
        else:
            if isinstance(lookup_value, str):
                if search_wildcard(str(val), lookup_value):
                    return result_vector[i]
            elif val == lookup_value:
                return result_vector[i]
    return NA_ERROR


def HLOOKUP(
    lookup_value: SimpleCellValue,
    cell_range: CellRange,
    row_index: int,
    range_lookup: BooleanValue = TRUE,
):
    """Looks in the top row of an array and returns the value of the indicated cell"""
    if row_index < 1 or row_index > len(cell_range):
        return VALUE_ERROR
    return _lookup(lookup_value, cell_range[0], cell_range[row_index - 1], range_lookup)


def LOOKUP(
    lookup_value: SimpleCellValue,
    lookup_vector: CellRange,
    result_vector: CellValue | None = None,
) -> SimpleCellValue:
    """Looks up values in a vector or array"""
    lookup_shape = lookup_vector.shape
    if result_vector is None:
        # Array form
        if len(lookup_shape) > 1:
            if lookup_shape[1] <= lookup_shape[0]:
                # Search by column
                result_vector = lookup_vector[:, -1]
                lookup_vector = lookup_vector[:, 0]
            else:
                # Search by row
                result_vector = lookup_vector[-1]
                lookup_vector = lookup_vector[0]
        else:
            result_vector = lookup_vector
    else:
        # Vector form
        if lookup_shape != result_vector.shape:
            return VALUE_ERROR

        if len(lookup_shape) > 1:
            if not any(x == 1 for x in lookup_shape):
                return VALUE_ERROR
            lookup_vector = lookup_vector[0]
            result_vector = result_vector[0]

    return _lookup(lookup_value, lookup_vector, result_vector)


def HSTACK(array1: CellRange, *arrays: tuple[CellRange]) -> CellRange:
    """Appends arrays horizontally and in sequence to return a larger array"""
    arrays = [array1, *arrays]
    num_rows = 1
    for ar in arrays:
        if ar.shape[0] > num_rows:
            num_rows = ar.shape[0]

    result = []

    for i in range(num_rows):
        row = []
        for ar in arrays:
            if i < ar.shape[0]:
                if ar.two_dimensional:
                    row += ar[i]._values
                else:
                    row.append(ar[i])
            else:
                row += [NA_ERROR] * (ar.shape[1] if ar.two_dimensional else 1)

        result.append(row)

    return CellRange(result)


def INDEX(range: CellRange, row_num: int = 0, column_num: int = 0):
    """Uses an index to choose a value from a reference or array"""
    if row_num < 0 or column_num < 0:
        return REF_ERROR
    if row_num == 0 and column_num == 0:
        return range
    try:
        if row_num == 0:
            return range[:, column_num - 1]
        if column_num == 0:
            return range[row_num - 1]
        return range[row_num - 1][column_num - 1]
    except IndexError:
        return REF_ERROR


def INDIRECT(ref_text: str, a1: BooleanValue = TRUE):
    """Returns a reference indicated by a text value"""
    cells = stack()[2][0].f_locals["self"].cells
    try:
        address = Address.from_a1(ref_text) if a1 else Address.from_r1c1(ref_text)
    except ValueError:
        return REF_ERROR
    return cells.get(address.sheet).get(address)


def MATCH(
    lookup_value: SimpleCellValue, cell_range: CellRange, match_type: int = 1
) -> int:
    """Looks up values in a reference or array"""

    if cell_range.two_dimensional or match_type not in [-1, 0, 1]:
        return NA_ERROR

    for i, val in enumerate(cell_range):
        if isinstance(val, type(lookup_value)):
            if match_type == 1:
                if val <= lookup_value and (
                    i == cell_range.shape[0] - 1
                    or (
                        isinstance(cell_range[i + 1], type(lookup_value))
                        and cell_range[i + 1] > lookup_value
                    )
                ):
                    return i + 1
            elif match_type == -1:
                if val >= lookup_value and (
                    i == cell_range.shape[0] - 1
                    or (
                        isinstance(cell_range[i + 1], type(lookup_value))
                        and cell_range[i + 1] <= lookup_value
                    )
                ):
                    return i + 1
            elif match_type == 0 and val == lookup_value:
                return i + 1
    return NA_ERROR


def OFFSET(
    reference: "DashRef", rows: int, cols: int, height: int = 1, width: int = 1
) -> CellRange:
    """Returns a reference offset from a given reference"""

    if height <= 0 or width <= 0:
        return VALUE_ERROR

    try:
        cells = reference.ref.dash.cells
        rng = reference.ref.range
    except AttributeError:
        return VALUE_ERROR

    start_col = rng.min_col + cols
    if start_col < 0:
        return REF_ERROR
    start_row = rng.min_row + rows
    if start_row < 0:
        return REF_ERROR

    if height == 1 and width == 1:
        addr = Address.from_list([start_col, start_row, rng.sheet])
        return cells.get(addr.sheet).get(addr)

    result = [
        [
            cells.get(Address.from_list([start_col + i, start_row + j, rng.sheet]))
            for i in range(width)
        ]
        for j in range(height)
    ]

    return CellRange(result)


def ROW(reference: Optional["DashRef"] = None) -> int | CellRange:
    """Returns the row number of a reference"""
    if reference is None:
        ip = get_ipython()  # type: ignore # noqa: F821
        return ip.parent_header["content"]["toRun"][0]["cellId"][1] + 1
    else:
        rng = reference.ref.range
        if isinstance(reference, CellRange):
            return CellRange(list(range(rng.min_row + 1, rng.max_row + 2)))
        else:
            return rng.min_row + 1


def ROWS(array: CellRange) -> int:
    """Returns the number of rows in a reference"""
    return array.shape[0] if array.two_dimensional else 1


def SORT(
    array: CellRange,
    sort_index: int = 1,
    sort_order: int = 1,
    by_col: BooleanValue = FALSE,
) -> CellRange:
    """Sorts the contents of a range or array"""
    if sort_order not in [-1, 1]:
        return VALUE_ERROR
    is2d = array.two_dimensional
    if by_col:
        if not (is2d and 1 <= sort_index <= len(array)):
            return VALUE_ERROR
    elif is2d:
        if not 1 <= sort_index <= array.shape[1]:
            return VALUE_ERROR
    elif sort_index != 1:
        return VALUE_ERROR

    reverse = sort_order == -1

    if by_col:
        array._values = list(
            zip(*sorted(zip(*array), key=itemgetter(sort_index - 1), reverse=reverse))
        )
    else:
        array = sorted(
            array, key=itemgetter(sort_index - 1) if is2d else None, reverse=reverse
        )
    return array


class reversor:
    def __init__(self, obj: Any):
        self.obj = obj

    def __eq__(self, other: Any):
        return other.obj == self.obj

    def __lt__(self, other: Any):
        return other.obj < self.obj


def SORTBY(
    array: CellRange,
    by_array1: CellRange,
    sort_order1: int = 1,
    *by_arrays_sort_orders: tuple[CellRange, int],
) -> CellRange:
    """Sorts the contents of a range or array based on the values in a corresponding range or array"""

    arrays_orders = (by_array1, sort_order1, *by_arrays_sort_orders)

    for order in arrays_orders[1::2]:
        if order not in [-1, 1]:
            return VALUE_ERROR

    length = len(array)
    for by_array in arrays_orders[::2]:
        if len(by_array) != length:
            return VALUE_ERROR

    def sort_by_key(x):
        for idx, elem in enumerate(array):
            eq = elem == x
            if hasattr(eq, "all"):
                eq = eq.all()
            if eq:
                break
        else:
            raise ValueError(f"Value {x} not found in array")
        return tuple(
            reversor(val[idx]) if order == -1 else val[idx]
            for val, order in zip(arrays_orders[::2], arrays_orders[1::2])
        )

    for sort_order in arrays_orders[1::2]:
        if sort_order not in [-1, 1]:
            return VALUE_ERROR

    return CellRange(sorted(array, key=sort_by_key))


def TAKE(
    array: CellRange, rows: int | None = None, columns: int | None = None
) -> CellRange:
    """Returns a specified number of contiguous rows or columns from the start or end of an array"""
    if rows == 0 or columns == 0:
        return VALUE_ERROR

    if rows is None:
        r = slice(len(array))
    else:
        r = slice(rows) if rows > 0 else slice(rows, len(array))
    if array.two_dimensional and columns is not None:
        c = slice(columns) if columns > 0 else slice(columns, array.shape[1])
        return array[r, c]
    return array[r]


def _tocol(
    array: CellRange,
    ignore: int = 0,
    scan_by_column: bool = False,
    return_row: bool = False,
) -> CellRange:
    if not 0 <= ignore <= 3:
        return VALUE_ERROR
    res = _IGNORE_CODES[ignore](
        _flatten_range(
            array, none_to_zero=ignore in [0, 2], scan_by_column=scan_by_column
        )
    )
    if return_row:
        res = [res]
    return CellRange(res)


def TOCOL(
    array: CellRange, ignore: int = 0, scan_by_column: BooleanValue = FALSE
) -> CellRange:
    """Returns the array in a single column"""
    return _tocol(array, ignore, scan_by_column, False)


def TOROW(
    array: CellRange, ignore: int = 0, scan_by_column: BooleanValue = FALSE
) -> CellRange:
    """Returns the array in a single row"""
    return _tocol(array, ignore, scan_by_column, True)


def TRANSPOSE(array: CellRange) -> CellRange:
    """Returns the transpose of an array"""
    if array.two_dimensional:
        return CellRange(list(zip(*array)))
    if _is_row(array):
        return CellRange(array)
    return CellRange([array])


def UNIQUE(
    array: CellRange, by_col: BooleanValue = FALSE, exactly_once: BooleanValue = FALSE
) -> CellRange:
    """Returns a list of unique values in a list or range"""

    def convert_val(val):
        if isinstance(val, tuple):
            if len(val) > 1:
                return list(val)
            return val[0]
        return val

    src = zip(*array) if by_col and array.two_dimensional else array
    res = [
        convert_val(val)
        for val, num in Counter(
            tuple(ar) if isinstance(ar, Iterable) else ar for ar in src
        ).items()
        if not exactly_once or num == 1
    ]
    if by_col and array.two_dimensional and len(array) == 1:
        res = [res]
    return CellRange(res)


def VLOOKUP(
    lookup_value: SimpleCellValue,
    cell_range: CellRange,
    col_index,
    range_lookup: BooleanValue = TRUE,
):
    """Searches down the first column of a range for a key and returns the value of a specified cell in the row found"""
    if col_index < 1 or col_index > cell_range.shape[1]:
        return VALUE_ERROR
    return _lookup(
        lookup_value, cell_range[:, 0], cell_range[:, col_index - 1], range_lookup
    )


def VSTACK(array1: CellRange, *arrays: tuple[CellRange]) -> CellRange:
    """Appends arrays vertically and in sequence to return a larger array"""
    arrays = [array1, *arrays]
    num_cols = 1
    for ar in arrays:
        if ar.two_dimensional and ar.shape[1] > num_cols:
            num_cols = ar.shape[1]
    result = []
    for ar in arrays:
        for row in ar:
            r = row._values if ar.two_dimensional else [row]
            add_cols = num_cols - (ar.shape[1] if ar.two_dimensional else 1)
            result.append(r + [NA_ERROR] * add_cols)
    return CellRange(result)


def _wraprows(
    vector: CellRange, wrap_count: int, pad_with: SimpleCellValue | None = None
):
    if pad_with is None:
        pad_with = NA_ERROR
    flatten = list(_flatten_range(vector))
    mod = len(flatten) % wrap_count
    if mod:
        flatten += [pad_with] * (wrap_count - mod)
    return (
        flatten[i * wrap_count : (i + 1) * wrap_count]
        for i in range(len(flatten) // wrap_count)
    )


def WRAPCOLS(
    vector: CellRange, wrap_count: int, pad_with: SimpleCellValue | None = None
) -> CellRange:
    """Wraps the provided row or column of values by columns after a specified number of elements"""
    if wrap_count < 1 or (
        vector.two_dimensional and all(coord != 1 for coord in vector.shape)
    ):
        return VALUE_ERROR
    result = list(zip(*_wraprows(vector, wrap_count, pad_with)))
    return CellRange(result)


def WRAPROWS(
    vector: CellRange, wrap_count: int, pad_with: SimpleCellValue | None = None
) -> CellRange:
    """Wraps the provided row or column of values by rows after a specified number of elements"""
    if wrap_count < 1 or (
        vector.two_dimensional and all(coord != 1 for coord in vector.shape)
    ):
        return VALUE_ERROR
    return CellRange(list(_wraprows(vector, wrap_count, pad_with)))


def _xlookup(
    lookup_value: SimpleCellValue,
    lookup_array: CellRange,
    match_mode: int = 0,
    search_mode: int = 1,
):
    if match_mode not in [-1, 0, 1, 2] or search_mode not in [-2, -1, 1, 2]:
        return VALUE_ERROR

    if lookup_array.two_dimensional:
        return VALUE_ERROR

    def binary_search(val, lookup_array, match_mode, search_mode):
        lookup_len = len(lookup_array)
        if search_mode == -2:
            lookup_array._values.reverse()

        if match_mode == 0:
            i = bisect_left(lookup_array, val)
            if i != lookup_len and lookup_array[i] == val:
                return i
        elif match_mode == -1:
            i = bisect_left(lookup_array, val)
            if i:
                return i - 1
        elif match_mode == 1:
            i = bisect_right(lookup_array, val)
            if i != lookup_len:
                return i

    def check_value(val, idx):
        nonlocal next_val
        nonlocal next_ret_idx

        if match_mode == 2 and isinstance(lookup_value, str):
            if search_wildcard(str(val), lookup_value):
                return True
        elif match_mode in [-1, 0, 1] and val == lookup_value:
            return True
        elif match_mode == 1 and val > lookup_value:
            if next_val is None or val < next_val:
                next_val = val
                next_ret_idx = idx
        elif match_mode == -1 and val < lookup_value:
            if next_val is None or val > next_val:
                next_val = val
                next_ret_idx = idx
        return False

    next_val = None
    next_ret_idx = None

    if search_mode in [-1, 1]:
        if search_mode == -1:
            lookup_array = reversed(lookup_array)
        for idx, val in enumerate(lookup_array):
            res = check_value(val, idx)
            if res:
                return idx
    else:
        if match_mode == 2:
            return VALUE_ERROR
        idx = binary_search(lookup_value, lookup_array, match_mode, search_mode)
        if idx is not None:
            return idx if search_mode == 2 else -(idx + 1)

    return next_val, (next_ret_idx if search_mode != -1 else -(next_ret_idx + 1))


def XLOOKUP(
    lookup_value: SimpleCellValue,
    lookup_array: CellRange,
    return_array: CellRange,
    if_not_found: str | None = None,
    match_mode: int = 0,
    search_mode: int = 1,
) -> CellValue:
    """Searches a range or an array, and returns an item corresponding to the first match it finds. If a match doesn't exist, then XLOOKUP can return the closest (approximate) match"""

    search_by_row = _is_row(lookup_array)

    if search_by_row:
        if len(lookup_array) != return_array.shape[1]:
            return VALUE_ERROR
    elif len(lookup_array) != len(return_array):
        return VALUE_ERROR

    def to_row_if_col(idx):
        if return_array.two_dimensional and search_by_row:
            return CellRange([row[idx] for row in return_array])
        if isinstance(return_array[idx], CellRange):
            return CellRange([return_array[idx]])
        return return_array[idx]

    res = _xlookup(lookup_value, lookup_array, match_mode, search_mode)

    if isinstance(res, SpreadsheetError):
        return res

    if not isinstance(res, tuple):
        return to_row_if_col(res)

    next_val, next_ret_idx = res

    if match_mode in [-1, 1] and next_val is not None:
        return to_row_if_col(next_ret_idx)
    return NA_ERROR if if_not_found is None else if_not_found


def XMATCH(
    lookup_value: SimpleCellValue,
    lookup_array: CellRange,
    match_mode: int = 0,
    search_mode: int = 1,
):
    """Returns the relative position of an item in an array or range of cells"""
    res = _xlookup(lookup_value, lookup_array, match_mode, search_mode)

    if isinstance(res, SpreadsheetError):
        return res

    if not isinstance(res, tuple):
        return (res + 1) if res >= 0 else len(lookup_array) + res

    next_val, next_ret_idx = res

    if match_mode in [-1, 1] and next_val is not None:
        return next_ret_idx + 1
    return NA_ERROR
