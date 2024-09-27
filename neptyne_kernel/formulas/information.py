from inspect import stack

from ..cell_address import format_cell
from ..cell_range import CellRange, CellRangeList, CellRangeRef
from ..spreadsheet_datetime import SpreadsheetDate, SpreadsheetDateTime, SpreadsheetTime
from ..spreadsheet_error import (
    GETTING_DATA_ERROR,
    NA_ERROR,
    NAME_ERROR,
    NULL_ERROR,
    NUM_ERROR,
    REF_ERROR,
    VALUE_ERROR,
    ZERO_DIV_ERROR,
    SpreadsheetError,
)
from .boolean import BooleanValue
from .helpers import CellValue, Numeric, SimpleCellValue

__all__ = [
    "CELL",
    "ERROR",
    "ISBLANK",
    "ISERR",
    "ISERROR",
    "ISEVEN",
    "ISLOGICAL",
    "ISNA",
    "ISNONTEXT",
    "ISNUMBER",
    "ISODD",
    "ISREF",
    "ISTEXT",
    "N",
    "NA",
    "SHEET",
    "SHEETS",
    "TYPE",
]

_ERROR_TYPES = {
    NULL_ERROR: 1,
    ZERO_DIV_ERROR: 2,
    VALUE_ERROR: 3,
    REF_ERROR: 4,
    NAME_ERROR: 5,
    NUM_ERROR: 6,
    NA_ERROR: 7,
    GETTING_DATA_ERROR: 8,
}


def _get_cell_type(value):
    if isinstance(value, str):
        return "l"
    elif value is None:
        return "b"
    return "v"


def _get_cell_contents(value):
    if isinstance(value, CellRange):
        return value[0][0] if value.two_dimensional else value[0]
    return 0 if value is None else value


def _get_row_col(reference, addr_row_or_col=-1):
    if reference is None:
        ip = get_ipython()  # type: ignore # noqa: F821
        col_num, row_num, _ = ip.parent_header["content"]["toRun"][0]["cellId"]
    elif hasattr(reference, "ref") and reference.ref is not None:
        rng = reference.ref.range
        col_num, row_num = rng.min_col, rng.min_row
    else:
        return VALUE_ERROR
    if addr_row_or_col == -1:
        # Address
        return format_cell(col_num, row_num)
    elif addr_row_or_col == 0:
        # Row
        return row_num + 1
    else:
        # Col
        return col_num + 1


_INFO_TYPES = {
    "address": lambda x: _get_row_col(x, -1),
    "col": lambda x: _get_row_col(x, 1),
    "contents": _get_cell_contents,
    "row": lambda x: _get_row_col(x, 0),
    "type": _get_cell_type,
}


def CELL(info_type: str, reference: CellValue | None = None) -> str:
    """Returns information about the formatting, location, or contents of a cell"""
    # TODO: info types 'color', 'filename', 'format', 'parentheses', 'prefix', 'protect', 'width' are not supported in Excel for the web
    # Supported info types: 'address', 'col', 'contents', 'row', 'type'
    return _INFO_TYPES[info_type](reference)


class ERROR:
    @staticmethod
    def TYPE(error_val: SpreadsheetError) -> int:
        """Returns a number corresponding to an error type"""
        return _ERROR_TYPES.get(error_val, NA_ERROR)


def ISBLANK(value: SimpleCellValue) -> bool:
    """Returns TRUE if the value is blank"""
    if hasattr(value, "ref"):
        return value.is_empty()
    return value is None


def ISERR(value: SimpleCellValue) -> bool:
    """Returns TRUE if the value is any error value except #N/A"""
    return isinstance(value, SpreadsheetError) and value != NA_ERROR


def ISERROR(value: SimpleCellValue) -> bool:
    """Returns TRUE if the value is any error value"""
    return isinstance(value, SpreadsheetError)


def ISEVEN(number: Numeric) -> bool:
    """Returns TRUE if the number is even"""
    try:
        return int(number) % 2 == 0
    except (TypeError, ValueError):
        return VALUE_ERROR


def ISLOGICAL(value: SimpleCellValue) -> bool:
    """Returns TRUE if the value is a logical value"""
    return isinstance(value, BooleanValue | bool)


def ISNA(value: SimpleCellValue) -> bool:
    """Returns TRUE if the value is the #N/A error value"""
    return isinstance(value, SpreadsheetError) and value == NA_ERROR


def ISNONTEXT(value: CellValue) -> bool:
    """Returns TRUE if the value is not text"""
    return not isinstance(value, str)


def ISNUMBER(value: SimpleCellValue) -> bool:
    """Returns TRUE if the value is a number"""
    return isinstance(value, int | float)


def ISODD(number: Numeric) -> bool:
    """Returns TRUE if the number is odd"""
    try:
        return int(number) % 2 != 0
    except (TypeError, ValueError):
        return VALUE_ERROR


def ISREF(value: CellValue) -> bool:
    """Returns TRUE if the value is a reference"""
    return hasattr(value, "ref") and value.ref is not None


def ISTEXT(value: CellValue) -> bool:
    """Returns TRUE if the value is text"""
    return isinstance(value, str)


def N(value: SimpleCellValue) -> int:
    """Returns a value converted to a number"""
    if isinstance(value, int | float | complex | SpreadsheetError):
        return value
    elif isinstance(value, BooleanValue | bool):
        return int(value)
    return 0


def NA() -> SpreadsheetError:
    """Returns the error value #N/A"""
    return NA_ERROR


def SHEET() -> int:
    """Returns the sheet number of the referenced sheet"""
    # TODO: Doesn't work for SHEET(sheet_name)
    try:
        return get_ipython().parent_header["content"]["toRun"][0]["cellId"][2]
    except NameError:
        return NAME_ERROR


def SHEETS() -> int:
    """Returns the number of sheets in a reference"""
    # TODO: Doesn't work for SHEETS(reference)
    return len(stack()[2][0].f_locals["self"].cells)


_TYPE_CODES = {
    **dict.fromkeys(
        (int, float, SpreadsheetDateTime, SpreadsheetDate, SpreadsheetTime, type(None)),
        1,
    ),
    str: 2,
    **dict.fromkeys((bool, BooleanValue), 4),
    SpreadsheetError: 16,
    **dict.fromkeys((CellRange, CellRangeList, CellRangeRef), 64),
}


def TYPE(value: CellValue) -> int:
    """Returns a number indicating the data type of value"""
    if isinstance(value, str) and value == "":
        return 1
    return _TYPE_CODES.get(type(value), 2)
