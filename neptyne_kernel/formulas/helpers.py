import math
import operator
import re
import statistics
from functools import wraps
from typing import Callable, Iterable

import numpy as np
import roman

from ..cell_range import CellRange
from ..primitives import check_none
from ..spreadsheet_datetime import SpreadsheetDateTime
from ..spreadsheet_error import (
    NA_ERROR,
    NUM_ERROR,
    VALUE_ERROR,
    ZERO_DIV_ERROR,
    SpreadsheetError,
)
from .boolean import BooleanValue

Numeric = int | float
Matrix = list | CellRange
CellValue = (
    str
    | int
    | float
    | SpreadsheetDateTime
    | BooleanValue
    | CellRange
    | SpreadsheetError
)
SimpleCellValue = (
    str | int | float | SpreadsheetDateTime | BooleanValue | SpreadsheetError
)

SimpleCellValueT = (
    float,
    int,
    str,
    BooleanValue,
    SpreadsheetDateTime,
    SpreadsheetError,
)

BOOL_OPERATORS = {
    ">=": operator.__ge__,
    "<=": operator.__le__,
    "<>": operator.__ne__,
    ">": operator.__gt__,
    "<": operator.__lt__,
    "=": operator.__eq__,
}

_MAX_COL_IDX = 18278
_ALPHABET_LEN = 26


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


def get_column_letter(idx: int) -> str:
    """Convert a column index into a column letter
    (3 -> 'C')
    """
    try:
        return _STRING_COL_CACHE[idx]
    except KeyError:
        raise ValueError(f"Invalid column index {idx}")


class SpreadsheetErrorException(Exception):
    pass


def _flatten_range(
    cell_range: CellRange, none_to_zero=False, scan_by_column=False, ignore_errors=False
):
    def get_value(val):
        return 0 if check_none(val) is None and none_to_zero else val

    def check_error(val):
        return not (ignore_errors and isinstance(val, SpreadsheetError))

    if not len(cell_range):
        return
    if isinstance(cell_range[0], CellRange):
        if scan_by_column:
            for j in range(cell_range.shape[1]):
                for val in cell_range[:, j]:
                    if check_error(val):
                        yield get_value(val)

        else:
            for row in cell_range:
                for val in row:
                    if check_error(val):
                        yield get_value(val)
    else:
        for val in cell_range:
            if check_error(val):
                yield get_value(val)


def sign(n: Numeric):
    return int(math.copysign(1, n)) if n else 0


def round_half_up(number: Numeric) -> int:
    if (float(number) % 1) >= 0.5:
        return math.ceil(number) if number >= 0 else -math.ceil(-number)
    else:
        return round(number)


def round_to_decimals(number: Numeric, num_digits: int = 2) -> Numeric:
    if num_digits:
        fac = 10**num_digits
    else:
        fac = 1.0
    if number < 0:
        number = math.ceil(number * fac - 0.5) / fac
    else:
        number = math.floor(number * fac + 0.5) / fac
    if num_digits < 0:
        num_digits = 0

    return number, num_digits


def round_to_digits_func(number: Numeric, num_digits: int, func: Callable) -> Numeric:
    tens = 10**num_digits
    return func(number * tens) / tens


def args2positive_int(func):
    def wrapper(*args):
        int_args = []
        for arg in args:
            if arg >= 0:
                try:
                    int_args.append(int(arg))
                except ValueError:
                    return VALUE_ERROR
            else:
                return NUM_ERROR
        return func(*int_args)

    return wrapper


def num_func(func, error=NUM_ERROR):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except TypeError:
                return VALUE_ERROR
            except (ValueError, roman.InvalidRomanNumeralError):
                return error

        return wrapper

    return decorator


def mat_func(
    func,
    error,
    eq_shapes: list | bool | None = None,
    args_to_numpy: list | bool = True,
    shape_error=None,
    count_bool=True,
):
    """
    :param func: function to appy
    :param error: error if func throws an exception
    :param eq_shapes: list of argument indexes with equal shapes, True if all arguments should have equal shapes
    :param args_to_numpy: which arguments will be converted to ndarray
    :param shape_error: error if matrix shapes are not equal and len(eq_shape)>0
    :return:
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args):
            to_np = (
                []
                if args_to_numpy is None
                else range(len(args))
                if args_to_numpy is True
                else args_to_numpy
            )
            try:
                new_args = [
                    arg
                    if arg is None or ind not in to_np
                    else cellrange2np(arg, count_bool)
                    if isinstance(arg, CellRange)
                    else cellrange2np(
                        CellRange(
                            [arg]
                            if isinstance(arg, float | int | BooleanValue)
                            else arg
                        ),
                        count_bool,
                    )
                    for ind, arg in enumerate(args)
                ]
            except SpreadsheetErrorException as e:
                return e.args[0]

            eq_shapes_ind = (
                []
                if eq_shapes is None
                else to_np  # range(len(new_args))
                if eq_shapes is True
                else eq_shapes
            )
            if eq_shapes_ind:
                shape = new_args[eq_shapes_ind[0]].shape
                for ind in eq_shapes_ind[1:]:
                    if ind >= len(new_args):
                        break
                    if new_args[ind] is not None and new_args[ind].shape != shape:
                        return error if shape_error is None else shape_error

            try:
                result = func(*new_args)
                if isinstance(result, np.ndarray):
                    result = CellRange(result.tolist())
                return result
            except SpreadsheetErrorException as e:
                return e.args[0]
            except (TypeError, ValueError):
                return error

        return wrapper

    return decorator


def agg_func(
    func,
    make_list=False,
    count_text=True,
    count_bool=True,
    count_empty=False,
    bool_as_num=False,
    result_if_zero=0,
):
    def decorator(f):
        @wraps(f)
        def wrapper(*args):
            def flatten(args):
                for arg in args:
                    if isinstance(arg, str):
                        if count_text:
                            yield 0
                        else:
                            continue
                    elif check_none(arg):
                        if count_empty:
                            yield 0
                        else:
                            continue
                    elif isinstance(arg, BooleanValue):
                        if count_bool:
                            yield int(arg) if bool_as_num else 0
                    elif isinstance(arg, Iterable):
                        yield from flatten(arg)
                    elif isinstance(arg, SpreadsheetError):
                        raise SpreadsheetErrorException(arg)
                    else:
                        yield arg

            flattened_args = flatten(args)
            if make_list:
                try:
                    flattened_args = [*flattened_args]
                except SpreadsheetErrorException as e:
                    return e.args[0]
                # This is not quite correct, Excel formulas return a value error on an empty list
                # but tend to return 0 on a list of strings where numbers are expected
                if not flattened_args:
                    return result_if_zero

            try:
                return func(flattened_args)
            except SpreadsheetErrorException as e:
                return e.args[0]
            except TypeError:
                return VALUE_ERROR
            except (RecursionError, ValueError):
                return NUM_ERROR
            except statistics.StatisticsError:
                return NA_ERROR

        return wrapper

    return decorator


def is_number(value: str) -> bool:
    """Check if value can be parsed into number"""
    stripped_value = value.strip()
    try:
        float(stripped_value)
        return True
    except ValueError:
        return False


def to_number(value: str):
    """Convert string value into a number"""
    stripped_value = value.strip()
    try:
        return int(stripped_value)
    except ValueError:
        try:
            return float(stripped_value)
        except ValueError:
            return VALUE_ERROR


def re_from_wildcard(wildcard: str) -> str:
    re_pattern = []
    i = 0
    while i < len(wildcard):
        if wildcard[i] == "~":
            if i < len(wildcard) - 1:
                if wildcard[i + 1] == "?":
                    re_pattern += ["\\", "?"]
                    i += 1
                elif wildcard[i + 1] == "*":
                    re_pattern += ["\\", "*"]
                    i += 1
                else:
                    re_pattern.append("~")
            else:
                re_pattern.append("~")
        elif wildcard[i] == "*":
            re_pattern += [".", "*"]
        elif wildcard[i] == "?":
            re_pattern.append(".")
        elif wildcard[i] in ".^$+{}\\[]|()":
            re_pattern += ["\\", wildcard[i]]
        else:
            re_pattern.append(wildcard[i])
        i += 1
    return "".join(re_pattern)


def search_wildcard(source: str, pattern: str, func=operator.__eq__) -> bool:
    """Wildcard search in source string"""
    regexp = re_from_wildcard(pattern)
    if regexp != pattern:
        try:
            r = re.compile(regexp)
            m = r.fullmatch(str(source)) is None
            return m if func == operator.__ne__ else not m
        except re.error:
            return func(source, pattern)
    return func(source, pattern)


def criteria_func(func, arg):
    try:
        return func(arg)
    except TypeError:
        return False


def parse_criteria(criteria: str | int | float, empty_as_none=False):
    """Parses criterion string and returns parsed function and argument"""
    if criteria is None:
        if empty_as_none:
            return None
        criteria = 0

    func = None

    arg = criteria
    if isinstance(criteria, str):
        if not criteria and empty_as_none:
            return None
            # return lambda x: True
        for bool_op, f in BOOL_OPERATORS.items():
            if criteria.startswith(bool_op):
                if is_number(criteria[len(bool_op) :]):
                    arg = to_number(criteria[len(bool_op) :])
                    func = f
                else:
                    arg = criteria[len(bool_op) :]

                    def func(s, p):
                        return search_wildcard(s, p, f)

                break

        if not func:
            func = search_wildcard
            arg = criteria
    elif isinstance(criteria, int | float):
        func = operator.__eq__
        arg = criteria

    return lambda x: func(x, arg)


def prepare_crit_ranges(
    criteria_range1: CellRange,
    criteria1: SimpleCellValue,
    *crit_ranges_criterias: tuple[CellRange, SimpleCellValue],
):
    crit_cranges = [criteria_range1, criteria1, *crit_ranges_criterias]
    crits = crit_cranges[1::2]
    cranges = crit_cranges[::2]

    if len(crits) != len(cranges):
        return VALUE_ERROR

    parsed_crits = []
    for c in crits:
        func = parse_criteria(c)
        if not func:
            return ZERO_DIV_ERROR
        parsed_crits.append(func)

    crit_args = []
    for args in zip(*cranges):
        cr = []
        for arg in zip(parsed_crits, args):
            cr.append(arg)
        crit_args.append(cr)

    return cranges, crit_args


def cellrange2np(cell_range, count_bool=True, raise_on_error=True):
    """Return self as a numpy array with 0 for non-numeric entries."""

    def as_number(n):
        if isinstance(n, float | int):
            return n
        elif isinstance(n, BooleanValue):
            return int(n)
        elif raise_on_error and isinstance(n, SpreadsheetError):
            raise SpreadsheetErrorException(n)
        return 0

    dtypes = [int, float]
    if count_bool:
        dtypes.append(BooleanValue)
    if raise_on_error:
        dtypes.append(SpreadsheetError)
    dtypes = tuple(dtypes)
    if cell_range.two_dimensional:
        return np.asarray([np.array(x) for x in cell_range])
    return np.asarray([as_number(n) for n in cell_range if isinstance(n, dtypes)])


def assert_equal(value, expected):
    if isinstance(value, CellRange) or isinstance(expected, CellRange):
        assert (value == expected).all()
    else:
        assert value == expected
