import operator
from typing import Any, Callable

from ..spreadsheet_error import NA_ERROR, SpreadsheetError

__all__ = [
    "TRUE",
    "FALSE",
    "IF",
    "NOT",
    "AND",
    "OR",
    "XOR",
    "IFS",
    "IFNA",
    "IFERROR",
    "LET",
]


class BooleanValue:
    """Boolean value class"""

    def __init__(self, value: bool):
        self._value = value

    def __bool__(self):
        return self._value

    def __call__(self):
        return self._value

    def __lt__(self, other):
        return self.apply_operator(other, operator.__lt__)

    def __gt__(self, other):
        return self.apply_operator(other, operator.__gt__)

    def __ge__(self, other):
        return self.apply_operator(other, operator.__ge__)

    def __le__(self, other):
        return self.apply_operator(other, operator.__le__)

    # Pairwise operators:
    def apply_operator(self, other, op, reverse=False):
        val = int(self._value)
        return op(other, val) if reverse else op(val, other)

    def __add__(self, other):
        return self.apply_operator(other, operator.__add__)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.apply_operator(other, operator.__sub__)

    def __rsub__(self, other):
        return self.apply_operator(other, operator.__sub__, reverse=True)

    def __mul__(self, other):
        return self.apply_operator(other, operator.__mul__)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __floordiv__(self, other):
        return self.apply_operator(other, operator.__floordiv__)

    def __rfloordiv__(self, other):
        return self.apply_operator(other, operator.__floordiv__, reverse=True)

    def __truediv__(self, other):
        return self.apply_operator(other, operator.__truediv__)

    def __rtruediv__(self, other):
        return self.apply_operator(other, operator.__truediv__, reverse=True)

    def __int__(self):
        return int(self._value)

    def __repr__(self):
        return "TRUE" if self._value else "FALSE"

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        if isinstance(other, BooleanValue):
            return self._value == other._value
        else:
            return self._value == other


TRUE = BooleanValue(True)
"""Returns the logical value `TRUE`."""

FALSE = BooleanValue(False)
"""Returns the logical value `FALSE`."""


def IF(condition: bool, when_true, when_false):
    """Returns one value if a logical expression is `TRUE` and another if it is `FALSE`."""
    if condition:
        return when_true
    else:
        return when_false


def NOT(condition: bool) -> bool:
    """Returns the opposite of a logical value - `NOT(TRUE)` returns `FALSE`; `NOT(FALSE)` returns `TRUE."""
    return not condition


def AND(condition1: bool, condition2: bool) -> bool:
    """Returns True if all of the provided arguments are logically true, and False if any of the provided arguments are logically false."""
    return condition1 and condition2


def OR(condition1: bool, condition2: bool) -> bool:
    """Returns True if any of the provided arguments are logically true, and False if all of the provided arguments are logically false."""
    return condition1 or condition2


def XOR(condition1: bool, condition2: bool) -> bool:
    """Returns True if an odd number of the provided arguments are logically true, and False otherwise."""
    return condition1 ^ condition2


def IFS(logical_test1: bool, value_if_true1, *args):
    """Evaluates multiple conditions and returns a value that corresponds to the first true condition."""

    # Check if args list is even:
    if len(args) % 2:
        return NA_ERROR

    args_with_required = [logical_test1, value_if_true1, *list(args)]
    for i, cond in list(enumerate(args_with_required))[::2]:
        if cond:
            return args_with_required[i + 1]
    return NA_ERROR


def IFNA(value: Any, value_if_na: Any) -> Any:
    """Returns the value you specify if a formula returns the '#N/A error'; otherwise returns the result of the formula."""
    if value == NA_ERROR:
        return value_if_na
    elif isinstance(value, Callable):
        try:
            result = value()
            return value_if_na if result == NA_ERROR else result
        except IndexError:
            return value_if_na
    return value


def IFERROR(value: Any, value_if_error: Any) -> Any:
    """Returns the value you specify if a formula returns an error; otherwise returns the result of the formula."""
    if isinstance(value, SpreadsheetError):
        return value_if_error
    elif isinstance(value, Callable):
        try:
            result = value()
            return value_if_error if isinstance(result, SpreadsheetError) else result
        except Exception:
            return value_if_error
    return value


def LET(
    name1: Any,
    name_value1: Any,
    calculation_or_name2: Any,
    *names_vals_calculation: tuple[Any],
) -> Any:
    """Assigns names to calculation results"""
    pass
