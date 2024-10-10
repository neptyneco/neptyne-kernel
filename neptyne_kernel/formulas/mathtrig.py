import math
import operator
import string
from functools import reduce
from random import randint, random
from typing import Iterable

import numpy as np
import roman

from ..cell_range import CellRange
from ..spreadsheet_error import (
    NA_ERROR,
    NUM_ERROR,
    VALUE_ERROR,
    ZERO_DIV_ERROR,
    SpreadsheetError,
)
from .boolean import BooleanValue
from .helpers import (
    CellValue,
    Matrix,
    Numeric,
    SimpleCellValue,
    SimpleCellValueT,
    agg_func,
    args2positive_int,
    criteria_func,
    mat_func,
    num_func,
    prepare_crit_ranges,
    round_half_up,
    round_to_digits_func,
    sign,
)

__all__ = [
    "ABS",
    "ACOS",
    "ACOSH",
    "ACOT",
    "ACOTH",
    "ARABIC",
    "ASIN",
    "ASINH",
    "ATAN",
    "ATAN2",
    "ATANH",
    "BASE",
    "CEILING",
    "COMBIN",
    "COMBINA",
    "COS",
    "COSH",
    "COT",
    "COTH",
    "CSC",
    "CSCH",
    "DECIMAL",
    "DEGREES",
    "EVEN",
    "EXP",
    "FACT",
    "FACTDOUBLE",
    "FLOOR",
    "GCD",
    "INT",
    "ISO",
    "LCM",
    "LN",
    "LOG",
    "LOG10",
    "MDETERM",
    "MINVERSE",
    "MMULT",
    "MOD",
    "MROUND",
    "MULTINOMIAL",
    "MUNIT",
    "ODD",
    "PI",
    "POWER",
    "PRODUCT",
    "QUOTIENT",
    "RADIANS",
    "RAND",
    "RANDARRAY",
    "RANDBETWEEN",
    "ROMAN",
    "ROUND",
    "ROUNDDOWN",
    "ROUNDUP",
    "SEC",
    "SECH",
    "SERIESSUM",
    "SEQUENCE",
    "SIGN",
    "SIN",
    "SINH",
    "SQRT",
    "SQRTPI",
    "SUM",
    "SUMIF",
    "SUMIFS",
    "SUMPRODUCT",
    "SUMSQ",
    "SUMX2MY2",
    "SUMX2PY2",
    "SUMXMY2",
    "TAN",
    "TANH",
    "TRUNC",
]

_digits = string.digits + string.ascii_uppercase

_roman_chars = ["M", "D", "C", "L", "X", "V", "I"]
_roman_values = [1000, 500, 100, 50, 10, 5, 1]


@num_func(abs)
def ABS(number: Numeric) -> Numeric:
    """Returns the absolute value of a number."""
    pass


@num_func(math.acos)
def ACOS(number: Numeric) -> Numeric:
    """Returns the arccosine of a number"""
    pass


@num_func(math.acosh)
def ACOSH(number: Numeric) -> Numeric:
    """Returns the inverse hyperbolic cosine of a number"""
    pass


@num_func(lambda n: math.pi / 2 - math.atan(n))
def ACOT(number: Numeric) -> Numeric:
    """Returns the arccotangent of a number"""
    pass


@num_func(lambda n: 0.5 * math.log((n + 1) / (n - 1)))
def ACOTH(number: Numeric) -> Numeric:
    """Returns the hyperbolic arccotangent of a number"""
    pass


@num_func(lambda t: roman.fromRoman(t.upper()), VALUE_ERROR)
def ARABIC():
    """Converts a Roman number to Arabic, as a number"""
    pass


@num_func(math.asin)
def ASIN(number: Numeric) -> Numeric:
    """Returns the arcsine of a number"""
    pass


@num_func(math.asinh)
def ASINH(number: Numeric) -> Numeric:
    """Returns the inverse hyperbolic sine of a number"""
    pass


@num_func(math.atan)
def ATAN(number: Numeric) -> Numeric:
    """Returns the arctangent of a number"""
    pass


@num_func(math.atan2)
def ATAN2(x_number: Numeric, y_number: Numeric) -> Numeric:
    """Returns the arctangent from x- and y-coordinates"""
    pass


@num_func(math.atanh)
def ATANH(number: Numeric) -> Numeric:
    """Returns the inverse hyperbolic tangent of a number"""
    pass


def BASE(number: int, radix: int, min_length: int = 0) -> str:
    """Converts a number into a text representation with the given radix (base)"""
    if radix < 2 or radix > 36:
        return NUM_ERROR
    if number == 0:
        return "0"
    digits = []
    while number:
        digits.append(_digits[int(number % radix)])
        number //= radix
    str_digits = "".join(digits[::-1])
    if len(str_digits) < min_length:
        str_digits = "0" * (min_length - len(str_digits)) + str_digits
    return str_digits


@num_func(lambda num, s: s * math.ceil(float(num) / s) if s else 0, VALUE_ERROR)
def CEILING(number: Numeric, significance: Numeric = 1) -> Numeric:
    """Rounds a number to the nearest integer or to the nearest multiple of significance"""
    pass


@num_func(
    lambda n, s, m: CEILING(n, s)
    if (n >= 0 or m >= 0 or not s)
    else s * math.floor(n / s)
)
def _ceiling_math(number: Numeric, significance: Numeric = 1, mode: int = 0) -> Numeric:
    """Rounds a number up, to the nearest integer or to the nearest multiple of significance"""
    pass


@num_func(lambda num, s: (math.ceil(num / abs(s)) * abs(s)) if s else 0)
def _ceiling_precise(number: Numeric, significance: Numeric = 1) -> Numeric:
    """Rounds a number the nearest integer or to the nearest multiple of significance. Regardless of the sign of the number, the number is rounded up."""
    pass


CEILING.MATH = _ceiling_math
CEILING.PRECISE = _ceiling_precise


@num_func(lambda n, c: math.comb(int(n), int(c)) if 0 <= c <= n else NUM_ERROR)
def COMBIN(number: int, number_chosen: int) -> int:
    """Returns the number of combinations for a given number of objects"""
    pass


@num_func(lambda num, num_c: COMBIN(num + num_c - 1, num_c))
def COMBINA(number: int, number_chosen: int) -> int:
    """Returns the number of combinations for a given number of objects"""
    pass


@num_func(math.cos)
def COS(number: Numeric) -> Numeric:
    """Returns the cosine of a number"""
    pass


@num_func(math.cosh)
def COSH(number: Numeric) -> Numeric:
    """Returns the hyperbolic cosine of a number"""
    pass


@num_func(lambda num: 1 / math.tan(num) if num != 0 else ZERO_DIV_ERROR)
def COT(number: Numeric) -> Numeric:
    """Returns the cotangent of an angle"""
    pass


@num_func(lambda num: 1 / math.tanh(num) if num != 0 else ZERO_DIV_ERROR)
def COTH(number: Numeric) -> Numeric:
    """Returns the hyperbolic cotangent of a number"""
    pass


@num_func(lambda num: 1 / math.sin(num) if num != 0 else ZERO_DIV_ERROR)
def CSC(number: Numeric) -> Numeric:
    """Returns the cosecant of an angle"""
    pass


@num_func(lambda num: 1 / math.sinh(num) if num != 0 else ZERO_DIV_ERROR)
def CSCH(number: Numeric) -> Numeric:
    """Returns the hyperbolic cosecant of an angle"""
    pass


@num_func(lambda text, radix: int(str(text), radix))
def DECIMAL(text: str, radix: int) -> Numeric:
    """Converts a text representation of a number in a given base into a decimal number"""
    pass


@num_func(math.degrees)
def DEGREES(angle: Numeric) -> Numeric:
    """Converts radians to degrees"""
    pass


@num_func(
    lambda num: int((math.ceil(num / 2) if num >= 0 else math.floor(num / 2)) * 2)
)
def EVEN(number: Numeric) -> int:
    """Rounds a number up to the nearest even integer"""
    pass


@num_func(math.exp)
def EXP(number: Numeric) -> Numeric:
    """Returns e raised to the power of a given number"""
    pass


@num_func(lambda num: math.factorial(math.trunc(num)) if num >= 0 else NUM_ERROR)
def FACT(number: Numeric) -> int:
    """Returns the factorial of a number"""
    pass


@num_func(
    lambda n: reduce(operator.mul, range(math.trunc(n), 0, -2)) if n >= 0 else NUM_ERROR
)
def FACTDOUBLE(number: Numeric) -> int:
    """Returns the double factorial of a number"""
    pass


@num_func(
    lambda n, s: (s * math.floor(float(n) / s)) if n <= 0 or s >= 0 else NUM_ERROR
)
def FLOOR(number: Numeric, significance: Numeric = 1) -> Numeric:
    """Rounds a number down, toward zero"""
    pass


def _floor_math(number: Numeric, significance: Numeric = 1, mode: int = 0) -> Numeric:
    """Rounds a number up, to the nearest integer or to the nearest multiple of significance"""
    res = FLOOR(number, significance)
    if isinstance(res, SpreadsheetError):
        return res
    return res if number >= 0 or mode >= 0 else -FLOOR(-number, significance)


@num_func(lambda num, s: math.floor(num / abs(s)) * abs(s))
def _floor_precise(number: Numeric, significance: Numeric = 1) -> Numeric:
    """Rounds a number down to the nearest integer or to the nearest multiple of significance. Regardless of the sign of the number, the number is rounded down."""
    pass


FLOOR.MATH = _floor_math
FLOOR.PRECISE = _floor_precise


@args2positive_int
@num_func(math.gcd)
def GCD(value1: Numeric, *values: Numeric):
    """Returns the greatest common divisor"""
    pass


@num_func(math.floor)
def INT(number: Numeric) -> int:
    """Rounds a number down to the nearest integer"""
    pass


class ISO:
    @staticmethod
    @num_func(lambda num, s: _ceiling_math(num, abs(s), 1))
    def CEILING(number: Numeric, significance: Numeric = 1) -> Numeric:
        """Returns a number that is rounded up to the nearest integer or to the nearest multiple of significance"""
        pass


@args2positive_int
@num_func(math.lcm)
def LCM(value1: Numeric, *values: Numeric):
    """Returns the least common multiple"""
    pass


@num_func(math.log)
def LN(number: Numeric) -> Numeric:
    """Returns the natural logarithm of a number"""
    pass


@num_func(math.log)
def LOG(number: Numeric, base: Numeric = 10) -> Numeric:
    """Returns the logarithm of a number to a specified base"""
    pass


@num_func(math.log10)
def LOG10(number: Numeric) -> Numeric:
    """Returns the base-10 logarithm of a number"""
    pass


@mat_func(np.linalg.det, VALUE_ERROR)
def MDETERM(array: Matrix) -> Numeric:
    """Returns the matrix determinant of an array"""
    pass


@mat_func(np.linalg.inv, NUM_ERROR)
def MINVERSE(array: Matrix) -> CellRange:
    """Returns the matrix inverse of an array"""
    pass


@mat_func(np.dot, VALUE_ERROR)
def MMULT(array1: Matrix, array2: Matrix) -> CellRange:
    """Returns the matrix product of two arrays"""
    pass


def MOD(number, divisor) -> int:
    """Returns the remainder from division"""
    try:
        number = int(number)
        divisor = int(divisor)
    except ValueError:
        return VALUE_ERROR

    try:
        result = number % divisor
        return result
    except ZeroDivisionError:
        return ZERO_DIV_ERROR


@num_func(lambda n, m: (round_half_up(n / m) * m) if sign(n) == sign(m) else NUM_ERROR)
def MROUND(number: Numeric, multiple: Numeric) -> Numeric:
    """Returns a number rounded to the desired multiple"""
    pass


@agg_func(sum)
def SUM(value1: CellValue, *values: CellValue) -> Numeric:
    """Returns the sum of a series of numbers and/or cells."""
    pass


def MULTINOMIAL(value1: CellValue, *values: CellValue) -> Numeric:
    """Returns the multinomial of a set of numbers"""
    args = [value1, *values]
    facs = []
    for arg in args:
        try:
            if isinstance(arg, CellRange):
                fac = arg.map(math.factorial)
            else:
                try:
                    val = int(arg)
                except ValueError:
                    return VALUE_ERROR
                fac = math.factorial(val)
        except ValueError:
            return NUM_ERROR
        facs.append(fac)
    prod = PRODUCT(facs)
    if isinstance(prod, SpreadsheetError):
        return prod
    s = SUM(*args)
    if isinstance(s, SpreadsheetError):
        return s
    return math.factorial(s) / prod


@num_func(lambda dim: CellRange(np.identity(dim).tolist()))
def MUNIT(dimension: int) -> CellRange:
    """Returns the unit matrix or the specified dimension"""
    pass


def ODD(number: Numeric) -> int:
    """Rounds a number up to the nearest odd integer"""
    rounded = round_half_up(number)
    return rounded + (sign(number) if not rounded % 2 else 0)


def PI() -> float:
    """Returns the value of pi"""
    return math.pi


@num_func(operator.pow)
def POWER(number: Numeric, power: Numeric) -> Numeric:
    """Returns the result of a number raised to a power."""
    pass


@agg_func(
    lambda *args: reduce(operator.__mul__, *args), make_list=True, count_text=False
)
def PRODUCT(value1: CellValue, *values: CellValue) -> Numeric:
    """Multiplies its arguments"""
    pass


@num_func(lambda n, d: int(n / d) if d != 0 else ZERO_DIV_ERROR)
def QUOTIENT(numerator: Numeric, denominator: Numeric):
    """Returns the integer portion of a division"""
    pass


@num_func(math.radians)
def RADIANS(angle: Numeric) -> Numeric:
    """Returns the integer portion of a division"""
    pass


def RAND() -> Numeric:
    """Returns a random number between 0 and 1"""
    return random()


def RANDARRAY(
    rows: int = 1,
    columns: int = 1,
    min_val: Numeric = 0,
    max_val: Numeric = 1,
    integer: bool = False,
) -> CellRange:
    """Returns an array of random numbers between 0 and 1.
    However, you can specify the number of rows and columns to fill,
    minimum and maximum values, and whether to return whole numbers or decimal values."""
    if min_val >= max_val:
        return VALUE_ERROR
    func = np.random.randint if integer else np.random.uniform
    return CellRange(func(min_val, max_val, (rows, columns)).tolist())


@num_func(randint)
def RANDBETWEEN(bottom: int, top: int) -> int:
    """Returns a random number between the numbers you specify"""
    pass


def ROMAN(number: int, form: int | BooleanValue = 0) -> str:
    """Converts an Arabic numeral to Roman, as text"""
    if number < 1 or number > 3999 or form < 0 or form > 4:
        return VALUE_ERROR
    if form is True:
        form = 0
    elif form is False:
        form = 4

    max_index = len(_roman_values) - 1
    result = []
    for i in range(0, max_index // 2 + 1):
        ind = i * 2
        digit = number // _roman_values[ind]
        if digit % 5 == 4:
            ind2 = (ind - 1) if digit == 4 else (ind - 2)
            steps = 0
            while steps < form and ind < max_index:
                steps += 1
                if _roman_values[ind2] - _roman_values[ind + 1] <= number:
                    ind += 1
                else:
                    steps = form
            result += [_roman_chars[ind], _roman_chars[ind2]]
            number += _roman_values[ind]
            number -= _roman_values[ind2]
        else:
            if digit > 4:
                result.append(_roman_chars[ind - 1])
            result += [_roman_chars[ind]] * (digit % 5)
            number %= _roman_values[ind]

    return "".join(result)


@num_func(lambda n, d: round_to_digits_func(n, d, round_half_up))
def ROUND(number, num_digits=0):
    """Rounds a number to a specified number of digits"""
    pass


@num_func(lambda n, d: round_to_digits_func(n, d, np.floor))
def ROUNDDOWN(number, num_digits=0):
    """Rounds a number down, toward zero"""
    pass


@num_func(lambda n, d: round_to_digits_func(n, d, np.ceil))
def ROUNDUP(number, num_digits=0):
    """Rounds a number up, away from zero"""
    pass


@num_func(lambda num: 1 / math.cos(num))
def SEC(number: Numeric) -> Numeric:
    """Returns the secant of an angle"""
    pass


@num_func(lambda num: 1 / math.cosh(num))
def SECH(number: Numeric) -> Numeric:
    """Returns the hyperbolic secant of an angle"""
    pass


@num_func(
    lambda x, n, m, c: sum(
        a * x ** (n + i * m)
        for i, a in enumerate(CellRange([c] if isinstance(c, int | float) else c))
    )
)
def SERIESSUM(
    x: Numeric, n: int, m: int, coeffs: list | tuple | CellRange | Numeric
) -> Numeric:
    """Returns the sum of a power series based on the formula"""
    pass


def SEQUENCE(rows: int, columns: int = 1, start: int = 1, step: int = 1) -> CellRange:
    """Generates a list of sequential numbers in an array, such as 1, 2, 3, 4"""
    return CellRange(
        np.arange(start, rows * columns * step + start, step)
        .reshape((rows, columns))
        .tolist()
    )


@num_func(sign)
def SIGN(number: Numeric) -> int:
    """Returns the sign of a number"""
    pass


@num_func(math.sin)
def SIN(number: Numeric) -> Numeric:
    """Returns the sine of a number"""
    pass


@num_func(math.sinh)
def SINH(number: Numeric) -> Numeric:
    """Returns the hyperbolic sine of a number"""
    pass


@num_func(lambda n: math.sqrt(n) if n >= 0 else NUM_ERROR)
def SQRT(number: Numeric) -> Numeric:
    """Returns a positive square root"""
    pass


@num_func(lambda n: math.sqrt(n * math.pi) if n >= 0 else NUM_ERROR)
def SQRTPI(number: Numeric) -> Numeric:
    """Returns the square root of (number * pi)"""
    pass


def _sumifs(
    sum_range,
    criteria_range1: CellRange,
    criteria1,
    *crit_ranges: tuple[CellRange, int | float | str],
):
    prep_result = prepare_crit_ranges(criteria_range1, criteria1, *crit_ranges)
    if isinstance(prep_result, SpreadsheetError):
        return prep_result
    cranges, crit_args = prep_result

    if not all(len(sum_range) == len(cr) for cr in cranges):
        return VALUE_ERROR

    def _sum(term, c_args):
        if isinstance(term, Iterable) and not isinstance(term, str):
            return sum(_sum(a, c_arg) for a, c_arg in zip(term, c_args))
        if isinstance(term, SimpleCellValueT):
            crit = all(criteria_func(func, c) for func, c in c_args)
            if crit:
                return term
        return 0

    try:
        return _sum(sum_range, crit_args)
    except TypeError:
        return VALUE_ERROR
    except (RecursionError, ValueError):
        return NUM_ERROR


def SUMIF(
    cell_range: CellRange, criteria: SimpleCellValue, sum_range: CellRange = None
) -> Numeric:
    """Adds the cells specified by a given criteria"""
    if sum_range is None:
        sum_range = cell_range
    return _sumifs(sum_range, cell_range, criteria)


def SUMIFS(
    sum_range: CellRange,
    criteria_range: CellRange,
    criteria: SimpleCellValue,
    *crit_ranges: tuple[CellRange, SimpleCellValue],
) -> Numeric:
    """Adds the cells in a range that meet multiple criteria"""

    return _sumifs(sum_range, criteria_range, criteria, *crit_ranges)


@mat_func(
    lambda *a: np.einsum(
        ",".join(["i" if len(a[0].shape) == 1 else "ij"] * len(a)), *a
    ),
    VALUE_ERROR,
    eq_shapes=True,
    shape_error=VALUE_ERROR,
)
def SUMPRODUCT(array1: Matrix, *arrays: Matrix) -> CellRange:
    """Returns the sum of the products of corresponding array components"""
    pass


@num_func(lambda *nums: sum(num**2 for num in nums))
def SUMSQ(number1: Numeric, *numbers: Numeric) -> Numeric:
    """Returns the sum of the squares of the arguments"""
    pass


@mat_func(lambda a1, a2: np.sum(a1**2 - a2**2), NA_ERROR, eq_shapes=True)
def SUMX2MY2(array1: Matrix, array2: Matrix):
    """Returns the sum of the difference of squares of corresponding values in two arrays"""
    pass


@mat_func(lambda a1, a2: np.sum(a1**2 + a2**2), NA_ERROR, eq_shapes=True)
def SUMX2PY2(array1: Matrix, array2: Matrix):
    """Returns the sum of the sum of squares of corresponding values in two arrays"""
    pass


@mat_func(lambda a1, a2: np.sum((a1 - a2) ** 2), NA_ERROR, eq_shapes=True)
def SUMXMY2(array1: Matrix, array2: Matrix):
    """Returns the sum of squares of differences of corresponding values in two arrays"""
    pass


@num_func(math.tan)
def TAN(number: Numeric) -> Numeric:
    """Returns the tangent of a number"""
    pass


@num_func(math.tanh)
def TANH(number: Numeric) -> Numeric:
    """Returns the hyperbolic tangent of a number"""
    pass


@num_func(lambda n, d: round_to_digits_func(abs(n), d, math.floor) * sign(n))
def TRUNC(number: Numeric, num_digits=0) -> int:
    """Truncates a number to an integer"""
    pass
