# ruff: noqa: F405
import pytest

from ..cell_range import CellRange
from ..spreadsheet_error import *  # noqa: F403
from .helpers import assert_equal
from .mathtrig import *  # noqa: F403
from .stats import SUBTOTAL
from .test_helpers import approx_or_error


@pytest.mark.parametrize(
    "args, result",
    [
        [(CellRange([1, 2, 3]),), 6],
        [(CellRange([[1, 1], [2, 2]]),), 6],
        [(CellRange([1, 2, 3]), 1, 1), 8],
        [(CellRange(["hello", "world", 10])), 10],
    ],
)
def test_SUM(args, result):
    assert SUM(*args) == pytest.approx(result)


@pytest.mark.parametrize("number,power,result", [(2, 3, 8), (0.3e1, 3, 27), (0, 0, 1)])
def test_POWER(number, power, result):
    assert POWER(number, power) == pytest.approx(result)


@pytest.mark.parametrize("number,result", [(-2, 2), (-0.3e1, 3), (-15000, 15000)])
def test_ABS(number, result):
    assert ABS(number) == pytest.approx(result)


def test_math_errors():
    assert CEILING("n2", "3.4hi") == VALUE_ERROR
    assert POWER("n2", "3.4hi") == VALUE_ERROR
    assert FLOOR(2.5, -2) == NUM_ERROR


@pytest.mark.parametrize(
    "args, result",
    [
        ((5, 15, 30), 2250),
        ((CellRange([5, 15, 30]), 2), 4500),
        ((CellRange(["hello", 2, "world"]), 3), 6),
        ((CellRange([[2, 3], [4, 5]]),), 120),
    ],
)
def test_PRODUCT(args, result):
    assert PRODUCT(*args) == pytest.approx(result)


@pytest.mark.parametrize(
    "args, result",
    [
        ((1, CellRange([1, 2, 3]), CellRange([4, 5, 6])), 3.5),
        ((2, CellRange([1, 2, 3]), CellRange([4, 5, 6])), 6),
        ((3, CellRange([1, 2, 3]), CellRange([4, 5, 6])), 6),
        ((4, CellRange([1, 2, 3]), CellRange([4, 5, 6])), 6),
        ((5, CellRange([1, 2, 3]), CellRange([4, 5, 6])), 1),
        ((6, CellRange([1, 2, 3]), CellRange([4, 5, 6])), 720),
        ((7, CellRange([1, 2, 3]), CellRange([4, 5, 6])), 1.87083),
        ((8, CellRange([1, 2, 3]), CellRange([4, 5, 6])), 1.7078251276599),
        ((9, CellRange([1, 2, 3]), CellRange([4, 5, 6])), 21),
        ((10, CellRange([1, 2, 3]), CellRange([4, 5, 6])), 3.5),
        ((11, CellRange([1, 2, 3]), CellRange([4, 5, 6])), 2.9166667),
    ],
)
def test_SUBTOTAL(args, result):
    assert SUBTOTAL(*args) == pytest.approx(result)


def test_PI():
    assert PI() == pytest.approx(3.1415926, abs=1e-3)
    assert PI() / 2 == pytest.approx(1.570796327, abs=1e-3)


def test_ACOS():
    assert ACOS(-0.5) == pytest.approx(2.094395102, abs=1e-3)
    assert ACOS(-0.5) * 180 / PI() == pytest.approx(120, abs=1e-3)
    assert DEGREES(ACOS(-0.5)) == pytest.approx(120, abs=1e-3)


@pytest.mark.parametrize(
    "number, result",
    [
        (1, 0),
        (10, 2.9932228),
    ],
)
def test_ACOSH(number, result):
    assert ACOSH(number) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "number, result",
    [
        (2, 0.4636),
    ],
)
def test_ACOT(number, result):
    assert ACOT(number) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "number, result",
    [
        (6, 0.168),
    ],
)
def test_ACOTH(number, result):
    assert ACOTH(number) == pytest.approx(result, abs=1e-3)


@pytest.mark.parametrize(
    "text, result", [("LVII", 57), ("mcmxii", 1912), ("hello", VALUE_ERROR)]
)
def test_ARABIC(text, result):
    assert ARABIC(text) == result


@pytest.mark.parametrize(
    "number, result",
    [(-0.5, -0.523598776), (1, 1.570796327), (0, 0), (-0.4, -0.4115168461)],
)
def test_ASIN(number, result):
    assert ASIN(number) == pytest.approx(result, abs=1e-3)


@pytest.mark.parametrize(
    "number, result",
    [
        (-2.5, -1.647231146),
        (10, 2.99822295),
        (1, 0.881373587),
        (0, 0),
        (-0.4, -0.3900353198),
    ],
)
def test_ASINH(number, result):
    assert ASINH(number) == pytest.approx(result, abs=1e-3)


@pytest.mark.parametrize(
    "number, result", [(1, 0.785398163), (0.5, 0.463647609), (0, 0), (-3, -1.249045772)]
)
def test_ATAN(number, result):
    assert ATAN(number) == pytest.approx(result, abs=1e-3)


@pytest.mark.parametrize(
    "x_num, y_num, result", [(1, 1, 0.785398163), (-1, -1, -2.35619449)]
)
def test_ATAN2(x_num, y_num, result):
    assert ATAN2(x_num, y_num) == pytest.approx(result, abs=1e-3)


@pytest.mark.parametrize(
    "number, result", [(0.76159416, 1.00000001), (-0.1, -0.100335348)]
)
def test_ATANH(number, result):
    assert ATANH(number) == pytest.approx(result, abs=1e-3)


@pytest.mark.parametrize(
    "number, radix, min_length, result",
    [
        (7, 2, 0, "111"),
        (100, 16, 0, "64"),
        (15, 2, 10, "0000001111"),
        (255, 16, 0, "FF"),
        (21, 2, 0, "10101"),
        (4095, 16, 6, "000FFF"),
    ],
)
def test_BASE(number, radix, min_length, result):
    assert BASE(number, radix, min_length) == result


@pytest.mark.parametrize(
    "number,significance,result",
    [
        (2.5, 1, 3),
        (-2.5, -2, -4),
        (-2.5, 2, -2),
        (1.5, 0.1, 1.5),
        (0.234, 0.01, 0.24),
    ],
)
def test_CEILING(number, significance, result):
    assert CEILING(number, significance) == pytest.approx(result)


@pytest.mark.parametrize(
    "number,significance,result",
    [
        (-10.5, 1, -10),
        (96, 10, 100),
        (-23.25, 0.1, -23.2),
        (4.3, 1, 5),
        (-4.3, 1, -4),
        (4.3, 2, 6),
        (4.3, -2, 6),
        (-4.3, 2, -4),
        (-4.3, -2, -4),
    ],
)
def test_CEILING_PRECISE(number, significance, result):
    assert CEILING.PRECISE(number, significance) == pytest.approx(result)


@pytest.mark.parametrize(
    "number,significance,mode, result",
    [
        (11.2, 1, 0, 12),
        (-8.8, 1, 0, -8),
        (7.7, 0.2, 0, 7.8),
        (-10.2, 2, -1, -12),
        (-42, 10, -1, -50),
        (24.3, 5, 0, 25),
        (6.7, 1, 0, 7),
        (-8.1, 2, 0, -8),
        (-5.5, 2, -1, -6),
    ],
)
def test_CEILING_MATH(number, significance, mode, result):
    assert CEILING.MATH(number, significance, mode) == pytest.approx(result)


@pytest.mark.parametrize(
    "number, number_chosen, result",
    [
        (8, 2, 28),
        (4, 2, 6),
        (10, 7, 120),
        (2, 10, NUM_ERROR),
        (3, 0, 1),
        (0, 5, NUM_ERROR),
    ],
)
def test_COMBIN(number, number_chosen, result):
    assert COMBIN(number, number_chosen) == result


@pytest.mark.parametrize(
    "number, number_chosen, result",
    [(2, 2, 3), (5, 3, 35), (2, 10, 11), (3, 0, 1), (4, 3, 20), (10, 3, 220)],
)
def test_COMBINA(number, number_chosen, result):
    assert COMBINA(number, number_chosen) == result


@pytest.mark.parametrize("number, result", [(1.047, 0.5001711), (60 * PI() / 180, 0.5)])
def test_COS(number, result):
    assert COS(number) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize("number, result", [(4, 27.308233), (2.718281828, 7.6101251)])
def test_COSH(number, result):
    assert COSH(number) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "number, result",
    [
        (30, -0.156),
        (45, 0.617),
        (0, ZERO_DIV_ERROR),
        (1, 0.6420926159),
        (-1, -0.6420926159),
        (4, 0.8636911545),
    ],
)
def test_COT(number, result):
    assert COT(number) == approx_or_error(result)


@pytest.mark.parametrize(
    "number, result",
    [
        (2, 1.037),
        (0, ZERO_DIV_ERROR),
        (1, 1.313035285),
        (-1, -1.313035285),
        (4, 1.00067115),
    ],
)
def test_COTH(number, result):
    assert COTH(number) == approx_or_error(result)


@pytest.mark.parametrize(
    "number, result",
    [
        (15, 1.538),
        (0, ZERO_DIV_ERROR),
        (1, 1.188395106),
        (-1, -1.188395106),
        (4, -1.321348709),
    ],
)
def test_CSC(number, result):
    assert CSC(number) == approx_or_error(result)


@pytest.mark.parametrize(
    "number, result",
    [
        (1.5, 0.4696),
        (0, ZERO_DIV_ERROR),
        (1, 0.8509181282),
        (-1, -0.8509181282),
        (4, 0.03664357033),
    ],
)
def test_CSCH(number, result):
    assert CSCH(number) == approx_or_error(result)


@pytest.mark.parametrize(
    "text, radix, result", [("FF", 16, 255), (111, 2, 7), ("zap", 36, 45745)]
)
def test_DECIMAL(text, radix, result):
    assert DECIMAL(text, radix) == pytest.approx(result)


def test_DEGREES():
    assert DEGREES(PI()) == 180


@pytest.mark.parametrize(
    "number, result",
    [(1.5, 2), (3, 4), (2, 2), (-1, -2), (826.745, 828), (-110.2, -112), (110.2, 112)],
)
def test_EVEN(number, result):
    assert EVEN(number) == result


@pytest.mark.parametrize(
    "number, result",
    [
        (1, 2.71828183),
        (2, 7.3890561),
        (5, 148.4131591),
        (5.1, 164.0219073),
        (-2, 0.1353352832),
    ],
)
def test_EXP(number, result):
    assert EXP(number) == pytest.approx(result)


@pytest.mark.parametrize(
    "number, result", [(5, 120), (1.9, 1), (0, 1), (-1, NUM_ERROR), (1, 1), (4.2, 24)]
)
def test_FACT(number, result):
    assert FACT(number) == result


@pytest.mark.parametrize(
    "number, result", [(7, 105), (6, 48), (3, 3), (-1, NUM_ERROR), (4, 8)]
)
def test_FACTDOUBLE(number, result):
    assert FACTDOUBLE(number) == result


@pytest.mark.parametrize(
    "number,significance,result",
    [
        (3.7, 2, 2),
        (-2.5, -2, -2),
        (1.58, 0.1, 1.5),
        (0.234, 0.01, 0.23),
    ],
)
def test_FLOOR(number, significance, result):
    assert FLOOR(number, significance) == pytest.approx(result)


@pytest.mark.parametrize(
    "number,significance,mode, result",
    [
        (24.3, 5, 0, 20),
        (6.7, 1, 0, 6),
        (-8.1, 2, 0, -10),
        (-5.5, 2, -1, -4),
        (11.2, 1, 0, 11),
        (-8.8, 1, 0, -9),
        (7.7, 0.2, 0, 7.6),
        (-10.2, 2, 0, -12),
        (-42, 10, -1, -40),
    ],
)
def test_FLOOR_MATH(number, significance, mode, result):
    assert FLOOR.MATH(number, significance, mode) == pytest.approx(result)


@pytest.mark.parametrize(
    "number,significance,result",
    [
        (-3.2, -1, -4),
        (3.2, 1, 3),
        (-3.2, 1, -4),
        (3.2, -1, 3),
        (3.2, 1, 3),
        (-10.5, 1, -11),
        (96, 10, 90),
        (-23.25, 0.1, -23.3),
    ],
)
def test_FLOOR_PRECISE(number, significance, result):
    assert FLOOR.PRECISE(number, significance) == pytest.approx(result)


@pytest.mark.parametrize(
    "args, result",
    [
        ((5, 2), 1),
        ((24, 36), 12),
        ((7, 1), 1),
        ((5, 0), 5),
        ((10, 100), 10),
        ((18, 24), 6),
        ((7, 9), 1),
        ((14.4, 21.7, 42.3), 7),
        ((-4, -8), NUM_ERROR),
    ],
)
def test_GCD(args, result):
    assert GCD(*args) == result


@pytest.mark.parametrize(
    "number, result",
    [(8.9, 8), (-8.9, -9), (10, 10), (6.18, 6), (-6.18, -7), (2.49, 2), (-2.49, -3)],
)
def test_INT(number, result):
    assert INT(number) == result


@pytest.mark.parametrize(
    "number,significance,result",
    [
        (4.3, 1, 5),
        (-4.3, 1, -4),
        (4.3, 2, 6),
        (4.3, -2, 6),
        (-4.3, 2, -4),
        (-4.3, -2, -4),
        (0, 1, 0),
        (1, 0, 0),
    ],
)
def test_ISO_CEILING(number, significance, result):
    assert ISO.CEILING(number, significance) == pytest.approx(result)


@pytest.mark.parametrize(
    "args, result",
    [
        ((5, 2), 10),
        ((24, 36), 72),
        ((10, 100), 100),
        ((12, 18), 36),
        ((12, 18, 24), 72),
        ((-4, -8), NUM_ERROR),
    ],
)
def test_LCM(args, result):
    assert LCM(*args) == result


@pytest.mark.parametrize(
    "number, result",
    [
        (86, 4.4543473),
        (2.7182818, 1),
        (EXP(3), 3),
        (100, 4.605170186),
        (50, 3.912023005),
    ],
)
def test_LN(number, result):
    assert LN(number) == pytest.approx(result)


@pytest.mark.parametrize(
    "number, base, result", [(10, 10, 1), (8, 2, 3), (86, 2.7182818, 4.4543473)]
)
def test_LOG(number, base, result):
    assert LOG(number, base) == pytest.approx(result)


@pytest.mark.parametrize("number, result", [(86, 1.9345), (10, 1), (100000, 5)])
def test_LOG10(number, result):
    assert LOG10(number) == pytest.approx(result)


@pytest.mark.parametrize(
    "cell_range, result",
    [
        (CellRange([[2, 1], [4, 3]]), 2),
        (CellRange([[1, 3, 8, 5], [1, 3, 6, 1], [1, 1, 1, 0], [7, 3, 10, 2]]), 88),
        ([[3, 6, 1], [1, 1, 0], [3, 10, 2]], 1),
        ([[3, 6], [1, 1]], -3),
    ],
)
def test_MDETERM(cell_range, result):
    assert MDETERM(cell_range) == pytest.approx(result)


@pytest.mark.parametrize(
    "cell_range,result",
    [
        (CellRange([[2, 1], [4, 3]]), CellRange([[1.5, -0.5], [-2, 1]])),
        (CellRange([[4, -1], [2, 0]]), CellRange([[0, 0.5], [-1, 2]])),
    ],
)
def test_MINVERSE(cell_range, result):
    assert (MINVERSE(cell_range) == result).all()


@pytest.mark.parametrize(
    "cell_range1, cell_range2, result",
    [
        (
            CellRange([[1, 3], [7, 2]]),
            CellRange([[2, 0], [0, 2]]),
            CellRange([[2, 6], [14, 4]]),
        ),
        (
            CellRange([[1, -2, 3], [-3, 1, 2]]),
            CellRange([[2, 2], [1, -3], [-1, 4]]),
            CellRange([[-3, 20], [-7, -1]]),
        ),
    ],
)
def test_MMULT(cell_range1, cell_range2, result):
    assert (MMULT(cell_range1, cell_range2) == result).all()


@pytest.mark.parametrize(
    "number, divisor, result",
    [
        (3, 2, 1),
        (-3, 2, 1),
        (3, -2, -1),
        (-3, -2, -1),
        (5, 0, ZERO_DIV_ERROR),
        ("one", 2, VALUE_ERROR),
        (1, "two", VALUE_ERROR),
        ("13", "10", 3),
    ],
)
def test_MOD(number, divisor, result):
    assert_equal(MOD(number, divisor), result)


@pytest.mark.parametrize(
    "number,multiple,result",
    [
        (34.74, 0.1, 34.7),
        (10, 3, 9),
        (-10, -3, -9),
        (1.3, 0.2, 1.4),
        (34.75, 0.1, 34.8),
        (12.24, 0.1, 12.2),
        (12.25, 0.1, 12.3),
        (11, 3, 12),
        (-9, -3, -9),
        (1.5, 0.2, 1.6),
    ],
)
def test_MROUND(number, multiple, result):
    assert_equal(MROUND(number, multiple), pytest.approx(result))


@pytest.mark.parametrize(
    "values, result",
    [((2, 3, 4), 1260), ((3,), 1), ((1, 2, 3), 60), ((CellRange([0, 2]), 4, 6), 13860)],
)
def test_MULTINOMIAL(values, result):
    assert_equal(MULTINOMIAL(*values), pytest.approx(result))


@pytest.mark.parametrize(
    "values, result", [(("hello", 3, 4), VALUE_ERROR), ((-1, 2, 3), NUM_ERROR)]
)
def test_MULTINOMIAL_errors(values, result):
    assert_equal(MULTINOMIAL(*values), result)


@pytest.mark.parametrize(
    "dimension, result",
    [
        (2, CellRange([[1, 0], [0, 1]])),
        (3, CellRange([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        (1, CellRange([[1]])),
    ],
)
def test_MUNIT(dimension, result):
    assert (MUNIT(dimension) == result).all()


@pytest.mark.parametrize(
    "number, result",
    [
        (1.5, 3),
        (3, 3),
        (2, 3),
        (-1, -1),
        (-2, -3),
        (826.745, 827),
        (-110.2, -111),
        (110.2, 111),
    ],
)
def test_ODD(number, result):
    assert_equal(ODD(number), result)


@pytest.mark.parametrize(
    "numerator, denominator, result", [(5, 2, 2), (4.5, 3.1, 1), (-10, 3, -3)]
)
def test_QUOTIENT(numerator, denominator, result):
    assert_equal(QUOTIENT(numerator, denominator), result)


@pytest.mark.parametrize(
    "angle, result",
    [(270, 4.712389), (45, 0.7853981634), (77, 1.343903524), (-30, -0.5235987756)],
)
def test_RADIANS(angle, result):
    assert_equal(RADIANS(angle), pytest.approx(result))


def test_RAND():
    for i in range(100):
        assert 1 > RAND() >= 0


@pytest.mark.parametrize(
    "rows,columns,min_val,max_val,integer",
    [
        (2, 3, 9, 20, True),
        (1, 10, 2, 3, False),
    ],
)
def test_RANDARRAY(rows, columns, min_val, max_val, integer):
    r = RANDARRAY(rows, columns, min_val, max_val, integer)
    assert r.shape == (rows, columns)
    for i in range(rows):
        for j in range(columns):
            assert max_val >= r[i][j] >= min_val
            assert isinstance(r[i][j], int if integer else float)


@pytest.mark.parametrize("bottom, top", [(-10, 10), (0, 1), (-100, 100)])
def test_RANDBETWEEN(bottom, top):
    for i in range(100):
        assert top >= RANDBETWEEN(bottom, top) >= bottom


@pytest.mark.parametrize("number, result", [(499, "CDXCIX"), (3999, "MMMCMXCIX")])
def test_ROMAN_noform(number, result):
    assert_equal(ROMAN(number), result)


@pytest.mark.parametrize(
    "number, form, result",
    [
        (499, 0, "CDXCIX"),
        (499, 1, "LDVLIV"),
        (499, 2, "XDIX"),
        (499, 3, "VDIV"),
        (499, 4, "ID"),
        (3999, 0, "MMMCMXCIX"),
        (3999, 1, "MMMLMVLIV"),
        (3999, 2, "MMMXMIX"),
        (3999, 3, "MMMVMIV"),
        (3999, 4, "MMMIM"),
    ],
)
def test_ROMAN(number, form, result):
    assert ROMAN(number, form) == result


@pytest.mark.parametrize(
    "number, num_digits, result",
    [
        (2.15, 1, 2.2),
        (2.149, 1, 2.1),
        (-1.475, 2, -1.48),
        (21.5, -1, 20),
        (626.3, -3, 1000),
        (1.98, -1, 0),
        (-50.55, -2, -100),
        (826.645, 0, 827),
        (826.645, 1, 826.6),
        (826.645, 2, 826.65),
        (826.645, 3, 826.645),
        (826.645, -1, 830),
        (826.645, -2, 800),
    ],
)
def test_ROUND(number, num_digits, result):
    assert ROUND(number, num_digits) == pytest.approx(result)


@pytest.mark.parametrize(
    "number, num_digits, result",
    [
        (826.646, 0, 826),
        (826.646, 1, 826.6),
        (826.646, 2, 826.64),
        (826.646, -1, 820),
        (826.646, -2, 800),
    ],
)
def test_ROUNDDOWN(number, num_digits, result):
    assert ROUNDDOWN(number, num_digits) == pytest.approx(result)


@pytest.mark.parametrize(
    "number, num_digits, result",
    [
        (826.446, 0, 827),
        (826.446, 1, 826.5),
        (826.446, 2, 826.45),
        (826.446, -1, 830),
        (826.446, -2, 900),
    ],
)
def test_ROUNDUP(number, num_digits, result):
    assert ROUNDUP(number, num_digits) == pytest.approx(result)


@pytest.mark.parametrize(
    "number, result", [(1, 1.850815718), (-1, 1.850815718), (4, -1.529885656), (0, 1)]
)
def test_SEC(number, result):
    assert SEC(number) == pytest.approx(result)


@pytest.mark.parametrize(
    "number, result", [(1, 0.6480542737), (-1, 0.6480542737), (4, 0.03661899347)]
)
def test_SECH(number, result):
    assert SECH(number) == pytest.approx(result)


@pytest.mark.parametrize(
    "x, n, m, coeffs, result",
    [(1, 0, 1, 1, 1), (2, 1, 0, [1, 2, 3], 12), (-3, 1, 1, [2, 4, 6], -132)],
)
def test_SERIESSUM(x, n, m, coeffs, result):
    assert SERIESSUM(x, n, m, coeffs) == pytest.approx(result)


@pytest.mark.parametrize(
    "rows, columns, start, step, result",
    [
        (2, 1, 1, 1, CellRange([[1], [2]])),
        (2, 3, 1, 1, CellRange([[1, 2, 3], [4, 5, 6]])),
        (2, 3, 3, 2, CellRange([[3, 5, 7], [9, 11, 13]])),
        # see: https://www.pivotaltracker.com/story/show/182291912
        # (2, 3, 10, -1, CellRange([[10, 9, 8], [7, 6, 5], [4, 3, 2]])),
    ],
)
def test_SEQUENCE(rows, columns, start, step, result):
    assert_equal(SEQUENCE(rows, columns, start, step), result)


@pytest.mark.parametrize("number, result", [(10, 1), (4 - 4, 0), (-0.00001, -1)])
def test_SIGN(number, result):
    assert_equal(SIGN(number), result)


@pytest.mark.parametrize(
    "number, result",
    [
        (PI(), 0.0),
        (PI() / 2, 1.0),
        (30 * PI() / 180, 0.5),
        (1, 0.8414709848),
        (-1, -0.8414709848),
        (4, -0.7568024953),
        (0, 0),
    ],
)
def test_SIN(number, result):
    assert SIN(number) == pytest.approx(result)


@pytest.mark.parametrize(
    "number, result", [(1, 1.175201194), (0, 0), (-0.4, -0.4107523258)]
)
def test_SINH(number, result):
    assert SINH(number) == pytest.approx(result)


@pytest.mark.parametrize("number, result", [(16, 4), (0, 0), (4, 2), (-4, NUM_ERROR)])
def test_SQRT(number, result):
    assert SQRT(number) == result


@pytest.mark.parametrize(
    "number, result", [(1, 1.772454), (2, 2.506628), (PI(), 3.141592654)]
)
def test_SQRTPI(number, result):
    assert SQRTPI(number) == pytest.approx(result)


TEST_SUMIF_RANGE = CellRange(
    [
        [100000, 7000, 250000],
        [200000, 14000, None],
        [300000, 21000, None],
        [400000, 28000, None],
    ]
)


@pytest.mark.parametrize(
    "cell_range,criterion,sum_range,result",
    [
        (TEST_SUMIF_RANGE[:, 0], ">160000", TEST_SUMIF_RANGE[:, 1], 63000),
        (TEST_SUMIF_RANGE[:, 0], ">160000", None, 900000),
        (TEST_SUMIF_RANGE[:, 0], 300000, TEST_SUMIF_RANGE[:, 1], 21000),
        (
            TEST_SUMIF_RANGE[:, 0],
            ">" + str(TEST_SUMIF_RANGE[0][2]),
            TEST_SUMIF_RANGE[:, 1],
            49000,
        ),
    ],
)
def test_SUMIF(cell_range, criterion, sum_range, result):
    assert SUMIF(cell_range, criterion, sum_range) == pytest.approx(result)


TEST_SUMIFS_RANGE = CellRange(
    [
        [5, "Apples", "Tom"],
        [4, "Apples", "Sarah"],
        [15, "Artichokes", "Tom"],
        [3, "Artichokes", "Sarah"],
        [22, "Bananas", "Tom"],
        [12, "Bananas", "Sarah"],
        [10, "Carrots", "Tom"],
        [33, "Carrots", "Sarah"],
    ]
)


@pytest.mark.parametrize(
    "sum_range,crit_range,result",
    [
        (
            TEST_SUMIFS_RANGE[:, 0],
            (TEST_SUMIFS_RANGE[:, 1], "=A*", TEST_SUMIFS_RANGE[:, 2], "Tom"),
            20,
        ),
        (
            TEST_SUMIFS_RANGE[:, 0],
            (TEST_SUMIFS_RANGE[:, 1], "<>Bananas", TEST_SUMIFS_RANGE[:, 2], "Tom"),
            30,
        ),
        (
            TEST_SUMIFS_RANGE[:, 0],
            (TEST_SUMIFS_RANGE[:, 1], "<>Ca??o*", TEST_SUMIFS_RANGE[:, 2], "Sarah"),
            19,
        ),
    ],
)
def test_SUMIFS(sum_range, crit_range, result):
    assert SUMIFS(sum_range, *crit_range) == pytest.approx(result)


TEST_SUMPRODUCT_RANGE = CellRange([[3, 4], [2, 5], [1, 6]])


@pytest.mark.parametrize(
    "arrays, result",
    [
        ((TEST_SUMPRODUCT_RANGE[:, 0], TEST_SUMPRODUCT_RANGE[:, 1]), 28),
        (([2, 4, 7], [1, 6, 9]), 89),
        (([1, 2, 3], [4]), VALUE_ERROR),
    ],
)
def test_SUMPRODUCT(arrays, result):
    assert SUMPRODUCT(*arrays) == result


@pytest.mark.parametrize("numbers, result", [((3, 4), 25), ((2, 6), 40)])
def test_SUMSQ(numbers, result):
    assert SUMSQ(*numbers) == result


TEST_SUMXY_RANGE = CellRange([[2, 6], [3, 5], [9, 11], [1, 7], [8, 5], [7, 4], [5, 4]])


@pytest.mark.parametrize(
    "array1, array2, result",
    [
        (TEST_SUMXY_RANGE[:, 0], TEST_SUMXY_RANGE[:, 1], -55),
        ([2, 3, 9, 1, 8, 7, 5], [6, 5, 11, 7, 5, 4, 4], -55),
    ],
)
def test_SUMX2MY2(array1, array2, result):
    assert SUMX2MY2(array1, array2) == pytest.approx(result)


@pytest.mark.parametrize(
    "array1, array2, result",
    [
        (TEST_SUMXY_RANGE[:, 0], TEST_SUMXY_RANGE[:, 1], 521),
        ([2, 3, 9, 1, 8, 7, 5], [6, 5, 11, 7, 5, 4, 4], 521),
    ],
)
def test_SUMX2PY2(array1, array2, result):
    assert SUMX2PY2(array1, array2) == pytest.approx(result)


@pytest.mark.parametrize(
    "array1, array2, result",
    [
        (TEST_SUMXY_RANGE[:, 0], TEST_SUMXY_RANGE[:, 1], 79),
        ([2, 3, 9, 1, 8, 7, 5], [6, 5, 11, 7, 5, 4, 4], 79),
    ],
)
def test_SUMXMY2(array1, array2, result):
    assert SUMXMY2(array1, array2) == pytest.approx(result)


@pytest.mark.parametrize(
    "number, result",
    [(1, 1.557407725), (-1, -1.557407725), (4, 1.157821282), (0, 0), (0.785, 0.99920)],
)
def test_TAN(number, result):
    assert TAN(number) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "number, result",
    [
        (-2, -0.964028),
        (0, 0),
        (0.5, 0.462117),
        (1, 0.761594156),
        (-1, -0.761594156),
        (4, 0.9993292997),
    ],
)
def test_TANH(number, result):
    assert TANH(number) == pytest.approx(result)


@pytest.mark.parametrize(
    "number, ndig, result",
    [
        (8.9, 0, 8),
        (-8.9, 0, -8),
        (0.45, 0, 0),
    ],
)
def test_TRUNC(number, ndig, result):
    assert TRUNC(number, ndig) == pytest.approx(result)
