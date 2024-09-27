# ruff: noqa: F405
import pytest

from ..cell_range import CellRange
from ..spreadsheet_datetime import *  # noqa: F403
from ..spreadsheet_error import *  # noqa: F403
from .boolean import *  # noqa: F403
from .helpers import assert_equal, cellrange2np
from .stats import *  # noqa: F403


@pytest.mark.parametrize(
    "args, result",
    [
        ((CellRange([4, 5, 6, 7, 5, 4, 3])), 1.020408),
        ((CellRange([5, 7, -2, 4, 0])), 3.04),
        ((5, 7, -2, 4, 0), 3.04),
    ],
)
def test_AVEDEV(args, result):
    return AVEDEV(*args) == pytest.approx(result)


@pytest.mark.parametrize(
    "args, result",
    [
        [(CellRange([1, 2, 3, None, None]),), 2],
        [(CellRange([1, 2, 3]),), 2],
        [(CellRange([[1, 1], [2, 2]]),), 1.5],
        [(CellRange([1, 2, 3]), 1, 1), 1.6],
        [(CellRange(["hello", "world", 10, -20])), -5],
    ],
)
def test_AVERAGE(args, result):
    assert AVERAGE(*args) == pytest.approx(result)


def test_AVERAGE_zero_div():
    assert AVERAGE(CellRange(["hello", "world"])) == ZERO_DIV_ERROR
    assert AVERAGE([]) == ZERO_DIV_ERROR


TEST_AVERAGEA_RANGE = CellRange([2, -1, 11, "Google", TRUE, FALSE])


@pytest.mark.parametrize(
    "args, result",
    [
        [(CellRange([1, 2, 3]),), 2],
        [(CellRange([[1, 1], [2, 2]]),), 1.5],
        [(CellRange([1, 2, 3]), 1, 1), 1.6],
        [(CellRange(["hello", "world", 10, -20])), -2.5],
        [(TEST_AVERAGEA_RANGE[:3]), 4],
        [(TEST_AVERAGEA_RANGE[:4]), 3],
        [
            (TEST_AVERAGEA_RANGE[:4], TEST_AVERAGEA_RANGE[3], TEST_AVERAGEA_RANGE[4]),
            2.166666667,
        ],
        [(CellRange([TRUE, FALSE])), 0.5],
    ],
)
def test_AVERAGEA(args, result):
    assert AVERAGEA(*args) == pytest.approx(result)


def test_AVERAGEA_zero_div():
    assert AVERAGEA(CellRange([None])) == ZERO_DIV_ERROR
    assert AVERAGEA([]) == ZERO_DIV_ERROR


TEST_AVERAGEIF_1 = CellRange(
    [[100000, 7000], [200000, 14000], [300000, 21000], [400000, 28000]]
)
TEST_AVERAGEIF_2 = CellRange(
    [
        ["East", 45678],
        ["West", 23789],
        ["North", -4789],
        ["South (New Office)", 0],
        ["MidWest", 9678],
    ]
)
TEST_AVERAGEIF_3 = CellRange(
    [
        ["East", 45678],
        ["<?PWest>", 23789],
        ["North", -4789],
        ["South (New Office)", 0],
        ["MidWest", 9678],
    ]
)


@pytest.mark.parametrize(
    "cell_range, criteria, average_range, result",
    [
        (TEST_AVERAGEIF_1[:, 1], "<23000", None, 14000),
        (TEST_AVERAGEIF_1[:, 0], "<250000", None, 150000),
        (TEST_AVERAGEIF_1[:, 0], ">250000", TEST_AVERAGEIF_1[:, 1], 24500),
        (TEST_AVERAGEIF_2[:, 0], "=*West", TEST_AVERAGEIF_2[:, 1], 16733.5),
        (TEST_AVERAGEIF_2[:, 0], "<>*(New Office)", TEST_AVERAGEIF_2[:, 1], 18589),
        (TEST_AVERAGEIF_3[:, 0], "=*West*", TEST_AVERAGEIF_3[:, 1], 16733.5),
    ],
)
def test_AVERAGEIF(cell_range, criteria, average_range, result):
    assert AVERAGEIF(cell_range, criteria, average_range) == pytest.approx(
        result, rel=1e-3
    )


@pytest.mark.parametrize(
    "cell_range, criteria, average_range, result",
    [
        (TEST_AVERAGEIF_1[:, 0], "<95000", None, ZERO_DIV_ERROR),
    ],
)
def test_AVERAGEIF_errors(cell_range, criteria, average_range, result):
    assert AVERAGEIF(cell_range, criteria, average_range) == result


TEST_AVERAGEIFS_1 = CellRange(
    [
        ["Emilio", 75, 85, 87],
        ["Julie", 94, 80, 88],
        ["Hans", 86, 93, "Incomplete"],
        ["Frederique", "Incomplete", 75, 75],
    ]
)


@pytest.mark.parametrize(
    "args, result",
    [
        (
            (
                TEST_AVERAGEIFS_1[:, 1],
                TEST_AVERAGEIFS_1[:, 1],
                ">70",
                TEST_AVERAGEIFS_1[:, 1],
                "<90",
            ),
            80.5,
        ),
        (
            (
                TEST_AVERAGEIFS_1[:, 3],
                TEST_AVERAGEIFS_1[:, 3],
                "<>Incomplete",
                TEST_AVERAGEIFS_1[:, 3],
                ">80",
            ),
            87.5,
        ),
    ],
)
def test_AVERAGEIFS(args, result):
    assert AVERAGEIFS(*args) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "args, result",
    [
        ((TEST_AVERAGEIFS_1[:, 2], TEST_AVERAGEIFS_1[:, 2], ">95"), ZERO_DIV_ERROR),
    ],
)
def test_AVERAGEIFS_errors(args, result):
    assert AVERAGEIFS(*args) == result


@pytest.mark.parametrize(
    "x, alpha, beta, c, A, B, result",
    [
        (2, 8, 10, TRUE, 1, 3, 0.6854706),
        (2, 8, 10, FALSE, 1, 3, 1.4837646),
    ],
)
def test_BETA_DIST(x, alpha, beta, c, A, B, result):
    assert BETA.DIST(x, alpha, beta, c, A, B) == pytest.approx(result)


@pytest.mark.parametrize(
    "p, alpha, beta, A, B, result",
    [(0.6854706, 8, 10, 1, 3, 2), (0.13, 5, 1, -1, 1, 0.33)],
)
def test_BETA_INV(p, alpha, beta, A, B, result):
    assert BETA.INV(p, alpha, beta, A, B) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "ns, nt, p, c, result",
    [
        (6, 10, 0.5, FALSE, 0.2050781),
        (9, 20, 0.5, 0, 0.1601791382),
        (9, 20, 0.5, TRUE, 0.411901474),
    ],
)
def test_BINOM_DIST(ns, nt, p, c, result):
    assert BINOM.DIST(ns, nt, p, c) == pytest.approx(result)


@pytest.mark.parametrize(
    "nt,ps,tp, result",
    [
        (6, 0.5, 0.75, 4),
        (8, 0.6, 0.9, 7),
        (100, 0.5, 0.2, 46),
        (100, 0.5, 0.5, 50),
        (100, 0.5, 0.9, 56),
    ],
)
def test_BINOM_INV(nt, ps, tp, result):
    assert BINOM.INV(nt, ps, tp) == pytest.approx(result)


@pytest.mark.parametrize(
    "nt, pb, ns, ns2, result",
    [
        (100, 0.5, 45, None, 0.04847429663),
        (100, 0.5, 30, 45, 0.1840847287),
        (100, 0.5, 30, None, 0.00002317069058),
        (60, 0.75, 48, None, 0.084),
        (60, 0.75, 45, 50, 0.524),
    ],
)
def test_BINOM_DIST_RANGE(nt, pb, ns, ns2, result):
    assert BINOM.DIST.RANGE(nt, pb, ns, ns2) == pytest.approx(result, rel=1e-2)


@pytest.mark.parametrize(
    "x, df, c, result",
    [
        (0.5, 1, TRUE, 0.52049988),
        (2, 3, FALSE, 0.20755375),
        (12.3, 5, FALSE, 0.01223870353),
    ],
)
def test_CHISQ_DIST(x, df, c, result):
    assert CHISQ.DIST(x, df, c) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, df, result", [(18.307, 10, 0.0500006), (12.3, 5, 0.03090046464)]
)
def test_CHISQ_DIST_RT(x, df, result):
    assert CHISQ.DIST.RT(x, df) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "p, df, result", [(0.050001, 10, 18.306973), (0.05, 4, 9.487729037)]
)
def test_CHISQ_INV_RT(p, df, result):
    assert CHISQ.INV.RT(p, df) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "act, exp, result",
    [
        (
            CellRange([[58, 35], [11, 25], [10, 23]]),
            CellRange([[45.35, 47.65], [17.56, 18.44], [16.09, 16.91]]),
            0.0003082,
        ),
        (
            CellRange([11, 15, 8, 10, 2, 14]),
            CellRange([10, 10, 10, 10, 10, 10]),
            0.05137998348,
        ),
    ],
)
def test_CHISQ_TEST(act, exp, result):
    assert CHISQ.TEST(act, exp) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "alpha, stdev, size, result",
    [(0.05, 2.5, 50, 0.692952), (0.05, 6.48, 27, 2.444225188)],
)
def test_CONFIDENCE_NORM(alpha, stdev, size, result):
    assert CONFIDENCE.NORM(alpha, stdev, size) == pytest.approx(result)


@pytest.mark.parametrize(
    "alpha, stdev, size, result",
    [(0.05, 1, 50, 0.284196855), (0.05, 6.43, 27, 2.543623284)],
)
def test_CONFIDENCE_T(alpha, stdev, size, result):
    assert CONFIDENCE.T(alpha, stdev, size) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "array1, array2, result",
    [
        ([3, 2, 4, 5, 6], [9, 7, 12, 15, 17], 0.997),
    ],
)
def test_CORREL(array1, array2, result):
    assert CORREL(array1, array2) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "args, result",
    [
        [(CellRange([1, 2, 3])), 3],
        [(CellRange([[1, 1], [2, 2]])), 4],
        [(CellRange([1, 2, 3]), 1, 1), 5],
        [(CellRange(["hello", "world", 10])), 1],
        [
            (
                SpreadsheetDateTime("10/10/2022"),
                1.2,
                3,
                "hello",
                SpreadsheetTime("21:40"),
            ),
            4,
        ],
        [(2 < 1, 7, "hi"), 2],
    ],
)
def test_COUNT(args, result):
    assert COUNT(args) == result


@pytest.mark.parametrize(
    "args, result",
    [
        [(CellRange([39790, 19, 22.24, TRUE, ZERO_DIV_ERROR])), 5],
        [(CellRange(["hello", "", 22.24, ZERO_DIV_ERROR, None])), 4],
    ],
)
def test_COUNTA(args, result):
    assert COUNTA(args) == result


@pytest.mark.parametrize(
    "args, result",
    [((CellRange([[6, ""], [None, 27], [4, 34]]),), 2), ((1, 2, 3, 4, 5), 0)],
)
def test_COUNTBLANK(args, result):
    assert COUNTBLANK(*args) == result


TEST_COUNTIF_RANGE = CellRange(
    [
        ["apples", 32],
        ["oranges", 54],
        ["peaches", 75],
        ["apples", 86],
        [None, None],
        [None, "apples"],
    ]
)


@pytest.mark.parametrize(
    "cell_range, criterion, result",
    [
        (TEST_COUNTIF_RANGE[:, 0], TEST_COUNTIF_RANGE[2][0], 1),
        (TEST_COUNTIF_RANGE[:, 0], TEST_COUNTIF_RANGE[0][0], 2),
        (TEST_COUNTIF_RANGE[:, 1], "<>" + str(TEST_COUNTIF_RANGE[2][1]), 4),
        (TEST_COUNTIF_RANGE[:, 1], ">55", 2),
        (TEST_COUNTIF_RANGE[:, 1], ">=32", 4),
        (TEST_COUNTIF_RANGE[:, 1], ">85", 1),
        (TEST_COUNTIF_RANGE[:, 0], "*", 4),
        (TEST_COUNTIF_RANGE[:, 0], "?????es", 2),
        (TEST_COUNTIF_RANGE, "apples", 3),
    ],
)
def test_COUNTIF(cell_range, criterion, result):
    assert COUNTIF(cell_range, criterion) == result


TEST_COUNTIFS_1 = CellRange(
    [
        ["Davidoski", "Yes", "No", "No"],
        ["Burke", "Yes", "Yes", "No"],
        ["Sundaram", "Yes", "Yes", "Yes"],
        ["Levitan", "No", "Yes", "Yes"],
    ]
)


@pytest.mark.parametrize(
    "args, result",
    [
        ((TEST_COUNTIFS_1[0][1:], "=Yes"), 1),
        (
            (
                TEST_COUNTIFS_1[:, 1],
                "=Yes",
                TEST_COUNTIFS_1[:, 2],
                "=Yes",
            ),
            2,
        ),
        (
            (
                TEST_COUNTIFS_1[-1][1:],
                "=Yes",
                TEST_COUNTIFS_1[1][1:],
                "=Yes",
            ),
            1,
        ),
    ],
)
def test_COUNTIFS(args, result):
    assert COUNTIFS(*args) == pytest.approx(result, rel=1e-3)


TEST_COVP_RANGE = CellRange([[3, 9], [2, 7], [4, 12], [5, 15], [6, 17]])


@pytest.mark.parametrize(
    "array1, array2, result",
    [
        (TEST_COVP_RANGE[:, 0], TEST_COVP_RANGE[:, 1], 5.2),
    ],
)
def test_COVARIANCE_P(array1, array2, result):
    assert COVARIANCE.P(array1, array2) == pytest.approx(result)


@pytest.mark.parametrize(
    "array1, array2, result",
    [
        ([2, 4, 8], [5, 11, 12], 9.666666667),
        (CellRange([2, 4, 8, TRUE]), CellRange([5, 11, 12, FALSE]), 9.666666667),
    ],
)
def test_COVARIANCE_S(array1, array2, result):
    assert COVARIANCE.S(array1, array2) == pytest.approx(result)


@pytest.mark.parametrize(
    "array1, array2, result",
    [
        (
            CellRange(
                [
                    1,
                ]
            ),
            CellRange([2, 3]),
            NA_ERROR,
        ),
        (CellRange([1, 2, 3]), CellRange([1, 2, 3, 3]), NA_ERROR),
        (
            CellRange(
                [
                    1,
                ]
            ),
            CellRange(
                [
                    1,
                ]
            ),
            ZERO_DIV_ERROR,
        ),
    ],
)
def test_COVARIANCE_errors(array1, array2, result):
    assert COVARIANCE.S(array1, array2) == result
    assert COVARIANCE.P(array1, array2) == result


@pytest.mark.parametrize(
    "args, result",
    [
        ((4, 5, 8, 7, 11, 4, 3), 48),
        ((3, 5, 6, 8, 10), 29.2),
        ((3, 5, 6, 8, 10, 4, 9), 41.71428571),
        ((2, 5, 8, 13, 10, TRUE, FALSE, "hello", "world"), 73.2),
        ((2, 5, 8, 13, 10, TRUE, FALSE), 73.2),
        ((2, 5, 8, 13, 10), 73.2),
    ],
)
def test_DEVSQ(args, result):
    assert DEVSQ(*args) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, lmb, c, result", [(0.2, 10, TRUE, 0.86466472), (0.2, 10, FALSE, 1.35335283)]
)
def test_EXPON_DIST(x, lmb, c, result):
    assert EXPON.DIST(x, lmb, c) == pytest.approx(result)


@pytest.mark.parametrize(
    "x, lmb, c, result", [(-0.2, 10, TRUE, NUM_ERROR), (0.2, 0, FALSE, NUM_ERROR)]
)
def test_EXPON_DIST_errors(x, lmb, c, result):
    assert EXPON.DIST(x, lmb, c) == result


@pytest.mark.parametrize(
    "x, df1, df2, c, result",
    [
        (15.2069, 6, 4, TRUE, 0.99),
        (15.2069, 6, 4, FALSE, 0.0012238),
    ],
)
def test_F_DIST(x, df1, df2, c, result):
    assert F.DIST(x, df1, df2, c) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, df1, df2, result", [(15.2068649, 6, 4, 0.01), (15.35, 7, 6, 0.001930553)]
)
def test_F_DIST_RT(x, df1, df2, result):
    assert F.DIST.RT(x, df1, df2) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, df1, df2, result", [(0.01, 6, 4, 0.10930991), (0.95, 4, 5, 5.192167773)]
)
def test_F_INV(x, df1, df2, result):
    assert F.INV(x, df1, df2) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "p, df1, df2, result", [(0.01, 6, 4, 15.20686), (0.05, 4, 5, 5.192167773)]
)
def test_F_INV_RT(p, df1, df2, result):
    assert F.INV.RT(p, df1, df2) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "a1, a2, result",
    [
        (CellRange([6, 7, 9, 15, 21]), CellRange([20, 28, 31, 38, 40]), 0.64831785),
        (
            CellRange([92, 75, 97, 85, 87, 82, 79]),
            CellRange([84, 89, 87, 95, 82, 71]),
            0.8600520777,
        ),
        (
            CellRange([92, 75, 97, 85, 87, 82, 79]),
            CellRange([84, 89, 87, 95, 82, 71, TRUE, FALSE]),
            0.8600520777,
        ),
        (
            [28, 26, 31, 23, 20, 27, 28, 14, 4, 0, 2, 8, 9],
            [19, 13, 12, 5, 34, 31, 31, 12, 24, 23, 19, 10, 33],
            0.6341228293,
        ),
    ],
)
def test_F_TEST(a1, a2, result):
    assert F.TEST(a1, a2) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize("x, result", [(0.75, 0.9729551), (0.67832234592208, 0.826)])
def test_FISHER(x, result):
    assert FISHER(x) == pytest.approx(result)


@pytest.mark.parametrize("y, result", [(0.9729551, 0.75), (0.826, 0.67832234592208)])
def test_FISHERINV(y, result):
    assert FISHERINV(y) == pytest.approx(result)


TEST_FORECAST_RANGE = CellRange(
    [
        [1, 15.53],
        [2, 19.99],
        [3, 20.43],
        [4, 21.18],
        [5, 25.93],
        [6, 30],
        [7, 30],
        [8, 34.01],
        [9, 36.47],
    ]
)


@pytest.mark.parametrize(
    "x, Y, X, result",
    [
        (30, [6, 7, 9, 15, 21], [20, 28, 31, 38, 40], 10.607253),
        (10, TEST_FORECAST_RANGE[:, 1], TEST_FORECAST_RANGE[:, 0], 38.76388889),
        (11, TEST_FORECAST_RANGE[:, 1], TEST_FORECAST_RANGE[:, 0], 41.32688889),
        (12, TEST_FORECAST_RANGE[:, 1], TEST_FORECAST_RANGE[:, 0], 43.88988889),
    ],
)
def test_FORECAST(x, Y, X, result):
    assert FORECAST(x, Y, X) == pytest.approx(result)
    assert FORECAST.LINEAR(x, Y, X) == pytest.approx(result)


@pytest.mark.parametrize(
    "da, ba, result",
    [
        (
            [79, 85, 78, 85, 50, 81, 95, 88, 97],
            [70, 79, 89],
            CellRange([[1], [2], [4], [2]]),
        ),
        (
            [18, 30, 90, 91, 35, 27, 75, 28, 58],
            [25, 50, 75],
            CellRange([[1], [4], [2], [2]]),
        ),
        # ([79,85,78,85,50,81,95,88,97],[],CellRange([[0],[9]])),
        # ([],[70,79,89],CellRange([[0],[0],[0],[0]])),
    ],
)
def test_FREQUENCY(da, ba, result):
    assert (FREQUENCY(da, ba) == result).all()


@pytest.mark.parametrize("x, result", [(2.5, 1.329), (-3.75, 0.268)])
def test_GAMMA(x, result):
    assert GAMMA(x) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, alpha, beta, c, result",
    [
        (10.00001131, 9, 2, FALSE, 0.032639),
        (10.00001131, 9, 2, TRUE, 0.068094),
        (5, 3.14, 2, FALSE, 0.1276550316),
        (2, 1, 1, TRUE, 0.8646647),
    ],
)
def test_GAMMA_DIST(x, alpha, beta, c, result):
    assert GAMMA.DIST(x, alpha, beta, c) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, alpha, beta, result",
    [(0.068094, 9, 2, 10.0000112), (0.3, 5, 1, 3.633609083)],
)
def test_GAMMA_INV(x, alpha, beta, result):
    assert GAMMA.INV(x, alpha, beta) == pytest.approx(result, rel=1e-3)


def test_GAMMA_errors():
    assert GAMMA(0) == NUM_ERROR
    assert GAMMA("number") == VALUE_ERROR
    assert GAMMA(-2) == NUM_ERROR
    assert GAMMA.DIST(-1, 1, 1, TRUE) == NUM_ERROR
    assert GAMMA.DIST(1, 0, 1, TRUE) == NUM_ERROR
    assert GAMMA.DIST(1, 1, 0, TRUE) == NUM_ERROR
    assert GAMMA.DIST("number", 1, 1, TRUE) == VALUE_ERROR
    assert GAMMA.INV(-1, 1, 1) == NUM_ERROR
    assert GAMMA.INV(1.0001, 1, 1) == NUM_ERROR
    assert GAMMA.INV(1, 0, 1) == NUM_ERROR
    assert GAMMA.INV(1, 1, 0) == NUM_ERROR
    assert GAMMA.INV("number", 1, 1) == VALUE_ERROR


@pytest.mark.parametrize(
    "x, result", [(4, 1.7917595), (4.5, 2.453736571), (3, 0.6931471806)]
)
def test_GAMMALN(x, result):
    assert GAMMALN(x) == pytest.approx(result, rel=1e-3)
    assert GAMMALN.PRECISE(x) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize("x, e", [(-1, NUM_ERROR), ("number", VALUE_ERROR)])
def test_GAMMALN_errors(x, e):
    assert GAMMALN(x) == e
    assert GAMMALN.PRECISE(x) == e


@pytest.mark.parametrize(
    "x, result", [(2, 0.47725), (1, 0.3413447461), (-1, -0.3413447461), (0, 0)]
)
def test_GAUSS(x, result):
    assert GAUSS(x) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "args, result",
    [
        ((4, 5, 8, 7, 11, 4, 3), 5.476987),
    ],
)
def test_GEOMEAN(args, result):
    assert GEOMEAN(*args) == pytest.approx(result)


@pytest.mark.parametrize(
    "args, result",
    [((-4, 5, 8, 7, 11, 4, 3), NUM_ERROR), (("hello", "world"), 0)],
)
def test_GEOMEAN_error(args, result):
    assert GEOMEAN(*args) == result


TEST_GROWTH_RANGE = CellRange(
    [
        [11, 33100],
        [12, 47300],
        [13, 69000],
        [14, 102000],
        [15, 150000],
        [16, 220000],
        [17, 320197],
        [18, 468536],
    ]
)


@pytest.mark.parametrize(
    "y, x, nx, b, result",
    [
        (
            TEST_GROWTH_RANGE[:6, 1],
            TEST_GROWTH_RANGE[:6, 0],
            TEST_GROWTH_RANGE[6:, 0],
            TRUE,
            CellRange([320197, 468536]),
        ),
        (
            TEST_GROWTH_RANGE[:6, 1],
            TEST_GROWTH_RANGE[:6, 0],
            None,
            TRUE,
            CellRange([32618, 47729, 69841, 102197, 149542, 218822]),
        ),
    ],
)
def test_GROWTH(y, x, nx, b, result):
    assert cellrange2np(GROWTH(y, x, nx, b)) == pytest.approx(
        cellrange2np(result), rel=1e-3
    )


def test_cellrange2np():
    arg = [
        CellRange([2310, 2333, 2356, 2379, 2402, 2425, 2448, 2471, 2494, 2517, 2540]),
        CellRange([2, 2, 3, 3, 2, 4, 2, 2, 3, 4, 2]),
        CellRange([2, 2, 1.5, 2, 3, 2, 1.5, 2, 3, 4, 3]),
        CellRange([20, 12, 33, 43, 53, 23, 99, 34, 23, 55, 22]),
    ]
    res = cellrange2np(CellRange(arg), True)
    assert res.shape == (4, 11)


@pytest.mark.parametrize(
    "args, result",
    [
        ((4, 5, 8, 7, 11, 4, 3), 5.028376),
    ],
)
def test_HARMEAN(args, result):
    assert HARMEAN(*args) == pytest.approx(result)


@pytest.mark.parametrize(
    "args, result",
    [((-4, 5, 8, 7, 11, 4, 3), NUM_ERROR), (("hello", "world"), NUM_ERROR)],
)
def test_HARMEAN_error(args, result):
    assert HARMEAN(*args) == result


@pytest.mark.parametrize(
    "s, ns, p, np, c, result",
    [
        (1, 4, 8, 20, TRUE, 0.4654),
        (1, 4, 8, 20, FALSE, 0.3633),
    ],
)
def test_HYPGEOM_DIST(s, ns, p, np, c, result):
    assert HYPGEOM.DIST(s, ns, p, np, c) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize("s, ns, p, np, c, result", [(-1, 4, 8, 20, TRUE, NUM_ERROR)])
def test_HYPGEOM_DIST_errors(s, ns, p, np, c, result):
    assert HYPGEOM.DIST(s, ns, p, np, c) == result


@pytest.mark.parametrize(
    "y, x, result",
    [
        ([2, 3, 9, 1, 8], [6, 5, 11, 7, 5], 0.0483871),
        (CellRange([4, 5, 10, 23]), CellRange([1, 1.7, 2.3, 5]), -1.891067538),
        ([2, 4, 6], [1, 2, 3], 0),
    ],
)
def test_INTERCEPT(y, x, result):
    assert INTERCEPT(y, x) == pytest.approx(result)


@pytest.mark.parametrize(
    "args, result",
    [
        ((3, 4, 5, 2, 3, 4, 5, 6, 4, 7), -0.151799637),
        ((2, 5, 8, 13, 10, TRUE, FALSE, "hello", "world"), -0.860879692),
        ((2, 5, 8, 13, 10, TRUE, FALSE), -0.860879692),
        ((2, 5, 8, 13, 10), -0.860879692),
    ],
)
def test_KURT(args, result):
    assert KURT(*args) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "args, result",
    [
        ((3, 3, 3, 3, 3), ZERO_DIV_ERROR),
        ((TRUE, FALSE, "hello", "world"), ZERO_DIV_ERROR),
        ((1, 2, 3), ZERO_DIV_ERROR),
    ],
)
def test_KURT_errors(args, result):
    assert KURT(*args) == result


TEST_LARGE_RANGE = CellRange([[3, 4], [5, 2], [3, 4], [5, 6], [4, 7]])


@pytest.mark.parametrize(
    "array, k, result",
    [
        (TEST_LARGE_RANGE, 3, 5),
        (TEST_LARGE_RANGE, 7, 4),
        (TEST_LARGE_RANGE, 14, NUM_ERROR),
        (TEST_LARGE_RANGE, -1, NUM_ERROR),
    ],
)
def test_LARGE(array, k, result):
    assert LARGE(array, k) == result


TEST_LINEST_1 = CellRange(
    [[1.5, 0, 0.1, 2], [9, 4, 4.1, 4], [5, 2, 2.1, 6], [7, 3, 3.1, 3]]
)
TEST_LINEST_2 = CellRange(
    [
        [2310, 2, 2, 20, 142000],
        [2333, 2, 2, 12, 144000],
        [2356, 3, 1.50, 33, 151000],
        [2379, 3, 2, 43, 150000],
        [2402, 2, 3, 53, 139000],
        [2425, 4, 2, 23, 169000],
        [2448, 2, 1.50, 99, 126000],
        [2471, 2, 2, 34, 142900],
        [2494, 3, 3, 23, 163000],
        [2517, 4, 4, 55, 169000],
        [2540, 2, 3, 22, 149000],
    ]
)


TEST_LINEST_2_TRUE_RESULT = CellRange(
    [
        [-234.2371645, 2553.21066, 12529.76817, 27.64138737, 52317.83051],
        [13.26801148, 530.6691519, 400.0668382, 5.429374042, 12237.3616],
        [0.996747993, 970.5784629, NA_ERROR, NA_ERROR, NA_ERROR],
        [459.7536742, 6, NA_ERROR, NA_ERROR, NA_ERROR],
        [1732393319, 5652135.316, NA_ERROR, NA_ERROR, NA_ERROR],
    ]
)

TEST_LINEST_2_FALSE_RESULT = CellRange(
    [
        [-250.1350324, 1248.795225, 12540.26186, 50.71193751, 0],
        [23.71906247, 808.5983029, 745.0407108, 1.11437735, NA_ERROR],
        [0.999907674, 1807.533164, NA_ERROR, NA_ERROR, NA_ERROR],
        [18952.72317, 7, NA_ERROR, NA_ERROR, NA_ERROR],
        [2.47688e11, 22870232.98, NA_ERROR, NA_ERROR, NA_ERROR],
    ]
)

TEST_LINEST_3 = CellRange(
    [
        [2310, 2333, 2356, 2379, 2402, 2425, 2448, 2471, 2494, 2517, 2540],
        [2, 2, 3, 3, 2, 4, 2, 2, 3, 4, 2],
        [2, 2, 1.5, 2, 3, 2, 1.5, 2, 3, 4, 3],
        [20, 12, 33, 43, 53, 23, 99, 34, 23, 55, 22],
        [
            142000,
            144000,
            151000,
            150000,
            139000,
            169000,
            126000,
            142900,
            163000,
            169000,
            149000,
        ],
    ]
)


TEST_LINEST_4 = CellRange([[1, 2, 3, 5], [3, 4, 7, 9], [5, 6, 11, 13]])
TEST_LINEST_4_FALSE_RESULT = CellRange(
    [
        [2.230769231, 0],
        [0.050357975, NA_ERROR],
        [0.997458489, 0.480384461],
        [1962.333333, 5],
        [452.8461538, 1.153846154],
    ]
)


@pytest.mark.parametrize(
    "y, x, c, s, result",
    [
        (
            TEST_LINEST_1[:, 0],
            TEST_LINEST_1[:, 1],
            TRUE,
            TRUE,
            CellRange(
                [
                    [1.871428571, 1.414285714],
                    [0.049487166, 0.133248272],
                    [0.998603433, 0.146385011],
                    [1430.083333, 2],
                    [30.64464286, 0.042857143],
                ]
            ),
        ),
        (
            TEST_LINEST_1[:, 0],
            TEST_LINEST_1[:, 1],
            TRUE,
            FALSE,
            CellRange([1.871428571, 1.414285714]),
        ),
        (
            TEST_LINEST_1[:, 0],
            TEST_LINEST_1[:, 1],
            FALSE,
            TRUE,
            CellRange(
                [
                    [2.310344828, 0],
                    [0.168048178, NA_ERROR],
                    [0.984375857, 0.904967136],
                    [189.0105263, 3],
                    [154.7931034, 2.456896552],
                ]
            ),
        ),
        (
            TEST_LINEST_1[:, 0],
            TEST_LINEST_1[:, 1],
            FALSE,
            FALSE,
            CellRange([2.310344828, 0]),
        ),
        # Multivariate
        (
            TEST_LINEST_2[:, -1],
            TEST_LINEST_2[:, :-1],
            TRUE,
            TRUE,
            TEST_LINEST_2_TRUE_RESULT,
        ),
        (
            TEST_LINEST_2[:, -1],
            TEST_LINEST_2[:, :-1],
            FALSE,
            TRUE,
            TEST_LINEST_2_FALSE_RESULT,
        ),
        (TEST_LINEST_3[-1], TEST_LINEST_3[:-1], TRUE, TRUE, TEST_LINEST_2_TRUE_RESULT),
        (
            TEST_LINEST_3[-1],
            TEST_LINEST_3[:-1],
            FALSE,
            TRUE,
            TEST_LINEST_2_FALSE_RESULT,
        ),
        # Multicolumn
        (
            TEST_LINEST_4[:, 2:],
            TEST_LINEST_4[:, :2],
            FALSE,
            TRUE,
            TEST_LINEST_4_FALSE_RESULT,
        ),
    ],
)
def test_LINEST(y, x, c, s, result):
    res = LINEST(y, x, c, s)
    for i, _ in enumerate(res):
        assert (
            cellrange2np(res[i], raise_on_error=False)
            if isinstance(res[i], CellRange)
            else res[i]
        ) == pytest.approx(
            cellrange2np(result[i], raise_on_error=False)
            if isinstance(result[i], CellRange)
            else result[i],
            rel=1e-3,
        )


@pytest.mark.parametrize(
    "y, x, c, s, result",
    [
        (CellRange([1, 2, 3, 4, 5]), CellRange([1, 2, 3]), TRUE, TRUE, REF_ERROR),
        (CellRange([1, 2, 3]), CellRange([1, 2, 3, 4, 5]), TRUE, TRUE, REF_ERROR),
        (CellRange([[1, 2], [3, 4]]), CellRange([1, 2, 3]), TRUE, TRUE, REF_ERROR),
    ],
)
def test_LINEST_error(y, x, c, s, result):
    assert LINEST(y, x, c, s) == result


TEST_LOGEST_2_TRUE_RESULT = CellRange(
    [
        [0.998307311, 1.018106214, 1.085434734, 1.000174374, 80387.14721],
        [8.22132e-05, 0.003288211, 0.002478954, 3.36423e-05, 0.075826957],
        [0.997224229, 0.006014042, NA_ERROR, NA_ERROR, NA_ERROR],
        [538.8903983, 6, NA_ERROR, NA_ERROR, NA_ERROR],
        [0.077963872, 0.000217012, NA_ERROR, NA_ERROR, NA_ERROR],
    ]
)
TEST_LOGEST_2_FALSE_RESULT = CellRange(
    [
        [0.99488689, 0.76823558, 1.087896493, 1.005168243, 1],
        [0.004443601, 0.151485243, 0.139578172, 0.000208771, NA_ERROR],
        [0.999485743, 0.33862871, NA_ERROR, NA_ERROR, NA_ERROR],
        [3401.219777, 7, NA_ERROR, NA_ERROR, NA_ERROR],
        [1560.063364, 0.80268582, NA_ERROR, NA_ERROR, NA_ERROR],
    ]
)


@pytest.mark.parametrize(
    "y, x, c, s, result",
    [
        (
            TEST_LINEST_2[:, -1],
            TEST_LINEST_2[:, :-1],
            TRUE,
            TRUE,
            TEST_LOGEST_2_TRUE_RESULT,
        ),
        (
            TEST_LINEST_2[:, -1],
            TEST_LINEST_2[:, :-1],
            FALSE,
            TRUE,
            TEST_LOGEST_2_FALSE_RESULT,
        ),
    ],
)
def test_LOGEST(y, x, c, s, result):
    res = LOGEST(y, x, c, s)
    for i, row in enumerate(res):
        for j, col in enumerate(row):
            if isinstance(col, SpreadsheetError):
                assert col == result[i][j]
            else:
                assert res[i][j] == pytest.approx(result[i][j], rel=1e-5)


@pytest.mark.parametrize(
    "x, m, s, c, result",
    [
        (4, 3.5, 1.2, TRUE, 0.0390836),
        (4, 3.5, 1.2, FALSE, 0.0176176),
    ],
)
def test_LOGNORM_DIST(x, m, s, c, result):
    assert LOGNORM.DIST(x, m, s, c) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "p, m, s, result",
    [
        (0.039084, 3.5, 1.2, 4.0000252),
    ],
)
def test_LOGNORM_INV(p, m, s, result):
    assert LOGNORM.INV(p, m, s) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "args, result",
    [
        [(CellRange([1, 2, None, 7, None])), 7],
        [(CellRange([[1, 2, None, 0, None], [1, 2, None, 4, None]])), 4],
        [(CellRange([1, 2, 3])), 3],
        [(CellRange([[1, 1], [2, 2]])), 2],
        [(CellRange([1, 2, 3]), 1, 1), 3],
        [(CellRange([FALSE, TRUE])), 0],
        [
            (SpreadsheetDate("07/10/2021"), SpreadsheetDate("08/10/2021")),
            SpreadsheetDate("08/10/2021"),
        ],
        [
            (SpreadsheetTime("18:10"), SpreadsheetTime("23:59")),
            SpreadsheetTime("23:59"),
        ],
        [(30, CellRange([10, 7, 9, 27, 2])), 30],
        [("hello", -1), -1],
        [("hello", "world"), 0],
    ],
)
def test_MAX(args, result):
    assert MAX(*args) == result


@pytest.mark.parametrize(
    "args, result", [[(CellRange([[0], [0.2], [0.5], [0.4], [TRUE]])), 1]]
)
def test_MAXA(args, result):
    assert MAXA(*args) == result


@pytest.mark.parametrize(
    "args, result",
    [
        ((CellRange([89, 93, 96, 85, 91, 88]), CellRange([1, 2, 2, 3, 1, 1]), 1), 91),
        (
            (CellRange([10, 1, 100, 1, 1]), CellRange(["a", "a", "b", "a", "a"]), "a"),
            10,
        ),
        (
            (
                CellRange([10, 1, 100, 1, 1, 50]),
                CellRange(["b", "a", "a", "b", "a", "b"]),
                "b",
                CellRange([100, 100, 200, 300, 100, 400]),
                ">100",
            ),
            50,
        ),
        # Should be an empty cell instead of None
        (
            (
                CellRange([10, 1, 100, 11, 1, 12]),
                CellRange(["b", "a", "a", "b", "a", "b"]),
                "b",
                CellRange([8, 8, 8, 0, 8, 0]),
                None,
            ),
            12,
        ),
        (
            (
                CellRange([10, 1, 100, 1]),
                CellRange(
                    [
                        ["b", None],
                        ["a", None],
                        ["a", None],
                        ["b", None],
                        ["a", None],
                        ["b", None],
                    ]
                ),
                "a",
            ),
            VALUE_ERROR,
        ),
        (
            (
                CellRange([10, 1, 100, 1, 1]),
                CellRange(["b", "a", "a", "b", "a"]),
                "a",
                CellRange([100, 100, 200, 300, 100]),
                ">200",
            ),
            0,
        ),
    ],
)
def test_MAXIFS(args, result):
    assert MAXIFS(*args) == result


TEST_MEDIAN = CellRange([*range(1, 7)])


@pytest.mark.parametrize(
    "args, result",
    [
        (TEST_MEDIAN[:5], 3),
        (TEST_MEDIAN, 3.5),
    ],
)
def test_MEDIAN(args, result):
    assert MEDIAN(*args) == pytest.approx(result)


@pytest.mark.parametrize(
    "args, result",
    [(("hello", "world"), 0)],
)
def test_MEDIAN_error(args, result):
    assert MEDIAN(*args) == result


@pytest.mark.parametrize(
    "args, result",
    [
        [(CellRange([1, 2, None, 0, None])), 0],
        [(CellRange([[1, 2, None, 0, None], [1, 2, None, -1, None]])), -1],
        [(CellRange([1, 2, 3])), 1],
        [(CellRange([[1, 1], [2, 2]])), 1],
        [(CellRange([1, 2, 3]), 1, 1), 1],
        [(CellRange([FALSE, TRUE])), 0],
        [
            (SpreadsheetDate("07/10/2021"), SpreadsheetDate("08/10/2021")),
            SpreadsheetDate("07/10/2021"),
        ],
        [
            (SpreadsheetTime("18:10"), SpreadsheetTime("23:59")),
            SpreadsheetTime("18:10"),
        ],
        [(30, CellRange([10, 7, 9, 27, 2])), 2],
        [("hello", -1), -1],
        [("hello", "world"), 0],
        ((FALSE, 0.2, 0.5, 0.4, 0.8), 0.2),
    ],
)
def test_MIN(args, result):
    assert MIN(*args) == result


@pytest.mark.parametrize(
    "args, result",
    [
        ((FALSE, 0.2, 0.5, 0.4, 0.8), 0),
        (("hello", 0.2, 0.5, 0.4, 0.8), 0),
        (("hello", -1, 0.5, 0.4, 0.8), -1),
    ],
)
def test_MINA(args, result):
    assert MINA(*args) == result


@pytest.mark.parametrize(
    "args, result",
    [
        ((CellRange([89, 93, 96, 85, 91, 88]), CellRange([1, 2, 2, 3, 1, 1]), 1), 88),
        ((CellRange([10, 11, 100, 111]), CellRange(["a", "a", "b", "a"]), "a"), 10),
        (
            (
                CellRange([10, 11, 12, 13, 14, 15]),
                CellRange(["b", "a", "a", "b", "b", "b"]),
                "b",
                CellRange([100, 100, 200, 300, 300, 400]),
                ">100",
            ),
            13,
        ),
        # Should be an empty cell instead of None
        (
            (
                CellRange([10, 1, 100, 11, 1, 1]),
                CellRange(["b", "a", "a", "b", "a", "b"]),
                "b",
                CellRange([8, 8, 8, 0, 8, 0]),
                None,
            ),
            1,
        ),
        (
            (
                CellRange([10, 1, 100, 1]),
                CellRange(
                    [
                        ["b", None],
                        ["a", None],
                        ["a", None],
                        ["b", None],
                        ["a", None],
                        ["b", None],
                    ]
                ),
                "a",
            ),
            VALUE_ERROR,
        ),
        (
            (
                CellRange([10, 1, 100, 1, 1]),
                CellRange(["b", "a", "a", "b", "a"]),
                "a",
                CellRange([100, 100, 200, 300, 100]),
                ">200",
            ),
            0,
        ),
    ],
)
def test_MINIFS(args, result):
    assert MINIFS(*args) == result


@pytest.mark.parametrize(
    "args, result",
    [
        (CellRange([1, 2, 1, 2, 3]), CellRange([[1], [2]])),
        (CellRange([1, 2, 3, 2]), NA_ERROR),
        ([10, 15, 20, 30, 10, 15], CellRange([[10], [15]])),
    ],
)
def test_MODE_MULT(args, result):
    assert_equal(MODE.MULT(*args), result)


@pytest.mark.parametrize(
    "args, result",
    [
        (CellRange([5.6, 4, 4, 3, 2, 4]), 4),
        (CellRange([3, 5, 6, 5, 7.6, 5, 2]), 5),
        (CellRange([1, 2, 3]), NA_ERROR),
    ],
)
def test_MODE_SNGL(args, result):
    assert MODE.SNGL(*args) == result


@pytest.mark.parametrize(
    "nf, ns, p, c, result",
    [
        (10, 5, 0.25, TRUE, 0.3135141),
        (10, 5, 0.25, FALSE, 0.0550487),
    ],
)
def test_NEGBINOM_DIST(nf, ns, p, c, result):
    assert NEGBINOM.DIST(nf, ns, p, c) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x,mean,standard_dev,cumulative, result",
    [
        (42, 40, 1.5, TRUE, 0.9087888),
        (42, 40, 1.5, FALSE, 0.10934),
        ("hello", 40, 1.5, FALSE, VALUE_ERROR),
        (42, "world", 1.5, FALSE, VALUE_ERROR),
        (42, 40, 1.5, FALSE, 0.10934),
    ],
)
def test_NORM_DIST(x, mean, standard_dev, cumulative, result):
    if isinstance(result, SpreadsheetError):
        assert NORM.DIST(x, mean, standard_dev, cumulative) == result
    else:
        assert NORM.DIST(x, mean, standard_dev, cumulative) == pytest.approx(result)


@pytest.mark.parametrize(
    "p,mean,standard_dev, result",
    [
        (0.9087888, 40, 1.5, 42.000002),
    ],
)
def test_NORM_INV(p, mean, standard_dev, result):
    assert NORM.INV(p, mean, standard_dev) == pytest.approx(result)


@pytest.mark.parametrize(
    "z, cumulative, result",
    [(1.333333, TRUE, 0.908788726), (1.333333, FALSE, 0.164010148)],
)
def test_NORM_S_DIST(z, cumulative, result):
    assert NORM.S.DIST(z, cumulative) == pytest.approx(result)


@pytest.mark.parametrize(
    "probability, result", [(0.908789, 1.3333347), (10, NUM_ERROR), (-1, NUM_ERROR)]
)
def test_NORM_S_INV(probability, result):
    if isinstance(result, SpreadsheetError):
        assert NORM.S.INV(probability) == result
    else:
        assert NORM.S.INV(probability) == pytest.approx(result)


@pytest.mark.parametrize(
    "a1, a2, result",
    [(CellRange([9, 7, 5, 3, 1]), CellRange([10, 6, 1, 5, 3]), 0.699379)],
)
def test_PEARSON(a1, a2, result):
    assert PEARSON(a1, a2) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "a1, a2, result",
    [
        # ([],CellRange([10,6,1,5,3]),NA_ERROR),
        # (CellRange([10,6,1,5,3]),[],NA_ERROR),
        (CellRange([1]), CellRange([10, 6, 1, 5, 3]), NA_ERROR),
    ],
)
def test_PEARSON_error(a1, a2, result):
    assert PEARSON(a1, a2) == result


TEST_PER_EXC_RANGE = CellRange([15, 20, 35, 40, 50])


@pytest.mark.parametrize(
    "array, k, result",
    [
        (CellRange([*range(1, 10)]), 0.25, 2.5),
        (CellRange([*range(1, 10)]), 0, NUM_ERROR),
        (CellRange([*range(1, 10)]), 0.01, NUM_ERROR),
        (CellRange([*range(1, 10)]), 2, NUM_ERROR),
        (TEST_PER_EXC_RANGE, 0.15, NUM_ERROR),
        (TEST_PER_EXC_RANGE, 0.4, 26),
        (TEST_PER_EXC_RANGE, 0.5, 35),
        (TEST_PER_EXC_RANGE, 0.9, NUM_ERROR),
    ],
)
def test_PERCENTILE_EXC(array, k, result):
    assert PERCENTILE.EXC(array, k) == result


@pytest.mark.parametrize(
    "array, k, result",
    [
        (CellRange([1, 3, 2, 4]), 0.3, 1.9),
        (CellRange([1, 3, 2, 4]), "hello", VALUE_ERROR),
        (CellRange([1, 3, 2, 4]), 10, NUM_ERROR),
        (CellRange([1, 3, 2, 4]), -10, NUM_ERROR),
    ],
)
def test_PERCENTILE_INC(array, k, result):
    assert PERCENTILE.INC(array, k) == result


TEST_PERCENTRANK_EXC = CellRange([1, 2, 3, 6, 6, 6, 7, 8, 9])
TEST_PERCENTRANK_EXC2 = CellRange([1, 1, 1, 1, 2, 3, 4, 5, 5, 5])


@pytest.mark.parametrize(
    "a, x, s, result",
    [
        (TEST_PERCENTRANK_EXC, 7, 3, 0.7),
        (TEST_PERCENTRANK_EXC, 5.43, 3, 0.381),
        (TEST_PERCENTRANK_EXC, 5.43, 1, 0.3),
        (TEST_PERCENTRANK_EXC2, 1, 2, 0.09),
        (TEST_PERCENTRANK_EXC2, 1.5, 2, 0.4),
        (TEST_PERCENTRANK_EXC2, 2.5, 2, 0.5),
        (TEST_PERCENTRANK_EXC2, 2, 2, 0.45),
        (TEST_PERCENTRANK_EXC2, 4.99, 3, 0.726),
        (TEST_PERCENTRANK_EXC2, 5, 3, 0.727),
    ],
)
def test_PERCENTRANK_EXC(a, x, s, result):
    assert PERCENTRANK.EXC(a, x, s) == pytest.approx(result, rel=1e-2)


TEST_PERCENTRANK_INC = CellRange([13, 12, 11, 8, 4, 3, 2, 1, 1, 1])


@pytest.mark.parametrize(
    "a, x, s, result",
    [
        (TEST_PERCENTRANK_INC, 2, 3, 0.333),
        (TEST_PERCENTRANK_INC, 4, 3, 0.555),
        (TEST_PERCENTRANK_INC, 8, 3, 0.666),
        (TEST_PERCENTRANK_INC, 5, 3, 0.583),
        (TEST_PERCENTRANK_EXC2, 1, 2, 0),
        (TEST_PERCENTRANK_EXC2, 1.5, 2, 0.38),
        (TEST_PERCENTRANK_EXC2, 2.5, 2, 0.5),
        (TEST_PERCENTRANK_EXC2, 2, 2, 0.44),
        (TEST_PERCENTRANK_EXC2, 4.99, 3, 0.776),
        (TEST_PERCENTRANK_EXC2, 5, 3, 0.777),
    ],
)
def test_PERCENTRANK_INC(a, x, s, result):
    assert PERCENTRANK.INC(a, x, s) == pytest.approx(result, rel=1e-2)


@pytest.mark.parametrize(
    "n, nc, result",
    [
        (100, 3, 970200),
        (3, 2, 6),
        (0, 1, NUM_ERROR),
        (2, 10, NUM_ERROR),
        (2, -1, NUM_ERROR),
    ],
)
def test_PERMUT(n, nc, result):
    assert PERMUT(n, nc) == result


@pytest.mark.parametrize(
    "n, nc, result",
    [(3, 2, 9), (2, 2, 4), (3.2, 2.3, 9), (0, 10, NUM_ERROR), (3, 5, NUM_ERROR)],
)
def test_PERMUTATIONA(n, nc, result):
    assert PERMUTATIONA(n, nc) == result


@pytest.mark.parametrize(
    "x, result", [(0.75, 0.301137432), (2, 0.053990967), (-2, 0.053990967)]
)
def test_PHI(x, result):
    assert PHI(x) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, m, c, result",
    [
        (2, 5, TRUE, 0.124652),
        (2, 5, FALSE, 0.084224),
    ],
)
def test_POISSON_DIST(x, m, c, result):
    assert POISSON.DIST(x, m, c) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, p, lo, up, result",
    [
        ([0, 1, 2, 3], [0.2, 0.3, 0.1, 0.4], 2, None, 0.1),
        ([0, 1, 2, 3], [0.2, 0.3, 0.1, 0.4], 1, 3, 0.8),
        ([0, 1, 2, 3], [0.2, 0.3, 0.1, 0.4], 5, 10, 0),
        ([0, 1, 2, 3], [0.2, 0.3, 0.1, 0.4], 5, None, 0),
    ],
)
def test_PROB(x, p, lo, up, result):
    assert PROB(x, p, lo, up) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, p, lo, up, result",
    [
        ([0, 1, 2, 3], [-0.2, 0.3, 0.1, 0.4], 2, None, NUM_ERROR),
        ([0, 1, 2, 3], [1.2, 0.3, 0.1, 0.4], 2, None, NUM_ERROR),
        ([0, 1, 2, 3, 4], [0.2, 0.3, 0.1, 0.4], 2, None, NA_ERROR),
    ],
)
def test_PROB_errors(x, p, lo, up, result):
    assert PROB(x, p, lo, up) == result


TEST_QUARTILE_EXC_RANGE = CellRange([6, 7, 15, 36, 39, 40, 41, 42, 43, 47, 49])


@pytest.mark.parametrize(
    "array, k, result",
    [
        (TEST_QUARTILE_EXC_RANGE, 1, 15),
        (TEST_QUARTILE_EXC_RANGE, 3, 43),
    ],
)
def test_QUARTILE_EXC(array, k, result):
    assert QUARTILE.EXC(array, k) == result


@pytest.mark.parametrize(
    "array, k, result",
    [
        (CellRange([1, 2, 4, 7, 8, 9, 10, 12]), 1, 3.5),
    ],
)
def test_QUARTILE_INC(array, k, result):
    assert QUARTILE.INC(array, k) == result


@pytest.mark.parametrize(
    "number,ref,order,result",
    [
        (88, CellRange([89, 88, 92, 101, 94, 97, 95]), TRUE, 1),
        (94, CellRange([89, 88, 92, 101, 94, 97, 95]), FALSE, 4),
        (-1, CellRange([89, 88, 92, 101, 94, 97, 95]), FALSE, NA_ERROR),
    ],
)
def test_RANK_AVG(number, ref, order, result):
    assert RANK.AVG(number, ref, order) == result


@pytest.mark.parametrize(
    "number,ref,order,result",
    [
        (7, CellRange([7, 3.5, 3.5, 1, 2]), 1, 5),
        (2, CellRange([7, 3.5, 3.5, 1, 2]), 0, 4),
        (3.5, CellRange([7, 3.5, 3.5, 1, 2]), 1, 3),
        (3.5, CellRange([7, 3.5, 3.5, 1, 2]), 0, 2),
    ],
)
def test_RANK_EQ(number, ref, order, result):
    assert RANK.EQ(number, ref, order) == result


@pytest.mark.parametrize(
    "y, x, result",
    [
        ([2, 3, 9, 1, 8, 7, 5], [6, 5, 11, 7, 5, 4, 4], 0.05795),
        (
            [5, 8.5, 10, 11.2, 14, 16, 16.8, 18.55, 20],
            [1, 2.5, 3.1, 4, 4.7, 5.3, 6, 7.1, 9],
            0.9558856309,
        ),
    ],
)
def test_RSQ(y, x, result):
    assert RSQ(y, x) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "args, result",
    [
        ((3, 4, 5, 2, 3, 4, 5, 6, 4, 7), 0.359543),
        ((2, 5, 8, 13, 10, 18, 23, 26), 0.3446154716),
    ],
)
def test_SKEW(args, result):
    assert SKEW(*args) == pytest.approx(result)


@pytest.mark.parametrize(
    "args, result",
    [
        ((3, 4, 5, 2, 3, 4, 5, 6, 4, 7), 0.303193),
        ((2, 5, 8, 13, 10, 18, 23, 26), 0.2763070768),
        ((CellRange([2, 5, 8, 13, 10, 18, 23, 26]), 30, 40), 0.4621754338),
        ((3, 4, 5, 2, 3, 4, 5, 6, 4, 7, "q", "v"), 0.303193),
        ((3, 4, 5, 2, 3, 4, 5, 6, 4, 7, TRUE, FALSE), -0.219190496),
    ],
)
def test_SKEW_P(args, result):
    assert SKEW.P(*args) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "y, x, result",
    [
        (
            [5, 8.5, 10, 11.2, 14, 16, 16.8, 18.55, 20],
            [1, 2.5, 3.1, 4, 4.7, 5.3, 6, 7.1, 9],
            1.997074937,
        )
    ],
)
def test_SLOPE(y, x, result):
    assert SLOPE(y, x) == pytest.approx(result)


TEST_SMALL_RANGE = CellRange(
    [[3, 1], [4, 4], [5, 8], [2, 3], [3, 7], [4, 12], [6, 54], [4, 8], [7, 23]]
)


@pytest.mark.parametrize(
    "array, k, result",
    [
        (TEST_SMALL_RANGE[:, 0], 4, 4),
        (TEST_SMALL_RANGE[:, 1], 2, 3),
        (TEST_SMALL_RANGE, 44, NUM_ERROR),
        (TEST_SMALL_RANGE, -1, NUM_ERROR),
    ],
)
def test_SMALL(array, k, result):
    assert SMALL(array, k) == result


@pytest.mark.parametrize(
    "x, m, s, result",
    [(42, 40, 1.5, 1.33333333), (7.5, 6, 2, 0.75), (-10.2, -3, 4.5, -1.6)],
)
def test_STANDARDIZE(x, m, s, result):
    assert STANDARDIZE(x, m, s) == pytest.approx(result)


@pytest.mark.parametrize(
    "args, result",
    [((1345, 1301, 1368, 1322, 1310, 1370, 1318, 1350, 1303, 1299), 26.05456)],
)
def test_STDEV_P(args, result):
    assert STDEV.P(*args) == pytest.approx(result)


@pytest.mark.parametrize(
    "args, result",
    [((1345, 1301, 1368, 1322, 1310, 1370, 1318, 1350, 1303, 1299), 27.46392)],
)
def test_STDEV(args, result):
    assert STDEV.S(*args) == pytest.approx(result)


TEST_VARA_RANGE = CellRange([2, 5, 8, 13, 10, TRUE, FALSE, "Google"])


@pytest.mark.parametrize(
    "args, result",
    [
        ((1345, 1301, 1368, 1322, 1310, 1370, 1318, 1350, 1303, 1299), 27.46392),
        ((TEST_VARA_RANGE[:5],), 4.277849927),
        ((TEST_VARA_RANGE[:9],), 4.969550138),
        ((TEST_VARA_RANGE[:5], 1, 0, 0), 4.969550138),
        ((TEST_VARA_RANGE[:5], [1, 0, 0]), 4.969550138),
    ],
)
def test_STDEVA(args, result):
    assert STDEVA(*args) == pytest.approx(result)


@pytest.mark.parametrize(
    "args, result",
    [
        ((1345, 1301, 1368, 1322, 1310, 1370, 1318, 1350, 1303, 1299), 26.05456),
        ((TEST_VARA_RANGE[:5],), 3.826225294),
        ((TEST_VARA_RANGE[:9],), 4.648588495),
        ((TEST_VARA_RANGE[:5], 1, 0, 0), 4.648588495),
        ((TEST_VARA_RANGE[:5], [1, 0, 0]), 4.648588495),
    ],
)
def test_STDEVPA(args, result):
    assert STDEVPA(*args) == pytest.approx(result)


@pytest.mark.parametrize(
    "y, x, result",
    [
        ([2, 3, 9, 1, 8, 7, 5], [6, 5, 11, 7, 5, 4, 4], 3.305719),
        (
            [5, 8.5, 10, 11.2, 14, 16, 16.8, 18.55, 20],
            [1, 2.5, 3.1, 4, 4.7, 5.3, 6, 7.1, 9],
            1.121834626,
        ),
    ],
)
def test_STEYX(y, x, result):
    assert STEYX(y, x) == pytest.approx(result)


@pytest.mark.parametrize(
    "x, df, c, result",
    [
        (60, 1, TRUE, 0.99469533),
        (8, 3, FALSE, 0.00073691),
    ],
)
def test_T_DIST(x, df, c, result):
    assert T.DIST(x, df, c) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, df, c, result",
    [
        (60, 0, TRUE, NUM_ERROR),
        ("hello", 1, TRUE, VALUE_ERROR),
    ],
)
def test_T_DIST_error(x, df, c, result):
    assert T.DIST(x, df, c) == result


@pytest.mark.parametrize(
    "x, df, result",
    [
        (1.959999998, 60, 0.054645),
        (1.96, 60, 0.054644929736529),
        (1, 2, 0.42264973081037),
    ],
)
def test_T_DIST_2T(x, df, result):
    assert T.DIST_2T(x, df) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, df, result",
    [
        (60, 0, NUM_ERROR),
        ("hello", 1, VALUE_ERROR),
    ],
)
def test_T_DIST_2T_error(x, df, result):
    assert T.DIST_2T(x, df) == result


@pytest.mark.parametrize(
    "x, df, result", [(1.959999998, 60, 0.027322), (-1.98, 2, 0.9068737480782105)]
)
def test_T_DIST_RT(x, df, result):
    assert T.DIST.RT(x, df) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, df, result",
    [
        (60, 0, NUM_ERROR),
        ("hello", 1, VALUE_ERROR),
    ],
)
def test_T_DIST_RT_error(x, df, result):
    assert T.DIST.RT(x, df) == result


@pytest.mark.parametrize(
    "p, df, result", [(0.75, 2, 0.8164966), (0.05735, 45.2, -1.608591458)]
)
def test_T_INV(p, df, result):
    assert T.INV(p, df) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "p, df, result",
    [
        (0.546449, 60, 0.606533),
    ],
)
def test_T_INV_2T(p, df, result):
    assert T.INV_2T(p, df) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "a1, a2, t, tt, result",
    [
        ([3, 4, 5, 8, 9, 1, 2, 4, 5], [6, 19, 3, 2, 14, 4, 5, 17, 1], 2, 1, 0.196016),
        (
            [28, 26, 31, 23, 20, 27, 28, 14, 4, 0, 2, 8, 9],
            [19, 13, 12, 5, 34, 31, 31, 12, 24, 23, 19, 10, 33],
            1,
            1,
            0.2097651442,
        ),
    ],
)
def test_T_TEST(a1, a2, t, tt, result):
    assert T.TEST(a1, a2, t, tt) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "a1, a2, t, tt, result",
    [
        ([3, 4, 5, 8, 9, 1, 2, 4], [6, 19, 3, 2, 14, 4, 5, 17, 1], 2, 1, NA_ERROR),
    ],
)
def test_T_TEST_error(a1, a2, t, tt, result):
    assert T.TEST(a1, a2, t, tt) == result


TEST_TREND_RANGE = CellRange(
    [
        [1, 15.53],
        [2, 19.99],
        [3, 20.43],
        [4, 21.18],
        [5, 25.93],
        [6, 30],
        [7, 30],
        [8, 34.01],
        [9, 36.47],
    ]
)


@pytest.mark.parametrize(
    "y, x, nx, b, result",
    [
        (
            CellRange([5, 7, 9, 11]),
            CellRange([1, 2, 3, 4]),
            CellRange([5, 6]),
            TRUE,
            CellRange([13, 15]),
        ),
        (CellRange([5, 7, 9, 11]), None, None, TRUE, CellRange([5, 7, 9, 11])),
        (
            TEST_TREND_RANGE[:, 1],
            TEST_TREND_RANGE[:, 0],
            CellRange([10, 11, 12]),
            TRUE,
            CellRange([38.76388889, 41.32688889, 43.88988889]),
        ),
        (
            TEST_TREND_RANGE[:, 1],
            TEST_TREND_RANGE[:, 0],
            CellRange([10, 11, 12]),
            FALSE,
            CellRange([46.3677193, 51.00449123, 55.64126316]),
        ),
        ([2, 2, 2], [1, 1, 1], None, TRUE, CellRange([2, 2, 2])),
        (
            CellRange([[1, 3], [2, 4]]),
            CellRange([[2, 6], [4, 8]]),
            [9, 1],
            FALSE,
            CellRange([4.5, 0.5]),
        ),
        (
            CellRange([[1, 3], [2, 4]]),
            CellRange([[2, 6], [4, 8]]),
            [9, 1],
            TRUE,
            CellRange([4.5, 0.5]),
        ),
    ],
)
def test_TREND(y, x, nx, b, result):
    assert cellrange2np(TREND(y, x, nx, b)) == pytest.approx(cellrange2np(result))


TEST_TRIMMEAN_RANGE = CellRange([0.1, 2, 3, 7, 8, 10, 13.5, 17, 20, 23])


@pytest.mark.parametrize(
    "array, percent, result",
    [
        (CellRange([4, 5, 6, 7, 2, 3, 4, 5, 1, 2, 3]), 0.2, 3.778),
        (TEST_TRIMMEAN_RANGE, 0.2, 10.0625),
        (TEST_TRIMMEAN_RANGE, 0.4, 9.75),
    ],
)
def test_TRIMMEAN(array, percent, result):
    assert TRIMMEAN(array, percent) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "args, result",
    [((1345, 1301, 1368, 1322, 1310, 1370, 1318, 1350, 1303, 1299), 754.2667)],
)
def test_VARS(args, result):
    assert VAR.S(*args) == pytest.approx(result)


@pytest.mark.parametrize(
    "args, result",
    [((1345, 1301, 1368, 1322, 1310, 1370, 1318, 1350, 1303, 1299), 678.84)],
)
def test_VARP(args, result):
    assert VAR.P(*args) == pytest.approx(result)


@pytest.mark.parametrize(
    "args, result",
    [
        ((1345, 1301, 1368, 1322, 1310, 1370, 1318, 1350, 1303, 1299), 754.2667),
        ((TEST_VARA_RANGE[:5],), 18.3),
        ((TEST_VARA_RANGE[:8],), 24.69642857),
        ((TEST_VARA_RANGE[:5], 1, 0, 0), 24.69642857),
        ((TEST_VARA_RANGE[:5], [1, 0, 0]), 24.69642857),
    ],
)
def test_VARA(args, result):
    assert VARA(*args) == pytest.approx(result)


@pytest.mark.parametrize(
    "args, result",
    [
        ((1345, 1301, 1368, 1322, 1310, 1370, 1318, 1350, 1303, 1299), 678.84),
        ((TEST_VARA_RANGE[:5],), 14.64),
    ],
)
def test_VARPA(args, result):
    assert VARPA(*args) == pytest.approx(result)


@pytest.mark.parametrize(
    "x, a, b, c, result",
    [
        (105, 20, 100, TRUE, 0.929581),
        (105, 20, 100, FALSE, 0.035589),
    ],
)
def test_WEIBULL_DIST(x, a, b, c, result):
    assert WEIBULL.DIST(x, a, b, c) == pytest.approx(result, rel=1e-3)


@pytest.mark.parametrize(
    "x, a, b, c, result",
    [
        (-1, 20, 100, FALSE, NUM_ERROR),
        (105, 0, 100, FALSE, NUM_ERROR),
        (105, 20, 0, FALSE, NUM_ERROR),
    ],
)
def test_WEIBULL_DIST_error(x, a, b, c, result):
    assert WEIBULL.DIST(x, a, b, c) == result


TEST_ZTEST_RANGE = CellRange([3, 6, 7, 8, 6, 5, 4, 2, 1, 9])
TEST_ZTEST_RANGE2 = CellRange([97, 90, 88, 92, 87, 83, 90, 89, 94, 80])


@pytest.mark.parametrize(
    "array, x, sigma, result",
    [
        (TEST_ZTEST_RANGE, 4, None, 0.090574),
        (TEST_ZTEST_RANGE, 6, None, 0.863043),
        (TEST_ZTEST_RANGE2, 88, 5, 0.2635446284),
        (TEST_ZTEST_RANGE2, 90, 5, 0.7364553716),
    ],
)
def test_ZTEST(array, x, sigma, result):
    assert Z.TEST(array, x, sigma) == pytest.approx(result, rel=1e-3)


TEST_AGGR1 = CellRange([3, 10, 1, ZERO_DIV_ERROR])
TEST_AGGR2 = CellRange([[3, 10, 1, ZERO_DIV_ERROR], [5, 6, None, NAME_ERROR]])
TEST_AGGR_EXCEL = CellRange(
    [
        [ZERO_DIV_ERROR, 82],
        [72, 65],
        [30, 95],
        [NUM_ERROR, 63],
        [31, 53],
        [96, 71],
        [32, 55],
        [81, 83],
        [33, 100],
        [53, 91],
        [34, 89],
    ]
)


@pytest.mark.parametrize(
    "function_num, options, args, result",
    [
        # Single row
        (1, 1, (TEST_AGGR1,), ZERO_DIV_ERROR),  # AVERAGE
        (1, 2, (TEST_AGGR1,), 14 / 3),  # AVERAGE / ignore errors
        (2, 1, (TEST_AGGR1,), 3),  # COUNT
        (3, 1, (TEST_AGGR1,), 4),  # COUNTA
        (3, 2, (TEST_AGGR1,), 3),  # COUNTA / ignore errors
        (4, 1, (TEST_AGGR1,), ZERO_DIV_ERROR),  # MAX
        (4, 2, (TEST_AGGR1,), 10),  # MAX / ignore errors
        (5, 1, (TEST_AGGR1,), ZERO_DIV_ERROR),  # MIN
        (5, 2, (TEST_AGGR1,), 1),  # MIN / ignore errors
        (6, 1, (TEST_AGGR1,), ZERO_DIV_ERROR),  # PRODUCT
        (6, 2, (TEST_AGGR1,), 30),  # PRODUCT / ignore errors
        (9, 1, (TEST_AGGR1,), ZERO_DIV_ERROR),  # SUM
        (9, 2, (TEST_AGGR1,), 14),  # SUM / ignore errors
        # Multiple rows
        (1, 1, (TEST_AGGR2,), ZERO_DIV_ERROR),  # AVERAGE
        (1, 2, (TEST_AGGR2,), 5),  # AVERAGE / ignore errors
        (2, 1, (TEST_AGGR2,), 5),  # COUNT
        (3, 1, (TEST_AGGR2,), 7),  # COUNTA
        (3, 2, (TEST_AGGR2,), 5),  # COUNTA / ignore errors
        (4, 1, (TEST_AGGR2,), ZERO_DIV_ERROR),  # MAX
        (4, 2, (TEST_AGGR2,), 10),  # MAX / ignore errors
        (5, 1, (TEST_AGGR2,), ZERO_DIV_ERROR),  # MIN
        (5, 2, (TEST_AGGR2,), 1),  # MIN / ignore errors
        (6, 1, (TEST_AGGR2,), ZERO_DIV_ERROR),  # PRODUCT
        (6, 2, (TEST_AGGR2,), 900),  # PRODUCT / ignore errors
        (9, 1, (TEST_AGGR2,), ZERO_DIV_ERROR),  # SUM
        (9, 2, (TEST_AGGR2,), 25),  # SUM / ignore errors
        # Two cell range arguments
        (1, 1, (TEST_AGGR2[0], TEST_AGGR2[1]), ZERO_DIV_ERROR),  # AVERAGE
        (1, 2, (TEST_AGGR2[0], TEST_AGGR2[1]), 5),  # AVERAGE / ignore errors
        (2, 1, (TEST_AGGR2[0], TEST_AGGR2[1]), 5),  # COUNT
        (3, 1, (TEST_AGGR2[0], TEST_AGGR2[1]), 7),  # COUNTA
        (3, 2, (TEST_AGGR2[0], TEST_AGGR2[1]), 5),  # COUNTA / ignore errors
        (4, 1, (TEST_AGGR2[0], TEST_AGGR2[1]), ZERO_DIV_ERROR),  # MAX
        (4, 2, (TEST_AGGR2[0], TEST_AGGR2[1]), 10),  # MAX / ignore errors
        (5, 1, (TEST_AGGR2[0], TEST_AGGR2[1]), ZERO_DIV_ERROR),  # MIN
        (5, 2, (TEST_AGGR2[0], TEST_AGGR2[1]), 1),  # MIN / ignore errors
        (6, 1, (TEST_AGGR2[0], TEST_AGGR2[1]), ZERO_DIV_ERROR),  # PRODUCT
        (6, 2, (TEST_AGGR2[0], TEST_AGGR2[1]), 900),  # PRODUCT / ignore errors
        (9, 1, (TEST_AGGR2[0], TEST_AGGR2[1]), ZERO_DIV_ERROR),  # SUM
        (9, 2, (TEST_AGGR2[0], TEST_AGGR2[1]), 25),  # SUM / ignore errors
        (2, 1, (TEST_AGGR_EXCEL,), 20),  # COUNT
        (2, 2, (TEST_AGGR_EXCEL,), 20),  # COUNT
        (3, 1, (TEST_AGGR_EXCEL,), 22),  # COUNTA
        (3, 2, (TEST_AGGR_EXCEL,), 20),  # COUNTA
        (4, 6, (TEST_AGGR_EXCEL[:, 0],), 96),  # MAX
        (5, 2, (TEST_AGGR_EXCEL,), 30),  # MIN
        (6, 2, (TEST_AGGR_EXCEL,), 439732094768489097580467990528000000),  # PRODUCT
        (7, 2, (TEST_AGGR_EXCEL,), 24.2084),  # STDEV.S
        (8, 2, (TEST_AGGR_EXCEL,), 23.5954),  # STDEV.P
        (9, 2, (TEST_AGGR_EXCEL,), 1309),  # SUM
        (10, 2, (TEST_AGGR_EXCEL,), 586.05),  # VAR.S
        (11, 2, (TEST_AGGR_EXCEL,), 556.7475),  # VAR.P
        (12, 2, (TEST_AGGR_EXCEL,), 68),  # MEDIAN
        (12, 6, (TEST_AGGR_EXCEL[:, 0], TEST_AGGR_EXCEL[:, 1]), 68),  # MEDIAN
        (13, 2, (TEST_AGGR_EXCEL,), 53),  # MODE.SNGL
        (14, 6, (TEST_AGGR_EXCEL[:, 0], 3), 72),  # LARGE
        (15, 6, (TEST_AGGR_EXCEL[:, 0],), VALUE_ERROR),  # SMALL / missing argument
        (15, 6, (TEST_AGGR_EXCEL[:, 0], 13), NUM_ERROR),  # SMALL / incorrect argument
        (15, 6, (TEST_AGGR_EXCEL[:, 0], 3), 32),  # SMALL
        (15, 6, (TEST_AGGR_EXCEL[:, 1], 3), 63),  # SMALL
        (
            15,
            1,
            (TEST_AGGR_EXCEL[:, 0], 3),
            ZERO_DIV_ERROR,
        ),  # SMALL / error in a column
        (15, 1, (TEST_AGGR_EXCEL[:, 1], 3), 63),  # SMALL
        (16, 1, (TEST_AGGR_EXCEL[:, 1], 3), NUM_ERROR),  # PERCENTILE.INC
        (16, 1, (TEST_AGGR_EXCEL[:, 1], 0.75), 90),  # PERCENTILE.INC
        (16, 1, (TEST_AGGR_EXCEL[:, 0], 0.75), ZERO_DIV_ERROR),  # PERCENTILE.INC
        (16, 2, (TEST_AGGR_EXCEL[:, 0], 0.75), 72),  # PERCENTILE.INC
        (17, 2, (TEST_AGGR_EXCEL[:, 0], 1), 32),  # QUARTILE.INC
        (18, 2, (TEST_AGGR_EXCEL[:, 0], 0.75), 76.5),  # PERCENTILE.EXC
        (19, 2, (TEST_AGGR_EXCEL[:, 0], 0.75), NUM_ERROR),  # QUARTILE.EXC
    ],
)
def test_AGGREGATE(function_num, options, args, result):
    assert AGGREGATE(function_num, options, *args) == (
        result if not isinstance(result, float) else pytest.approx(result, 1e-3)
    )
