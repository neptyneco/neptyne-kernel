import math
import operator
import statistics
from typing import Callable, Iterable

import numpy as np
import statsmodels.api as smapi
from scipy.stats import (
    beta,
    binom,
    chi2,
    chisquare,
    expon,
    f,
    gamma,
    hypergeom,
    linregress,
    lognorm,
    median_abs_deviation,
    nbinom,
    norm,
    pearsonr,
    poisson,
    rankdata,
    t,
    trim_mean,
    ttest_ind,
    ttest_rel,
)

from ..cell_range import CellRange
from ..spreadsheet_datetime import SpreadsheetDateTime
from ..spreadsheet_error import (
    NA_ERROR,
    NUM_ERROR,
    REF_ERROR,
    VALUE_ERROR,
    ZERO_DIV_ERROR,
    SpreadsheetError,
)
from .boolean import FALSE, TRUE, BooleanValue
from .helpers import (
    CellValue,
    Matrix,
    Numeric,
    SimpleCellValue,
    SimpleCellValueT,
    _flatten_range,
    agg_func,
    criteria_func,
    mat_func,
    num_func,
    prepare_crit_ranges,
    round_to_digits_func,
)
from .mathtrig import PRODUCT, SUM

__all__ = [
    "AGGREGATE",
    "AVEDEV",
    "AVERAGE",
    "AVERAGEA",
    "AVERAGEIF",
    "AVERAGEIFS",
    "BETA",
    "BINOM",
    "CHISQ",
    "CONFIDENCE",
    "CORREL",
    "COUNT",
    "COUNTA",
    "COUNTBLANK",
    "COUNTIF",
    "COUNTIFS",
    "COVARIANCE",
    "DEVSQ",
    "EXPON",
    "F",
    "FISHER",
    "FISHERINV",
    "FORECAST",
    "FREQUENCY",
    "GAMMA",
    "GAMMALN",
    "GAUSS",
    "GEOMEAN",
    "GROWTH",
    "HARMEAN",
    "HYPGEOM",
    "INTERCEPT",
    "KURT",
    "LARGE",
    "LINEST",
    "LOGEST",
    "LOGNORM",
    "MAX",
    "MAXA",
    "MAXIFS",
    "MEDIAN",
    "MIN",
    "MINA",
    "MINIFS",
    "MODE",
    "NEGBINOM",
    "NORM",
    "PEARSON",
    "PERCENTILE",
    "PERCENTRANK",
    "PERMUT",
    "PERMUTATIONA",
    "PHI",
    "POISSON",
    "PROB",
    "QUARTILE",
    "RANK",
    "RSQ",
    "SKEW",
    "SLOPE",
    "SMALL",
    "STANDARDIZE",
    "STDEV",
    "STDEVP",
    "STDEVA",
    "STDEVPA",
    "STEYX",
    "SUBTOTAL",
    "T",
    "TREND",
    "TRIMMEAN",
    "VAR",
    "VARP",
    "VARPA",
    "VARA",
    "WEIBULL",
    "Z",
]

from ..primitives import Empty


@agg_func(median_abs_deviation)
def AVEDEV(number: CellValue, *numbers: CellValue) -> Numeric:
    """Returns the average of the absolute deviations of data points from their mean"""
    pass


def _beta(
    func: Callable,
    x: Numeric,
    alpha: Numeric,
    beta: Numeric,
    A: Numeric = 0,
    B: Numeric = 1,
):
    if alpha <= 0 or beta <= 0:
        return NUM_ERROR
    return func(x, alpha, beta, A, B)


class BETA:
    @staticmethod
    @num_func(
        lambda x, a, b, c, A, B: NUM_ERROR
        if x < A or x > B or A == B
        else _beta(beta.cdf, x, a, b, A, B - A)
        if c
        else _beta(beta.pdf, x, a, b, A, B - A)
    )
    def DIST(
        x: Numeric,
        alpha: Numeric,
        beta: Numeric,
        cumulative: BooleanValue,
        A: Numeric = 0,
        B: Numeric = 1,
    ) -> Numeric:
        """Returns the beta cumulative distribution function"""
        pass

    @staticmethod
    @num_func(
        lambda p, a, b, A, B: NUM_ERROR
        if p <= 0 or p > 1
        else _beta(beta.ppf, p, a, b, A, B - A)
    )
    def INV(
        probability: float,
        alpha: Numeric,
        beta: Numeric,
        A: Numeric = 0,
        B: Numeric = 1,
    ) -> Numeric:
        """Returns the inverse of the cumulative distribution function for a specified beta distribution"""
        pass


def _dist_range(nt, pb, ns, ns2):
    if not (0 < pb < 1 and 0 < ns < nt):
        return NUM_ERROR
    ns = int(ns)
    nt = int(nt)
    if ns2:
        ns2 = int(ns2)
        if not (ns < ns2 < nt):
            return NUM_ERROR
        else:
            return sum(binom.pmf(k, nt, pb) for k in range(ns, ns2 + 1))
    return binom.pmf(ns, nt, pb)


class BINOM:
    @staticmethod
    @num_func(
        lambda ns, nt, p, c: NUM_ERROR
        if ns < 0 or ns > nt or p < 0 or p > 1
        else binom.cdf(int(ns), int(nt), p)
        if c
        else binom.pmf(int(ns), int(nt), p)
    )
    def DIST(
        num_successes: Numeric,
        num_trials: Numeric,
        prob_success: float,
        cumulative: BooleanValue,
    ) -> Numeric:
        """Returns the individual term binomial distribution probability"""
        pass

    @staticmethod
    @num_func(_dist_range)
    def _dist_range(
        num_trials: Numeric,
        prob_success: float,
        num_success: Numeric,
        num_success2: Numeric = None,
    ) -> Numeric:
        """Returns the probability of a trial result using a binomial distribution"""
        pass

    @staticmethod
    @num_func(
        lambda nt, ps, tp: binom.ppf(tp, int(nt), ps)
        if 0 < ps < 1 and 0 < tp < 1 and nt >= 0
        else NUM_ERROR
    )
    def INV(num_trials: Numeric, prob_success: float, target_prob: float) -> Numeric:
        """Returns the smallest value for which the cumulative binomial distribution is less than or equal to a criterion value"""
        pass


BINOM.DIST.RANGE = BINOM._dist_range


def _chisq(func, x, df):
    if x < 0 or df < 1 or df > 1e10:
        return NUM_ERROR
    return func(x, int(df))


def _chisq_test(a, e):
    if len(a.shape) == 1:
        r = 1
        c = len(a)
    else:
        r, c = a.shape
    if r > 1 and c > 1:
        df = (r - 1) * (c - 1)
    elif r == 1 and c > 1:
        df = c - 1
    elif r > 1 and c == 1:
        df = r - 1
    else:
        return NA_ERROR
    return 1 - chi2.cdf(chisquare(a, e, axis=None, ddof=df)[0], df=df)


class CHISQ:
    @staticmethod
    @num_func(lambda x, df, c: _chisq(chi2.cdf if c else chi2.pdf, x, df))
    def DIST(x: Numeric, deg_freedom: int, cumulative: BooleanValue) -> Numeric:
        """Returns the cumulative beta probability density function"""
        pass

    @staticmethod
    @num_func(lambda x, df: _chisq(chi2.sf, x, df))
    def _dist_rt(x: Numeric, deg_freedom: int) -> Numeric:
        """Returns the one-tailed probability of the chi-squared distribution"""
        pass

    @staticmethod
    @num_func(lambda x, df: _chisq(chi2.ppf, x, df))
    def INV(probability: float, deg_freedom: int) -> Numeric:
        """Returns the cumulative beta probability density function"""
        pass

    @staticmethod
    @num_func(lambda x, df: _chisq(chi2.isf, x, df))
    def _inv_rt(probability: float, deg_freedom: int) -> Numeric:
        """Returns the inverse of the one-tailed probability of the chi-squared distribution"""
        pass

    @staticmethod
    @mat_func(_chisq_test, NUM_ERROR, eq_shapes=True, shape_error=NA_ERROR)
    def TEST(actual_range: Matrix, expected_range: Matrix) -> Numeric:
        """Returns the test for independence"""
        pass


CHISQ.DIST.RT = CHISQ._dist_rt
CHISQ.INV.RT = CHISQ._inv_rt


class CONFIDENCE:
    @staticmethod
    @num_func(
        lambda a, s, sz: NUM_ERROR
        if a <= 0 or a >= 1 or s <= 0 or sz < 1
        else norm.ppf(1 - a / 2) * s / np.sqrt(int(sz))
    )
    def NORM(alpha: float, stdev: Numeric, size: int) -> Numeric:
        """Returns the confidence interval for a population mean"""
        pass

    @staticmethod
    @num_func(
        lambda a, s, sz: NUM_ERROR
        if a <= 0 or a >= 1 or s <= 0 or sz < 1
        else ZERO_DIV_ERROR
        if sz == 1
        else t.ppf(1 - a / 2, df=sz - 1) * s / np.sqrt(int(sz))
    )
    def T(alpha: float, stdev: Numeric, size: int) -> Numeric:
        """Returns the confidence interval for a population mean, using a Student's t distribution"""
        pass


@mat_func(
    lambda *args: pearsonr(*args)[0],
    ZERO_DIV_ERROR,
    eq_shapes=True,
    shape_error=NA_ERROR,
)
def CORREL(array1: Matrix, array2: Matrix):
    """Returns the correlation coefficient between two data sets"""
    pass


def COUNTA(value1: CellValue, *values: CellValue) -> int:
    """Counts how many values are in the list of arguments"""

    def _cnt(elem):
        if isinstance(elem, Iterable) and not isinstance(elem, str):
            return sum(_cnt(x) for x in elem)
        if elem is not None:
            return 1
        return 0

    try:
        args = [value1, *values]
        return _cnt(args)
    except (RecursionError, TypeError, ValueError):
        return VALUE_ERROR


def COUNTBLANK(value1: CellValue, *values: CellValue) -> int:
    """Counts the number of blank cells within a range"""

    def _cnt(elem):
        if isinstance(elem, Iterable) and not isinstance(elem, str):
            return sum(_cnt(x) for x in elem)
        if elem is None or elem == "":
            return 1
        return 0

    try:
        args = [value1, *values]
        return _cnt(args)
    except (RecursionError, TypeError, ValueError):
        return VALUE_ERROR


class COVARIANCE:
    @staticmethod
    @mat_func(
        lambda a1, a2: np.cov(a1, a2, ddof=0)[0][1] if a1.size >= 2 else ZERO_DIV_ERROR,
        VALUE_ERROR,
        count_bool=False,
        eq_shapes=True,
        shape_error=NA_ERROR,
    )
    def P(array1: Matrix, array2: Matrix) -> Numeric:
        """Returns covariance, the average of the products of paired deviations"""
        pass

    @staticmethod
    @mat_func(
        lambda a1, a2: np.cov(a1, a2)[0][1] if a1.size >= 2 else ZERO_DIV_ERROR,
        VALUE_ERROR,
        count_bool=False,
        eq_shapes=True,
        shape_error=NA_ERROR,
    )
    def S(array1: Matrix, array2: Matrix) -> Numeric:
        """Returns the sample covariance, the average of the products deviations for each data point pair in two data sets"""
        pass


@agg_func(
    statistics.mean,
    count_text=True,
    bool_as_num=True,
    count_bool=True,
    make_list=True,
    result_if_zero=ZERO_DIV_ERROR,
)
def AVERAGEA(value1: CellValue, *values: CellValue) -> float:
    """Returns the average of its arguments."""
    pass


def _f(func, x, df1, df2):
    if x < 0 or df1 < 1 or df2 < 1 or df1 > 1e10 or df2 > 1e10:
        return NUM_ERROR
    return func(x, int(df1), int(df2))


def _ftest(a1, a2):
    var1 = np.var(a1, ddof=1)
    if not var1:
        return ZERO_DIV_ERROR
    var2 = np.var(a2, ddof=1)
    if not var2:
        return ZERO_DIV_ERROR
    F = var1 / var2

    cd = f.cdf(F, len(a1) - 1, len(a2) - 1)
    return 2 * min(cd, 1 - cd)


class F:
    @staticmethod
    @num_func(lambda x, df1, df2, c: _f(f.cdf if c else f.pdf, x, df1, df2))
    def DIST(
        x: Numeric, deg_freedom1: int, deg_freedom2: int, cumulative: BooleanValue
    ) -> Numeric:
        """Returns the F probability distribution"""
        pass

    @staticmethod
    @num_func(lambda x, df1, df2: _f(f.sf, x, df1, df2))
    def _dist_rt(x: Numeric, deg_freedom1: int, deg_freedom2: int) -> Numeric:
        """Returns the F probability distribution"""
        pass

    @staticmethod
    @num_func(lambda x, df1, df2: _f(f.ppf, x, df1, df2))
    def INV(probability: float, deg_freedom1: int, deg_freedom2: int) -> Numeric:
        """Returns the inverse of the F probability distribution"""
        pass

    @staticmethod
    @num_func(lambda x, df1, df2: _f(f.isf, x, df1, df2))
    def _inv_rt(probability: float, deg_freedom: int, deg_freedom2: int) -> Numeric:
        """Returns the inverse of the F probability distribution"""
        pass

    @staticmethod
    @mat_func(_ftest, ZERO_DIV_ERROR, args_to_numpy=[0, 1], count_bool=False)
    def TEST(array1: Matrix, array2: Matrix) -> Numeric:
        """Returns the result of an F-test"""
        pass


F.DIST.RT = F._dist_rt
F.INV.RT = F._inv_rt


@num_func(lambda x: math.atanh(x) if -1 < x < 1 else NUM_ERROR)
def FISHER(x: Numeric) -> Numeric:
    """Returns the Fisher transformation"""
    pass


@num_func(math.tanh)
def FISHERINV(y: Numeric) -> Numeric:
    """Returns the inverse of the Fisher transformation"""
    pass


@agg_func(statistics.median, count_text=True)
def MEDIAN(value1: CellValue, *values: CellValue):
    """Returns the median of the given numbers"""
    pass


@agg_func(statistics.multimode)
def mode(*arg):
    pass


class MODE:
    @staticmethod
    def SNGL(value1: CellValue, *values: CellValue) -> Numeric:
        """Returns the most common value in a data set"""
        modes = mode(value1, *values)
        return modes[0] if isinstance(modes, list) and len(modes) <= 1 else NA_ERROR

    @staticmethod
    def MULT(value1: CellValue, *values: CellValue) -> CellRange:
        """Returns a vertical array of the most frequently occurring, or repetitive values in an array or range of data"""
        modes = mode(value1, *values)
        # We have to return a vertical array
        return (
            CellRange([[mode] for mode in modes])
            if isinstance(modes, list) and len(modes) > 1
            else NA_ERROR
        )


@mat_func(
    lambda a, k: np.partition(a.flatten(), -k)[-k] if k > 0 else NUM_ERROR,
    NUM_ERROR,
    args_to_numpy=[0],
)
def LARGE(array: Matrix, k: int) -> Numeric:
    """Returns the k-th largest value in a data set"""
    pass


@mat_func(
    lambda a, k: np.partition(a.flatten(), k - 1)[k - 1] if k > 0 else NUM_ERROR,
    NUM_ERROR,
    args_to_numpy=[0],
)
def SMALL(array: Matrix, k: int) -> Numeric:
    """Returns the k-th smallest value in a data set"""
    pass


def _percentile_exc(a, k):
    n = len(a)
    n1 = 1 / (n + 1)
    if 0 < k < 1 and len(a) and n1 <= k <= n * n1:
        return np.percentile(a, int(k * 100), method="weibull")
    return NUM_ERROR


class PERCENTILE:
    @staticmethod
    @mat_func(
        lambda a, k: np.percentile(a, int(k * 100))
        if 0 <= k <= 1 and len(a)
        else NUM_ERROR,
        VALUE_ERROR,
        args_to_numpy=[0],
    )
    def INC(array: Matrix, k: Numeric) -> Numeric:
        """Returns the k-th percentile of values in a range"""
        pass

    @staticmethod
    @mat_func(_percentile_exc, VALUE_ERROR, args_to_numpy=[0])
    def EXC(array: Matrix, k: Numeric) -> Numeric:
        """Returns the k-th percentile of values in a range, where k is in the range 0..1, exclusive"""
        pass


def _percent_rank(arr, score, sig_digits=3, exc=False):
    data_len = len(arr)
    if not data_len or sig_digits < 1:
        return NUM_ERROR
    op = operator.__lt__  # operator.__le__ if exc else operator.__lt__
    bound_len = data_len + 1 if exc else data_len - 1
    arr = np.asarray(arr)
    if score in arr:
        small = op(arr, score).sum() + int(exc)
        return round_to_digits_func(small / bound_len, sig_digits, int)
    else:
        if score < arr.min() or score > arr.max():
            return NA_ERROR
        else:
            arr = np.sort(arr)
            position = np.searchsorted(arr, score)
            small = arr[position - 1]
            large = arr[position]
            small_rank = (op(arr, score).sum() - int(not exc)) / bound_len
            large_rank = (op(arr, score).sum() + int(exc)) / bound_len
            step = (score - small) / (large - small)
            rank = small_rank + step * (large_rank - small_rank)

            return round_to_digits_func(rank, sig_digits, int)


class PERCENTRANK:
    @staticmethod
    @mat_func(
        lambda a, x, s: _percent_rank(a, x, s, exc=False), NUM_ERROR, args_to_numpy=[0]
    )
    def INC(array: Matrix, x: Numeric, significance: int = 3) -> Numeric:
        """Returns the percentage rank of a value in a data set"""
        pass

    @staticmethod
    @mat_func(
        lambda a, x, s: _percent_rank(a, x, s, exc=True), NUM_ERROR, args_to_numpy=[0]
    )
    def EXC(array: Matrix, x: Numeric, significance: int = 3) -> Numeric:
        """Returns the percentage rank of a value in a data set"""
        pass


@mat_func(
    lambda a1, a2: pearsonr(a1, a2)[0], NA_ERROR, shape_error=NA_ERROR, eq_shapes=True
)
def PEARSON(array1: Matrix, array2: Matrix) -> Numeric:
    """Returns the Pearson product moment correlation coefficient"""
    pass


class QUARTILE:
    @staticmethod
    @mat_func(
        lambda a, k: np.percentile(a, k * 25) if 0 <= k <= 4 and len(a) else NUM_ERROR,
        VALUE_ERROR,
        args_to_numpy=[0],
    )
    def INC(array: Matrix, k: int) -> Numeric:
        """Returns the quartile of a data set"""
        pass

    @staticmethod
    @mat_func(
        lambda a, k: _percentile_exc(a, k * 0.25) if 1 <= k <= 3 else NUM_ERROR,
        VALUE_ERROR,
        args_to_numpy=[0],
    )
    def EXC(array: Matrix, k: int) -> Numeric:
        """Returns the quartile of the data set, based on percentile values from 0..1, exclusive"""
        pass


class NORM:
    @staticmethod
    @num_func(lambda x, m, s, c: norm.cdf(x, m, s) if c else norm.pdf(x, m, s))
    def DIST(
        x: Numeric, mean: Numeric, standard_dev: Numeric, cumulative: BooleanValue
    ) -> Numeric:
        """Returns the normal cumulative distribution"""
        pass

    @staticmethod
    @num_func(
        lambda p, m, sd: norm.ppf(p, m, sd) if 0 <= p <= 1 and sd > 0 else NUM_ERROR
    )
    def INV(probability: Numeric, mean: Numeric, stdev: Numeric) -> Numeric:
        """Returns the inverse of the normal cumulative distribution"""
        pass

    class S:
        @staticmethod
        @num_func(lambda z, c: norm.cdf(z) if c else norm.pdf(z))
        def DIST(z: Numeric, cumulative: BooleanValue) -> Numeric:
            """Returns the standard normal cumulative distribution"""
            pass

        @staticmethod
        @num_func(lambda p: norm.ppf(p) if 0 <= p <= 1 else NUM_ERROR)
        def INV(probability: Numeric) -> Numeric:
            """Returns the inverse of the standard normal cumulative distribution"""
            pass


def COUNT(value1: CellValue, *values: CellValue) -> int:
    """Counts how many numbers, dates, boolean values or a text representation of numbers are in the list of arguments"""

    def _cnt(elem):
        if isinstance(elem, Iterable) and not isinstance(elem, str):
            return sum(_cnt(x) for x in elem)
        if isinstance(elem, float | int | BooleanValue):
            return 1
        return 0

    try:
        args = [value1, *list(values)]
        return _cnt(args)
    except (RecursionError, TypeError, ValueError):
        return VALUE_ERROR


@agg_func(
    statistics.mean,
    count_text=False,
    count_bool=False,
    count_empty=False,
    make_list=True,
    result_if_zero=ZERO_DIV_ERROR,
)
def AVERAGE(value1: CellValue, *values: CellValue) -> Numeric:
    """Returns the average of its arguments"""
    pass


@mat_func(lambda y, x: linregress(x, y).intercept, ZERO_DIV_ERROR)
def INTERCEPT(known_ys: Matrix, known_xs: Matrix) -> Numeric:
    """Returns the intercept of the linear regression line"""
    pass


@agg_func(
    lambda args: max(args) if len(args) else 0,
    make_list=True,
    count_text=False,
    count_bool=False,
)
def MAX(value1: CellValue, *values: CellValue) -> Numeric:
    """Returns the maximum value in a list of arguments"""
    pass


@agg_func(
    lambda args: max(args) if len(args) else 0,
    make_list=True,
    count_bool=True,
    count_text=True,
    bool_as_num=True,
)
def MAXA(value1: CellValue, *values: CellValue) -> Numeric:
    """Returns the maximum value in a list of arguments, including numbers, text, and logical values"""
    pass


@agg_func(
    lambda args: min(args) if len(args) else 0,
    make_list=True,
    count_text=False,
    count_bool=False,
)
def MIN(value1: CellValue, *values: CellValue) -> Numeric:
    """Returns the minimum value in a list of arguments"""
    pass


@agg_func(
    lambda args: min(args) if len(args) else 0,
    make_list=True,
    count_bool=True,
    count_text=True,
    bool_as_num=True,
)
def MINA(value1: CellValue, *values: CellValue) -> Numeric:
    """Returns the minimum value in a list of arguments, including numbers, text, and logical values"""
    pass


def _rank(n, rng, o, method="average"):
    flat_rng = list(rng if o else -rng)

    ranks = rankdata(flat_rng, method=method)
    try:
        return ranks[flat_rng.index(n if o else -n)]
    except ValueError:
        return NA_ERROR


class RANK:
    @staticmethod
    @mat_func(lambda n, r, o: _rank(n, r, o, "average"), NUM_ERROR)
    def AVG(number: Numeric, ref: CellValue, order: int = 0) -> Numeric:
        """Returns the rank of a number in a list of numbers"""
        pass

    @staticmethod
    @mat_func(lambda n, r, o: _rank(n, r, o, "min"), NUM_ERROR)
    def EQ(number: Numeric, ref: CellValue, order: int = 0) -> Numeric:
        """Returns the rank of a number in a list of numbers"""
        pass


@mat_func(
    lambda y, x: ZERO_DIV_ERROR
    if x.shape != y.shape or len(y) < 2
    else linregress(x, y).rvalue ** 2,
    ZERO_DIV_ERROR,
)
def RSQ(known_ys: Matrix, known_xs: Matrix) -> Numeric:
    """Returns the square of the Pearson product moment correlation coefficient"""
    pass


def _skew(args, p=False):
    n = len(args)
    if n < 3:
        return ZERO_DIV_ERROR
    s = statistics.pstdev(args) if p else statistics.stdev(args)
    if not s:
        return ZERO_DIV_ERROR
    u = statistics.mean(args)
    if p:
        return sum(((x - u) / s) ** 3 for x in args) / n
    else:
        return n / ((n - 1) * (n - 2)) * sum(((x - u) / s) ** 3 for x in args)


@num_func(norm.pdf)
def PHI(number: Numeric) -> Numeric:
    """Returns the value of the density function for a standard normal distribution"""
    pass


@agg_func(
    lambda args: _skew(args, False), make_list=True, bool_as_num=True, count_text=False
)
def SKEW(number: CellValue, *numbers: CellValue) -> Numeric:
    """Returns the skewness of a distribution"""
    pass


@agg_func(
    lambda args: _skew(args, True), make_list=True, bool_as_num=True, count_text=False
)
def _skew_p(number: CellValue, *numbers: CellValue) -> Numeric:
    """Returns the skewness of a distribution based on a population: a characterization of the degree of asymmetry of a distribution around its mean"""
    pass


SKEW.P = _skew_p


@mat_func(lambda y, x: linregress(x, y).slope, ZERO_DIV_ERROR)
def SLOPE(known_ys: Matrix, known_xs: Matrix) -> Numeric:
    """Returns the slope of the linear regression line"""
    pass


@agg_func(statistics.stdev)
def STDEV(value1: CellValue, *values: CellValue) -> Numeric:
    """Estimates standard deviation based on a sample"""
    pass


@agg_func(statistics.pstdev)
def STDEVP(value1: CellValue, *values: CellValue) -> Numeric:
    """Calculates standard deviation based on the entire population"""
    pass


STDEV.S = STDEV
STDEV.P = STDEVP


@agg_func(statistics.stdev, count_text=True, bool_as_num=True)
def STDEVA(value1: CellValue, *values: CellValue) -> Numeric:
    """Estimates standard deviation based on a sample, including numbers, text, and logical values"""
    pass


@agg_func(statistics.pstdev, count_text=True, bool_as_num=True)
def STDEVPA(value1: CellValue, *values: CellValue) -> Numeric:
    """Calculates standard deviation based on the entire population, including numbers, text, and logical values"""
    pass


def _steyx(y, x):
    res = linregress(x, y)
    y_pred = res.slope * x + res.intercept
    return math.sqrt(((y - y_pred) ** 2).sum() / (len(x) - 2))


@mat_func(_steyx, ZERO_DIV_ERROR)
def STEYX(known_ys: Matrix, known_xs: Matrix) -> Numeric:
    """Returns the standard error of the predicted y-value for each x in the regression"""
    pass


@num_func(lambda x, m, s: (x - m) / s if s > 0 else ZERO_DIV_ERROR)
def STANDARDIZE(x: Numeric, mean: Numeric, st_dev: Numeric) -> Numeric:
    """Returns a normalized value"""
    pass


@agg_func(statistics.variance)
def VAR(value1: CellValue, *values: CellValue) -> Numeric:
    """Estimates variance based on a sample"""
    pass


@agg_func(statistics.pvariance)
def VARP(value1: CellValue, *values: CellValue) -> Numeric:
    """Calculates variance based on the entire population"""
    pass


VAR.S = VAR
VAR.P = VARP


@agg_func(statistics.variance, count_text=True, bool_as_num=True)
def VARA(value1: CellValue, *values: CellValue) -> Numeric:
    """Estimates variance based on a sample, including numbers, text, and logical values"""
    pass


@agg_func(statistics.pvariance, count_text=True, bool_as_num=True)
def VARPA(value1: CellValue, *values: CellValue) -> Numeric:
    """Calculates variance based on the entire population, including numbers, text, and logical values"""
    pass


_subtotal_func_codes = {
    1: AVERAGE,
    2: COUNT,
    3: COUNTA,
    4: MAX,
    5: MIN,
    6: PRODUCT,
    7: STDEV.S,
    8: STDEV.P,
    9: SUM,
    10: VAR.S,
    11: VAR.P,
    101: AVERAGE,
    102: COUNT,
    103: COUNTA,
    104: MAX,
    105: MIN,
    106: PRODUCT,
    107: STDEV.S,
    108: STDEV.P,
    109: SUM,
    110: VAR.S,
    111: VAR.P,
}

_aggregate_func_codes = {
    1: AVERAGE,
    2: COUNT,
    3: COUNTA,
    4: MAX,
    5: MIN,
    6: PRODUCT,
    7: STDEV.S,
    8: STDEV.P,
    9: SUM,
    10: VAR.S,
    11: VAR.P,
    12: MEDIAN,
    13: MODE.SNGL,
    14: LARGE,
    15: SMALL,
    16: PERCENTILE.INC,
    17: QUARTILE.INC,
    18: PERCENTILE.EXC,
    19: QUARTILE.EXC,
}


def AGGREGATE(
    function_num: int, options: int, ref1: CellRange, *refs: tuple[CellValue]
):
    """Returns an aggregate in a list or database"""
    # TODO: Ignoring nested SUBTOTAL, AGGREGATE and hidden rows is not supported
    if function_num not in _aggregate_func_codes or not 0 <= options <= 7:
        return VALUE_ERROR
    if function_num >= 14:
        # Second argument required
        if len(refs) != 1 or not isinstance(refs[0], int | float):
            return VALUE_ERROR
        args = [ref1]
    else:
        args = [ref1, *refs]
    try:
        if options in [2, 3, 6, 7]:
            args = [_flatten_range(arg, ignore_errors=True) for arg in args]
        return (
            _aggregate_func_codes[function_num](*args)
            if function_num <= 13
            else _aggregate_func_codes[function_num](list(args[0]), refs[0])
        )
    except TypeError:
        return VALUE_ERROR


def SUBTOTAL(function_code: int, range1: CellRange, *ranges: CellRange) -> Numeric:
    """Returns a subtotal in a list or database"""
    if function_code not in _subtotal_func_codes:
        return VALUE_ERROR
    return _subtotal_func_codes[function_code](range1, *ranges)


class WEIBULL:
    @staticmethod
    @num_func(
        lambda x, a, b, c: NUM_ERROR
        if x < 0 or a <= 0 or b <= 0
        else (1 - math.exp(-((x / b) ** a)))
        if c
        else ((a / b**a) * (x ** (a - 1)) * math.exp(-((x / b) ** a)))
    )
    def DIST(x: Numeric, alpha: Numeric, beta: Numeric, cumulative: BooleanValue):
        """Returns the Weibull distribution"""
        pass


class Z:
    @staticmethod
    def TEST(array: Matrix, x: Numeric, sigma=None) -> Numeric:
        """Returns the one-tailed probability-value of a z-test"""
        if not len(array):
            return NA_ERROR
        n = COUNT(array)
        if isinstance(n, SpreadsheetError):
            return n
        avg = AVERAGE(array)
        if isinstance(avg, SpreadsheetError):
            return avg
        if sigma is None:
            return 1 - NORM.S.DIST((avg - x) / (STDEV.S(array) / math.sqrt(n)), TRUE)
        else:
            return 1 - NORM.S.DIST((avg - x) / (sigma / math.sqrt(n)), TRUE)


def _ttest(a1, a2, tails, ttype):
    tails = int(tails)
    ttype = int(ttype)
    if tails < 1 or tails > 2:
        return NUM_ERROR
    if ttype == 1:
        if a1.shape != a2.shape:
            return NA_ERROR
        pvalue = ttest_rel(a1, a2)[1]
    elif ttype in [2, 3]:
        pvalue = ttest_ind(a1, a2, equal_var=ttype == 2)[1]
    else:
        return NUM_ERROR
    return pvalue / 2 if tails == 1 else pvalue


@num_func(
    lambda x, df, c: NUM_ERROR
    if df < 1
    else t.cdf(x, int(df))
    if c
    else t.pdf(x, int(df))
)
def _t_dist(x: Numeric, deg_freedom: int, cumulative: BooleanValue) -> Numeric:
    """Returns the Percentage Points (probability) for the Student t-distribution"""
    pass


@num_func(lambda x, df: NUM_ERROR if df < 1 else t.sf(x, int(df)) * 2)
def _t_dist_2t(x: Numeric, deg_freedom: int) -> Numeric:
    """Returns the Percentage Points (probability) for the Student t-distribution"""
    pass


@num_func(lambda x, df: NUM_ERROR if df < 1 else t.sf(x, int(df)))
def _t_dist_rt(x: Numeric, deg_freedom: int) -> Numeric:
    """Returns the Student's t-distribution"""
    pass


@num_func(lambda p, df: NUM_ERROR if p <= 0 or p > 1 or df < 1 else t.ppf(p, int(df)))
def _t_inv(probability: Numeric, deg_freedom: int) -> Numeric:
    """Returns the t-value of the Student's t-distribution as a function of the probability and the degrees of freedom"""
    pass


@num_func(
    lambda p, df: NUM_ERROR if p <= 0 or p > 1 or df < 1 else t.ppf(1 - p / 2, int(df))
)
def _t_inv_2t(probability: Numeric, deg_freedom: int) -> Numeric:
    """Returns the inverse of the Student's t-distribution"""
    pass


@mat_func(_ttest, VALUE_ERROR, args_to_numpy=[0, 1])
def _t_test(array1: Matrix, array2: Matrix, tails: int, test_type: int) -> Numeric:
    """Returns the probability associated with a Student's t-test"""
    pass


def T(value: CellValue) -> str:
    """Converts its arguments to text"""
    if isinstance(value, str):
        return value
    return ""


T.DIST = _t_dist
T.DIST_2T = _t_dist_2t
T.DIST.RT = _t_dist_rt
T.INV = _t_inv
T.INV_2T = _t_inv_2t
T.TEST = _t_test


def _linregress(
    known_data_y: Matrix,
    known_data_x: Matrix = None,
    new_data_x: Matrix = None,
    b: BooleanValue = TRUE,
    log=False,
    stats=False,
    advanced_stats=False,
):
    y_shape = known_data_y.shape
    n = len(known_data_y)

    if known_data_x is None:
        x = np.arange(len(known_data_y))
        y = np.ravel(known_data_y)
    elif y_shape != known_data_x.shape:
        if len(y_shape) > 1:
            return REF_ERROR
        if len(known_data_x.shape) > 1:
            if known_data_x.shape[1] == n:
                x = known_data_x.T
            elif known_data_x.shape[0] == n:
                x = known_data_x
            else:
                return REF_ERROR
        else:
            return REF_ERROR
        y = known_data_y
    else:
        y = np.ravel(known_data_y)
        # scipy.stats.linregress fails when all X values are equal
        if np.all(known_data_x == known_data_x.flat[0]):
            x = known_data_x.astype(float)
            x.flat[0] = x.flat[0] + 1e-12
        else:
            x = np.ravel(known_data_x)

    if new_data_x is None:
        new_data_x = x

    if log:
        y = np.log(y)

    if b:
        x = smapi.add_constant(x)

    model = smapi.OLS(y, x, hasconst=b)
    res = model.fit()
    if stats:
        coeffs = list(res.params)[::-1]
        if not b:
            coeffs.append(0)
        if log:
            coeffs = np.exp(coeffs)
        if not advanced_stats:
            return CellRange(coeffs)

        errors = list(res.bse)[::-1]
        if not b:
            errors.append(NA_ERROR)
        err_to_add = [NA_ERROR] * (len(coeffs) - 2)
        sey = (res.ssr / res.df_resid) ** 0.5
        return CellRange(
            [
                coeffs,
                errors,
                [res.rsquared, sey, *err_to_add],
                [res.fvalue, res.df_resid, *err_to_add],
                [res.ess, res.ssr, *err_to_add],
            ]
        )
    else:
        if b:
            pred_y = np.flip(res.params)[:-1] * new_data_x + res.params[0]
        else:
            pred_y = np.flip(res.params) * new_data_x
        return np.exp(pred_y) if log else pred_y


@mat_func(
    lambda y, x, b, s: _linregress(y, x, None, b, False, True, s),
    VALUE_ERROR,
    args_to_numpy=[0, 1],
)
def LINEST(
    known_data_y: Matrix,
    known_data_x: Matrix = None,
    b: BooleanValue = TRUE,
    stats: BooleanValue = FALSE,
) -> CellRange:
    """Returns the parameters of a linear trend"""
    pass


@mat_func(
    lambda y, x, b, s: _linregress(y, x, None, b, True, True, s),
    VALUE_ERROR,
    args_to_numpy=[0, 1],
)
def LOGEST(
    known_data_y: Matrix,
    known_data_x: Matrix = None,
    b: BooleanValue = TRUE,
    stats: BooleanValue = FALSE,
) -> CellRange:
    """Returns the parameters of an exponential trend"""
    pass


@mat_func(_linregress, VALUE_ERROR, args_to_numpy=[0, 1, 2])
def TREND(
    known_data_y: Matrix,
    known_data_x: Matrix = None,
    new_data_x: Matrix = None,
    b: BooleanValue = TRUE,
) -> CellRange:
    """Returns values along a linear trend"""
    pass


@mat_func(
    lambda x, Y, X: _linregress(Y, X, x)[0],
    VALUE_ERROR,
    args_to_numpy=[1, 2],
    eq_shapes=[1, 2],
)
def FORECAST(
    x: Numeric,
    known_data_y: Matrix,
    known_data_x: Matrix,
) -> Numeric:
    """Returns a future value based on existing values"""
    pass


FORECAST.LINEAR = FORECAST


@mat_func(lambda a, p: trim_mean(a, p / 2), VALUE_ERROR, args_to_numpy=[0])
def TRIMMEAN(array: Matrix, percent: float):
    """Returns the mean of the interior of a data set"""
    pass


def FORECAST_ETS(
    target_date: SpreadsheetDateTime,
    values: Matrix,
    timeline: Matrix,
    seasonality: int = 1,
    data_completion: int = 1,
    aggregation: int = 1,
):
    """Returns a future value based on existing (historical) values by using the AAA version of the Exponential Smoothing (ETS) algorithm"""
    raise NotImplementedError


def _frequency(data, bins):
    bins = bins.ravel()
    bins.sort()
    freqs = [0] * (len(bins) + 1)
    for item in data.ravel():
        for i, bin_val in enumerate(bins):
            if item <= bin_val:
                freqs[i] += 1
                break
        else:
            freqs[len(bins)] += 1
    if not len(bins):
        freqs = [0, *freqs]
    return freqs


@mat_func(
    lambda d, b: np.atleast_2d(_frequency(d, b)).T, VALUE_ERROR, args_to_numpy=True
)
def FREQUENCY(data_array: Matrix, bins_array: Matrix) -> CellRange:
    """Returns a frequency distribution as a vertical array"""
    pass


@num_func(
    lambda x: NUM_ERROR if x == 0 or (x < 0 and isinstance(x, int)) else math.gamma(x)
)
def GAMMA(number: Numeric) -> Numeric:
    """Returns the Gamma function value"""
    pass


def _gamma(func, x, a, b):
    if x < 0 or a <= 0 or b <= 0:
        return NUM_ERROR
    return func(x, a, scale=b, loc=0)


@num_func(
    lambda x, a, b, c: _gamma(gamma.cdf, x, a, b) if c else _gamma(gamma.pdf, x, a, b)
)
def _gamma_dist(
    x: Numeric, alpha: Numeric, beta: Numeric, cumulative: BooleanValue
) -> Numeric:
    """Returns the gamma distribution"""
    pass


@num_func(lambda x, a, b: _gamma(gamma.ppf, x, a, b) if 0 <= x <= 1 else NUM_ERROR)
def _gamma_inv(probability: Numeric, alpha: Numeric, beta: Numeric) -> Numeric:
    """Returns the inverse of the gamma cumulative distribution"""
    pass


GAMMA.DIST = _gamma_dist
GAMMA.INV = _gamma_inv


@num_func(lambda x: math.log(math.gamma(x)) if x > 0 else NUM_ERROR)
def GAMMALN(number: Numeric) -> Numeric:
    """Returns the natural logarithm of the gamma function, Γ(x)"""
    pass


GAMMALN.PRECISE = GAMMALN


@num_func(lambda x: norm.cdf(x) - 0.5)
def GAUSS(number: Numeric) -> Numeric:
    """Returns 0.5 less than the standard normal cumulative distribution"""
    pass


@agg_func(
    lambda args: statistics.geometric_mean(args) if any(args) else 0,
    count_text=True,
    make_list=True,
)
def GEOMEAN(value1: CellValue, *values: CellValue):
    """Returns the geometric mean"""
    pass


@mat_func(
    lambda y, x, xn, b: _linregress(y, x, xn, b, log=True),
    VALUE_ERROR,
    args_to_numpy=[0, 1, 2],
)
def GROWTH(
    known_data_y: Matrix,
    known_data_x: Matrix = None,
    new_data_x: Matrix = None,
    b: BooleanValue = TRUE,
) -> CellRange:
    """Returns values along an exponential trend"""
    pass


@agg_func(
    lambda args: statistics.harmonic_mean(args)
    if any(a > 0 for a in args)
    else NUM_ERROR,
    count_text=True,
    make_list=True,
)
def HARMEAN(value1: CellValue, *values: CellValue):
    """Returns the harmonic mean"""
    pass


class HYPGEOM:
    @staticmethod
    @num_func(
        lambda s, ns, p, np, c: NUM_ERROR
        if s < 0
        or s > min(ns, p)
        or s < max(0, ns - np + p)
        or ns <= 0
        or ns > np
        or p <= 0
        or p > np
        or np <= 0
        else hypergeom.cdf(s, np, p, ns)
        if c
        else hypergeom.pmf(s, np, p, ns)
    )
    def DIST(
        sample_s: int,
        number_sample: int,
        population_s: int,
        number_pop: int,
        cumulative: BooleanValue,
    ) -> Numeric:
        """Returns the hypergeometric distribution"""
        pass


def _kurt(args):
    n = len(args)
    if len(args) < 4:
        return ZERO_DIV_ERROR
    s = statistics.stdev(args)
    if not s:
        return ZERO_DIV_ERROR
    xi = statistics.mean(args)
    n1 = n - 1
    n2 = n - 2
    n3 = n - 3
    return n * (n + 1) / (n1 * n2 * n3) * np.sum(
        ((np.array(args) - xi) / s) ** 4
    ) - 3 * n1**2 / (n2 * n3)


@agg_func(
    _kurt,
    count_text=False,
    bool_as_num=False,
    count_bool=False,
    make_list=True,
    result_if_zero=ZERO_DIV_ERROR,
)
def KURT(value1: CellValue, *values: CellValue) -> Numeric:
    """Returns the kurtosis of a data set"""
    pass


@agg_func(
    lambda a: np.sum((np.asarray(a) - np.mean(a)) ** 2),
    count_text=False,
    bool_as_num=False,
    count_bool=False,
    make_list=True,
    result_if_zero=ZERO_DIV_ERROR,
)
def DEVSQ(value1: CellValue, *values: CellValue) -> Numeric:
    """Returns the sum of squares of deviations"""
    pass


class EXPON:
    @staticmethod
    @num_func(
        lambda x, lmb, c: NUM_ERROR
        if x < 0 or lmb <= 0
        else expon.cdf(x, scale=1 / lmb)
        if c
        else expon.pdf(x, scale=1 / lmb)
    )
    def DIST(x: Numeric, Lambda: Numeric, cumulative: BooleanValue) -> Numeric:
        """Returns the exponential distribution"""
        pass


def _permut(n, nc):
    n = int(n)
    nc = int(nc)
    return (
        int(math.factorial(n) / math.factorial(n - nc)) if 0 <= nc <= n else NUM_ERROR
    )


@num_func(_permut)
def PERMUT(number: int, number_chosen: int) -> int:
    """Returns the number of permutations for a given number of objects"""
    pass


@num_func(lambda n, nc: int(n) ** int(nc) if 0 <= nc <= n else NUM_ERROR)
def PERMUTATIONA(number: int, number_chosen: int) -> int:
    """Returns the number of permutations for a given number of objects (with repetitions) that can be selected from the total objects"""
    pass


class NEGBINOM:
    @staticmethod
    @num_func(
        lambda nf, ns, p, c: NUM_ERROR
        if ns < 1 or nf < 0 or p > 1 or p < 0
        else nbinom.cdf(int(nf), int(ns), p)
        if c
        else nbinom.pmf(int(nf), int(ns), p)
    )
    def DIST(
        number_f: int, number_s: int, probability_s: Numeric, cumulative: BooleanValue
    ) -> Numeric:
        """Returns the negative binomial distribution"""
        pass


class POISSON:
    @staticmethod
    @num_func(
        lambda x, m, c: NUM_ERROR
        if x < 0 or m < 0
        else poisson.cdf(int(x), int(m))
        if c
        else poisson.pmf(int(x), int(m))
    )
    def DIST(x: int, mean: int, cumulative: BooleanValue) -> Numeric:
        """Returns the Poisson distribution"""
        pass


class LOGNORM:
    @staticmethod
    @num_func(
        lambda x, m, s, c: NUM_ERROR
        if x <= 0 or s <= 0
        else lognorm.cdf(x, s, scale=math.exp(m))
        if c
        else lognorm.pdf(x, s, scale=math.exp(m))
    )
    def DIST(
        x: Numeric, mean: Numeric, stdev: Numeric, cumulative: BooleanValue
    ) -> Numeric:
        """Returns the cumulative lognormal distribution"""
        pass

    @staticmethod
    @num_func(
        lambda p, m, s: NUM_ERROR
        if p <= 0 or p >= 1 or s <= 0
        else lognorm.ppf(p, s, scale=math.exp(m))
    )
    def INV(probability: Numeric, mean: Numeric, stdev: Numeric) -> Numeric:
        """Returns the inverse of the lognormal cumulative distribution"""
        pass


def _prob(x_range, prob_range, lower_limit, upper_limit):
    if not x_range.size or not prob_range.size:
        return NA_ERROR
    if (
        np.any((prob_range > 1) | (prob_range < 0))
        or abs(np.sum(prob_range) - 1) > 0.0001
    ):
        return NUM_ERROR

    if upper_limit is None:
        upper_limit = lower_limit

    if lower_limit > upper_limit:
        lower_limit, upper_limit = upper_limit, lower_limit

    return prob_range[(x_range >= lower_limit) & (x_range <= upper_limit)].sum()


@mat_func(
    _prob, NUM_ERROR, eq_shapes=[0, 1], shape_error=NA_ERROR, args_to_numpy=[0, 1]
)
def PROB(
    x_range: Matrix,
    prob_range: Matrix,
    lower_limit: Numeric,
    upper_limit: Numeric = None,
) -> Numeric:
    """Returns the probability that values in a range are between two limits"""
    pass


def _averageifs(
    avg_range: CellRange, criteria_range1: CellRange, criteria1, *crit_ranges_criterias
):
    if not len(avg_range) or isinstance(avg_range, str):
        return ZERO_DIV_ERROR

    prep_result = prepare_crit_ranges(
        criteria_range1, criteria1, *crit_ranges_criterias
    )
    if isinstance(prep_result, SpreadsheetError):
        return prep_result
    cranges, crit_args = prep_result

    if not all(len(avg_range) == len(cr) for cr in cranges):
        return VALUE_ERROR

    s = 0

    def _sum_cnt(avg_elem, c_args):
        if isinstance(avg_elem, Iterable) and not isinstance(avg_elem, str):
            return sum(_sum_cnt(a, c_arg) for a, c_arg in zip(avg_elem, c_args))
        if isinstance(avg_elem, SimpleCellValueT):
            crit = all(criteria_func(func, c) for func, c in c_args)
            if crit:
                nonlocal s
                s += avg_elem
            return int(crit)
        return 0

    try:
        cnt = _sum_cnt(avg_range, crit_args)
        if cnt:
            return s / cnt
        else:
            return ZERO_DIV_ERROR
    except TypeError:
        return VALUE_ERROR
    except (RecursionError, ValueError):
        return NUM_ERROR


def AVERAGEIFS(
    avg_range: CellRange,
    criteria_range1: CellRange,
    criteria1: SimpleCellValue,
    *crit_ranges_criterias: tuple[CellRange, SimpleCellValue],
) -> Numeric:
    """Returns the average (arithmetic mean) of all cells that meet multiple criteria"""
    return _averageifs(avg_range, criteria_range1, criteria1, *crit_ranges_criterias)


def AVERAGEIF(
    cell_range: CellRange, criteria: SimpleCellValue, average_range: CellRange = None
) -> Numeric:
    """Returns the average (arithmetic mean) of all the cells in a range that meet a given criteria"""
    if average_range is None:
        average_range = cell_range
    return _averageifs(average_range, cell_range, criteria)


def _countifs(
    criteria_range1: CellRange,
    criteria1: SimpleCellValue,
    *crit_ranges_criterias: tuple[CellRange, SimpleCellValue],
):
    prep_result = prepare_crit_ranges(
        list(_flatten_range(criteria_range1, none_to_zero=True, ignore_errors=True)),
        criteria1,
        *crit_ranges_criterias,
    )
    if isinstance(prep_result, SpreadsheetError):
        return prep_result
    cranges, crit_args = prep_result

    if not all(len(cranges[0]) == len(cr) for cr in cranges[1:]):
        return VALUE_ERROR

    def _cnt(elem):
        if isinstance(elem[0][-1], Iterable) and not isinstance(
            elem[0][-1], str | Empty
        ):
            return sum(_cnt(x) for x in elem)
        if isinstance(elem[0][-1], SimpleCellValueT):
            return int(all(criteria_func(func, c) for func, c in elem))
        return 0

    try:
        return _cnt(crit_args)
    except TypeError:
        return VALUE_ERROR
    except (RecursionError, ValueError):
        return NUM_ERROR


def COUNTIFS(
    criteria_range1: CellRange,
    criteria1: SimpleCellValue,
    *crit_ranges_criterias: tuple[CellRange, SimpleCellValue],
) -> int:
    """Counts the number of cells within a range that meet multiple criteria"""
    return _countifs(criteria_range1, criteria1, *crit_ranges_criterias)


def COUNTIF(cell_range: CellRange, criterion: SimpleCellValue) -> int:
    """Counts the number of cells within a range that meet the given criteria"""
    return _countifs(cell_range, criterion)


def _minmaxifs(
    func,
    extremum_value,
    op_range,
    criteria_range1: CellRange,
    criteria1,
    *crit_ranges: tuple[CellRange, int | float | str],
):
    prep_result = prepare_crit_ranges(criteria_range1, criteria1, *crit_ranges)
    if isinstance(prep_result, SpreadsheetError):
        return prep_result
    cranges, crit_args = prep_result

    if not all(len(op_range) == len(cr) for cr in cranges):
        return VALUE_ERROR

    def _max(term, c_args):
        if isinstance(term, Iterable) and not isinstance(term, str):
            return func(_max(a, c_arg) for a, c_arg in zip(term, c_args))
        if isinstance(term, SimpleCellValueT):
            crit = all(criteria_func(cfunc, c) for cfunc, c in c_args)
            if crit:
                return term
        return extremum_value

    try:
        res = _max(op_range, crit_args)
        if res == extremum_value:
            return 0
        return res
    except TypeError:
        return VALUE_ERROR
    except (RecursionError, ValueError):
        return NUM_ERROR


def MAXIFS(
    max_range: CellRange,
    criteria_range: CellRange,
    criteria: SimpleCellValue,
    *crit_ranges: tuple[CellRange, SimpleCellValue],
) -> Numeric:
    """Returns the maximum value among cells specified by a given set of conditions or criteria"""

    return _minmaxifs(max, -math.inf, max_range, criteria_range, criteria, *crit_ranges)


def MINIFS(
    min_range: CellRange,
    criteria_range: CellRange,
    criteria: SimpleCellValue,
    *crit_ranges: tuple[CellRange, SimpleCellValue],
) -> Numeric:
    """Returns the minimum value among cells specified by a given set of conditions or criteria"""

    return _minmaxifs(min, math.inf, min_range, criteria_range, criteria, *crit_ranges)
