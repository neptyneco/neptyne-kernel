from collections import defaultdict
from functools import reduce, wraps
from operator import __mul__
from statistics import StatisticsError, mean, pstdev, pvariance, stdev, variance

from ..cell_range import CellRange
from ..spreadsheet_error import NUM_ERROR, VALUE_ERROR
from .helpers import Numeric, SimpleCellValue, criteria_func, parse_criteria

DBField = int | str


def _dget(seq):
    value = None
    for val in seq:
        if value is None:
            value = val
        else:
            return NUM_ERROR

    return value if value is not None else VALUE_ERROR


def _get_column_ind_by_name(rng: CellRange, col: str, is2d: bool):
    if is2d:
        try:
            return list(rng[0]).index(col)
        except ValueError:
            return
    else:
        return 0 if rng[0] == col else None


def true_func(x):
    return True


def _get_criteria(rng: CellRange) -> list:
    def calc_criteria_col(value):
        return (
            true_func
            if value is None or value == ""
            else parse_criteria(value, empty_as_none=True)
        )

    result = []
    if len(rng.shape) == 2:
        for val in rng[1:]:
            res = defaultdict(list)
            for j, col in enumerate(rng[0]):
                cr = calc_criteria_col(val[j])
                if cr:
                    res[col].append(cr)
            result.append(res)
        return list(rng[0]), result
    else:
        for val in rng[1:]:
            cr = calc_criteria_col(val)
            if cr:
                result.append({rng[0]: [cr]})
        return [rng[0]], result


def db_iter(database, crits, ind_col, inds_crit, is2d):
    for row in database[1:]:
        if any(
            all(
                all(criteria_func(c, row[i] if is2d else row) for c in crit_row[key])
                for i, key in inds_crit.items()
            )
            for crit_row in crits
        ):
            yield row[ind_col] if is2d else row


def db_func(func):
    def decorator(f):
        @wraps(f)
        def wrapper(database: CellRange, field, criteria):
            is2d = len(database.shape) == 2
            if isinstance(field, str):
                ind_col = _get_column_ind_by_name(database, field, is2d)
                if ind_col is None:
                    return VALUE_ERROR
            elif isinstance(field, int):
                if field < 1 or (is2d and field > database.shape[1]):
                    return VALUE_ERROR
                ind_col = field - 1

            keys, crits = _get_criteria(criteria)
            inds_crit = {}

            for key in keys:
                if key not in database[0]:
                    return VALUE_ERROR
                inds_crit[_get_column_ind_by_name(database, key, is2d)] = key

            try:
                return func(db_iter(database, crits, ind_col, inds_crit, is2d))
            except TypeError:
                return VALUE_ERROR
            except (ValueError, StatisticsError):
                return NUM_ERROR

        return wrapper

    return decorator


@db_func(mean)
def DAVERAGE(database: CellRange, field: DBField, criteria: CellRange) -> Numeric:
    """Returns the average of selected database entries"""
    pass


@db_func(
    lambda seq: len(
        [val for val in seq if val is not None and not isinstance(val, str)]
    )
)
def DCOUNT(database: CellRange, field: DBField, criteria: CellRange) -> Numeric:
    """Counts the cells that contain numbers in a database"""
    pass


@db_func(lambda seq: len(list(seq)))
def DCOUNTA(database: CellRange, field: DBField, criteria: CellRange) -> Numeric:
    """Counts nonblank cells in a database"""
    pass


@db_func(_dget)
def DGET(database: CellRange, field: DBField, criteria: CellRange) -> SimpleCellValue:
    """Extracts from a database a single record that matches the specified criteria"""
    pass


@db_func(max)
def DMAX(database: CellRange, field: DBField, criteria: CellRange) -> Numeric:
    """Returns the maximum value from selected database entries"""
    pass


@db_func(min)
def DMIN(database: CellRange, field: DBField, criteria: CellRange) -> Numeric:
    """Returns the minimum value from selected database entries"""
    pass


@db_func(lambda seq: reduce(__mul__, seq))
def DPRODUCT(database: CellRange, field: DBField, criteria: CellRange) -> Numeric:
    """Multiplies the values in a particular field of records that match the criteria in a database"""
    pass


@db_func(stdev)
def DSTDEV(database: CellRange, field: DBField, criteria: CellRange) -> Numeric:
    """Estimates the standard deviation based on a sample of selected database entries"""
    pass


@db_func(pstdev)
def DSTDEVP(database: CellRange, field: DBField, criteria: CellRange) -> Numeric:
    """Calculates the standard deviation based on the entire population of selected database entries"""
    pass


@db_func(sum)
def DSUM(database: CellRange, field: DBField, criteria: CellRange) -> Numeric:
    """Adds the numbers in the field column of records in the database that match the criteria"""
    pass


@db_func(variance)
def DVAR(database: CellRange, field: DBField, criteria: CellRange) -> Numeric:
    """Estimates variance based on a sample from selected database entries"""
    pass


@db_func(pvariance)
def DVARP(database: CellRange, field: DBField, criteria: CellRange) -> Numeric:
    """Calculates variance based on the entire population of selected database entries"""
    pass
