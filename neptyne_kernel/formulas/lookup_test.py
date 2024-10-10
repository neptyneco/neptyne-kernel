# ruff: noqa: F405
import pytest

from ..cell_range import CellRange
from ..spreadsheet_error import CALC_ERROR, NA_ERROR, REF_ERROR, VALUE_ERROR
from .boolean import FALSE, TRUE
from .helpers import assert_equal, search_wildcard
from .lookup import (
    ADDRESS,
    CHOOSE,
    CHOOSECOLS,
    CHOOSEROWS,
    COLUMNS,
    DROP,
    EXPAND,
    FILTER,
    HLOOKUP,
    HSTACK,
    INDEX,
    LOOKUP,
    MATCH,
    ROWS,
    SORT,
    SORTBY,
    TAKE,
    TOCOL,
    TOROW,
    TRANSPOSE,
    UNIQUE,
    VLOOKUP,
    VSTACK,
    WRAPCOLS,
    WRAPROWS,
    XLOOKUP,
    XMATCH,
)
from .mathtrig import *  # noqa: F403

TEST_LOOKUP_1 = CellRange([["world", "1", 2], ["hello", "3", 4]])

TEST_VLOOKUP_1 = CellRange([["N444", 90], ["N333", 100], ["N222", 85], ["N111", 80]])


@pytest.mark.parametrize(
    "search_key, c_range, index, range_lookup, result",
    [
        ("hello", TEST_LOOKUP_1, 6, FALSE, VALUE_ERROR),
        ("hello", TEST_LOOKUP_1, 3, FALSE, 4),
        ("lorem", TEST_LOOKUP_1, 3, FALSE, NA_ERROR),
        ("he?lo", TEST_LOOKUP_1, 3, FALSE, 4),
        ("*lo", TEST_LOOKUP_1, 3, FALSE, 4),
        (2, CellRange([[1, 2], [3, 4]]), 2, TRUE, 2),
        ("N111", TEST_VLOOKUP_1, 2, FALSE, 80),
        ("N222", TEST_VLOOKUP_1, 2, FALSE, 85),
        ("N333", TEST_VLOOKUP_1, 2, FALSE, 100),
        ("N444", TEST_VLOOKUP_1, 2, FALSE, 90),
    ],
)
def test_VLOOKUP(search_key, c_range, index, range_lookup, result):
    assert VLOOKUP(search_key, c_range, index, range_lookup) == result


TEST_HLOOKUP_1 = CellRange(
    [["Axles", "Bearings", "Bolts"], [4, 4, 9], [5, 7, 10], [6, 8, 11]]
)


@pytest.mark.parametrize(
    "search_key, c_range, index,  range_lookup, result",
    [
        ("world", TEST_LOOKUP_1, 6, FALSE, VALUE_ERROR),
        ("world", TEST_LOOKUP_1, 2, FALSE, "hello"),
        ("lorem", TEST_LOOKUP_1, 2, FALSE, NA_ERROR),
        ("wo?ld", TEST_LOOKUP_1, 2, FALSE, "hello"),
        ("*ld", TEST_LOOKUP_1, 2, FALSE, "hello"),
        (4, CellRange([[1, 3, 5], [3, 4, 7]]), 2, TRUE, 4),
        ("Axles", TEST_HLOOKUP_1, 2, TRUE, 4),
        ("Bearings", TEST_HLOOKUP_1, 3, FALSE, 7),
        ("B", TEST_HLOOKUP_1, 3, TRUE, 5),
        ("Bolts", TEST_HLOOKUP_1, 4, TRUE, 11),
        (3, CellRange([[1, 2, 3], ["a", "b", "c"], ["d", "e", "f"]]), 2, TRUE, "c"),
    ],
)
def test_HLOOKUP(search_key, c_range, index, range_lookup, result):
    assert HLOOKUP(search_key, c_range, index, range_lookup) == result


TEST_RANGE = CellRange(
    [
        ["David", "Vegetarian", "No", 3],
        ["Bob", "None", "No", 5],
        ["david", "None", "Yes", 1],
        ["Nancy", "None", "No", 4],
        ["Mary", "Vegetarian", "Yes", 2],
    ]
)


@pytest.mark.parametrize(
    "range, row_num, column_num, result",
    [
        (TEST_RANGE, 2, 0, CellRange(["Bob", "None", "No", 5])),
        (TEST_RANGE, 0, 4, CellRange([3, 5, 1, 4, 2])),
        (TEST_RANGE, 0, 0, TEST_RANGE),
        (TEST_RANGE, -1, 4, REF_ERROR),
    ],
)
def test_INDEX(range, row_num, column_num, result):
    assert_equal(INDEX(range, row_num, column_num), result)


MATCH_TEST_RANGE1 = CellRange(
    [["Bananas", 25], ["Oranges", 38], ["Apples", 40], ["Pears", 41]]
)
MATCH_TEST_RANGE2 = CellRange([1030, 1032, 1033, 1036])


@pytest.mark.parametrize(
    "lookup_value, cell_range, match_type, result",
    [
        (12, CellRange(["hello", 12, PRODUCT("hello", 12)]), 1, 3),
        (12, CellRange(["hello", 12, PRODUCT("hello", 12)]), -1, 2),
        (39, MATCH_TEST_RANGE1[:, 1], 1, 2),
        (41, MATCH_TEST_RANGE1[:, 1], 0, 4),
        (40, MATCH_TEST_RANGE1[:, 1], -1, 4),
        (1034, MATCH_TEST_RANGE2, 1, 3),
        (1034, MATCH_TEST_RANGE2, 1, 3),
        (1036, MATCH_TEST_RANGE2, 1, 4),
        (1034, MATCH_TEST_RANGE2, 0, NA_ERROR),
        (1036, MATCH_TEST_RANGE2, 0, 4),
    ],
)
def test_MATCH(lookup_value, cell_range, match_type, result):
    assert MATCH(lookup_value, cell_range, match_type) == result


@pytest.mark.parametrize(
    "range, row_num, column_num, result",
    [
        (TEST_RANGE, MATCH("Mary", TEST_RANGE[:, 0], 0), 2, "Vegetarian"),
    ],
)
def test_INDEX_MATCH(range, row_num, column_num, result):
    assert INDEX(range, row_num, column_num) == result


@pytest.mark.parametrize(
    "source,pattern,result",
    [
        ("color", "co*r", True),
        ("color", "?olor", True),
        ("color", "?o*r", True),
        ("color~", "*~", True),
        ("color*", "*~*", True),
        ("co~*", "*~~*", True),
    ],
)
def test_search_wildcard(source, pattern, result):
    assert search_wildcard(source, pattern) == result


@pytest.mark.parametrize(
    "row_num, column_num, abs_num, a1, sheet_text, result",
    [
        (2, 3, 1, TRUE, "", "$C$2"),
        (2, 3, 2, TRUE, "", "C$2"),
        (2, 3, 2, FALSE, "", "R2C[3]"),
        (2, 3, 1, FALSE, "[Book1]Sheet1", "'[Book1]Sheet1'!R2C3"),
        (2, 3, 1, FALSE, "EXCEL SHEET", "'EXCEL SHEET'!R2C3"),
        (4, 2, 1, TRUE, "", "$B$4"),
        (4, 2, 2, TRUE, "", "B$4"),
        (4, 2, 3, TRUE, "", "$B4"),
        (4, 2, 4, TRUE, "", "B4"),
        (4, 2, 1, FALSE, "", "R4C2"),
        (4, 2, 3, FALSE, "", "R[4]C2"),
        (4, 2, 4, FALSE, "", "R[4]C[2]"),
        (2, 3, -1, TRUE, "", VALUE_ERROR),
        (2, 3, 5, TRUE, "", VALUE_ERROR),
    ],
)
def test_ADDRESS(row_num, column_num, abs_num, a1, sheet_text, result):
    assert ADDRESS(row_num, column_num, abs_num, a1, sheet_text) == result


@pytest.mark.parametrize(
    "index_num, values, result",
    [
        (3, ("Wide", 115, "world", 8), "world"),
        (0, ("Wide", 115, "world", 8), VALUE_ERROR),
        (5, ("Wide", 115, "world", 8), VALUE_ERROR),
    ],
)
def test_CHOOSE(index_num, values, result):
    assert CHOOSE(index_num, *values) == result


CHOOSE_RANGE = CellRange(
    [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
        [26, 27, 28, 29, 30],
    ]
)


@pytest.mark.parametrize(
    "array, col_nums, result",
    [
        (
            CHOOSE_RANGE,
            (1, 3, 5, 1),
            CellRange(
                [
                    [1, 3, 5, 1],
                    [6, 8, 10, 6],
                    [11, 13, 15, 11],
                    [16, 18, 20, 16],
                    [21, 23, 25, 21],
                    [26, 28, 30, 26],
                ]
            ),
        ),
        (
            CHOOSE_RANGE,
            (-5, -3, -1, 1),
            CellRange(
                [
                    [1, 3, 5, 1],
                    [6, 8, 10, 6],
                    [11, 13, 15, 11],
                    [16, 18, 20, 16],
                    [21, 23, 25, 21],
                    [26, 28, 30, 26],
                ]
            ),
        ),
        (CHOOSE_RANGE[:, 0], (1,), CHOOSE_RANGE[:, 0]),
        (CHOOSE_RANGE, (3,), CHOOSE_RANGE[:, 2]),
    ],
)
def test_CHOOSECOLS(array, col_nums, result):
    assert (CHOOSECOLS(array, *col_nums) == result).all()


@pytest.mark.parametrize(
    "array, row_nums, result",
    [
        (
            CHOOSE_RANGE,
            (1, 3, 5, 1),
            CellRange(
                [
                    [1, 2, 3, 4, 5],
                    [11, 12, 13, 14, 15],
                    [21, 22, 23, 24, 25],
                    [1, 2, 3, 4, 5],
                ]
            ),
        ),
        (
            CHOOSE_RANGE,
            (-5, -3, -1, 1),
            CellRange(
                [
                    [6, 7, 8, 9, 10],
                    [16, 17, 18, 19, 20],
                    [26, 27, 28, 29, 30],
                    [1, 2, 3, 4, 5],
                ]
            ),
        ),
        (CellRange([CHOOSE_RANGE[0]]), (1,), CellRange([CHOOSE_RANGE[0]])),
        (CellRange([CHOOSE_RANGE[0]]), (1, 2), VALUE_ERROR),
        (
            CellRange([CHOOSE_RANGE[0]]),
            (1, -1),
            CellRange([CHOOSE_RANGE[0], CHOOSE_RANGE[0]]),
        ),
        (CHOOSE_RANGE, (3,), CellRange([CHOOSE_RANGE[2]])),
    ],
)
def test_CHOOSEROWS(array, row_nums, result):
    assert_equal(CHOOSEROWS(array, *row_nums), result)


@pytest.mark.parametrize(
    "array, result",
    [(CellRange([1, 2, 3, 4]), 4), (CellRange([[1, 2, 3], [4, 5, 6]]), 3)],
)
def test_COLUMNS(array, result):
    assert COLUMNS(array) == result


DROP_RANGE = CellRange([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.mark.parametrize(
    "array,rows,cols,result",
    [
        (DROP_RANGE, 2, None, DROP_RANGE[2:]),
        (DROP_RANGE, None, 2, DROP_RANGE[:, 2:]),
        (DROP_RANGE, -2, None, DROP_RANGE[:1, :]),
        (DROP_RANGE, 2, 2, DROP_RANGE[2:, 2:]),
    ],
)
def test_DROP(array, rows, cols, result):
    assert (DROP(array, rows, cols) == result).all()


EXPAND_RANGE = CellRange([[1, 2], [3, 4]])


@pytest.mark.parametrize(
    "array,rows,columns,pad_with,result",
    [
        (
            EXPAND_RANGE,
            3,
            3,
            None,
            CellRange([[1, 2, None], [3, 4, None], [None, None, None]]),
        ),
        (
            EXPAND_RANGE,
            3,
            3,
            "-",
            CellRange([[1, 2, "-"], [3, 4, "-"], ["-", "-", "-"]]),
        ),
        (EXPAND_RANGE, 1, 3, "-", VALUE_ERROR),
        (EXPAND_RANGE, 3, 1, "-", VALUE_ERROR),
        (
            EXPAND_RANGE[0],
            3,
            3,
            "*",
            CellRange([[1, "*", "*"], [2, "*", "*"], ["*", "*", "*"]]),
        ),
        (
            EXPAND_RANGE[:, 0],
            3,
            3,
            "*",
            CellRange([[1, "*", "*"], [3, "*", "*"], ["*", "*", "*"]]),
        ),
    ],
)
def test_EXPAND(array, rows, columns, pad_with, result):
    assert_equal(EXPAND(array, rows, columns, pad_with), result)


@pytest.mark.parametrize(
    "array, result",
    [
        (
            CellRange([["Jan", 100], ["Feb", 200], ["Mar", 150], ["Apr", 300]]),
            CellRange([["Jan", "Feb", "Mar", "Apr"], [100, 200, 150, 300]]),
        ),
        (CellRange([1, 2, 3]), CellRange([[1, 2, 3]])),
    ],
)
def test_TRANSPOSE(array, result):
    return (TRANSPOSE(array) == result).all()


@pytest.mark.parametrize(
    "array, result",
    [
        (CellRange([[1, 2, 3], [4, 5, 6]]), 2),
        (CellRange([1, 2, 3, 4]), 1),
    ],
)
def test_ROWS(array, result):
    assert ROWS(array) == result


@pytest.mark.parametrize(
    "arrays, result",
    [
        (
            (
                CellRange([["A", "B", "C"], ["D", "E", "F"]]),
                CellRange([["AA", "BB", "CC"], ["DD", "EE", "FF"]]),
            ),
            CellRange(
                [
                    ["A", "B", "C"],
                    ["D", "E", "F"],
                    ["AA", "BB", "CC"],
                    ["DD", "EE", "FF"],
                ]
            ),
        ),
        (
            (
                CellRange([[1, 2], [3, 4], [5, 6]]),
                CellRange([["A", "B"], ["C", "D"]]),
                CellRange([["X", "Y"]]),
            ),
            CellRange([[1, 2], [3, 4], [5, 6], ["A", "B"], ["C", "D"], ["X", "Y"]]),
        ),
        (
            (
                CellRange([1, 2, 3]),
                CellRange([["A", "B"], ["C", "D"]]),
                CellRange([["X", "Y"]]),
            ),
            CellRange(
                [
                    [1, NA_ERROR],
                    [2, NA_ERROR],
                    [3, NA_ERROR],
                    ["A", "B"],
                    ["C", "D"],
                    ["X", "Y"],
                ]
            ),
        ),
        ((CellRange([1]), CellRange([3, 4, 5])), CellRange([[1], [3], [4], [5]])),
        (
            (CellRange([1]), CellRange([[3, 4, 5]])),
            CellRange([[1, NA_ERROR, NA_ERROR], [3, 4, 5]]),
        ),
    ],
)
def test_VSTACK(arrays, result):
    assert (VSTACK(*arrays) == result).all()


@pytest.mark.parametrize(
    "arrays, result",
    [
        (
            (
                CellRange([[1, 2], [3, 4], [5, 6]]),
                CellRange([["A", "B", "C"], ["D", "E", "F"]]),
                CellRange([[1]]),
            ),
            CellRange(
                [
                    [1, 2, "A", "B", "C", 1],
                    [3, 4, "D", "E", "F", NA_ERROR],
                    [5, 6, NA_ERROR, NA_ERROR, NA_ERROR, NA_ERROR],
                ]
            ),
        ),
        (
            (
                CellRange([[1, 2], [3, 4], [5, 6]]),
                CellRange([["A", "B"], ["C", "D"]]),
                CellRange([["X", "Y"]]),
            ),
            CellRange(
                [
                    [1, 2, "A", "B", "X", "Y"],
                    [3, 4, "C", "D", NA_ERROR, NA_ERROR],
                    [5, 6, NA_ERROR, NA_ERROR, NA_ERROR, NA_ERROR],
                ]
            ),
        ),
        (
            (
                CellRange([["A", "B", "C"], ["D", "E", "F"]]),
                CellRange([["AA", "BB", "CC"], ["DD", "EE", "FF"]]),
            ),
            CellRange(
                [["A", "B", "C", "AA", "BB", "CC"], ["D", "E", "F", "DD", "EE", "FF"]]
            ),
        ),
    ],
)
def test_HSTACK(arrays, result):
    assert (HSTACK(*arrays) == result).all()


@pytest.mark.parametrize(
    "array, rows, columns, result",
    [
        (
            CellRange([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            2,
            None,
            CellRange([[1, 2, 3], [4, 5, 6]]),
        ),
        (
            CellRange([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            -2,
            None,
            CellRange([[4, 5, 6], [7, 8, 9]]),
        ),
        (
            CellRange([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            -2,
            -2,
            CellRange([[5, 6], [8, 9]]),
        ),
        (CellRange([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 0, None, VALUE_ERROR),
        (CellRange([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), None, 0, VALUE_ERROR),
        (
            CellRange([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            None,
            -1,
            CellRange([[3], [6], [9]]),
        ),
        (
            CellRange([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            None,
            None,
            CellRange([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ),
        (
            CellRange([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            -4,
            -4,
            CellRange([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ),
        (
            CellRange([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            10,
            10,
            CellRange([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ),
    ],
)
def test_TAKE(array, rows, columns, result):
    assert_equal(TAKE(array, rows, columns), result)


TOCOL_RANGE1 = CellRange(
    [
        ["Ben", "Peter", "Mary", "Sam"],
        ["John", "Hillary", "Jenny", "James"],
        ["Agnes", "Harry", None, None],
    ]
)
TOCOL_RANGE2 = CellRange(
    [["A", "B", None], ["D", "E", "F"], ["G", NA_ERROR, "H"], [None, "J", None]]
)


@pytest.mark.parametrize(
    "array, ignore, scan_by_column, result",
    [
        (
            TOCOL_RANGE1,
            0,
            FALSE,
            CellRange(
                [
                    "Ben",
                    "Peter",
                    "Mary",
                    "Sam",
                    "John",
                    "Hillary",
                    "Jenny",
                    "James",
                    "Agnes",
                    "Harry",
                    None,
                    None,
                ]
            ),
        ),
        (
            TOCOL_RANGE1,
            1,
            FALSE,
            CellRange(
                [
                    "Ben",
                    "Peter",
                    "Mary",
                    "Sam",
                    "John",
                    "Hillary",
                    "Jenny",
                    "James",
                    "Agnes",
                    "Harry",
                ]
            ),
        ),
        (
            TOCOL_RANGE1,
            1,
            TRUE,
            CellRange(
                [
                    "Ben",
                    "John",
                    "Agnes",
                    "Peter",
                    "Hillary",
                    "Harry",
                    "Mary",
                    "Jenny",
                    "Sam",
                    "James",
                ]
            ),
        ),
        (
            TOCOL_RANGE2,
            0,
            FALSE,
            CellRange(
                ["A", "B", None, "D", "E", "F", "G", NA_ERROR, "H", None, "J", None]
            ),
        ),
        (
            TOCOL_RANGE2,
            1,
            FALSE,
            CellRange(["A", "B", "D", "E", "F", "G", NA_ERROR, "H", "J"]),
        ),
        (
            TOCOL_RANGE2,
            2,
            FALSE,
            CellRange(["A", "B", None, "D", "E", "F", "G", "H", None, "J", None]),
        ),
        (TOCOL_RANGE2, 3, FALSE, CellRange(["A", "B", "D", "E", "F", "G", "H", "J"])),
        (TOCOL_RANGE2, -1, FALSE, VALUE_ERROR),
        (TOCOL_RANGE2, 4, FALSE, VALUE_ERROR),
    ],
)
def test_TOCOL(array, ignore, scan_by_column, result):
    assert_equal(TOCOL(array, ignore, scan_by_column), result)


@pytest.mark.parametrize(
    "array, ignore, scan_by_column, result",
    [
        (
            TOCOL_RANGE1,
            0,
            FALSE,
            CellRange(
                [
                    [
                        "Ben",
                        "Peter",
                        "Mary",
                        "Sam",
                        "John",
                        "Hillary",
                        "Jenny",
                        "James",
                        "Agnes",
                        "Harry",
                        None,
                        None,
                    ]
                ]
            ),
        ),
        (
            TOCOL_RANGE1,
            1,
            FALSE,
            CellRange(
                [
                    [
                        "Ben",
                        "Peter",
                        "Mary",
                        "Sam",
                        "John",
                        "Hillary",
                        "Jenny",
                        "James",
                        "Agnes",
                        "Harry",
                    ]
                ]
            ),
        ),
        (
            TOCOL_RANGE1,
            1,
            TRUE,
            CellRange(
                [
                    [
                        "Ben",
                        "John",
                        "Agnes",
                        "Peter",
                        "Hillary",
                        "Harry",
                        "Mary",
                        "Jenny",
                        "Sam",
                        "James",
                    ]
                ]
            ),
        ),
        (
            TOCOL_RANGE2,
            0,
            FALSE,
            CellRange(
                [["A", "B", None, "D", "E", "F", "G", NA_ERROR, "H", None, "J", None]]
            ),
        ),
        (
            TOCOL_RANGE2,
            1,
            FALSE,
            CellRange([["A", "B", "D", "E", "F", "G", NA_ERROR, "H", "J"]]),
        ),
        (
            TOCOL_RANGE2,
            2,
            FALSE,
            CellRange([["A", "B", None, "D", "E", "F", "G", "H", None, "J", None]]),
        ),
        (TOCOL_RANGE2, 3, FALSE, CellRange([["A", "B", "D", "E", "F", "G", "H", "J"]])),
        (TOCOL_RANGE2, -1, FALSE, VALUE_ERROR),
        (TOCOL_RANGE2, 4, FALSE, VALUE_ERROR),
    ],
)
def test_TOROW(array, ignore, scan_by_column, result):
    assert_equal(TOROW(array, ignore, scan_by_column), result)


@pytest.mark.parametrize(
    "array, by_col, exactly_once, result",
    [
        (CellRange([1, 1, 2, 2, 3]), FALSE, FALSE, CellRange([1, 2, 3])),
        (CellRange([1, 1, 2, 2, 3]), FALSE, TRUE, CellRange([3])),
        (CellRange([[1, 1, 2, 2, 3]]), FALSE, FALSE, CellRange([[1, 1, 2, 2, 3]])),
        (CellRange([[1, 1, 2, 2, 3]]), TRUE, TRUE, CellRange([[3]])),
        (CellRange([[1, 1, 2, 2, 3]]), TRUE, FALSE, CellRange([[1, 2, 3]])),
        (
            CellRange([[1, 1, 3], [2, 2, 3], [3, 3, 6]]),
            TRUE,
            FALSE,
            CellRange([[1, 2, 3], [3, 3, 6]]),
        ),
        (
            CellRange([[1, 1, 3], [2, 2, 3], [3, 3, 6]]),
            TRUE,
            TRUE,
            CellRange([[3, 3, 6]]),
        ),
    ],
)
def test_UNIQUE(array, by_col, exactly_once, result):
    assert (UNIQUE(array, by_col, exactly_once) == result).all()


@pytest.mark.parametrize(
    "array, sort_index, sort_order, by_col, result",
    [
        (CellRange([2, 1, 3]), 1, 2, FALSE, VALUE_ERROR),
        (CellRange([2, 1, 3]), 1, 1, TRUE, VALUE_ERROR),
        (CellRange([2, 1, 3]), 1, -1, FALSE, CellRange([3, 2, 1])),
        (CellRange([[2, 1, 3]]), 1, -1, TRUE, CellRange([[3, 2, 1]])),
        (CellRange([[2, 1, 3]]), 1, -1, FALSE, CellRange([[2, 1, 3]])),
        (
            CellRange([622, 961, 691, 445, 378, 483, 650, 783, 142, 404]),
            1,
            -1,
            FALSE,
            CellRange([961, 783, 691, 650, 622, 483, 445, 404, 378, 142]),
        ),
        (
            CellRange([[6, 4], [3, 5], [2, 6]]),
            1,
            1,
            FALSE,
            CellRange([[2, 6], [3, 5], [6, 4]]),
        ),
        (
            CellRange([[6, 4], [3, 5], [2, 6]]),
            1,
            1,
            TRUE,
            CellRange([[4, 6], [5, 3], [6, 2]]),
        ),
        (
            CellRange([[6, 4], [3, 5], [2, 6]]),
            2,
            1,
            TRUE,
            CellRange([[6, 4], [3, 5], [2, 6]]),
        ),
        (CellRange([[6, 4], [3, 5], [2, 6]]), 4, 1, TRUE, VALUE_ERROR),
        (CellRange([[6, 4], [3, 5], [2, 6]]), 0, 1, TRUE, VALUE_ERROR),
    ],
)
def test_SORT(array, sort_index, sort_order, by_col, result):
    assert_equal(SORT(array, sort_index, sort_order, by_col), result)


MULTI_COLUMNS = CellRange(
    [
        ["Edward", 79, "Blue"],
        ["Hannah", 93, "Red"],
        ["Miranda", 85, "Green"],
        ["Willam", 64, "Green"],
        ["Joanna", 81, "Blue"],
        ["Collin", 85, "Green"],
        ["Mallory", 81, "Red"],
        ["Oscar", 63, "Blue"],
        ["Arturo", 79, "Green"],
        ["Annie", 72, "Red"],
    ]
)


@pytest.mark.parametrize(
    "array, arrays_orders, result",
    [
        (
            CellRange(["a", "b", "c", "d"]),
            (CellRange([3, 4, 2, 1]), 1),
            CellRange(["d", "c", "a", "b"]),
        ),
        (
            CellRange(["a", "b", "c", "d"]),
            (CellRange([3, 4, 2, 1]), -1),
            CellRange(["b", "a", "c", "d"]),
        ),
        (
            CellRange(["a", "b", "c", "d"]),
            (CellRange([3, 4, 2, 1]), -1),
            CellRange(["b", "a", "c", "d"]),
        ),
        (
            MULTI_COLUMNS,
            (MULTI_COLUMNS[:, 2], 1, MULTI_COLUMNS[:, 1], -1),
            CellRange(
                [
                    ["Joanna", 81, "Blue"],
                    ["Edward", 79, "Blue"],
                    ["Oscar", 63, "Blue"],
                    ["Miranda", 85, "Green"],
                    ["Collin", 85, "Green"],
                    ["Arturo", 79, "Green"],
                    ["Willam", 64, "Green"],
                    ["Hannah", 93, "Red"],
                    ["Mallory", 81, "Red"],
                    ["Annie", 72, "Red"],
                ]
            ),
        ),
        (
            MULTI_COLUMNS,
            (MULTI_COLUMNS[:, 2], 1, MULTI_COLUMNS[:, 1], 1),
            CellRange(
                [
                    ["Oscar", 63, "Blue"],
                    ["Edward", 79, "Blue"],
                    ["Joanna", 81, "Blue"],
                    ["Willam", 64, "Green"],
                    ["Arturo", 79, "Green"],
                    ["Miranda", 85, "Green"],
                    ["Collin", 85, "Green"],
                    ["Annie", 72, "Red"],
                    ["Mallory", 81, "Red"],
                    ["Hannah", 93, "Red"],
                ]
            ),
        ),
        (CellRange([3, 1, 2]), (CellRange(["z", "y", "x"]), 1), CellRange([2, 1, 3])),
        (CellRange([3, 1, 2]), (CellRange(["z", "y", "x"]), -2), VALUE_ERROR),
        (CellRange([3, 1, 2]), (CellRange(["z", "y", "x"]), 0), VALUE_ERROR),
        (CellRange([3, 1, 2]), (CellRange(["z", "y"]), 1), VALUE_ERROR),
        (CellRange([3, 1]), (CellRange(["z", "y", "x"]), 1), VALUE_ERROR),
    ],
)
def test_SORTBY(array, arrays_orders, result):
    assert_equal(SORTBY(array, *arrays_orders), result)


@pytest.mark.parametrize(
    "vector, wrap_count, pad_with, result",
    [
        (CellRange([[1, 2, 3], [4, 5, 6]]), 2, "x", VALUE_ERROR),
        (CellRange([1, 2, 3, 4, 5, 6]), 2, "x", CellRange([[1, 3, 5], [2, 4, 6]])),
        (CellRange([1, 2, 3, 4, 5]), 2, "x", CellRange([[1, 3, 5], [2, 4, "x"]])),
        (CellRange([[1, 2, 3, 4, 5]]), 2, "x", CellRange([[1, 3, 5], [2, 4, "x"]])),
        (
            CellRange([1, 2, 3, 4, 5]),
            4,
            "x",
            CellRange([[1, 5], [2, "x"], [3, "x"], [4, "x"]]),
        ),
        (
            CellRange(
                [
                    1,
                ]
            ),
            4,
            "x",
            CellRange([[1], ["x"], ["x"], ["x"]]),
        ),
        (CellRange([1, 2, 3, 4, 5, 6]), 0, "x", VALUE_ERROR),
    ],
)
def test_WRAPCOLS(vector, wrap_count, pad_with, result):
    assert_equal(WRAPCOLS(vector, wrap_count, pad_with), result)


@pytest.mark.parametrize(
    "vector, wrap_count, pad_with, result",
    [
        (CellRange([[1, 2, 3], [4, 5, 6]]), 2, "x", VALUE_ERROR),
        (CellRange([1, 2, 3, 4, 5]), 2, "x", CellRange([[1, 2], [3, 4], [5, "x"]])),
        (CellRange([[1, 2, 3, 4, 5]]), 2, "x", CellRange([[1, 2], [3, 4], [5, "x"]])),
        (
            CellRange([1, 2, 3, 4, 5]),
            4,
            "x",
            CellRange([[1, 2, 3, 4], [5, "x", "x", "x"]]),
        ),
        (
            CellRange(
                [
                    1,
                ]
            ),
            4,
            "x",
            CellRange([[1, "x", "x", "x"]]),
        ),
        (CellRange([1, 2, 3, 4, 5, 6]), 0, "x", VALUE_ERROR),
    ],
)
def test_WRAPROWS(vector, wrap_count, pad_with, result):
    assert_equal(WRAPROWS(vector, wrap_count, pad_with), result)


TEST_LOOKUP_2 = CellRange(
    [
        ["Andrew", 1257, 32],
        ["Bethany", 1470, 80],
        ["Charles", 1367, 42],
        ["David", 1497, 65],
        ["Emily", 1730, 50],
        ["Ferdinand", 1414, 41],
        ["Georgia", 1364, 17],
        ["Haley", 1778, 46],
        ["Ian", 1769, 90],
        ["Jennifer", 1530, 35],
    ]
)

TEST_LOOKUP_3 = CellRange(
    [
        [-1, FALSE, TRUE, 3, "Cloud", "Rain", "Raincoat", "Sun"],
        [None] * 8,
        [f"Pos {i}" for i in range(1, 9)],
    ]
)

TEST_LOOKUP_4 = CellRange(
    [
        [4.14, "red"],
        [4.19, "orange"],
        [5.17, "yellow"],
        [5.77, "green"],
        [6.39, "blue"],
    ]
)


@pytest.mark.parametrize(
    "lookup_value, lookup_vector, result_vector, result",
    [
        ("Haley", TEST_LOOKUP_2, None, 46),
        ("Haley", TEST_LOOKUP_2[:, 0], TEST_LOOKUP_2[:, -1], 46),
        ("Alisa", TEST_LOOKUP_2, None, NA_ERROR),
        ("Ken", TEST_LOOKUP_2, None, 35),
        ("Ken", TEST_LOOKUP_2[:, 0], TEST_LOOKUP_2[:, -1], 35),
        ("Evan", TEST_LOOKUP_2[:, 0], None, "Emily"),
        ("David", TEST_LOOKUP_2[:, 0], TEST_LOOKUP_2[:, 1], 1497),
        (-0.5, TEST_LOOKUP_3[0], TEST_LOOKUP_3[2], "Pos 1"),
        (TRUE, TEST_LOOKUP_3[0], TEST_LOOKUP_3[2], "Pos 3"),
        ("Area", TEST_LOOKUP_3[0], TEST_LOOKUP_3[2], NA_ERROR),
        (
            5,
            CellRange([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            CellRange(
                [
                    "Hydrogen",
                    "Helium",
                    "Lithium",
                    "Beryllium",
                    "Boron",
                    "Carbon",
                    "Nitrogen",
                    "Oxygen",
                    "Fluorine",
                    "Neon",
                ]
            ),
            "Boron",
        ),
        (4.19, TEST_LOOKUP_4[:, 0], TEST_LOOKUP_4[:, 1], "orange"),
        (5.75, TEST_LOOKUP_4[:, 0], TEST_LOOKUP_4[:, 1], "yellow"),
        (7.66, TEST_LOOKUP_4[:, 0], TEST_LOOKUP_4[:, 1], "blue"),
        (0, TEST_LOOKUP_4[:, 0], TEST_LOOKUP_4[:, 1], NA_ERROR),
        (7.66, TEST_LOOKUP_4, TEST_LOOKUP_4, VALUE_ERROR),
    ],
)
def test_LOOKUP(lookup_value, lookup_vector, result_vector, result):
    assert_equal(LOOKUP(lookup_value, lookup_vector, result_vector), result)


TEST_LOOKUP_5 = CellRange(
    [
        [0.1, 9700],
        [0.22, 39475],
        [0.24, 84200],
        [0.32, 160726],
        [0.35, 204100],
        [0.37, 510300],
    ]
)
TEST_LOOKUP_6 = CellRange(
    [
        ["Fargo", 1996, 5, 61000000],
        ["L.A. Confidential", 1997, 4, 126000000],
        ["The Sixth Sense", 1999, 1, 673000000],
        ["Toy Story", 1995, 2, 362000000],
        ["Unforgiven", 1992, 3, 159000000],
    ]
)

TEST_LOOKUP_7 = CellRange(
    [
        [0, 0],
        [10, 0.05],
        [25, 0.1],
        [50, 0.2],
        [100, 0.25],
    ]
)

TEST_LOOKUP_8 = CellRange(
    [
        [610, "Janet", "Farley", "Fulfillment"],
        [798, "Steven", "Batista", "Sales"],
        [841, "Evelyn", "Monet", "Fulfillment"],
        [886, "Marilyn", "Bradley", "Fulfillment"],
        [622, "Jonathan", "Adder", "Marketing"],
        [601, "Adrian", "Birt", "Engineering"],
        [869, "Julie", "Irons", "Sales"],
        [867, "Erica", "Tan", "Fulfillment"],
    ]
)

TEST_LOOKUP_9 = CellRange(
    [
        [None, "A", "B", "C", "D"],
        ["Vinyl", 10, 11.5, 13.23, 15.21],
        ["Wood", 12, 13.8, 15.87, 18.25],
        ["Glass", 15, 17.25, 19.84, 22.81],
        ["Steel", 18, 20.7, 23.81, 27.28],
        ["Titanium", 23, 26.45, 30.42, 34.98],
    ]
)

TEST_LOOKUP_10 = CellRange(
    [
        [None, "A", "B", "C", "D"],
        ["Wood", 12, 13.8, 15.87, 18.25],
        ["Vinyl", 10, 11.5, 13.23, 15.21],
        ["Titanium", 23, 26.45, 30.42, 34.98],
        ["Steel", 18, 20.7, 23.81, 27.28],
        ["Glass", 15, 17.25, 19.84, 22.81],
    ]
)


@pytest.mark.parametrize(
    "lookup_value, lookup_array, return_array, if_not_found, match_mode, search_mode, result",
    [
        (46573, TEST_LOOKUP_5[:, 1], TEST_LOOKUP_5[:, 0], 0, 1, 1, 0.24),
        ("Toy Story", TEST_LOOKUP_6[:, 0], TEST_LOOKUP_6[:, -1], None, 0, 1, 362000000),
        (28, TEST_LOOKUP_7[:, 0], TEST_LOOKUP_7[:, 1], None, -1, 1, 0.1),
        (
            841,
            TEST_LOOKUP_8[:, 0],
            TEST_LOOKUP_8[:, 1:],
            None,
            0,
            1,
            CellRange([["Evelyn", "Monet", "Fulfillment"]]),
        ),
        (
            "B",
            CellRange(TEST_LOOKUP_9[0][1:]),
            XLOOKUP("Glass", TEST_LOOKUP_9[1:, 0], TEST_LOOKUP_9[1:, 1:], None, 0, 1)[
                0
            ],
            None,
            0,
            1,
            17.25,
        ),
        (
            "Godzilla",
            TEST_LOOKUP_6[:, 0],
            TEST_LOOKUP_6[:, 3],
            "Not found",
            0,
            1,
            "Not found",
        ),
        (46523, TEST_LOOKUP_5[:, 1], TEST_LOOKUP_5[:, 0], 0, 1, 1, 0.24),
        ("Gl*", TEST_LOOKUP_9[1:, 0], TEST_LOOKUP_9[1:, -1], "Not found", 2, 1, 22.81),
        (
            "Gl*",
            TEST_LOOKUP_9[1:, 0],
            TEST_LOOKUP_9[1:, -1],
            "Not found",
            2,
            2,
            VALUE_ERROR,
        ),
        (
            "Globe",
            TEST_LOOKUP_9[1:, 0],
            TEST_LOOKUP_9[1:, -1],
            "Not found",
            1,
            1,
            27.28,
        ),
        (
            "Globe",
            TEST_LOOKUP_9[1:, 0],
            TEST_LOOKUP_9[1:, -1],
            "Not found",
            -1,
            1,
            22.81,
        ),
        (
            "Globe",
            TEST_LOOKUP_9[1:, 0],
            TEST_LOOKUP_9[1:, -1],
            "Not found",
            0,
            1,
            "Not found",
        ),
        (
            "Globe",
            TEST_LOOKUP_9[1:, 0],
            TEST_LOOKUP_9[1:, -1],
            "Not found",
            1,
            -1,
            27.28,
        ),
        (
            "Globe",
            TEST_LOOKUP_10[1:, 0],
            TEST_LOOKUP_10[1:, -1],
            "Not found",
            1,
            -2,
            27.28,
        ),
        (
            "Globe",
            TEST_LOOKUP_9[1:, 0],
            TEST_LOOKUP_9[1:, -1],
            "Not found",
            -1,
            -1,
            22.81,
        ),
        (11, TEST_LOOKUP_9[1:, 1], TEST_LOOKUP_9[1:, 0], "Not found", 1, 2, "Wood"),
        ("Tit", TEST_LOOKUP_10[1:, 0], TEST_LOOKUP_10[1:, 1], "Not found", -1, -2, 18),
    ],
)
def test_XLOOKUP(
    lookup_value,
    lookup_array,
    return_array,
    if_not_found,
    match_mode,
    search_mode,
    result,
):
    assert_equal(
        XLOOKUP(
            lookup_value,
            lookup_array,
            return_array,
            if_not_found,
            match_mode,
            search_mode,
        ),
        result,
    )


TEST_FILTER_1 = CellRange(
    [
        ["East", "Tom", "Apple", 6380],
        ["West", "Fred", "Grape", 5619],
        ["North", "Amy", "Pear", 4565],
        ["South", "Sal", "Banana", 5323],
        ["East", "Fritz", "Apple", 4394],
        ["West", "Sravan", "Grape", 7195],
        ["North", "Xi", "Pear", 5231],
        ["South", "Hector", "Banana", 2427],
        ["East", "Tom", "Banana", 4213],
        ["West", "Fred", "Pear", 3239],
        ["North", "Amy", "Grape", 6420],
        ["South", "Sal", "Apple", 1310],
        ["East", "Fritz", "Banana", 6274],
        ["West", "Sravan", "Pear", 4894],
        ["North", "Xi", "Grape", 7580],
        ["South", "Hector", "Apple", 9814],
    ]
)


@pytest.mark.parametrize(
    "array, include, if_empty, result",
    [
        (
            TEST_FILTER_1,
            CellRange([row[2] == "Apple" for row in TEST_FILTER_1]),
            None,
            CellRange(
                [
                    ["East", "Tom", "Apple", 6380],
                    ["East", "Fritz", "Apple", 4394],
                    ["South", "Sal", "Apple", 1310],
                    ["South", "Hector", "Apple", 9814],
                ]
            ),
        ),
        (
            TEST_FILTER_1,
            CellRange([row[2] == "Pineapple" for row in TEST_FILTER_1]),
            None,
            CALC_ERROR,
        ),
        (
            TEST_FILTER_1,
            CellRange([row[2] == "Pineapple" for row in TEST_FILTER_1]),
            "Not found",
            "Not found",
        ),
    ],
)
def test_FILTER(array, include, if_empty, result):
    assert_equal(FILTER(array, include, if_empty), result)


@pytest.mark.parametrize(
    "lookup_value, lookup_array, match_mode, search_mode, result",
    [
        ("Gra?", CellRange(["Apple", "Grape", "Pear", "Banana", "Cherry"]), 1, 1, 2),
        (3.1, CellRange(list(range(1, 8))), 0, 1, NA_ERROR),
        (3.1, CellRange(list(range(1, 8))), -1, 1, 3),
        (3.1, CellRange(list(range(1, 8))), 1, 1, 4),
    ],
)
def test_XMATCH(lookup_value, lookup_array, match_mode, search_mode, result):
    assert_equal(XMATCH(lookup_value, lookup_array, match_mode, search_mode), result)
