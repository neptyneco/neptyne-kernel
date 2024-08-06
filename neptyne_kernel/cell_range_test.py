from operator import __eq__, __ge__, __gt__, __le__, __lt__, __ne__

import numpy as np
import pytest

from .cell_range import CellRange, slice_or_int_to_range
from .formulas.helpers import assert_equal
from .primitives import Empty


def test_binary_operators():
    assert_equal(CellRange([1, 2, 3]) + [1, 2, 3], CellRange([2, 4, 6]))  # noqa: RUF005
    assert_equal(CellRange([1, 2, 3]) - [1, 2, 3], CellRange([0, 0, 0]))
    assert_equal(CellRange([1, 2, 3]) / [1, 2, 3], CellRange([1, 1, 1]))
    assert_equal(CellRange([1, 2, 3]) // [2, 2, 2], CellRange([0, 1, 1]))
    assert_equal(CellRange([1, 2, 3]) * [1, 2, 3], CellRange([1, 4, 9]))
    assert_equal(CellRange([[1, 2], [3, 4]]) + 1, CellRange([[2, 3], [4, 5]]))

    assert_equal(CellRange([1, 2, 3]) * 2, CellRange([2, 4, 6]))
    assert_equal(2 * CellRange([1, 2, 3]), CellRange([2, 4, 6]))

    assert_equal(6 // CellRange([1, 2, 3]), CellRange([6, 3, 2]))
    assert_equal(6 + CellRange([1, 2, 3]), CellRange([7, 8, 9]))
    assert_equal(6 - CellRange([1, 2, 3]), CellRange([5, 4, 3]))
    assert_equal(4 / CellRange([1, 2, 5]), CellRange([4, 2, 0.8]))


def test_in():
    assert 1 not in CellRange([0, 0, 0])
    assert 1 in CellRange([0, 1, 0])

    assert 1 not in CellRange([[0, 0, 0], [0, 0, 0]])
    assert 1 in CellRange([[0, 0, 0], [0, 1, 0]])


def test_cell_range_map():
    assert_equal(CellRange([1, 2, 3]).map(lambda x: x + 1), CellRange([2, 3, 4]))
    assert_equal(
        CellRange([[1, 2], [2, 3]]).map(lambda x: x + 1), CellRange([[2, 3], [3, 4]])
    )


def test_cell_range_iter():
    cr = CellRange([[1, 2], [3, 4]])
    i = 0
    for row in cr:
        for n in row:
            i += 1
            assert n == i
    assert i == 4


def test_slicing():
    cr = CellRange([*range(1, 11)])
    assert_equal(cr[[0, 2]], CellRange([1, 3]))
    assert_equal(cr[1:], [*range(2, 11)])

    cr = CellRange([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert_equal(cr[:, 0], CellRange([1, 4, 7]))
    assert_equal(cr[:, 2], CellRange([3, 6, 9]))
    assert_equal(cr[0:2, 0:2], CellRange([[1, 2], [4, 5]]))

    assert_equal(cr[:, [0, 2]], CellRange([[1, 3], [4, 6], [7, 9]]))


TEST_2D_RANGE = CellRange([[1, 2], [3, 4]])


@pytest.mark.parametrize(
    "cell_range, value, op, result",
    [
        (TEST_2D_RANGE, 2, __gt__, CellRange([[False, False], [True, True]])),
        (TEST_2D_RANGE, 2, __ge__, CellRange([[False, True], [True, True]])),
        (TEST_2D_RANGE, 2, __lt__, CellRange([[True, False], [False, False]])),
        (TEST_2D_RANGE, 2, __le__, CellRange([[True, True], [False, False]])),
        (TEST_2D_RANGE, 2, __ne__, CellRange([[True, False], [True, True]])),
        (TEST_2D_RANGE, 2, __eq__, CellRange([[False, True], [False, False]])),
    ],
)
def test_boolean_range(cell_range, value, op, result):
    assert_equal(op(cell_range, value), result)


def test_empty_to_nan():
    cr = CellRange([Empty(None), 1, 2])
    arr = cr.__array__()
    assert np.isnan(arr[0])


@pytest.mark.parametrize(
    "cell_range, dtype",
    [
        (CellRange([1, 2, 3]), np.int64),
        (CellRange([1, 2, 3, Empty(None)]), np.dtype("float64")),
        (CellRange([1, 2, 3, Empty(None), "a"]), np.dtype("O")),
    ],
)
def test_array_dtypes(cell_range, dtype):
    assert cell_range.__array__().dtype == dtype


@pytest.mark.parametrize(
    "key, start, end, result",
    [
        (0, 0, 10, (0, 0)),
        (slice(2, 5), 0, 10, (2, 4)),
        (slice(2, -1), 0, 10, (2, 9)),
        (slice(2, 7), 0, 10, (2, 6)),
        (slice(1, 5), 0, 10, (1, 4)),
        (slice(3, 5), 0, 10, (3, 4)),
        (slice(4, 6), 0, 19, (4, 5)),
        (slice(5, 7), 0, 10, (5, 6)),
        (slice(-5, 7), 0, 9, (5, 6)),
        (slice(-7, -5), 0, 9, (3, 4)),
        (-7, 0, 9, (3, 3)),
    ],
)
def test_slice_or_int_to_range(key, start, end, result):
    assert result == slice_or_int_to_range(key, start, end + 1)


@pytest.mark.parametrize("np_type", [np.int32, np.int64])
def test_numpy_indices(np_type):
    ind = np_type(0)
    assert_equal(TEST_2D_RANGE[ind], CellRange([1, 2]))
    assert_equal(TEST_2D_RANGE[ind, :], CellRange([1, 2]))
    assert_equal(TEST_2D_RANGE[:, ind], CellRange([1, 3]))


def test_to_data_frame():
    cr = CellRange([[1, 2], [3, 4]])
    df = cr.to_dataframe(header=False)
    assert df.shape == (2, 2)
    assert df.iloc[0, 1] == 2
    assert df.iloc[1, 0] == 3


def test_bool():
    with pytest.raises(ValueError):
        bool(CellRange([1, 2, 3]))
    assert CellRange([1, 2, 3]).any()
    assert not CellRange([0, 0, 0]).any()
    assert CellRange([1, 2, 3]).all()
    assert not CellRange([1, 0, 3]).all()

    assert CellRange([[1, 2], [3, 4]]).all()
    assert not CellRange([[1, 2], [3, 0]]).all()
    assert CellRange([[0, 0], [0, 4]]).any()


def test_to_list():
    assert CellRange([1, 2, 3]).to_list() == [1, 2, 3]
    assert CellRange([[1, 2], [3, 4]]).to_list() == [[1, 2], [3, 4]]
    assert CellRange([1, 2, 3]).to_list() == [1, 2, 3]
    assert CellRange([[1, 2], [3, 4]]).to_list() == [[1, 2], [3, 4]]
    assert CellRange([1, 2, 3]).to_list() == [1, 2, 3]
    assert CellRange([[1, 2], [3, 4]]).to_list() == [[1, 2], [3, 4]]


def test_equal():
    eq1 = CellRange([1, 2, 3]) == CellRange([1, 2, 3])
    assert len(eq1) == 3
    assert eq1.all()

    eq2 = CellRange([1, 2, 3]) == CellRange([1, 2, 4])
    assert len(eq2) == 3
    assert not eq2.all()

    eq3 = CellRange([[1, 2, 3], [4, 5, 6]]) == CellRange([[1, 2, 3], [4, 5, 6]])
    assert len(eq3) == 2
    assert len(eq3[0]) == 3
    assert eq3.all()
