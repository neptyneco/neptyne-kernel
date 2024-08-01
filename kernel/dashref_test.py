import numpy as np
import pytest

from .cell_range import CellRange
from .dash_ref import DashRef
from .formulas.helpers import assert_equal
from .test_utils import a1


def test_dash_ref_get_item(dash):
    dash[a1("A1:F6")] = [[ch1 + ch2 for ch1 in "ABCDEF"] for ch2 in "123456"]

    assert dash[a1("C1")] == "C1"

    dr = DashRef(dash, a1("B2:E5"))

    assert dr.getitem(2).getitem(0) == "B4"
    assert dr.getitem(slice(1, 3)).getitem(1).getitem(1) == "C4"

    drh = DashRef(dash, a1("B1:E1"))
    assert drh.getitem(1) == "C1"
    assert drh.getitem(slice(2, 3)).getitem(0) == "D1"

    drv = DashRef(dash, a1("C1:C5"))
    assert drv.getitem(1) == "C2"
    assert drv.getitem(slice(2, 4)).getitem(0) == "C3"


@pytest.mark.parametrize("ind", [0.5, {}, 1j + 5])
def test_invalid_indices(dash, ind):
    dash[a1("A1:F6")] = [[ch1 + ch2 for ch1 in "ABCDEF"] for ch2 in "123456"]
    dr = DashRef(dash, a1("A1:F6"))
    cell_range = CellRange(dr)

    # getitem
    with pytest.raises(IndexError):
        cell_range[ind]
    with pytest.raises(IndexError):
        cell_range[ind, :]
    with pytest.raises(IndexError):
        cell_range[:, ind]

    # setitem
    with pytest.raises(IndexError):
        cell_range[ind] = 1
    with pytest.raises(IndexError):
        cell_range[ind, :] = CellRange([1, 2, 3, 4, 5])
    with pytest.raises(IndexError):
        cell_range[:, ind] = CellRange([1, 2, 3, 4, 5])


def test_dash_ref_set_item(dash):
    dr = DashRef(dash, a1("A1:H10"))

    with pytest.raises(ValueError):
        dr.setitem((slice(0, 3), slice(0, 2)), [[1, 2, 3], [4, 5, 6]])

    dr.setitem((slice(0, 2), slice(0, 3)), [[1, 2, 3], [4, 5, 6]])

    assert dr.getitem(0).getitem(2) == 3


@pytest.mark.parametrize("np_type", [np.int32, np.int64])
def test_numpy_indices(dash, np_type):
    ind = np_type(0)
    dash[a1("A1:F6")] = [[ch1 + ch2 for ch1 in "ABCDEF"] for ch2 in "123456"]
    dr = DashRef(dash, a1("A1:F6"))
    cell_range = CellRange(dr)

    # getitem
    assert_equal(cell_range[ind], CellRange(["A1", "B1", "C1", "D1", "E1", "F1"]))
    assert_equal(cell_range[ind, :], CellRange(["A1", "B1", "C1", "D1", "E1", "F1"]))
    assert_equal(cell_range[:, ind], CellRange(["A1", "A2", "A3", "A4", "A5", "A6"]))

    # setitem
    ind_start = np_type(1)
    ind_end = np_type(3)

    cell_range = CellRange(dash[a1("B2:D4")])
    cell_range[ind_start:ind_end, ind_start:ind_end] = [[1, 2], [3, 4]]

    assert dash[a1("C3")] == 1
    assert dash[a1("D3")] == 2
    assert dash[a1("C4")] == 3
    assert dash[a1("D4")] == 4
