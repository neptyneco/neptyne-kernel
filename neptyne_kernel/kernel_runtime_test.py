from .cell_range import CellRange
from .formulas.helpers import assert_equal


def test_vectorize_cells(dash):
    from .neptyne_api.vectorize import vectorize_cells

    @vectorize_cells
    def add_ints(n1, n2, n3):
        return int(n1) + int(n2) + int(n3)

    assert_equal(add_ints(2, 2, 2), 6)

    assert_equal(add_ints(CellRange([2, 3, 4]), 1, 1), CellRange([4, 5, 6]))
    assert_equal(add_ints(1, CellRange([2, 3, 4]), 1), CellRange([4, 5, 6]))
    assert_equal(add_ints(1, 1, CellRange([2, 3, 4])), CellRange([4, 5, 6]))

    assert_equal(add_ints(1, n2=CellRange([2, 3, 4]), n3=1), CellRange([4, 5, 6]))

    assert_equal(
        add_ints(1, CellRange([1, 2]), n3=CellRange([1, 1])),
        CellRange([[3, 3], [4, 4]]),
    )
