import pytest

from ..cell_address import Range
from ..tyne_model.table_for_ai import TableForAI, ai_tables_for_sheet, grid_to_values
from .cell import CellMetadata


@pytest.mark.parametrize(
    ("cells", "expected_range", "expected_columns"),
    [
        (
            [
                [None, None, None, None, None],
                [None, "", "One", 2, None],
                [None, 0, 1, 2, 3],
                [None, 2, 3, 4, 5],
            ],
            "C2:D4",
            [("One", "C3:C4"), (2, "D3:D4")],
        ),
        (
            [
                ["One", "Two"],
                [1, 2],
                [3, 4],
            ],
            "A1:B3",
            [("One", "A2:A3"), ("Two", "B2:B3")],
        ),
        (
            [
                [None, None, None, None],
                [None, "One", "Two", None],
                [None, 1, 2, None],
                [None, 3, 4, None],
            ],
            "B2:C4",
            [("One", "B3:B4"), ("Two", "C3:C4")],
        ),
        (
            [
                [None, None, None, None],
                [None, "One", 2, None],
                [None, 1, 2, None],
                [None, 3, 4, None],
            ],
            "B2:C4",
            [("One", "B3:B4"), (2, "C3:C4")],
        ),
        (
            [
                [None, None, None, None],
                [None, "One", None, None],
                [None, 1, 2, None],
                [None, 3, 4, None],
            ],
            "B2:B4",
            [("One", "B3:B4")],
        ),
        (
            [
                [None, None, None, None],
                [None, None, None, None],
                [None, None, None, None],
                [None, None, None, None],
            ],
            None,
            None,
        ),
        (
            [
                [None, None, None, None],
                [None, "One", None, None],
                [None, None, None, None],
                [None, None, None, None],
            ],
            None,
            None,
        ),
        (
            [
                [None, None, None, None],
                [None, "One", "Two", None],
                [None, None, None, None],
                [None, None, None, None],
            ],
            None,
            None,
        ),
    ],
)
def test_find_tables(cells, expected_range, expected_columns):
    cells = grid_to_values(cells)
    tables = [*ai_tables_for_sheet(cells, "Sheet0")]

    for table in tables:
        d = table.to_dict()
        assert TableForAI.from_dict(d) == table

    if expected_range is None and expected_columns is None:
        assert len(tables) == 0
    else:
        assert len(tables) == 1
        table = tables[0]
        assert str(table.range) == expected_range
        assert table.columns == expected_columns


@pytest.mark.parametrize(
    ("cells", "fill_in", "result"),
    [
        (
            [
                [None, None, None, None],
                [None, "One", 1, None],
                [None, "Two", None, None],
            ],
            "C3:C3",
            [
                ["One", "1"],
                ["Two", TableForAI.PLACE_HOLDER],
            ],
        ),
        (
            [
                [None, None, None, None],
                [None, "One", "=len(B2)", None],
                [None, "Two", None, None],
            ],
            "C3:C3",
            None,
        ),
        (
            [
                [None, None, None, None],
                [None, "One", 1, None],
                [None, "One", None, None],
            ],
            "C3:C3",
            None,
        ),
    ],
)
def test_to_fill_in(cells, fill_in, result):
    cells = grid_to_values(cells)
    meta = {
        cell_id: CellMetadata(raw_code=cell_value)
        for cell_id, cell_value in cells.items()
        if isinstance(cell_value, str) and cell_value.startswith("=")
    }
    for table in ai_tables_for_sheet(cells, "Sheet0"):
        assert table.to_fill_in(cells, meta, Range.from_a1(fill_in)) == result
