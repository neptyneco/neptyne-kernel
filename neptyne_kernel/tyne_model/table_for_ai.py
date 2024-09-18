from dataclasses import dataclass
from numbers import Number
from typing import Any, Iterator

import numpy as np

from ..cell_address import Address, Range
from ..expression_compiler import is_cell_formula
from ..mime_handling import is_number_string
from ..tyne_model.cell import SheetCell
from .cell import CellMetadata

DECLARE_TABLE_KEYWORD = "DECLARE_TABLE"


@dataclass
class TableForAI:
    """A table found in a sheet, used for to tell an AI about it"""

    sheet_name: str
    range: Range
    columns: list[tuple[Any, str]]

    PLACE_HOLDER = "<|placeholder|>"

    def _prompt(self) -> str:
        return f"{self.sheet_name!r}, range={self.range!r}, columns={self.columns!r}"

    def as_prompt(self) -> str:
        return f"{DECLARE_TABLE_KEYWORD}({self._prompt()})"

    def __iter__(self) -> Iterator[list[Address]]:
        return self.range.__iter__()

    def to_fill_in(
        self,
        sheet_cells: dict[Address, SheetCell],
        cell_meta: dict[Address, CellMetadata],
        fill_in: Range,
    ) -> list[list[str]] | None:
        def escape(cell: Any) -> str:
            return str(cell).replace("|", "<|pipe|>")

        cells: list[list[Any]] = []
        fill_in_cells = set(cell_id for row in fill_in for cell_id in row)
        strings_seen = set()
        for row in self:
            next_row = []
            for cell_id in row:
                if cell_id in fill_in_cells:
                    next_row.append(TableForAI.PLACE_HOLDER)
                    continue
                value = sheet_cells.get(cell_id)
                if value is None:
                    continue
                metadata = cell_meta.get(cell_id)
                if metadata:
                    raw_code = metadata.raw_code
                    if raw_code and is_cell_formula(raw_code):
                        return None
                if isinstance(value, str) and not is_number_string(value):
                    strings_seen.add(value)
                next_row.append(escape(value))
            cells.append(next_row)

        if len(strings_seen) <= 1:
            return None

        return cells

    def to_dict(self) -> dict[str, Any]:
        return {
            "sheet_name": self.sheet_name,
            "range": self.range.to_dict(),
            "columns": self.columns,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TableForAI":
        return TableForAI(
            sheet_name=d["sheet_name"],
            range=Range.from_dict(d["range"]),
            columns=d["columns"],
        )


def grid_to_values(grid: list[list[Any]]) -> dict[Address, Any]:
    return {
        Address(col, row, 0): grid[row][col]
        for col in range(len(grid[0]))
        for row in range(len(grid))
        if grid[row][col] is not None
    }


def ai_tables_for_sheet(
    cells: dict[Address, Any],
    sheet_name: str,
    assume_filled: Range | None = None,
) -> Iterator[TableForAI]:
    from scipy.ndimage import find_objects, label

    if not cells:
        return []
    sheet_id = next(iter(cells)).sheet

    def get_value(col: int, row: int) -> Any:
        return cells.get(Address(col, row, sheet_id))

    max_row, max_col = 0, 0
    for cell_id in cells:
        max_row = max(max_row, cell_id.row)
        max_col = max(max_col, cell_id.column)
    if assume_filled:
        max_col = max(max_col, assume_filled.max_col)
        max_row = max(max_row, assume_filled.max_row)

    mask = np.full((max_row + 1, max_col + 1), False)
    for cell_id, value in cells.items():
        if value is not None:
            mask[cell_id.row, cell_id.column] = True
    if assume_filled:
        for row in assume_filled:
            for cell_id in row:
                mask[cell_id.row, cell_id.column] = True

    labeled, num_labels = label(mask)
    for rowslice, colslice in find_objects(labeled):

        def is_empty(value: Any) -> bool:
            return value is None or value == ""

        # Remove any headerless columns:
        while (
            is_empty(get_value(colslice.start, rowslice.start))
            and colslice.start < colslice.stop
        ):
            colslice = slice(colslice.start + 1, colslice.stop)

        while (
            is_empty(get_value(colslice.stop - 1, rowslice.start))
            and colslice.start < colslice.stop
        ):
            colslice = slice(colslice.start, colslice.stop - 1)

        if rowslice.stop - rowslice.start < 2:
            continue

        def stringify_and_truncate(value: Any) -> str | Number:
            if isinstance(value, Number):
                return value
            return str(value)[:40]

        headers = [
            stringify_and_truncate(get_value(col, rowslice.start))
            for col in range(colslice.start, colslice.stop)
        ]

        def col_range(col: int, row_start: int, row_end: int) -> str:
            start = Address(col, row_start, sheet_id).to_a1()
            stop = Address(col, row_end, sheet_id).to_a1()
            return f"{start}:{stop}"

        columns = [
            (
                header,
                col_range(column, rowslice.start + 1, rowslice.stop - 1),
            )
            for header, column in zip(headers, range(colslice.start, colslice.stop))
        ]
        yield TableForAI(
            sheet_name,
            Range(
                colslice.start, colslice.stop - 1, rowslice.start, rowslice.stop - 1, 0
            ),
            columns,
        )
