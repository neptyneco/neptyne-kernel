from copy import deepcopy
from typing import Any, Iterator

import numpy as np

from ..cell_address import Address, Range
from ..expression_compiler import DEFAULT_GRID_SIZE
from ..tyne_model.cell import SheetCell


class Sheet:
    cells: dict[Address, SheetCell]
    attributes: dict[str, Any]
    grid_size: tuple[int, int]
    id: int
    name: str

    def __init__(self, sheet_id: int, name: str) -> None:
        self.id = sheet_id
        self.cells = {}
        self.attributes = {}
        self.grid_size = DEFAULT_GRID_SIZE
        self.name = name

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells": ([cell.to_dict() for cell_id, cell in self.cells.items()]),
            "attributes": self.attributes,
            "grid_size": self.grid_size,
            "id": self.id,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sheet":
        sheet = cls(data["id"], data["name"])
        sheet.cells = {
            Address.from_coord(cell["cell_id"]): SheetCell.from_dict(
                cell, copy_dict=False
            )
            for cell in data["cells"]
        }
        sheet.attributes = data["attributes"]
        sheet.grid_size = data["grid_size"]
        return sheet

    def __contains__(self, item: Address) -> bool:
        return item in self.cells

    def copy(self, without_cells: bool = False) -> "Sheet":
        sheet = Sheet(self.id, self.name)
        sheet.cells = {} if without_cells else deepcopy(self.cells)
        sheet.attributes = deepcopy(self.attributes)
        sheet.grid_size = self.grid_size
        return sheet

    def export(self, compact: bool = False) -> dict[str, Any]:
        return {
            "n_rows": self.grid_size[1],
            "n_cols": self.grid_size[0],
            "sheet_attributes": self.attributes,
            "cells": [
                cell_info.export(compact) for cell_id, cell_info in self.cells.items()
            ],
            "sheet_id": self.id,
            "name": self.name,
        }

    def to_numpy_mask(self, assume_filled: Range | None = None) -> np.ndarray:
        cols, rows = self.grid_size
        mask = np.full(
            (rows + 1, cols + 1), False
        )  # grid_size had an off-by-one bug for a while
        for cell_id, cell in self.cells.items():
            if cell.output is not None and cell_id.row < rows and cell_id.column < cols:
                mask[cell_id.row, cell_id.column] = True
        if assume_filled:
            for row in assume_filled:
                for cell_id in row:
                    mask[cell_id.row, cell_id.column] = True
        return mask


class TyneSheets:
    sheets: dict[int, Sheet]
    next_sheet_id: int

    def __init__(self) -> None:
        self.sheets = {0: Sheet(0, "Sheet0")}
        self.next_sheet_id = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "sheets": [sheet.to_dict() for sheet in self.sheets.values()],
            "next_sheet_id": self.next_sheet_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TyneSheets":
        result = cls()
        result.sheets = {
            sheet.id: sheet for sheet in (Sheet.from_dict(s) for s in data["sheets"])
        }
        result.next_sheet_id = data["next_sheet_id"]
        return result

    def __contains__(self, item: Address) -> bool:
        sheet = self.sheets.get(item.sheet)
        if sheet is None:
            return False
        return item in sheet

    def sheet_ids_by_name(self) -> dict[str, int]:
        return {sheet.name: sheet.id for sheet in self.sheets.values()}

    def new_sheet_id(self) -> int:
        new_id = self.next_sheet_id
        self.next_sheet_id += 1
        if new_id in self.sheets:
            return self.new_sheet_id()
        return new_id

    def new_sheet(self, name: str | None = None) -> tuple[str, Sheet]:
        new_id = self.new_sheet_id()
        existing_names = {sheet.name for sheet in self.sheets.values()}
        if name is not None:
            if name in existing_names:
                raise KeyError(name)
        else:
            index = new_id
            while (name := f"Sheet{index}") in existing_names:
                index += 1
        assert name is not None
        new_sheet = Sheet(new_id, name)
        self.sheets[new_id] = new_sheet
        return name, new_sheet

    def rename_sheet(self, sheet_id: int, name: str) -> None:
        self.sheets[sheet_id].name = name

    def export(self, compact: bool = False) -> list[dict[str, Any]]:
        return [
            {
                **sheet.export(compact),
            }
            for sheet in self.sheets.values()
        ]

    def copy(self) -> "TyneSheets":
        result = TyneSheets()
        result.sheets = {id: sheet.copy() for id, sheet in self.sheets.items()}
        return result

    def get(
        self, cell_id: Address, create_if_missing: bool = False
    ) -> SheetCell | None:
        sheet_cells = self.sheets[cell_id.sheet].cells
        if cell_id not in sheet_cells:
            if create_if_missing:
                cell = SheetCell(cell_id=cell_id)
                sheet_cells[cell_id] = cell
                return cell
            return None
        return sheet_cells[cell_id]

    def get_grid_size(self, sheet_id: int) -> tuple[int, int]:
        return self.sheets[sheet_id].grid_size

    def set(self, cell_id: Address, cell: SheetCell) -> None:
        self.sheets[cell_id.sheet].cells[cell_id] = cell

    def delete(self, cell_id: Address) -> None:
        del self.sheets[cell_id.sheet].cells[cell_id]

    def delete_sheet(self, sheet_id: int) -> None:
        del self.sheets[sheet_id]

    def all_cells(self) -> Iterator[tuple[Address, SheetCell]]:
        for sheet_id, sheet in self.sheets.items():
            yield from sheet.cells.items()
