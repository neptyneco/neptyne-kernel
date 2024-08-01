from typing import Any, Callable

from .api_ref import ApiRef
from .cell_address import Address, Range, replace_negative_bounds_with_grid_size
from .cell_range import (
    Int,
    IntOrSlice,
    slice_or_int_to_range,
)
from .dash import Dash
from .neptyne_protocol import CellAttribute
from .primitives import proxy_val


class DashRef(ApiRef):
    dash: Dash
    range: Range

    def __init__(self, dash: Dash, ref: Address | Range):
        super().__init__(ref)
        self.dash = dash

    def __repr__(self) -> str:
        return f"DashRef({self.range})"

    def resolve(self) -> "DashRef":
        """Returns a new DashRef with the max_col and max_row resolved (not unbounded)"""
        col, row = self.dash.resolve_max_col_row(self.range)
        return DashRef(
            self.dash,
            Range(self.range.min_col, col, self.range.min_row, row, self.range.sheet),
        )

    def current_max_col_row(self) -> tuple[int, int]:
        return self.dash.resolve_max_col_row(self.range)

    def _write_value(self, col: int, row: int, value: Any) -> None:
        self.dash[(col, row, self.range.sheet)] = value

    def row_entry(self, idx: int) -> Any:
        if self.range.min_col == self.range.max_col:
            row = idx + self.range.min_row
            if row > self.range.max_row > -1:
                raise IndexError(f"Trying to read outside of cellrange ({row})")
            addr = Address(self.range.min_col, row, self.range.sheet)
        else:
            col = idx + self.range.min_col
            if col > self.range.max_col > 0:
                raise IndexError(f"Trying to read outside of cellrange ({col})")
            addr = Address(col, self.range.min_row, self.range.sheet)
        return proxy_val(
            self.dash.cells[addr.sheet].get(addr), DashRef(self.dash, addr)
        )

    def getitem(self, key: IntOrSlice) -> Any:
        range_max_col, range_max_row = self.current_max_col_row()

        if self.range.dimensions() <= 1:
            col = 0 if self.range.min_col == self.range.max_col else key
            row = 0 if self.range.min_row == self.range.max_row else key
            min_col, max_col = slice_or_int_to_range(
                col, self.range.min_col, range_max_col + 1
            )
            min_row, max_row = slice_or_int_to_range(
                row, self.range.min_row, range_max_row + 1
            )

            if min_col == max_col and min_row == max_row and isinstance(key, Int):  # type: ignore
                return self.dash[min_col, min_row, self.range.sheet]

            return DashRef(
                dash=self.dash,
                ref=Range(min_col, max_col, min_row, max_row, self.range.sheet),
            )
        else:
            min_row, max_row = slice_or_int_to_range(
                key, self.range.min_row, range_max_row + 1
            )
            if self.range.max_row > 0 and max_row > range_max_row:
                raise IndexError(f"{max_row} is out of range")
            return DashRef(
                dash=self.dash,
                ref=Range(
                    self.range.min_col,
                    range_max_col,
                    min_row,
                    max_row,
                    self.range.sheet,
                ),
            )

    def set_attributes(self, attributes: dict[CellAttribute, str]) -> None:
        attribute_updates = [
            (attr.value, attribute_value)
            for attr, attribute_value in attributes.items()
        ]

        changed: set[Address] = set()
        sheet = self.dash.sheets[self.range.sheet]
        for row in replace_negative_bounds_with_grid_size(
            self.range, (sheet.n_cols, sheet.n_rows)
        ):
            for address in row:
                changed.add(address)
                for attr, value in attribute_updates:
                    self.dash.update_cell_attribute(address, attr, value)
        self.dash.notify_client_cells_have_changed(changed)

    def get_attribute(
        self,
        attribute: str,
        modifier: Callable[[str], str] | None = None,
        default_value: int | str | None = None,
    ) -> list[list[str | None]] | list[str | None] | str | None:
        def attribute_for_address(address: Address) -> str | None:
            meta = self.dash.cell_meta.get(address)
            attr = (
                meta.attributes.get(attribute, default_value) if meta else default_value
            )
            return modifier(attr) if modifier else attr

        d = self.range.dimensions()
        if d == 0:
            return attribute_for_address(self.range.origin())
        elif d == 1:
            return [
                attribute_for_address(address) for row in self.range for address in row
            ]
        else:
            return [
                [attribute_for_address(address) for address in row]
                for row in self.range
            ]

    def clear(self) -> None:
        max_col, max_row = self.current_max_col_row()
        r = Range(
            self.range.min_col,
            max_col,
            self.range.min_row,
            max_row,
            self.range.sheet,
        )

        cell_ids = [cell for row in r for cell in row]
        self.dash.clear_cells_with_cascade(cell_ids)
