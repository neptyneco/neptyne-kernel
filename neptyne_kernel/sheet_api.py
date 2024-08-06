import contextvars
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import pandas as pd
from googleapiclient.discovery import Resource

from . import gsheets_api, sheet_context
from .api_ref import ApiRef
from .cell_address import Range
from .cell_range import CellRange, CellRangeRef, IntOrSlice
from .expression_compiler import DEFAULT_N_COLS, DEFAULT_N_ROWS
from .neptyne_protocol import (
    DeleteSheetContent,
    MessageTypes,
    RenameSheetContent,
    SheetAttribute,
    SheetAttributeUpdate,
)
from .tyne_model.sheet import Sheet
from .upgrade_model import DEFAULT_COL_WIDTH, DEFAULT_ROW_HEIGHT

if TYPE_CHECKING:
    pass


class NeptyneRowCols(CellRangeRef):
    _sheet: "NeptyneSheet"
    is_row: bool
    start_indx: int
    end_indx: int

    def __init__(
        self,
        sheet: "NeptyneSheet",
        *,
        is_row: bool,
        start_indx: int,
        end_indx: int,
    ):
        from .dash_ref import DashRef

        self.is_row = is_row
        self.start_indx = start_indx
        self.end_indx = end_indx
        self._sheet = sheet

        if is_row:
            dr = DashRef(
                sheet._collection.dash,
                Range(0, -1, start_indx, end_indx, sheet.sheet_id),
            )
        else:
            dr = DashRef(
                sheet._collection.dash,
                Range(start_indx, end_indx, 0, -1, sheet.sheet_id),
            )
        super().__init__(dr)

    def _set_sheet_attribute(self, attr: str, value: Any) -> None:
        self._sheet._collection.dash.update_sheet_attributes(
            SheetAttributeUpdate(attr, self._sheet.sheet_id, value).to_dict(),
            undo=False,
        )

    def _get_sizes(self) -> dict[str, int]:
        if self.is_row:
            sizes = self._sheet.attributes.get(SheetAttribute.ROWS_SIZES.value)
        else:
            sizes = self._sheet.attributes.get(SheetAttribute.COLS_SIZES.value)
        if not sizes:
            return {}
        return sizes

    def _update_and_set_sizes(self, sizes: dict[str, int], new_size: int) -> None:
        for i in range(self.start_indx, self.end_indx + 1):
            sizes[str(i)] = new_size
        if self.is_row:
            self._set_sheet_attribute(SheetAttribute.ROWS_SIZES.value, sizes)
        else:
            self._set_sheet_attribute(SheetAttribute.COLS_SIZES.value, sizes)

    def set_width(self, width: int) -> None:
        assert width > 0
        if self.is_row:
            raise ValueError("Can only set width on columns. Try set_height instead.")
        self._update_and_set_sizes(self._get_sizes(), width)

    def set_height(self, height: int) -> None:
        assert height > 0
        if not self.is_row:
            raise ValueError("Can only set height on rows. Try set_width instead.")
        self._update_and_set_sizes(self._get_sizes(), height)

    def get_width(self) -> int:
        if self.is_row:
            raise ValueError("Can only get width on columns. Try get_height instead.")
        return self._get_sizes().get(str(self.start_indx), DEFAULT_COL_WIDTH)

    def get_height(self) -> int:
        if not self.is_row:
            raise ValueError("Can only get height on rows. Try get_width instead.")
        return self._get_sizes().get(str(self.start_indx), DEFAULT_ROW_HEIGHT)

    width = property(get_width, set_width)
    height = property(get_height, set_height)

    def _get_hidden_set(self) -> set[int]:
        if self.is_row:
            hidden = self._sheet.attributes.get("rowsHiddenHeaders")
        else:
            hidden = self._sheet.attributes.get("colsHiddenHeaders")
        if not hidden:
            hidden = []
        return set(hidden)

    def _set_hidden_set(self, hidden: set[int]) -> None:
        hidden_list = list(hidden)
        sorted_hidden = sorted(hidden_list)
        if self.is_row:
            self._set_sheet_attribute("rowsHiddenHeaders", sorted_hidden)
        else:
            self._set_sheet_attribute("colsHiddenHeaders", sorted_hidden)

    def hide(self) -> None:
        hidden = self._get_hidden_set()
        hidden = hidden.union(set(range(self.start_indx, self.end_indx + 1)))
        self._set_hidden_set(hidden)

    def unhide(self) -> None:
        hidden = self._get_hidden_set()
        hidden = hidden.difference(set(range(self.start_indx, self.end_indx + 1)))
        self._set_hidden_set(hidden)

    def freeze(self) -> None:
        if self.start_indx != 0:
            raise ValueError("Can only freeze starting from the first row or column")
        if self.is_row:
            self._set_sheet_attribute("rowsFrozenCount", self.end_indx + 1)
        else:
            self._set_sheet_attribute("colsFrozenCount", self.end_indx + 1)

    def unfreeze(self) -> None:
        if self.start_indx != 0:
            raise ValueError("Can only un-freeze starting from the first row or column")
        if self.is_row:
            self._set_sheet_attribute("rowsFrozenCount", 0)
        else:
            self._set_sheet_attribute("colsFrozenCount", 0)


class NeptyneSheet(CellRangeRef):
    class RowColSliceAdapater:
        def __init__(self, sheet: "NeptyneSheet", is_row: bool):
            self.sheet = sheet
            self.is_row = is_row

        def __getitem__(self, indices: slice | int) -> NeptyneRowCols:
            if isinstance(indices, slice):
                start = indices.start
                if indices.stop < 0:
                    if self.is_row:
                        stop = self.sheet.n_rows + indices.stop
                    else:
                        stop = self.sheet.n_cols + indices.stop
                else:
                    stop = indices.stop - 1
            else:
                start = indices
                stop = indices
            return NeptyneRowCols(
                self.sheet, is_row=self.is_row, start_indx=start, end_indx=stop
            )

    def __init__(self, sheet_id: int, name: str, collection: "NeptyneSheetCollection"):
        from .dash_ref import DashRef

        self.sheet_id = sheet_id
        self._name = name
        self._collection = collection
        self.n_rows = DEFAULT_N_ROWS
        self.n_cols = DEFAULT_N_COLS
        self.attributes: dict[str, Any] = {}

        self.rows = NeptyneSheet.RowColSliceAdapater(self, True)
        self.cols = NeptyneSheet.RowColSliceAdapater(self, False)

        self._sheet_override_context_token: contextvars.Token[str | None] | None = None

        if service := self._collection._gsheet_service():
            cr = gsheets_api.get_item(
                service,
                collection.dash.gsheets_spreadsheet_id,
                (0, -1, 0, -1, name),
            )
            assert isinstance(cr, CellRange)
            dr = cr.ref
            assert isinstance(dr, ApiRef)
        else:
            dr = DashRef(collection.dash, Range(0, -1, 0, -1, sheet_id))
        super().__init__(dr)  # type: ignore

    def __getitem__(self, key: tuple | list | IntOrSlice | str) -> Any:
        try:
            return super().__getitem__(key)
        except IndexError:
            return None

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._collection.rename_sheet(self.sheet_id, value)

    def __repr__(self) -> str:
        return f"NeptyneSheet({self.sheet_id})"

    def __eq__(self, other: Any):  # type: ignore
        if isinstance(other, NeptyneSheet) and (
            self.sheet_id != other.sheet_id or self.name != other.name
        ):
            return False
        return super().__eq__(other)

    def to_serializable(self) -> Sheet:
        serializable_sheet = Sheet(self.sheet_id, self.name)
        serializable_sheet.attributes = self.attributes
        serializable_sheet.grid_size = self.n_cols, self.n_rows
        return serializable_sheet

    @classmethod
    def from_serializable(
        cls, sheet: Sheet, collection: "NeptyneSheetCollection"
    ) -> "NeptyneSheet":
        neptyne_sheet = cls(sheet.id, sheet.name, collection)
        # TODO: Support more sophisticated way to do this so that it doesn't run every time...
        neptyne_sheet.attributes = sheet.attributes
        neptyne_sheet.n_cols, neptyne_sheet.n_rows = sheet.grid_size
        return neptyne_sheet

    def width(self) -> int:
        return self.n_cols

    def height(self) -> int:
        return self.n_rows

    def __enter__(self) -> "NeptyneSheet":
        self._sheet_override_context_token = sheet_context.sheet_name_override.set(
            self.name
        )
        return self

    def __exit__(self, *_args: Any) -> None:
        if token := self._sheet_override_context_token:
            sheet_context.sheet_name_override.reset(token)


class NeptyneSheetCollection:
    _sheets: list[NeptyneSheet]
    _idx: dict[str | int, NeptyneSheet]

    if TYPE_CHECKING:
        from .dash import Dash

    def __init__(self, dash: "Dash") -> None:
        """@private"""
        self.dash = dash
        """@private"""
        self._sheets = []
        self._idx = {}

    def _gsheet_service(self) -> Resource | None:
        return self.dash.gsheet_service

    def _available_sheets(self) -> list[tuple[str, int]] | None:
        if gsheet_service := self._gsheet_service():
            return gsheets_api.available_sheets(
                gsheet_service, self.dash.gsheets_spreadsheet_id
            )
        return None

    def __getitem__(self, name_or_id: str | int) -> NeptyneSheet:
        if (available_sheets := self._available_sheets()) is not None:
            for title, sheet_id in available_sheets:
                if name_or_id in (title, sheet_id):
                    return NeptyneSheet(sheet_id, title, self)
            raise KeyError(f"Sheet {name_or_id} not found")
        return self._idx[name_or_id]

    def __iter__(self) -> Iterator[NeptyneSheet]:
        if (available_sheets := self._available_sheets()) is not None:
            for title, sheet_id in available_sheets:
                yield NeptyneSheet(sheet_id, title, self)
        else:
            yield from self._sheets

    def __len__(self) -> int:
        if (available_sheets := self._available_sheets()) is not None:
            return len(list(available_sheets))
        return len(self._sheets)

    def __delitem__(self, key: str | int) -> None:
        if service := self._gsheet_service():
            gsheets_api.delete_sheet(service, self.dash.gsheets_spreadsheet_id, key)
        else:
            self.delete_sheet(key)

    def __contains__(self, item_or_id: str | int) -> bool:
        if (available_sheets := self._available_sheets()) is not None:
            for title, sheet_id in available_sheets:
                if item_or_id in (title, sheet_id):
                    return True
            return False
        return item_or_id in self._idx

    def _reply_to_client(self, msg_type: MessageTypes, content: dict[str, Any]) -> None:
        self.dash.reply_to_client(msg_type, content)

    def _update_index(self, sheet: NeptyneSheet, *, do_add: bool) -> None:
        if do_add:
            self._idx[sheet.name] = sheet
            self._idx[sheet.sheet_id] = sheet
        else:
            del self._idx[sheet.name]
            del self._idx[sheet.sheet_id]
        if sheet.name.isidentifier():
            glbs = self.dash.shell.user_global_ns
            if do_add:
                glbs[sheet.name] = sheet
            elif sheet.name in glbs:
                del glbs[sheet.name]

    def _reset_for_testing(self) -> None:
        for sheet in self._sheets:
            self._update_index(sheet, do_add=False)
        self._sheets = []

    def new_sheet(self, name: str | None = None) -> NeptyneSheet:
        """Create a new sheet - a name will be generated if not provided."""
        if service := self._gsheet_service():
            title, sheet_id = gsheets_api.new_sheet(
                service, self.dash.gsheets_spreadsheet_id, name
            )
            return NeptyneSheet(sheet_id, title, collection=self)

        if name is not None and name in self._idx:
            raise KeyError(f"Sheet {name} already exists")
        if name is None:
            name = "Sheet"
            i = 0
            while name + str(i) in self._idx:
                i += 1
            name += str(i)
        # TODO: dont reuse sheet ids:
        sheet_id = max((sheet.sheet_id for sheet in self._sheets), default=-1) + 1
        sheet = NeptyneSheet(sheet_id, name, collection=self)
        self._sheets.append(sheet)
        self._update_index(sheet, do_add=True)

        content = {
            "name": sheet.name,
            **sheet.to_serializable().export(),
        }
        self._reply_to_client(MessageTypes.CREATE_SHEET, content)
        return sheet

    def delete_sheet(self, name_or_id: str | int) -> None:
        """Delete a sheet by **name_or_id**."""
        if service := self._gsheet_service():
            gsheets_api.delete_sheet(
                service, self.dash.gsheets_spreadsheet_id, name_or_id
            )
            return

        sheet = self[name_or_id]
        if sheet.sheet_id == 0:
            raise ValueError("Cannot delete the default sheet")
        cells_to_reevaluate = self.dash.clear_sheet(sheet.sheet_id)
        self._update_index(sheet, do_add=False)
        self._sheets.remove(sheet)
        self._reply_to_client(
            MessageTypes.DELETE_SHEET, DeleteSheetContent(sheet.sheet_id).to_dict()
        )
        self.dash.run_cells_with_cascade(cell_ids=list(cells_to_reevaluate))

    def rename_sheet(self, name_or_id: str | int, new_name: str) -> None:
        """Rename a sheet by **name_or_id** to **new_name**."""
        if service := self._gsheet_service():
            gsheets_api.rename_sheet(
                service, self.dash.gsheets_spreadsheet_id, name_or_id, new_name
            )
            return
        if new_name in self._idx:
            raise KeyError(f"Sheet {new_name} already exists")
        sheet = self[name_or_id]
        self.dash.rename_sheet_reference(sheet.name, new_name)
        self._update_index(sheet, do_add=False)
        sheet._name = new_name
        self._update_index(sheet, do_add=True)
        self._reply_to_client(
            MessageTypes.RENAME_SHEET,
            RenameSheetContent(name=new_name, sheet_id=sheet.sheet_id).to_dict(),
        )

    def sheet_from_dataframe(self, df: pd.DataFrame, sheet_name: str) -> NeptyneSheet:
        """Create a new sheet from a pandas DataFrame **df** with the name **sheet_name**."""
        i = 0

        while True:
            try:
                sheet = self.new_sheet(sheet_name if not i else f"{sheet_name}_{i}")
            except KeyError:
                i += 1
            else:
                break

        self.dash[0, 0, sheet.sheet_id] = [[*df]] + [
            list(t) for t in df.itertuples(index=False)
        ]
        return sheet

    def sheet_from_csv(self, path_or_buffer: str, sheet_name: str) -> NeptyneSheet:
        """Create a new sheet from a CSV file at **path_or_buffer** with the name **sheet_name**."""
        return self.sheet_from_dataframe(pd.read_csv(path_or_buffer), sheet_name)

    @contextmanager
    def use_gsheet(self, spreadsheet_id: str) -> Iterator[None]:
        """Within this context, all sheet operations will be performed on the Google Sheet with the
        given **spreadsheet_id**."""
        token = sheet_context.gsheet_id_override.set(spreadsheet_id)
        try:
            yield
        finally:
            sheet_context.gsheet_id_override.reset(token)

    @contextmanager
    def use_sheet(self, sheet_name: str) -> Iterator[None]:
        """Within this context, all sheet operations will be performed on the sheet with the given **sheet_name**."""
        token = sheet_context.sheet_name_override.set(sheet_name)
        try:
            yield
        finally:
            sheet_context.sheet_name_override.reset(token)

    def _get_sheet_name_to_id(self) -> dict[str, int]:
        return {sheet.name: sheet.sheet_id for sheet in self._idx.values()}

    def _load_serializable_sheets(self, sheets: Iterable[Sheet]) -> None:
        for sh in self._sheets:
            self._update_index(sh, do_add=False)
        self._sheets = []
        for sheet in sheets:
            self._sheets.append(NeptyneSheet.from_serializable(sheet, collection=self))
            self._update_index(self._sheets[-1], do_add=True)

    def _register_sheet(self, sheet_id: int, name: str) -> None:
        sheet = NeptyneSheet(sheet_id, name, self)
        self._sheets.append(sheet)
        self._update_index(sheet, do_add=True)
