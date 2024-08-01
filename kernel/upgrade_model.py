from typing import Any

from .neptyne_protocol import SheetAttribute
from .tyne_model.sheet import TyneSheets

DEFAULT_COL_WIDTH = 100
DEFAULT_ROW_HEIGHT = 20


def upgrade_model(sheets: TyneSheets) -> None:
    for sheet in sheets.sheets.values():
        upgrade_sheet_attributes(sheet.attributes)


def upgrade_sheet_attributes(sheet_attributes: dict[str, Any]) -> None:
    for attr in SheetAttribute.ROWS_SIZES.value, SheetAttribute.COLS_SIZES.value:
        if isinstance(value := sheet_attributes.get(attr), list):
            sheet_attributes[attr] = transform_header_sizes_attribute(
                value, attr == SheetAttribute.ROWS_SIZES.value
            )


def transform_header_sizes_attribute(
    header_sizes: list[int], is_row: bool
) -> dict[str, int]:
    default_size = DEFAULT_ROW_HEIGHT if is_row else DEFAULT_COL_WIDTH
    return {
        str(i): header_sizes[i]
        for i in range(len(header_sizes))
        if header_sizes[i] != default_size
    }
