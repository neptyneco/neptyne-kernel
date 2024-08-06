from .neptyne_protocol import SheetAttribute
from .tyne_model.sheet import TyneSheets
from .upgrade_model import (
    DEFAULT_COL_WIDTH,
    DEFAULT_ROW_HEIGHT,
    upgrade_model,
)


def test_upgrade_sheet_attributes():
    sheets = TyneSheets()
    sheets.sheets[0].attributes = {
        SheetAttribute.ROWS_SIZES.value: [
            DEFAULT_ROW_HEIGHT,
            40,
            DEFAULT_ROW_HEIGHT,
            50,
            DEFAULT_ROW_HEIGHT,
            DEFAULT_ROW_HEIGHT,
            DEFAULT_ROW_HEIGHT,
            DEFAULT_ROW_HEIGHT,
            DEFAULT_ROW_HEIGHT,
            DEFAULT_ROW_HEIGHT,
        ],
        SheetAttribute.COLS_SIZES.value: [
            DEFAULT_COL_WIDTH,
            DEFAULT_COL_WIDTH,
            200,
            DEFAULT_COL_WIDTH,
            DEFAULT_COL_WIDTH,
            DEFAULT_COL_WIDTH,
            DEFAULT_COL_WIDTH,
            DEFAULT_COL_WIDTH,
            DEFAULT_COL_WIDTH,
            DEFAULT_COL_WIDTH,
        ],
    }
    upgrade_model(sheets)
    assert sheets.sheets[0].attributes == {
        SheetAttribute.ROWS_SIZES.value: {"1": 40, "3": 50},
        SheetAttribute.COLS_SIZES.value: {"2": 200},
    }
    upgrade_model(sheets)
    assert sheets.sheets[0].attributes == {
        SheetAttribute.ROWS_SIZES.value: {"1": 40, "3": 50},
        SheetAttribute.COLS_SIZES.value: {"2": 200},
    }
