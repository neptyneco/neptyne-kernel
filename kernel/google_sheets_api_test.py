from unittest.mock import Mock

from .cell_range import CellRange
from .gsheets_api import GSheetRef
from .gsheets_api_test import MockGoogleSheetsService
from .test_utils import a1


class MockService(MockGoogleSheetsService):
    def __init__(self):
        super().__init__()
        self.calls = []

    def batchUpdate(self, spreadsheetId, body):
        self.calls.append(("batchUpdate", spreadsheetId, body))
        return Mock(execute=lambda: None)


def cell_range(a1_str: str, dash):
    address = a1(a1_str)
    return CellRange(
        GSheetRef(
            dash.gsheet_service,
            dash.gsheets_spreadsheet_id,
            "",
            address,
            [0],
        )
    )


def test_google_sheets_api(dash):
    from .neptyne_api.google.sheets import cut_paste

    dash.in_gs_mode = True
    spreadsheet_id = "spreadsheetid"
    dash.gsheets_spreadsheet_id = spreadsheet_id
    dash._gsheet_service = service = MockService()
    cut_paste(cell_range("A1:B2", dash), cell_range("C1", dash))
    sheet_id = hash("Sheet1")
    assert service.calls == [
        (
            "batchUpdate",
            spreadsheet_id,
            {
                "requests": [
                    {
                        "cutPaste": {
                            "source": {
                                "sheetId": sheet_id,
                                "startRowIndex": 0,
                                "endRowIndex": 2,
                                "startColumnIndex": 0,
                                "endColumnIndex": 2,
                            },
                            "destination": {
                                "sheetId": sheet_id,
                                "rowIndex": 0,
                                "columnIndex": 2,
                            },
                        }
                    }
                ]
            },
        )
    ]
