from collections import defaultdict
from dataclasses import replace
from datetime import date, datetime
from unittest import mock

import pytest

from .cell_address import Address, Range
from .cell_range import CellRangeGSheet
from .dash import Dash
from .datetime_conversions import datetime_to_serial
from .gsheets_api import Formula
from .neptyne_protocol import Dimension
from .primitives import NeptynePrimitive, unproxy_val

SPREADSHEET_ID = "spreadsheet_id"


class Executor:
    def __init__(self, func):
        self.func = func

    def execute(self):
        return self.func()


def result_for_values(values):
    def value_to_cell(value):
        cell = {}
        if isinstance(value, (datetime, date)):
            cell["effectiveValue"] = {"numberValue": datetime_to_serial(value)}
            cell["effectiveFormat"] = {"numberFormat": {"type": "DATE_TIME"}}
        elif isinstance(value, float):
            cell["effectiveValue"] = {"numberValue": value}
        elif isinstance(value, int):
            cell["effectiveValue"] = {"numberValue": float(value)}
        elif isinstance(value, str):
            cell["effectiveValue"] = {"stringValue": value}
        return cell

    return {
        "rowData": [{"values": [value_to_cell(cell) for cell in row]} for row in values]
    }


class MockGoogleSheetsService:
    sheets: dict[str, dict[tuple[int, int], str]]
    named_ranges: dict[str, str]

    def __init__(self):
        self.sheets = defaultdict(dict)
        self.sheets["Sheet1"] = self.empty_sheet()
        self.named_ranges = {}

    def spreadsheets(self):
        return self

    def split_range(self, range):
        if "!" in range:
            sheet, range = range.split("!", 1)
        else:
            sheet = next(iter(self.sheets.keys()))
        if ":" not in range:
            range = f"{range}:{range}"
        if not range[-1].isdigit():
            range += str(max(1000, max(y for x, y in self.sheets[sheet].keys()) + 1))
        p = range.find(":")
        if range[p + 1].isdigit():
            address = Address(
                max(x for x, y in self.sheets[sheet].keys()), int(range[p + 1 :]) - 1, 0
            )
            range = range[: p + 1] + address.to_a1()
        return range, sheet

    def get(
        self,
        spreadsheetId,
        range=None,
        ranges=None,
        fields=None,
        valueRenderOption="UNFORMATTED_VALUE",
        includeGridData=True,
    ):
        def execute():
            nonlocal range, ranges, fields

            if range is None:
                if ranges is not None:
                    range = ranges[0]
                    extra = {}
                    if range in self.named_ranges:
                        named_range = range
                        range = self.named_ranges[range]
                        r = Range.from_a1(range)
                        extra["namedRanges"] = [
                            {
                                "name": named_range,
                                "range": {
                                    "startColumnIndex": r.min_col,
                                    "endColumnIndex": r.max_col + 1,
                                    "startRowIndex": r.min_row,
                                    "endRowIndex": r.max_row + 1,
                                },
                                "rangeId": hash(named_range),
                            }
                        ]
                    r, sheet = self.split_range(range)
                    ranges_value = [self.get(spreadsheetId, range=range).execute()]
                    return {
                        "sheets": [
                            {
                                "data": [
                                    result_for_values(v["values"]) for v in ranges_value
                                ],
                                "properties": {"sheetId": hash(sheet), "title": sheet},
                            }
                        ],
                        **extra,
                    }
                else:
                    return {
                        "sheets": [
                            {"properties": {"sheetId": hash(name), "title": name}}
                            for name in self.sheets
                        ]
                    }

            if range in self.sheets:
                sheet = range
                values = []
                for x, y in sorted(self.sheets[range]):
                    if y >= len(values):
                        values.append([])
                    row = values[y]
                    if x >= len(row):
                        row.append(None)
                    row[x] = self.sheets[range][(x, y)]
                range = None
            else:
                r, sheet = self.split_range(range)

                values = []
                for addresses in Range.from_a1(r):
                    row = [
                        self.sheets[sheet].get((address.column, address.row))
                        for address in addresses
                    ]
                    while row and row[-1] is None:
                        row.pop()
                    values.append(row)
                if "!" not in range:
                    sheet = next(iter(self.sheets.keys()))
                    range = f"{sheet}!{range}"

            while len(values) > 1 and not values[-1]:
                values.pop()

            return {
                "values": values,
                "range": range,
                "properties": {"sheetId": hash(sheet), "title": sheet},
            }

        return Executor(execute)

    def values(self):
        return self

    def update(self, spreadsheetId, range, body, valueInputOption):
        def execute():
            r, sheet = self.split_range(range)

            for row_idx, row in enumerate(Range.from_a1(r)):
                for col_idx, address in enumerate(row):
                    value = body["values"][row_idx][col_idx]
                    if value is None:
                        continue
                    if (
                        valueInputOption == "RAW"
                        and isinstance(value, str)
                        and value.startswith("=")
                    ):
                        value = f"'{value}"

                    self.sheets[sheet][(address.column, address.row)] = value
            return {"updatedRange": range, "updatedValues": body["values"]}

        return Executor(execute)

    def empty_sheet(self):
        return {(x, y): None for x in range(26) for y in range(1000)}

    def _sheet_id_to_name(self, sheet_id: int):
        """Test use only, based on the fact that get() returns hash of the sheet name"""
        hash_to_key = {hash(key): key for key in self.sheets}
        return hash_to_key.get(sheet_id, 0)

    def batchUpdate(self, spreadsheetId, body):
        def execute():
            for request in body["requests"]:
                if props := request.get("addSheet"):
                    title = props["properties"]["title"]
                    self.sheets[title] = self.empty_sheet()
                    return {
                        "replies": [
                            {
                                "addSheet": {
                                    "properties": {
                                        "title": title,
                                        "sheetId": hash(title),
                                    }
                                }
                            }
                        ]
                    }
                elif props := request.get("deleteSheet"):
                    for title in self.sheets:
                        if hash(title) == props["sheetId"]:
                            break
                    else:
                        raise ValueError("Sheet not found")
                    del self.sheets[title]
                elif props := request.get("cutPaste"):
                    src_range = props["source"]
                    dst_range = props["destination"]

                    src_sheet_name = self._sheet_id_to_name(src_range["sheetId"])
                    start_column_index = src_range["startColumnIndex"]
                    start_row_index = src_range["startRowIndex"]
                    end_column_index = src_range["endColumnIndex"]
                    if end_column_index is None:
                        end_column_index = max(
                            x for x, y in self.sheets[src_sheet_name]
                        )
                    end_row_index = src_range["endRowIndex"]
                    if end_row_index is None:
                        end_row_index = max(y for x, y in self.sheets[src_sheet_name])
                    src = {
                        (x, y): self.sheets[src_sheet_name].get((x, y))
                        for x in range(start_column_index, end_column_index)
                        for y in range(start_row_index, end_row_index)
                    }
                    for c in src:
                        self.sheets[src_sheet_name][c] = None

                    dest_sheet_name = self._sheet_id_to_name(dst_range["sheetId"])
                    for (x, y), value in src.items():
                        self.sheets[dest_sheet_name][
                            (
                                x + dst_range["columnIndex"] - start_column_index,
                                y + dst_range["rowIndex"] - start_row_index,
                            )
                        ] = value
                elif props := request.get("updateCells"):
                    start = props["start"]
                    sheet_name = self._sheet_id_to_name(start["sheetId"])
                    for row_idx, row in enumerate(props["rows"]):
                        for col_idx, cell in enumerate(row["values"]):
                            self.sheets[sheet_name][
                                (
                                    start["columnIndex"] + col_idx,
                                    start["rowIndex"] + row_idx,
                                )
                            ] = cell["userEnteredValue"].get(
                                "stringValue",
                                cell["userEnteredValue"].get("numberValue"),
                            )
                elif props := request.get("copyPaste"):
                    pass
                else:
                    raise ValueError(
                        f"Request not yet supported. Please add it to MockGoogleSheetsService keys: {request.keys()}"
                    )
            return {}

        return Executor(execute)

    def clear(self, spreadsheetId, range):
        def execute():
            r, sheet = self.split_range(range)

            for row_idx, row in enumerate(Range.from_a1(r)):
                for col_idx, address in enumerate(row):
                    self.sheets[sheet][(address.column, address.row)] = None
            return {}

        return Executor(execute)


@pytest.fixture
def dash():
    prev = Dash._instance
    Dash._instance = None
    with mock.patch("neptyne_kernel.dash.get_ipython_mockable"):
        d = Dash(True)
        d.in_gs_mode = True
        d.gsheets_spreadsheet_id = SPREADSHEET_ID
        d._gsheet_service = MockGoogleSheetsService()
        d.named_ranges.setup(d.gsheet_service, d.gsheets_spreadsheet_id)
        yield d
    Dash._instance = prev


def test_set_get_item(dash):
    address = Address.from_a1("A1")
    dash[address] = "Test Value"
    assert dash[address] == "Test Value"


def test_set_get_item_range(dash):
    address = Address.from_a1("B2")
    dash[address] = [[1, 2], [3, 4]]

    assert dash[address] == 1
    assert dash[Address.from_a1("B3")] == 3

    r = dash[Range.from_a1("B2:C3")]
    assert r[1][1] == 4


def test_types(dash):
    address = Address.from_a1("A1")
    dash[address] = [[1, 2, 3], [3, 4, 5]]
    v = dash[address]
    assert isinstance(v, NeptynePrimitive)

    r = dash[Range.from_a1("A1:A3")]
    assert isinstance(r[1], NeptynePrimitive)
    for v in r:
        assert isinstance(v, NeptynePrimitive)

    r = dash[Range.from_a1("A1:C1")]
    assert isinstance(r[1], NeptynePrimitive)
    for v in r:
        assert isinstance(v, NeptynePrimitive)

    r2 = dash[Range.from_a1("A1:C3")]
    assert isinstance(r2[2][1], NeptynePrimitive)
    for r in r2:
        for v in r:
            assert isinstance(v, NeptynePrimitive)


def test_sheet_set_get_item(dash):
    dash[(0, 0, "Sheet1")] = "Test Value"
    assert dash[(0, 0, "Sheet1")] == "Test Value"


def test_insert_delete(dash):
    cell_range = dash[Range.from_a1("A1:B4")]
    result = cell_range.insert_row(0, "test")
    assert isinstance(result, CellRangeGSheet)
    assert result[0] == "test"
    assert result[1].is_empty()
    assert dash[Range.from_a1("A1")] == "test"

    cell_range.insert_row(0, "test2")
    assert dash[Range.from_a1("A1")] == "test2"


def test_insert_with_empty_row_shift(dash):
    cell_range = dash[Range.from_a1("A1:B4")]
    cell_range.insert_row(0, [1, 2])
    cell_range.insert_row(0, ["test", None])
    row_3 = [4, 5]
    assert [*cell_range.insert_row(0, row_3)] == row_3
    assert dash[Range.from_a1("A1")] == row_3[0]
    assert dash[Range.from_a1("B1")] == row_3[1]
    assert dash[Range.from_a1("A2")] == "test"
    assert not dash[Range.from_a1("B2")]
    assert dash[Range.from_a1("A3")] == 1
    assert dash[Range.from_a1("B3")] == 2


def test_insert_with_multi_dim_array(dash):
    cell_range = dash[Range.from_a1("A1:B4")]
    cell_range.insert_row(1, [1, 2])
    cell_range.insert_row(0, [[3, 4], [5, 6]])
    assert dash[Range.from_a1("A1")] == 3
    assert dash[Range.from_a1("B1")] == 4
    assert dash[Range.from_a1("A2")] == 5
    assert dash[Range.from_a1("B2")] == 6
    assert not dash[Range.from_a1("A3")]
    assert not dash[Range.from_a1("B3")]
    assert dash[Range.from_a1("A4")] == 1
    assert dash[Range.from_a1("B4")] == 2

    # Make sure no spillover outside of range
    cell_range.insert_row(1, [1, 2])
    assert not dash[Range.from_a1("A5")]


def test_insert_single_column(dash):
    cell_range = dash[Range.from_a1("A1:A4")]
    cell_range.insert_row(0, "test1")
    cell_range.insert_row(0, "test2")
    assert [*cell_range.insert_row(1, "test3")] == ["test3"]
    assert dash[Range.from_a1("A1")] == "test2"
    assert dash[Range.from_a1("A2")] == "test3"
    assert dash[Range.from_a1("A3")] == "test1"


def test_delete_row(dash):
    address = Address.from_a1("A1")
    dash[address] = [[1, 2], [3, 4], [5, "six"]]
    cell_range = dash[Range.from_a1("A1:B4")]
    result = cell_range.delete_row(1)
    assert result is None
    assert dash[Range.from_a1("A1")] == 1
    assert dash[Range.from_a1("B1")] == 2
    assert dash[Range.from_a1("A2")] == 5
    assert dash[Range.from_a1("B2")] == "six"
    assert not dash[Range.from_a1("A3")]
    assert not dash[Range.from_a1("B3")]


def test_delete_single_column(dash):
    address = Address.from_a1("B2")
    dash[address] = [1, 2, 3]
    cell_range = dash[Range.from_a1("B2:C4")]
    result = cell_range.delete_row(1)
    assert result is None
    assert dash[Range.from_a1("B2")] == 1
    assert dash[Range.from_a1("B3")] == 3
    assert not dash[Range.from_a1("B4")]


def test_sheets(dash):
    sheets = dash.sheets
    assert [sheet.name for sheet in sheets] == ["Sheet1"]

    sheets.new_sheet("Aap")
    sheets["Aap"][0, 0] = "Noot"
    assert sheets["Aap"][0, 0] == "Noot"

    del sheets["Aap"]
    assert [sheet.name for sheet in sheets] == ["Sheet1"]


def apply_values(dash, values):
    for cell, value in values:
        dash[Address.from_a1(cell)] = value


def range_from_a1(a1):
    c1, c2 = a1.split(":")
    infinite = None
    if c2.isdigit():
        c2 = f"ZZ{c2}"
        infinite = Dimension.COL
    elif c2.isalpha():
        c2 = f"{c2}10000"
        infinite = Dimension.ROW
    r = Range.from_a1(f"{c1}:{c2}")
    if infinite == Dimension.ROW:
        r = replace(r, max_row=-1)
    elif infinite == Dimension.COL:
        r = replace(r, max_col=-1)
    return r


@pytest.mark.parametrize(
    "a1_notation, values, expected",
    [
        ("B10:B", [], []),
        ("C10:D", [("C10", "header1"), ("D10", "header2")], [["header1", "header2"]]),
        (
            "C10:D",
            [
                ("C10", "header1"),
                ("D10", "header2"),
                ("C11", "val 1"),
                ("D11", "val 2"),
            ],
            [["header1", "header2"], ["val 1", "val 2"]],
        ),
        ("B10:10", [], []),
        (
            "C10:11",
            [("C10", "header1"), ("C11", "header2")],
            [["header1"], ["header2"]],
        ),
        (
            "C10:11",
            [
                ("C10", "header1"),
                ("D10", "header2"),
                ("C11", "val 1"),
                ("D11", "val 2"),
            ],
            [["header1", "header2"], ["val 1", "val 2"]],
        ),
    ],
)
def test_infinite_row_cols_values(dash, a1_notation, values, expected):
    apply_values(dash, values)
    cr = dash[range_from_a1(a1_notation)]
    assert cr.to_list() == expected


@pytest.mark.parametrize(
    "a1_notation, values, row, check, expected",
    [
        ("F1:F5", [("F1", "a"), ("F2", "b"), ("F3", "c")], ["d"], "F4:F5", ["d", None]),
        (
            "B10:C20",
            [("B10", "b"), ("C10", "c")],
            ["c1", "c2"],
            "B11:C11",
            ["c1", "c2"],
        ),
        (
            "C2:D",
            [("C2", "c"), ("D2", "d")],
            ["e", "f"],
            "C3:D3",
            ["e", "f"],
        ),
    ],
)
def test_append_row(dash, a1_notation, values, row, check, expected):
    apply_values(dash, values)
    cr = dash[range_from_a1(a1_notation)]
    cr.append_row(row)
    assert to_list(dash[Range.from_a1(check)]) == expected


@pytest.mark.parametrize(
    "a1_notation, index, expected",
    [
        ("B2:B6", -1, "B6"),
        ("B2:D5", (-1, -1), "D5"),
        ("B2:D5", (1, 1), "C3"),
        ("A1:A5", 0, "A1"),
        ("B2:B6", 1, "B3"),
        ("C2:D8", 1, ["C3", "D3"]),
        ("C2:C8", slice(1, 3), ["C3", "C4"]),
        ("C2:D8", slice(1, 3), [["C3", "D3"], ["C4", "D4"]]),
        ("B2:B6", -2, "B5"),
        ("C2:D8", -2, ["C7", "D7"]),
        ("C2:D8", slice(-2, -1), ["C7", "D7"]),
        ("C2:D8", slice(-2, -1), ["C7", "D7"]),
        ("B2:B6", slice(-3, None), ["B4", "B5", "B6"]),
        (
            "C2:D8",
            slice(-3, None),
            [
                ["C6", "D6"],
                ["C7", "D7"],
                ["C8", "D8"],
            ],
        ),
    ],
)
def test_indexes(dash, a1_notation, index, expected):
    dash[Address.from_a1("A1")] = [
        [ch1 + ch2 for ch1 in "ABCDEF"] for ch2 in "12345678"
    ]
    cr = dash[Range.from_a1(a1_notation)]
    val = cr[index]
    if isinstance(val, CellRangeGSheet):
        val = val.to_list()
    assert val == expected


def to_list(cr):
    def unproxy_row(row):
        return [unproxy_val(cell) for cell in row]

    if cr.two_dimensional:
        return [unproxy_row(row) for row in cr]
    return unproxy_row(cr)


def test_infinite_range_read(dash):
    dash[Address.from_a1("B2")] = [["b2", "c2"], ["b3", "c3"]]
    one_column = dash[range_from_a1("B2:B")]
    assert one_column[0] == "b2"
    assert one_column[2].is_empty()

    one_row = dash[range_from_a1("B2:2")]
    assert one_row[0] == "b2"
    assert one_row[2].is_empty()

    two_columns = dash[range_from_a1("B2:C")]
    assert to_list(two_columns[0]) == ["b2", "c2"]
    assert to_list(two_columns[2]) == [None, None]

    two_columns_10 = two_columns[:10]
    assert to_list(two_columns_10[0]) == ["b2", "c2"]

    four_cells = dash[range_from_a1("B2:C3")]
    assert to_list(four_cells) == [["b2", "c2"], ["b3", "c3"]]

    four_cells = dash[range_from_a1("B3:C4")]
    assert to_list(four_cells) == [["b3", "c3"], [None, None]]


@pytest.mark.parametrize(
    "rng, idx, val_addr",
    [
        ("B3:B7", -1, "B7"),
        ("B3:C7", (1, 1), "C4"),
        ("B3:B7", 4, "B7"),
        ("B3:E3", 3, "E3"),
        ("A1:A5", 0, "A1"),
        ("D1:D10", -2, "D9"),
        ("E5:E10", 2, "E7"),
    ],
)
def test_write_cells(dash, rng, idx, val_addr):
    cr = dash[range_from_a1(rng)]
    cr[idx] = val_addr
    assert dash[Address.from_a1(val_addr)] == val_addr
    assert cr[idx] == val_addr


def test_infinite_range_write(dash):
    dash[Address.from_a1("B2")] = [["b2", "c2"], ["b3", "c3"]]
    one_column = dash[range_from_a1("B2:B")]
    one_column[3] = "b5"
    assert dash[Address.from_a1("B5")] == "b5"
    assert one_column[2].is_empty()
    assert one_column[3] == "b5"

    one_row = dash[range_from_a1("B2:2")]
    one_row[3] = "e2"
    assert dash[Address.from_a1("E2")] == "e2"
    assert one_row[2].is_empty()
    assert one_row[3] == "e2"

    two_columns = dash[range_from_a1("B2:C")]
    two_columns[(3, 1)] = "c5"
    assert dash[Address.from_a1("C5")] == "c5"

    two_rows = dash[range_from_a1("B2:3")]
    two_rows[(1, 2)] = "d3"
    assert dash[Address.from_a1("D3")] == "d3"


def test_named_ranges(dash):
    named_ranges = dash._gsheet_service.named_ranges
    named_ranges["one_cell"] = "B1:B3"

    dash.named_ranges["one_cell"][1] = "b2"

    assert dash[Address.from_a1("B2")] == "b2"


def test_clear(dash):
    sheets = dash.sheets
    sheets.new_sheet("Sheet2")

    dash[(0, 0, "Sheet1")] = 2
    dash[(0, 0, "Sheet2")] = 3
    assert dash[(0, 0, "Sheet2")] == 3
    dash[(0, 0, "Sheet2")].clear()
    assert not dash[(0, 0, "Sheet2")]
    assert dash[(0, 0, "Sheet1")] == 2


def test_insert_formula(dash):
    dt = datetime(2022, 1, 1)
    dash[Address.from_a1("B4")] = [
        42,
        Formula("=SUM(B1:B3)"),
        "=SUM(B1:B3)",
        dt,
    ]

    assert dash[Address.from_a1("B4")] == 42
    assert dash[Address.from_a1("B5")] == "=SUM(B1:B3)"
    assert dash[Address.from_a1("B6")] == "'=SUM(B1:B3)"
    assert dash[Address.from_a1("B7")] == dt.strftime("%Y-%m-%d")


def test_insert_numpy_array_df(dash):
    import numpy as np
    import pandas as pd

    cell_range = dash[Range.from_a1("A1:C4")]
    cell_range.insert_row(0, np.array([[1, 2], [3, 4]]))

    assert dash[Range.from_a1("A1")] == 1
    assert dash[Range.from_a1("B1")] == 2
    assert dash[Range.from_a1("A2")] == 3
    assert dash[Range.from_a1("B2")] == 4

    assert not dash[Range.from_a1("A3")]
    assert not dash[Range.from_a1("B3")]

    cell_range = dash[Range.from_a1("A1:C10")]
    cell_range.append_row(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    assert dash[Range.from_a1("A3")] == "a"
    assert dash[Range.from_a1("A4")] == 1
