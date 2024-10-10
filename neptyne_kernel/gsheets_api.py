import asyncio
import datetime
import json
import math
import os
from collections.abc import Mapping
from typing import Any, Callable, Iterator, Sequence

import google.auth.credentials
import google.auth.transport
import google_auth_httplib2
import httplib2
import jwt
import numpy as np
import pandas as pd
from googleapiclient.discovery import Resource
from googleapiclient.errors import HttpError
from googleapiclient.http import HttpRequest
from IPython import get_ipython

from . import sheet_context, streamlit_server
from .api_ref import ApiRef
from .cell_address import Address, Range, format_cell
from .cell_api import CellApiMixin
from .cell_range import CellRange, IntOrSlice, slice_or_int_to_range
from .dash_traceback import CoordinateTuple
from .datetime_conversions import datetime_to_serial
from .kernel_runtime import get_api_token
from .neptyne_protocol import CellAttribute, Dimension, SheetTransform
from .primitives import proxy_val
from .proxied_apis import NEEDS_GSHEET_ADVANCED_FEATURES_HTTP_CODE, TOKEN_HEADER_NAME
from .spreadsheet_datetime import excel2datetime
from .spreadsheet_error import GSheetError, GSheetNotAuthorized, SpreadsheetError
from .util import list_like
from .widgets.color import Color


def do_execute(to_execute: HttpRequest) -> dict[str, Any]:
    try:
        result = to_execute.execute()
    except HttpError as e:
        if e.resp.status == NEEDS_GSHEET_ADVANCED_FEATURES_HTTP_CODE:
            if streamlit_server.is_running_in_streamlit():
                from streamlit.platform import post_parent_message

                post_parent_message(json.dumps({"type": "gsheet_not_authorized"}))
            raise GSheetNotAuthorized(
                "To use this feature, you need to enable advanced features. "
                "You can find this in the Neptyne menu in the Google Sheets add-on."
            ) from None
        else:
            raise GSheetError(e) from None
    return result


def execute(to_execute: HttpRequest) -> dict[str, Any]:
    try:
        kernel = get_ipython().kernel
    except AttributeError:
        # mostly for compatibility with tests, we make this work without a kernel
        kernel = None

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if not os.getenv("NEPTYNE_LOCAL_REPL") and kernel and loop:
        parent = kernel.shell.get_parent()
        future = loop.run_in_executor(None, do_execute, to_execute)

        try:
            while not future.done():
                loop.run_until_complete(kernel.do_one_iteration())
        finally:
            kernel.shell.set_parent(parent)

        if future.exception():
            raise future.exception() from None  # type: ignore

        return future.result()

    return do_execute(to_execute)


class Formula:
    """Represents a Google Sheets formula.

    Use this to set a cell value to a formula that will be evaluated.

    Example:
    ```python
    A11 = Formula("=SUM(A1:A10)")
    ```
    """

    def __init__(self, value: str):
        self.value = value


class Credentials(google.auth.credentials.Credentials):
    def refresh(self, request: google.auth.transport.Request) -> None:
        if not os.getenv("NEPTYNE_LOCAL_REPL"):
            return

        from .dash import Dash

        api_token, gsheet_id = Dash.instance().get_api_token()
        claims = jwt.decode(api_token, options={"verify_signature": False})
        self.token = api_token
        self.expiry = datetime.datetime.fromtimestamp(
            claims["exp"], datetime.timezone.utc
        ).replace(tzinfo=None)

    def apply(self, headers: dict, token: str | None = None) -> None:
        from .dash import Dash

        if not os.getenv("NEPTYNE_LOCAL_REPL"):
            token = (
                Dash.instance().api_token_override
                if Dash.instance().api_token_override
                else get_api_token()
            )
        if token is None:
            token = self.token
        headers[TOKEN_HEADER_NAME] = token


class ProxiedHttp(httplib2.Http):
    def request(
        self,
        uri: str,
        method: str = "GET",
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        redirections: int | None = 5,
        connection_type: type | None = None,
    ) -> tuple[httplib2.Response, bytes]:
        if uri.startswith("https://sheets.googleapis.com"):
            host_port = os.getenv("API_PROXY_HOST_PORT", "localhost:8888")
            if host_port.startswith("https://") or host_port.startswith("http://"):
                protocol = ""
            elif ":" in host_port:
                protocol = "http://"
            else:
                protocol = "https://"
            uri = uri.replace(
                "https://sheets.googleapis.com",
                f"{protocol}{host_port}/google_sheets_api",
            )
        res = super().request(uri, method, body, headers, redirections, connection_type)

        return res


def request_builder(credentials: Credentials) -> Callable[..., HttpRequest]:
    def build_request(_http: httplib2.Http, *args: Any, **kwargs: Any) -> HttpRequest:
        """A thread-safe alternative to the default google-api-client behavior.

        See https://googleapis.github.io/google-api-python-client/docs/thread_safety.html"""
        new_http = google_auth_httplib2.AuthorizedHttp(credentials, http=ProxiedHttp())
        return HttpRequest(new_http, *args, **kwargs)

    return build_request


def sheets_for_spreadsheet(service: Resource, spreadsheet_id: str) -> list[dict]:
    spreadsheet = execute(service.spreadsheets().get(spreadsheetId=spreadsheet_id))
    return spreadsheet.get("sheets", [])


def title_for_sheet_id(service: Resource, spreadsheet_id: str, sheet_id: int) -> str:
    sheets = sheets_for_spreadsheet(service, spreadsheet_id)
    for sheet in sheets:
        properties = sheet.get("properties", {})
        if properties.get("sheetId") == sheet_id:
            return properties.get("title", "")
    raise ValueError(f"No sheet matches for '{sheet_id}'")


def sheet_id_for_title(service: Resource, sheet_title: str, spreadsheet_id: str) -> int:
    sheets = sheets_for_spreadsheet(service, spreadsheet_id)
    for sheet in sheets:
        properties = sheet.get("properties", {})
        if sheet_title == properties.get("title", ""):
            return properties.get("sheetId")
    raise ValueError(f"No sheet matches for '{sheet_title}'")


def sheet_prefix_to_id(
    service: Resource, spreadsheet_id: str, sheet_prefix: str
) -> int:
    if sheet_prefix:
        sheet_title = sheet_prefix[:-1]
        return sheet_id_for_title(service, sheet_title, spreadsheet_id)
    sheets = execute(service.spreadsheets().get(spreadsheetId=spreadsheet_id))["sheets"]
    return sheets[0]["properties"]["sheetId"]


def api_result(
    service: Resource,
    spreadsheet_id: str,
    sheet_prefix: str,
    rng: Range,
    values: list[list[Any]],
) -> CellRange | CellApiMixin:
    row_count = rng.max_row - rng.min_row + 1
    col_count = rng.max_col - rng.min_col + 1

    if len(values) < row_count:
        values.extend([[None] * col_count for _ in range(row_count - len(values))])
    values = [row + [None] * (col_count - len(row)) for row in values]

    if len(values) == 1 and rng.max_row != -1:
        if len(values[0]) == 1 and rng.max_col != -1:
            return proxy_val(
                values[0][0],
                GSheetRef(
                    service,
                    spreadsheet_id,
                    sheet_prefix,
                    Address(rng.min_col, rng.min_row, 0),
                    [values[0][0]],
                ),
            )
        else:
            values = values[0]
    elif values and len(values[0]) == 1 and rng.max_col != -1:
        values = [v[0] if v else None for v in values]

    return CellRange(
        GSheetRef(
            service,
            spreadsheet_id,
            sheet_prefix,
            rng,
            values,
        )
    )


def get_item(
    service: Resource, spreadsheet_id: str, item: CoordinateTuple | Address | Range
) -> CellRange | CellApiMixin:
    sheet: int | str

    # sheet can either be a string or an int; Address or Range can't handle strings as sheets though
    # but we'd like to use them for formatting anyway, so we keep sheet separate and use a place holder
    # in those objects:
    if isinstance(item, tuple):
        min_col, min_row, *rest, sheet = item
        if rest:
            item = Range(min_col, min_row, rest[0], rest[1], 0)
        else:
            item = Address(min_col, min_row, 0)
    else:
        sheet = item.sheet

    if isinstance(item, Address):
        min_col = max_col = item.column
        min_row = max_row = item.row
    else:
        min_col = item.min_col
        max_col = item.max_col
        min_row = item.min_row
        max_row = item.max_row

    complete_sheet = max_col == -1 and max_row == -1

    if isinstance(sheet, int):
        if sheet == 0 and (override := sheet_context.sheet_name_override.get()):
            sheet_name = override
        elif sheet > 0 or complete_sheet:
            sheet_name = title_for_sheet_id(service, spreadsheet_id, sheet)
        else:
            sheet_name = ""
    else:
        sheet_name = sheet

    if complete_sheet:
        min_col = min_row = 0
        a1 = sheet_name
        sheet_prefix = a1 + "!"
    else:
        sheet_prefix = sheet_name + "!" if sheet_name else ""
        a1 = sheet_prefix + item.to_a1()

    result = execute(
        service.spreadsheets().get(
            spreadsheetId=spreadsheet_id,
            ranges=[a1],
            fields=",".join(
                [
                    "sheets.data.rowData.values.effectiveValue",
                    "sheets.data.rowData.values.effectiveFormat.numberFormat",
                    "sheets.data.rowData.values.userEnteredValue",
                ]
            ),
            includeGridData=True,
        ),
    )

    values = values_for_result(result)

    if complete_sheet:
        if rr := result.get("range"):
            rr = rr.split("!", 1)[-1]
            as_range = Range.from_a1(rr)
            min_row = as_range.min_row
            min_col = as_range.min_col
            max_row = as_range.max_row
            max_col = as_range.max_col
        elif not values:
            max_row = max_col = -1
        else:
            max_row = len(values)
            max_col = len(values[0]) if max_row else 0

    if values == [[]] and max_row == -1:
        values = []

    return api_result(
        service,
        spreadsheet_id,
        sheet_prefix,
        Range(min_col, max_col, min_row, max_row, 0),
        values,
    )


def values_for_result(result: dict[str, Any]) -> list[list[Any]]:
    values = []

    def cell_to_value(
        cell: dict[str, Any],
    ) -> SpreadsheetError | float | str | int | None:
        value = cell.get("effectiveValue")
        if value is not None:
            if error_value := value.get("errorValue"):
                return SpreadsheetError(
                    "#" + error_value.get("type"), error_value.get("message")
                )
            cell_format = (
                cell.get("effectiveFormat", {}).get("numberFormat", {}).get("type")
            )
            if isinstance(value, Mapping) and value:
                value = next(iter(value.values()))
                if cell_format == "DATE_TIME" or cell_format == "DATE":
                    value = excel2datetime(value)
        return value

    if result["sheets"]:
        sheet_data = result["sheets"][0]["data"]
        if sheet_data:
            values = [
                [cell_to_value(cell) for cell in row.get("values", [])]
                for row in sheet_data[0].get("rowData", [])
            ]
    return values


def get_named_range(
    service: Resource, spreadsheet_id: str, named_range: str
) -> CellRange | CellApiMixin:
    result = execute(
        service.spreadsheets().get(
            spreadsheetId=spreadsheet_id,
            ranges=[named_range],
            fields=",".join(
                [
                    "sheets.properties.title",
                    "sheets.data.rowData.values.effectiveValue",
                    "sheets.data.rowData.values.effectiveFormat.numberFormat",
                    "namedRanges",
                ]
            ),
            includeGridData=True,
        ),
    )

    values = values_for_result(result)
    sheet = result["sheets"][0]["properties"]["title"]
    name_range_range = result["namedRanges"][0]["range"]
    a1 = Range(
        name_range_range["startColumnIndex"],
        name_range_range["endColumnIndex"] - 1,
        name_range_range["startRowIndex"],
        name_range_range["endRowIndex"] - 1,
        0,
    )

    return api_result(
        service,
        spreadsheet_id,
        sheet + "!",
        a1,
        values,
    )


def value_for_gsheet(
    val: Any,
) -> float | int | str | datetime.datetime | datetime.date | Formula:
    if val is None:
        return ""
    if isinstance(val, float) and (math.isinf(val) or math.isnan(val)):
        return ""
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, float | int | datetime.datetime | datetime.date | str | Formula):
        return val
    return str(val)


def set_item(
    service: Resource,
    spreadsheet_id: str,
    address: Address | str,
    value: Any,
    *,
    sheet_prefix: str = "",
) -> list[list[Any]]:
    if list_like(value) and list_like(value[0]):
        num_rows = len(value)
        num_cols = max(len(sublist) for sublist in value)
        write_value = [[value_for_gsheet(val) for val in sublist] for sublist in value]
    else:
        num_cols = 1
        if list_like(value):
            num_rows = len(value)
            write_value = [[value_for_gsheet(val)] for val in value]
        else:
            num_rows = 1
            write_value = [[value_for_gsheet(value)]]
    if isinstance(address, Address):
        a1 = (
            format_cell(address.column, address.row)
            + ":"
            + format_cell(address.column + num_cols - 1, address.row + num_rows - 1)
        )
        if sheet_prefix:
            a1 = sheet_prefix + a1
        elif address.sheet:
            a1 = title_for_sheet_id(service, spreadsheet_id, address.sheet) + "!" + a1
    else:
        a1 = address

    has_raw = False
    raw = []
    has_user_entered = False
    user_entered = []
    for row in write_value:
        raw_row: list[None | float | int | str] = []
        user_entered_row: list[None | float | int | str] = []
        for cell in row:
            if isinstance(cell, Formula):
                user_entered_row.append(cell.value)
                raw_row.append(None)
                has_user_entered = True
            elif isinstance(cell, (datetime.datetime, datetime.date)):
                if not isinstance(cell, datetime.datetime) or (
                    cell.time() == cell.min.time()
                ):
                    cell = cell.strftime("%Y-%m-%d")
                else:
                    cell = cell.strftime("%Y-%m-%d %H:%M:%S")
                user_entered_row.append(cell)
                raw_row.append(None)
                has_user_entered = True
            else:
                raw_row.append(cell)
                user_entered_row.append(None)
                has_raw = True
        raw.append(raw_row)
        user_entered.append(user_entered_row)

    for format, has_format, values in [
        ("RAW", has_raw, raw),
        ("USER_ENTERED", has_user_entered, user_entered),
    ]:
        if has_format:
            body = {"values": values, "majorDimension": "ROWS"}
            execute(
                service.spreadsheets()
                .values()
                .update(
                    spreadsheetId=spreadsheet_id,
                    range=a1,
                    body=body,
                    valueInputOption=format,
                )
            )

    return write_value


def set_attributes(
    service: Resource,
    spreadsheet_id: str,
    r: Range,
    sheet_prefix: str,
    attributes: dict[CellAttribute, str],
) -> None:
    update_dict: dict[str, Any] = {}

    def set(key: str, val: Any) -> None:
        d = update_dict
        keys = key.split(".")
        for k in keys[:-1]:
            if k not in d:
                update_dict[k] = {}
            d = d[k]
        d[keys[-1]] = val

    note = None

    sheet_id = sheet_prefix_to_id(service, spreadsheet_id, sheet_prefix)

    for attribute, value in attributes.items():
        if (
            attribute == CellAttribute.BACKGROUND_COLOR
            or attribute == CellAttribute.COLOR
        ):
            color = Color.from_webcolor(value)
            key = (
                "backgroundColor"
                if attribute == CellAttribute.BACKGROUND_COLOR
                else "textFormat.foregroundColor"
            )
            set(
                key,
                {
                    "red": color.r / 255,
                    "green": color.g / 255,
                    "blue": color.b / 255,
                },
            )
        elif attribute == CellAttribute.FONT:
            set(
                "textFormat.fontFamily",
                value,
            )
        elif attribute == CellAttribute.FONT_SIZE:
            set(
                "textFormat.fontSize",
                int(value),
            )
        elif attribute == CellAttribute.TEXT_STYLE:
            for style in value.split(" "):
                set(
                    "textFormat.%s" % style.lower(),
                    True,
                )
        elif attribute == CellAttribute.NOTE:
            origin_range = {
                "sheetId": sheet_id,
                "startRowIndex": r.min_row,
                "endRowIndex": r.min_row + 1,
                "startColumnIndex": r.min_col,
                "endColumnIndex": r.max_col + 1,
            }
            note = {
                "rows": [{"values": [{"note": str(value) if value else None}]}],
                "fields": "note",
                "range": origin_range,
            }

    grid_range = {
        "sheetId": sheet_id,
        "startRowIndex": r.min_row,
        "endRowIndex": r.max_row + 1,
        "startColumnIndex": r.min_col,
        "endColumnIndex": r.max_col + 1,
    }

    requests: list[dict] = []
    if update_dict:
        requests.append(
            {
                "repeatCell": {
                    "range": grid_range,
                    "cell": {"userEnteredFormat": update_dict},
                    "fields": f"userEnteredFormat({', '.join(update_dict.keys())})",
                }
            }
        )
    if note:
        requests.append({"updateCells": note})

    if not requests:
        return

    body = {"requests": requests}
    execute(service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body))


def exec_insert_delete_sheet_row_col(
    service: Resource,
    spreadsheet_id: str,
    sheet_id: int,
    dimension: Dimension,
    transform: SheetTransform,
    index: int,
    amount: int,
) -> None:
    operation = (
        "insertDimension"
        if transform == SheetTransform.INSERT_BEFORE
        else "deleteDimension"
    )

    body = {
        "requests": [
            {
                operation: {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": (
                            "ROWS" if dimension == Dimension.ROW else "COLUMNS"
                        ),
                        "startIndex": index,
                        "endIndex": index + amount,
                    },
                },
            }
        ]
    }

    execute(
        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=body,
        )
    )


def get_attributes(
    service: Resource,
    spreadsheet_id: str,
    r: Range,
    sheet_prefix: str,
) -> dict[CellAttribute, str]:
    # This unfortunately doesn't work. Leaving it in to get started with the API:
    # a1 = sheet_prefix + r.to_a1()
    # fields = (
    #     "sheets(data(rowData(values("
    #     "userEnteredFormat.backgroundColor,"
    #     "userEnteredFormat.textFormat,"
    #     "userEnteredFormat.horizontalAlignment,"
    #     "userEnteredFormat.verticalAlignment,"
    #     "effectiveFormat.backgroundColor,"
    #     "effectiveFormat.textFormat,"
    #     "effectiveFormat.horizontalAlignment,"
    #     "effectiveFormat.verticalAlignment"
    #     "))))"
    # )
    response = service.spreadsheets()

    return response.get("values", [])


def insert_delete_row_column(
    service: Resource,
    spreadsheet_id: str,
    r: Range,
    dimension: Dimension,
    transform: SheetTransform,
    index: int,
    amount: int,
    data: Any,
    sheet_prefix: str,
) -> CellRange | None:
    requests: list[dict[str, Any]] = []
    sheet_id = sheet_prefix_to_id(service, spreadsheet_id, sheet_prefix)
    start_row_index = r.min_row
    end_row_index: int | None = r.max_row + 1
    start_col_index = r.min_col
    end_col_index: int | None = r.max_col + 1

    # rows and cols are the same just flipped:
    if dimension == Dimension.COL:
        start_row_index, start_col_index = start_col_index, start_row_index
        end_row_index, end_col_index = end_col_index, end_row_index

    if transform == SheetTransform.INSERT_BEFORE:
        start_row_index += index
        to_row_index = start_row_index + amount
        if end_row_index == 0:  # unbounded:
            end_row_index = None
        else:
            assert end_row_index is not None
            end_row_index -= amount
    elif transform == SheetTransform.DELETE:
        to_row_index = start_row_index + index
        start_row_index += index + amount
    else:
        raise ValueError(f"Unsupported transform: {transform}")
    to_col_index = start_col_index

    if dimension == Dimension.COL:
        start_row_index, start_col_index = start_col_index, start_row_index
        end_row_index, end_col_index = end_col_index, end_row_index
        to_row_index, to_col_index = to_col_index, to_row_index

    if (
        end_row_index is None
        or end_col_index is None
        or (end_row_index > start_row_index or end_col_index > start_col_index)
    ):
        requests.append(
            {
                "cutPaste": {
                    "source": {
                        "sheetId": sheet_id,
                        "startRowIndex": start_row_index,
                        "endRowIndex": end_row_index,
                        "startColumnIndex": start_col_index,
                        "endColumnIndex": end_col_index,
                    },
                    "destination": {
                        "sheetId": sheet_id,
                        "rowIndex": to_row_index,
                        "columnIndex": to_col_index,
                    },
                    "pasteType": "PASTE_NORMAL",
                }
            }
        )

    if transform == SheetTransform.INSERT_BEFORE:
        data_row = r.min_row + (index if dimension == Dimension.ROW else 0)
        data_col = r.min_col + (index if dimension == Dimension.COL else 0)
        data_width = (
            amount if dimension == Dimension.COL else end_col_index - start_col_index  # type: ignore
        )
        data_height = (
            amount if dimension == Dimension.ROW else end_row_index - start_row_index  # type: ignore
        )

        if isinstance(data, pd.DataFrame):
            data = [data.columns.tolist(), *data.values.tolist()]

        if isinstance(data, str) or not hasattr(data, "__len__"):
            data = [
                [data if x == 0 and y == 0 else None for x in range(data_width)]
                for y in range(data_height)
            ]
        elif isinstance(data[0], str) or not hasattr(data[0], "__len__"):
            if dimension == Dimension.COL:
                data = [
                    [
                        data[y] if x == 0 and y < len(data) else None
                        for x in range(data_width)
                    ]
                    for y in range(data_height)
                ]
            else:
                data = [
                    [data[y] if x == 0 else None for x in range(data_width)]
                    for y in range(data_height)
                ]
        else:
            data = [
                [
                    data[y][x] if y < len(data) and x < len(data[y]) else None
                    for x in range(data_width)
                ]
                for y in range(data_height)
            ]

        values = []
        for row_data in data:

            def gsheet_dict(val: Any) -> dict[str, dict[str, Any]]:
                val = value_for_gsheet(val)
                if isinstance(val, bool):
                    return {"userEnteredValue": {"boolValue": val}}
                elif isinstance(val, (float, int)):
                    return {"userEnteredValue": {"numberValue": val}}
                elif isinstance(val, (datetime.datetime, datetime.date)):
                    if not isinstance(val, datetime.datetime) or (
                        val.time() == val.min.time()
                    ):
                        format = "DATE"
                    else:
                        format = "DATE_TIME"
                    return {
                        "userEnteredValue": {"numberValue": datetime_to_serial(val)},
                        "userEnteredFormat": {"numberFormat": {"type": format}},
                    }
                elif isinstance(val, Formula):
                    return {"userEnteredValue": {"formulaValue": val.value}}
                else:
                    return {"userEnteredValue": {"stringValue": str(val)}}

            row_values = [gsheet_dict(cell) for cell in row_data]
            values.append({"values": row_values})
        data_request = {
            "updateCells": {
                "start": {
                    "sheetId": sheet_id,
                    "rowIndex": data_row,
                    "columnIndex": data_col,
                },
                "rows": values,
                "fields": "userEnteredValue,userEnteredFormat.numberFormat",
            }
        }
        requests.append(data_request)
        if len(data) == 1:
            data = data[0]
        elif len(data[0]) == 1:
            data = [row[0] for row in data]
        res = CellRange(
            GSheetRef(
                service,
                spreadsheet_id,
                sheet_prefix,
                Range(
                    data_col,
                    data_col + data_width - 1,
                    data_row,
                    data_row + data_height - 1,
                    0,
                ),
                data,
            )
        )
    else:
        res = None

    body = {"requests": requests}
    execute(
        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=body,
        )
    )
    return res


def available_sheets(service: Resource, spreadsheet_id: str) -> list[tuple[str, int]]:
    sheets = execute(service.spreadsheets().get(spreadsheetId=spreadsheet_id)).get(
        "sheets", []
    )
    return [
        (title, sheet_id)
        for sheet in sheets
        if (props := sheet.get("properties"))
        and (title := props.get("title"))
        and (sheet_id := props.get("sheetId")) is not None
    ]


def new_sheet(
    service: Resource, spreadsheet_id: str, name: str | None = None
) -> tuple[str, int]:
    body: dict[str, Any] = {"requests": [{"addSheet": {}}]}
    if name:
        body["requests"][0]["addSheet"]["properties"] = {"title": name}

    response = execute(
        service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body)
    )
    # Extract the added sheet's properties from the response
    added_sheet_properties = (
        response.get("replies", [])[0].get("addSheet", {}).get("properties", {})
    )
    sheet_title = added_sheet_properties.get("title", "")
    sheet_id = added_sheet_properties.get("sheetId", 0)

    return sheet_title, sheet_id


def delete_sheet(
    service: Resource, spreadsheet_id: str, name_or_idx: str | int
) -> None:
    if isinstance(name_or_idx, int):
        sheet_id = name_or_idx
    else:
        sheet_id = sheet_id_for_title(service, name_or_idx, spreadsheet_id)
    body = {"requests": [{"deleteSheet": {"sheetId": sheet_id}}]}
    execute(service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body))


def rename_sheet(
    service: Resource, spreadsheet_id: str, name_or_idx: str | int, new_name: str
) -> None:
    if isinstance(name_or_idx, int):
        sheet_id = name_or_idx
    else:
        sheet_id = sheet_id_for_title(service, name_or_idx, spreadsheet_id)
    body = {
        "requests": [
            {
                "updateSheetProperties": {
                    "properties": {"sheetId": sheet_id, "title": new_name},
                    "fields": "title",
                }
            }
        ]
    }
    execute(service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body))


class GSheetRef(ApiRef):
    service: Resource
    spreadsheet_id: str
    sheet_prefix: str
    values: list[list] | list

    def __init__(
        self,
        service: Resource,
        spreadsheet_id: str,
        sheet_prefix: str,
        ref: Address | Range,
        values: list[list] | list,
    ):
        super().__init__(ref)
        self.service = service
        self.spreadsheet_id = spreadsheet_id
        self.sheet_prefix = sheet_prefix
        self.values = values

    def with_range(self, new_range: Range, values: list[list] | list) -> "GSheetRef":
        return GSheetRef(
            service=self.service,
            spreadsheet_id=self.spreadsheet_id,
            sheet_prefix=self.sheet_prefix,
            ref=new_range,
            values=values,
        )

    def with_value(self, address: Address, value: Any) -> "CellApiMixin":
        return proxy_val(
            value,
            GSheetRef(
                service=self.service,
                spreadsheet_id=self.spreadsheet_id,
                sheet_prefix=self.sheet_prefix,
                ref=address,
                values=value,
            ),
        )

    def address(self) -> str:
        if (
            self.range.min_col != self.range.max_col
            or self.range.min_row != self.range.max_row
        ):
            a1 = self.range.to_a1()
        else:
            a1 = self.range.origin().to_a1()

        if self.sheet_prefix:
            a1 = self.sheet_prefix + a1

        return a1

    def set_attributes(self, attributes: dict[CellAttribute, str]) -> None:
        set_attributes(
            self.service, self.spreadsheet_id, self.range, self.sheet_prefix, attributes
        )

    def get_attribute(
        self,
        attribute: str,
        modifier: Callable[[str], str] | None = None,
        default_value: int | str | None = None,
    ) -> list[list[str | None]] | list[str | None] | str | None:
        # all_attributes = get_attributes(
        #     self.service, self.spreadsheet_id, self.range, self.sheet_prefix
        # )
        return None

    def clear(self) -> None:
        execute(
            self.service.spreadsheets()
            .values()
            .clear(spreadsheetId=self.spreadsheet_id, range=self.address())
        )

    def _write_value(self, col: int, row: int, value: Any) -> None:
        written = set_item(
            self.service,
            self.spreadsheet_id,
            Address(col, row, self.range.sheet),
            value,
            sheet_prefix=self.sheet_prefix,
        )

        col -= self.range.min_col
        row -= self.range.min_row

        if (
            self.range.min_col == self.range.max_col
            or self.range.min_row == self.range.max_row
        ):
            if self.range.min_row == self.range.max_row:
                written_1d = written[0]
            else:
                written_1d = [row[0] for row in written]
            offset = max(col, row)
            if len(self.values) < offset + len(written_1d):
                self.values += [None] * (offset + len(written_1d) - len(self.values))
            self.values[offset : offset + len(written_1d)] = written_1d
        else:
            for row_idx, row_written in enumerate(written):
                if len(self.values[row_idx + row]) < col + len(row_written):
                    self.values[row_idx + row] += [None] * (
                        col + len(row_written) - len(self.values[row_idx + row])
                    )
                self.values[row_idx + row][col : col + len(row_written)] = row_written

    def values_or_none(self, key: IntOrSlice) -> list[list] | list | Any:
        # Google sheets will drop trailing empty cells, which means that sometimes our values
        # will be shorter than the range we're trying to access. We need to pad the values with
        # None to match the range.
        r = self.range

        # min/max in_range represent the min max in values we're trying to access. This is the
        # columns if we the range is 1 dimensional in the column direction, otherwise it's the
        # rows.
        if r.min_row == r.max_row:
            min_in_range = r.min_col
            max_in_range = r.max_col
        else:
            min_in_range = r.min_row
            max_in_range = r.max_row
        if isinstance(key, slice):
            if max_in_range < 0:
                start = key.start or 0
                stop = key.stop
                if start < 0 or stop is None or stop < 0:
                    raise IndexError(
                        "Negative slices are not supported on infinite ranges"
                    )
            else:
                start, stop, _ = key.indices(max_in_range - min_in_range + 1)
            res = [self.values_or_none(i) for i in range(start, stop)]
            if list_like(res[0]) and len(res) == 1:
                res = res[0]
            return res
        else:
            if key < 0:
                if max_in_range < 0:
                    raise IndexError(
                        "Negative indices are not supported on infinite ranges"
                    )
                key += max_in_range
            if key >= len(self.values):
                if list_like(self.values[0]):
                    return [None] * len(self.values[0])
                return None
            return self.values[key]

    def getitem(self, key: IntOrSlice) -> Any:
        if self.range.dimensions() <= 1:
            col = 0 if self.range.min_col == self.range.max_col else key
            row = 0 if self.range.min_row == self.range.max_row else key
            min_col, max_col = slice_or_int_to_range(
                col, self.range.min_col, self.range.max_col + 1
            )
            min_row, max_row = slice_or_int_to_range(
                row, self.range.min_row, self.range.max_row + 1
            )
            if (
                min_col == max_col
                and min_row == max_row
                and isinstance(key, int | np.integer)
            ):
                return self.values[min_row][min_col]

            return self.with_range(
                Range(min_col, max_col, min_row, max_row, self.range.sheet),
                self.values[key],
            )
        else:
            min_row, max_row = slice_or_int_to_range(
                key, self.range.min_row, self.range.max_row + 1
            )
            return self.with_range(
                Range(
                    self.range.min_col,
                    self.range.max_col,
                    min_row,
                    max_row,
                    self.range.sheet,
                ),
                self.values_or_none(key),
            )

    def row_entry(self, idx: int) -> Any:
        try:
            if not list_like(self.values[0]):
                address = Address(self.range.min_col, self.range.min_row + idx, 0)
                return self.with_value(address, self.values[idx])

            if len(self.values) == 1:
                return self.values[0][idx]
            else:
                return self.values[idx][0]
        except IndexError:
            r = self.range
            horizontal = r.min_row == r.max_row
            infinite = (r.max_col if horizontal else r.max_row) == -1
            if infinite:
                address = Address(
                    self.range.min_col + (idx if horizontal else 0),
                    self.range.min_row + (idx if not horizontal else 0),
                    0,
                )
                return self.with_value(address, None)
            raise

    def current_max_col_row(self) -> tuple[int, int]:
        return self.range.max_col, self.range.max_row

    def insert_delete_sheet_row_col(
        self,
        dimension: Dimension,
        transform: SheetTransform,
        index: int,
        amount: int,
        data: Any,
    ) -> None:
        sheet_id = sheet_prefix_to_id(
            self.service, self.spreadsheet_id, self.sheet_prefix
        )
        exec_insert_delete_sheet_row_col(
            self.service,
            self.spreadsheet_id,
            sheet_id,
            dimension,
            transform,
            index,
            amount,
        )

    def to_gsheet(self, type: str, sheet_names_to_id: Mapping[str, int]) -> dict:
        if type == "GridRange":
            return {
                "sheetId": sheet_names_to_id[self.sheet_prefix[:-1]],
                "startRowIndex": self.range.min_row,
                "endRowIndex": self.range.max_row + 1,
                "startColumnIndex": self.range.min_col,
                "endColumnIndex": self.range.max_col + 1,
            }
        elif type == "GridCoordinate":
            return {
                "sheetId": sheet_names_to_id[self.sheet_prefix[:-1]],
                "rowIndex": self.range.min_row,
                "columnIndex": self.range.min_col,
            }
        else:
            raise ValueError(f"Unknown type {type}")


class GSheetNameRegistry(Mapping):
    _spreadsheet_id: str
    _gsheet_service: Resource
    _cache: dict[str, int]

    def __init__(self, gsheet_service: Resource, spreadsheet_id: str):
        self._gsheet_service = gsheet_service
        self._spreadsheet_id = spreadsheet_id
        self._cache = {}

    def __getitem__(self, item: str) -> int:
        if not self._cache:
            self._load_sheets()
        return self._cache[item]

    def __len__(self) -> int:
        if not self._cache:
            self._load_sheets()
        return len(self._cache)

    def __iter__(self) -> Iterator[str]:
        if not self._cache:
            self._load_sheets()
        return iter(self._cache)

    def clear_cache(self) -> None:
        self._cache = {}

    def _load_sheets(self) -> None:
        sheets = sheets_for_spreadsheet(self._gsheet_service, self._spreadsheet_id)
        self.clear_cache()
        for idx, sheet in enumerate(sheets):
            properties = sheet["properties"]
            sheet_id = properties["sheetId"]
            self._cache[properties["title"]] = sheet_id
            if idx == 0:
                self._cache[""] = sheet_id


def replace_refs(
    value: Any,
    mappable_types: dict,
    sheet_names: Mapping[str, int],
    path: Sequence[str] = (),
) -> Any:
    if (
        hasattr(value, "ref")
        and isinstance(value.ref, GSheetRef)
        and (typ := mappable_types.get(".".join(path))) is not None
    ):
        return value.ref.to_gsheet(typ, sheet_names)
    if isinstance(value, list):
        return [replace_refs(v, mappable_types, sheet_names, path) for v in value]
    elif isinstance(value, dict):
        return {
            k: replace_refs(v, mappable_types, sheet_names, (*path, k))
            for k, v in value.items()
        }
    else:
        return value


class SheetsAPIError(Exception):
    pass


class GSheetNamedRanges:
    service: Resource | None
    spreadsheet_id: str

    def __init__(self) -> None:
        self.service = None
        self.spreadsheet_id = ""

    def setup(self, service: Resource, spreadsheet_id: str) -> None:
        self.service = service
        self.spreadsheet_id = spreadsheet_id

    def __getitem__(self, name: str) -> Any:
        if not self.service:
            raise SheetsAPIError(
                "Named ranges only available in Google Sheets context."
            )
        return get_named_range(self.service, self.spreadsheet_id, name)

    def __setitem__(self, name: str, value: Any) -> None:
        if not self.service:
            raise SheetsAPIError(
                "Named ranges only available in Google Sheets context."
            )
        set_item(self.service, self.spreadsheet_id, name, value)
