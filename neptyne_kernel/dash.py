import asyncio
import base64
import bisect
import copy
import csv
import dataclasses
import datetime
import decimal
import inspect
import io
import json
import math
import os
import pickle
import re
import sys
import threading
import time
import traceback
import types
import uuid
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, fields, replace
from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
)
from unittest.mock import patch
from urllib.parse import urlparse

import dateutil.parser
import dateutil.tz
import ipykernel
import numpy as np
import pandas as pd
import plotly.io as pio
import requests
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError
from ipykernel.inprocess.ipkernel import InProcessKernel
from ipykernel.kernelbase import Kernel
from IPython import InteractiveShell
from IPython.core.interactiveshell import ExecutionResult
from PIL import Image
from plotly.basedatatypes import BaseFigure
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import gsheets_api, sheet_context, streamlit_server
from .api_ref import assure_compatible_shape
from .bokeh_formatter import maybe_format_bokeh
from .cell_address import Address, CoordAddr, Range
from .cell_api import CellApiMixin
from .cell_copier import pre_copy_adjust
from .cell_execution_graph import CellExecutionGraph
from .cell_range import CellRange, shape
from .dash_traceback import (
    AddressTuple,
    CoordinateTuple,
    DashAutoFormattedTB,
    DashSyntaxTB,
)
from .download import Download
from .expression_compiler import (
    DEFAULT_GRID_SIZE,
    CompileResult,
    compile_expression,
    compile_mime_typed_expression,
    is_cell,
    is_cell_formula,
    parse_widget_code,
    process_sheet_rename,
    process_sheet_transformation,
    rename_variable_in_code,
    replace_n_with_a1,
    replace_n_with_a1_match,
    tokenize_with_ranges,
    try_parse_capitalized_range,
)
from .get_ipython_mockable import get_ipython_mockable
from .gsheets_api import (
    GSheetNamedRanges,
    GSheetNameRegistry,
    SheetsAPIError,
    execute,
    replace_refs,
)
from .insert_delete_helper import add_delete_cells_helper
from .json_tools import json_clean
from .kernel_runtime import get_kernel, send_sync_request
from .linter import TyneCachingCompiler
from .mime_handling import (
    as_json,
    datetime_bundle,
    encode_for_gsheets,
    encode_pickled_b64,
    encode_pil_image,
    encode_plotly_figure,
    gsheet_spreadsheet_error_from_python_exception,
    maybe_format_common_values,
    output_to_value,
)
from .mime_types import (
    BYTES_MIME_KEY,
    CSV_MIME_KEY,
    DECIMAL_MIME_KEY,
    GSHEET_ERROR_KEY,
    JSON_MIME_KEY,
    NUMBER_MIME_KEY,
    SVG_MIME_KEY,
    TEXT_MIME_KEY,
    WELL_KNOWN_TEXT_KEY,
)
from .mutex_manager import MutexManager
from .neptyne_protocol import (
    CellAttribute,
    CellAttributesUpdate,
    CellChange,
    CopyCellsContent,
    Dimension,
    DownloadRequest,
    DragRowColumnContent,
    InsertDeleteContent,
    InsertDeleteReplyCellType,
    KernelInitState,
    MessageTypes,
    MIMETypes,
    RerunCellsContent,
    RunCellsContent,
    SheetAttributeUpdate,
    SheetAutofillContent,
    SheetTransform,
    SheetUpdateContent,
    StreamlitAppConfig,
    TickReplyContent,
    TracebackFrame,
    WidgetDefinition,
    WidgetParamType,
)
from .ops import ClearOp, ExecOp
from .pandas_unrolling import dataframe_to_grid
from .pip import neptyne_pip_install
from .primitives import Empty, proxy_val, unproxy_val
from .proxied_apis import get_api_error_service, start_api_proxying
from .renderers import InlineWrapper, WithSourceMixin
from .session_info import NeptyneSessionInfo
from .sheet_api import NeptyneSheetCollection
from .spreadsheet_datetime import SpreadsheetDateTimeBase
from .spreadsheet_error import (
    PYTHON_ERROR,
    SheetDoesNotExist,
    SpreadsheetError,
)
from .transformation import (
    Transformation,
    insert_delete_content_to_sheet_transform,
)
from .tyne_model.cell import (
    CODEPANEL_CELL_ID,
    CellMetadata,
    SheetCell,
    represents_simple_value,
)
from .tyne_model.dash_graph import DashGraph
from .tyne_model.jupyter_notebook import Output, OutputType
from .tyne_model.kernel_init_data import (
    InitPhase1Payload,
    InitPhase2Payload,
)
from .tyne_model.save_message import V1DashSaveMessage, json_encode
from .tyne_model.sheet import TyneSheets
from .tyne_model.table_for_ai import (
    TableForAI,
    ai_tables_for_sheet,
)
from .upgrade_model import upgrade_model
from .widgets.base_widget import (
    BaseWidget,
    decode_callable,
    validate_widget_params,
)
from .widgets.input_widgets import Autocomplete
from .widgets.output_widgets import (
    DEFAULT_OUTPUT_WIDGET_HEIGHT,
    DEFAULT_OUTPUT_WIDGET_WIDTH,
    maybe_render_widget,
)
from .widgets.register_widget import (
    get_widget_param_type,
    widget_name_from_code,
    widget_registry,
)

IPYKERNEL_MAJOR_VERSION = int(ipykernel.__version__.split(".")[0])

try:
    from .formulas.sheets import SvgImage
except ImportError:
    SvgImage = None  # type: ignore

try:
    from shapely.geometry.base import BaseGeometry as ShapelyBaseGeometry
    from shapely.geometry.point import Point as ShapelyPoint

    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False

try:
    from matplotlib_inline import backend_inline as matplotlib_backend_inline
except ImportError:
    matplotlib_backend_inline = None

if TYPE_CHECKING:
    from .sheet_api import NeptyneSheet

MAX_COL = 700
MAX_COL_EXCEPTION = IndexError(f"Neptyne currently only supports {MAX_COL} columns")

CELL_ATTRIBUTES_TO_CLEAR_ON_VALUE_CHANGE = {
    CellAttribute.WIDGET.value,
    CellAttribute.WIDGET_NAME.value,
}

CELL_ATTRIBUTES_TO_CLEAR_ON_CLEAR = {
    CellAttribute.LINK.value,
    CellAttribute.NOTE.value,
    CellAttribute.RENDER_WIDTH.value,
    CellAttribute.RENDER_HEIGHT.value,
    CellAttribute.EXECUTION_POLICY.value,
    CellAttribute.SOURCE.value,
}.union(CELL_ATTRIBUTES_TO_CLEAR_ON_VALUE_CHANGE)

MAX_CASCADE_COUNT = 10

UPDATE_CELL_METADATA_CHANGE = "update_cell_metadata_change"
CLEAR_CELL_METADATA_CHANGE = "clear_cell_metadata_change"


pio.templates.default = "plotly"


def get_parent_shim(self: Kernel, channel: str | None = None) -> dict:
    return self._parent_header


def check_max_col_for_address(item: Address) -> None:
    if item.column >= MAX_COL:
        raise MAX_COL_EXCEPTION


def check_max_col_for_range(item: Range) -> None:
    if item.min_col >= MAX_COL or item.max_col >= MAX_COL:
        raise MAX_COL_EXCEPTION


def hash_function(value: types.FunctionType | types.CodeType) -> int:
    code_obj: types.CodeType = value.__code__ if inspect.isfunction(value) else value  # type: ignore
    constants = tuple(
        hash_function(co) if inspect.iscode(co) else co for co in code_obj.co_consts
    )
    stable = (
        code_obj.co_code,
        code_obj.co_name,
    )
    return hash(stable) ^ hash(str(constants))


@dataclass
class APIRequest:
    body: bytes
    headers: dict[str, str]
    query: dict[str, str]


@dataclass
class TickCell:
    address: Address
    execution_policy: float

    def next_execution_time(self, now: float) -> float:
        return now + self.execution_policy


@dataclass
class Cron:
    expression: str
    schedule: str
    timezone: datetime.tzinfo
    alert_email: str | None = None

    def next_execution_time_datetime(self, now: float) -> datetime.datetime:
        from croniter import croniter

        cron = croniter(
            self.schedule, datetime.datetime.fromtimestamp(now, self.timezone)
        )
        return cron.get_next(datetime.datetime)

    def next_execution_time(self, now: float) -> float:
        return self.next_execution_time_datetime(now).timestamp()

    def to_dict(self) -> dict:
        timezone_name = self.timezone.key  # type: ignore
        return {**dataclasses.asdict(self), "timezone": timezone_name}


@dataclass(order=True)
class TickItem:
    next_execution_time: float
    item: TickCell | Cron = field(compare=False)

    def next(self, now: float) -> "TickItem":
        return replace(self, next_execution_time=self.item.next_execution_time(now))


class TickCellQueue:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.queue: list[TickItem] = []
        self.by_address: dict[Address, TickItem] = {}

    def pop_ready(self) -> list[TickItem]:
        with self.lock:
            ready = []
            now = time.time()
            while self.queue and self.queue[0].next_execution_time <= now:
                ready.append(self.queue.pop(0))
            for item in ready:
                if isinstance(item.item, TickCell):
                    if item.item.execution_policy > 0:
                        item_next = item.next(now)
                        self.by_address[item.item.address] = item_next
                        bisect.insort(self.queue, item_next)
                else:
                    item_next = item.next(now)
                    bisect.insort(self.queue, item_next)
            return ready

    def put(
        self, address: Address, execution_policy: int, next_execution_time: float
    ) -> None:
        with self.lock:
            self._put(address, execution_policy, next_execution_time)

    def initialize(self, cell_metas: dict[Address, "CellMetadata"]) -> None:
        with self.lock:
            self.queue = []
            self.by_address = {}
            for address, metadata in cell_metas.items():
                if metadata.execution_policy > 0:
                    self._put(
                        address, metadata.execution_policy, metadata.next_execution_time
                    )

    def _put(
        self, address: Address, execution_policy: int, next_execution_time: float
    ) -> None:
        assert execution_policy > 0
        if (item := self.by_address.get(address)) is not None:
            assert isinstance(item.item, TickCell)
            item.item.execution_policy = execution_policy
            item.next_execution_time = next_execution_time
        else:
            item = TickItem(execution_policy, TickCell(address, execution_policy))
            self.by_address[address] = item
            self.queue.append(item)
        self.queue.sort()

    def drop(self, address: Address) -> None:
        with self.lock:
            nonempty = bool(self.queue)
            if address in self.by_address:
                del self.by_address[address]
                self.queue = [
                    item
                    for item in self.queue
                    if isinstance(item.item, Cron) or item.item.address != address
                ]
            if nonempty and not self.queue:
                # a tick reply with time=0 tells the server we no longer have ticking cells
                self.queue.append(TickItem(0, TickCell(address, 0)))

    def clear_crons(self) -> None:
        with self.lock:
            self.queue = [
                item for item in self.queue if isinstance(item.item, TickCell)
            ]

    def put_cron(self, cron: Cron) -> None:
        now = time.time()
        with self.lock:
            item = TickItem(0, cron).next(now)
            bisect.insort(self.queue, item)

    def next_tick(self) -> float:
        with self.lock:
            if not self.queue:
                return 0
            return self.queue[0].next_execution_time

    def get_crons(self) -> list[Cron]:
        with self.lock:
            return [item.item for item in self.queue if isinstance(item.item, Cron)]


class RequestCancelledError(Exception):
    pass


def client_callable(f: Callable) -> Callable:
    # We don't use @wraps here to make is_callable_from_client work:
    def wrapper(*args: Any, **kwargs: Any) -> None:
        res = f(*args, **kwargs)
        # Check whether the dash is set to silent - we can't rely on mocking here since this happens
        # when dash is imported, before the mock is set up:
        if res is not None and not (isinstance(args[0], Dash) and args[0].silent):
            k = get_kernel()
            k.send_response(
                k.iopub_socket,
                MessageTypes.RPC_RESULT.value,
                {"method": f.__name__, "result": res},
            )
        return res

    return wrapper


@dataclass
class OnValueChangeRule:
    range: CellRange
    callback: Callable[[Any], None]
    apply_on_full_range: bool


APIFunction = Callable[[APIRequest], Any]


class DashRecursionError(Exception):
    pass


class Dash:
    _instance: "Dash | None" = None

    in_post_execute_hook = False

    cells: dict[int, dict[Address, Any]]
    cell_meta: dict[Address, CellMetadata]
    graph: DashGraph
    sheets: NeptyneSheetCollection
    _named_ranges: GSheetNamedRanges | None

    dirty_cells: set[Address]
    resized_sheets: set[int]

    kernel: Kernel
    function_hashes: dict[str, int] | None
    on_value_change_rules: list[OnValueChangeRule]
    scheduled_undo: dict[str, Any] | None
    initialized: bool
    api_functions: dict[str | None, APIFunction]
    in_gs_mode: bool
    _gsheets_spreadsheet_id: str
    time_zone: str
    streamlit_base_url_path: str
    cur_streamlit_info: tuple[Callable, StreamlitAppConfig] | None
    prev_streamlit_info: tuple[Callable, StreamlitAppConfig] | None
    _gsheet_service: Resource | None

    _cell_execution_stack: list[Address]
    local_repl_mode: bool = False

    def __init__(self, silent: bool = False) -> None:
        if Dash._instance is not None:
            raise RuntimeError("Dash is a singleton")

        Dash._instance = self

        self.cells = defaultdict(dict)
        self.cell_meta = defaultdict()
        self.graph = DashGraph()
        self.sheets = NeptyneSheetCollection(self)
        self.silent = silent

        self.dirty_cell_flush_lock = threading.Lock()
        self.dirty_cells = set()
        self.resized_sheets = set()
        self.scheduled_undo = None

        self.side_effect_cells: set[Address] = set()
        ip = get_ipython_mockable()
        self.shell = ip
        self.function_hashes = None
        self.on_value_change_rules = []
        self.initialized = False
        self.api_functions = {}
        self.in_gs_mode = False
        self._gsheets_spreadsheet_id = ""
        self.time_zone = "UTC"
        self.api_token_override: str | None = None
        self._named_ranges = None
        self._gsheet_service = None
        self._gsheet_name_registry: GSheetNameRegistry | None = None

        self._cell_execution_stack = []
        self._pending_display_msg: dict[str, Any] | None = None
        self._mutex_manager = MutexManager()

        if ip is not None:
            parent = ip if isinstance(ip, InteractiveShell) else None
            if IPYKERNEL_MAJOR_VERSION >= 6:
                ip.InteractiveTB = DashAutoFormattedTB(
                    mode="Plain",
                    color_scheme="LightBG",
                    tb_offset=ip.InteractiveTB.tb_offset,
                    check_cache=ip.InteractiveTB.check_cache,
                    debugger_cls=ip.debugger_cls,
                    parent=parent,
                )
                ip.InteractiveTB.set_mode(mode="Context")
                ip.SyntaxTB = DashSyntaxTB(color_scheme="LightBG", parent=parent)

            try:
                import matplotlib  # noqa: F401

                ip.run_line_magic("matplotlib", "inline")
            except ImportError:
                pass

            ip.events.register("pre_execute", self.pre_execute)
            ip.events.register("post_execute", self.post_execute)
            ip.events.register("post_run_cell", self.post_run_cell)

            if IPYKERNEL_MAJOR_VERSION >= 6:
                ip.__class__.compiler_class = TyneCachingCompiler
                ip.compile = ip.compiler_class()

            self.kernel = ip.kernel
            if not hasattr(self.kernel, "get_parent"):
                setattr(
                    self.kernel,
                    "get_parent",
                    get_parent_shim.__get__(self.kernel, Kernel),
                )

            self.patch_do_complete(self.kernel)

        self.message_publisher = Thread(
            name="msg-publisher", daemon=True, target=self.flush_loop
        )
        self.message_publisher.start()

        self.tick_cell_queue = TickCellQueue()
        self.tick_event = threading.Event()
        self.tick_thread = Thread(
            name="tick-thread", daemon=True, target=self.tick_loop
        )
        self.tick_thread.start()

        self.cur_streamlit_info = None
        self.prev_streamlit_info = None
        self.local_repl_mode = bool(os.getenv("NEPTYNE_LOCAL_REPL"))
        self.api_key = ""
        self.api_host = ""

    @classmethod
    def instance(cls) -> "Dash":
        if not cls._instance:
            cls._instance = cls()
        return cls._instance

    @property
    def gsheets_spreadsheet_id(self) -> str:
        if override := sheet_context.gsheet_id_override.get():
            return override
        return self._gsheets_spreadsheet_id

    @gsheets_spreadsheet_id.setter
    def gsheets_spreadsheet_id(self, value: str) -> None:
        self._gsheets_spreadsheet_id = value

    def get_raw_code(self, address: Address) -> str:
        meta = self.cell_meta.get(address)
        if meta:
            return meta.raw_code
        if address in self.graph.calculated_by:
            return ""
        cell = self.cells[address.sheet].get(address)
        if cell is None:
            return ""
        return str(cell)

    def has_formula(self, address: Address) -> bool:
        meta = self.cell_meta.get(address)
        if not meta or not meta.raw_code:
            return False
        return is_cell_formula(meta.raw_code)

    def set_raw_code(self, address: Address, code: str) -> None:
        meta = (
            self.get_or_create_cell_meta(address)
            if is_cell_formula(code)
            else self.cell_meta.get(address)
        )
        if meta:
            meta.raw_code = code
        else:
            cell = self.cells[address.sheet].get(address)
            if (not cell and not code) or (cell and str(cell) == code):
                return
            self.get_or_create_cell_meta(address).raw_code = code

    def exec_header(self) -> None:
        """Execute the code in the message header.

        Mainly useful to bypass iPython's code transformation. Removes all the iPython magic.
        """
        code = self.shell.parent_header["header"]["code"]
        exec(
            code,
            self.shell.user_global_ns,
            self.shell.user_ns,
        )

    def track_function_changes(self) -> None:
        if not self.kernel.get_parent("shell").get("header", {}).get("cellId"):
            return
        first = self.function_hashes is None
        if first:
            self.function_hashes = {}
        changes = []
        for name, value in self.shell.user_global_ns.items():
            if inspect.isfunction(value) and value.__module__ == "__main__":
                hsh = hash_function(value)
                assert self.function_hashes is not None
                if self.function_hashes.get(name) != hsh:
                    self.function_hashes[name] = hsh
                    if not first:
                        changes.append(name)
        if changes:
            self.process_function_changes(changes)

    def post_run_cell(self, result: ExecutionResult) -> None:
        client_traceback = []
        if exception := (result.error_in_exec or result.error_before_exec):
            if isinstance(exception, SyntaxError):
                client_traceback = [
                    TracebackFrame(
                        exec_count=result.execution_count,
                        current_cell=True,
                        lineno=exception.lineno,  # type: ignore
                        line=exception.text or "",
                    ).to_dict()
                ]
            else:
                tb = traceback.extract_tb(exception.__traceback__)
                this_exec_count = None
                for frame in tb[1:]:
                    if frame.line == "N_.flush_side_effects()":
                        continue
                    try:
                        exec_count = self.shell.compile._filename_map.get(
                            frame.filename
                        )
                    except AttributeError:
                        exec_count = 1
                    if this_exec_count is None and exec_count is not None:
                        this_exec_count = exec_count
                    client_traceback.append(
                        TracebackFrame(
                            exec_count=exec_count,
                            current_cell=exec_count is not None
                            and this_exec_count == exec_count,
                            lineno=frame.lineno or -1,
                            line=frame.line or "",
                        ).to_dict()
                    )
            if client_traceback:
                self.reply_to_client(
                    MessageTypes.TRACEBACK, {"traceback": client_traceback}
                )

    @classmethod
    def is_callable_from_client(cls, name: str) -> bool:
        if hasattr(cls, name):
            return getattr(cls, name).__qualname__.startswith(client_callable.__name__)
        return False

    def resolve_max_col_row(self, range: Range) -> tuple[int, int]:
        on_empty_value = -1

        if range.max_col >= 0 and range.max_row >= 0:
            return range.max_col, range.max_row

        cells = self.cells[range.sheet]
        if range.max_row >= 0:
            return (
                max(
                    (
                        k.column
                        for k in cells
                        if range.max_row >= k.row >= range.min_row
                    ),
                    default=on_empty_value,
                ),
                range.max_row,
            )
        elif range.max_col >= 0:
            return (
                range.max_col,
                max(
                    (
                        k.row
                        for k in cells
                        if range.max_col >= k.column >= range.min_col
                    ),
                    default=on_empty_value,
                ),
            )

        # [-1,-1]
        return (
            max((a.column for a in cells), default=on_empty_value),
            max((a.row for a in cells), default=on_empty_value),
        )

    def all_keys(self) -> set[Address]:
        res: set[Address] = set()
        for sheet in self.cells.values():
            res.update(sheet.keys())
        graph = self.graph
        res.update(self.cell_meta)
        res.update(graph.depends_on)
        res.update(graph.feeds_into)
        res.update(graph.calculated_by)
        return res

    def from_coordinate(
        self, coord: CoordinateTuple | Address | Range
    ) -> Address | Range:
        if isinstance(coord, Address) or isinstance(coord, Range):
            return coord

        sheet_id: int | str = coord[-1]
        if sheet_id == 0 and (override := sheet_context.sheet_name_override.get()):
            sheet_id = override
        if isinstance(sheet_id, str):
            if self.in_gs_mode:
                sheet_id = gsheets_api.sheet_id_for_title(
                    self.gsheet_service, sheet_id, self.gsheets_spreadsheet_id
                )
            else:
                if sheet_id not in self.sheets:
                    raise SheetDoesNotExist(sheet_id)
                sheet_id = self.sheets[sheet_id].sheet_id

            coord = (*coord[:-1], sheet_id)  # type: ignore

        if len(coord) == 3:
            return Address(*coord)
        elif len(coord) == 5:
            return Range(*coord)
        raise TypeError(
            "Invalid coordinate type: expected Address (3-tuple) or Range (5-tuple)"
        )

    def drag_row_column(self, drag_row_column: dict) -> None:
        content = DragRowColumnContent.from_dict(drag_row_column)
        inverse_drag_content = self.drag_row_column_internal(
            content.dimension,
            int(content.from_index),
            int(content.to_index),
            int(content.amount),
            int(content.sheet_id),
        )
        self.reply_to_client(
            MessageTypes.DRAG_ROW_COLUMN,
            drag_row_column,
            undo_msg=self.undo_msg(
                MessageTypes.DRAG_ROW_COLUMN, inverse_drag_content.to_dict()
            ),
        )

    def drag_row_column_internal(
        self,
        dimension: Dimension,
        from_index: int,
        to_index: int,
        amount: int,
        sheet_id: int,
    ) -> DragRowColumnContent:
        inverse_drag_content = DragRowColumnContent(
            amount=amount,
            dimension=dimension,
            to_index=from_index,
            from_index=to_index,
            sheet_id=sheet_id,
        )
        delete_transformation = Transformation(
            dimension, SheetTransform.DELETE, from_index, amount, sheet_id
        )
        _, cells_to_populate = self.compute_inverse_insert_delete_transformation(
            delete_transformation
        )
        self.add_delete_cells_internal(delete_transformation)

        shift_amount = to_index - from_index

        def transform_address(address: Address) -> Address:
            if sheet_id != address.sheet:
                return address
            if dimension == Dimension.ROW:
                if from_index <= address.row < from_index + amount:
                    return replace(address, row=address.row + shift_amount)
                elif from_index + amount <= address.row < to_index:
                    return replace(address, row=address.row - amount)
            else:
                if from_index <= address.column < from_index + amount:
                    return replace(address, column=address.column + shift_amount)
                elif from_index + amount <= address.column < to_index:
                    return replace(address, column=address.column - amount)
            return address

        for cell in cells_to_populate:
            # NOTE: We clear the graph and don't update the code
            #   There are a bunch of challenging cases there with no clear answer when splitting cell ranges.
            cell["cell_id"] = transform_address(
                Address.from_coord(cell["cell_id"])
            ).to_coord()
            if calculated_by := cell.get("calculated_by"):
                cell["calculated_by"] = transform_address(
                    Address.from_coord(calculated_by)
                ).to_coord()
            cell["feeds_into"] = []
            cell["depends_on"] = []

        if to_index > from_index:
            to_index -= amount

        insert_transform = Transformation(
            dimension, SheetTransform.INSERT_BEFORE, to_index, amount, sheet_id
        )

        self.add_delete_cells_internal(insert_transform, cells_to_populate)

        return inverse_drag_content

    def add_delete_cells(self, insert_delete_content: dict) -> None:
        insert_delete = InsertDeleteContent.from_dict(insert_delete_content)
        transformation, cells_to_populate = insert_delete_content_to_sheet_transform(
            insert_delete
        )

        self.add_delete_cells_internal(
            transformation, cells_to_populate, send_undo=True
        )

    def add_delete_cells_internal(
        self,
        transformation: Transformation,
        cells_to_populate: list[dict] | None = None,
        send_undo: bool = False,
    ) -> None:
        return add_delete_cells_helper(
            self, transformation, cells_to_populate, send_undo
        )

    def undo_msg(self, msg_type: MessageTypes, payload: dict) -> dict[str, Any]:
        if not self.kernel.session:
            return {}
        msg = self.kernel.session.msg(msg_type.value, payload)
        parent_header = self.shell.parent_header["header"]
        msg["header"]["msg_id"] = parent_header["msg_id"]
        msg["header"]["operation_id"] = parent_header.get(
            "operation_id", str(uuid.uuid4())
        )
        return msg

    def widget_triggered(self, cell_id: str, value: Any) -> None:
        address = Address.from_a1_or_str(cell_id)
        widget = self[address]
        if isinstance(widget, BaseWidget):
            try:
                self.evaluate_and_trigger_widget(address, value)
            except Exception as exception:
                self.shell.parent_header["header"]["cellId"] = None
                self.reply_to_client(
                    "error",
                    {
                        "ename": exception.__class__.__name__,
                        "evalue": str(exception),
                        "traceback": self.shell.SyntaxTB.structured_traceback(
                            *sys.exc_info(), tb_offset=0
                        ),
                    },
                )

    def evaluate_and_trigger_widget(
        self, address: Address, value: Any, do_copy_from_code: bool = True
    ) -> None:
        assert isinstance(self[address], BaseWidget)

        if self.has_formula(address):
            self.run_cells_with_cascade(cell_ids=[address])
        widget = self[address]
        assert isinstance(widget, BaseWidget)
        widget.trigger(value, widget._get_event(), do_copy_from_code)

    def load_values(self, sheets: TyneSheets) -> None:
        upgrade_model(sheets)
        for cell_id, cell in sheets.all_cells():
            if isinstance(cell.output, Output):
                value = output_to_value(cell.output.data)
                output_needs_meta = not represents_simple_value(cell.output)
            else:
                value = cell.output
                output_needs_meta = False
            if value is not None:
                self.cells[cell_id.sheet][cell_id] = value

            if cell.calculated_by:
                self.graph.calculated_by[cell_id] = cell.calculated_by

            if cell.feeds_into:
                self.graph.feeds_into[cell_id] = cell.feeds_into

            if cell.depends_on:
                self.graph.depends_on[cell_id] = cell.depends_on

            if (
                cell.attributes
                or cell.compiled_code
                or cell.mime_type
                or cell.execution_policy != -1
                or is_cell_formula(cell.raw_code)
                or output_needs_meta
            ):
                self.cell_meta[cell_id] = CellMetadata(
                    attributes=cell.attributes or {},
                    raw_code=cell.raw_code,
                    compiled_code=cell.compiled_code,
                    mime_type=cell.mime_type,
                    execution_policy=cell.execution_policy,
                    next_execution_time=cell.next_execution_time,
                    output=(cell.output if output_needs_meta else value),
                )
        if not self.in_gs_mode:
            self.sheets._load_serializable_sheets(sheets.sheets.values())
        self.tick_cell_queue.initialize(self.cell_meta)

    def broadcast_init_stage(self, state: KernelInitState) -> None:
        parent = self.kernel.get_parent("shell")
        parent["header"]["init_phase"] = state.value
        if self.kernel.session:
            self.kernel.session.send(
                self.kernel.iopub_socket,
                "status",
                {"execution_state": "busy"},
                parent=parent,
                ident=f"kernel.{self.kernel.ident}.status".encode(),
            )

    def pip_install(self, requirements_txt: str, silent: bool = False) -> None:
        neptyne_pip_install(requirements_txt, silent=silent)

    def write_streamlit_config(self) -> None:
        os.makedirs(".streamlit", exist_ok=True)
        with open(".streamlit/config.toml", "w") as f:
            for line in (
                "[server]",
                f'baseUrlPath = "{self.streamlit_base_url_path}"',
                'folderWatchBlacklist = ["**/neptyne_kernel"]',
                "enableXsrfProtection = false",
                "enableCORS = false",
                "headless = true",
                "[theme]",
                'primaryColor = "#0F9D58"',
                'backgroundColor = "#FFFFFF"',
                'secondaryBackgroundColor = "#F4F4F4"',
                'textColor = "#5F6368"',
                'font = "sans serif"',
            ):
                f.write(line + "\n")

    def initialize_phase_1(self, payload: bytes) -> None:
        """Phase 1 of initialization occurs before user code runs"""
        if self.initialized:
            return

        init_payload = InitPhase1Payload.from_bytes(payload)
        self.initialize_phase_1_decoded(init_payload)

    def initialize_phase_1_decoded(self, init_payload: InitPhase1Payload) -> None:
        self.in_gs_mode = init_payload.in_gs_mode
        self.gsheets_spreadsheet_id = init_payload.gsheets_sheet_id
        if self.gsheets_spreadsheet_id:
            self.named_ranges.setup(self.gsheet_service, self.gsheets_spreadsheet_id)
        self.time_zone = init_payload.time_zone
        self.set_env(init_payload.env)

        if init_payload.streamlit_base_url_path:
            self.streamlit_base_url_path = init_payload.streamlit_base_url_path

        if init_payload.requirements:
            self.broadcast_init_stage(KernelInitState.INSTALLING_REQUIREMENTS)
            self.pip_install(init_payload.requirements, silent=True)

        self.broadcast_init_stage(KernelInitState.RUN_CODE_PANEL)
        if not self.in_gs_mode:
            for sheet in init_payload.sheets:
                self.sheets._register_sheet(sheet.id, sheet.name)

    def initialize_phase_2(self, payload: bytes) -> None:
        init_payload = InitPhase2Payload.from_bytes(payload)
        self.initialize_phase_2_decoded(init_payload)

    def initialize_phase_2_decoded(self, init_payload: InitPhase2Payload) -> None:
        self.broadcast_init_stage(KernelInitState.LOADING_SHEET_VALUES)
        self.load_values(init_payload.sheets)

        if init_payload.requires_recompile:
            self.recompile_everything()
        start_api_proxying()
        self.initialized = True

    def clear_cells_internal(self, cell_ids: Iterable[Address]) -> None:
        for cell_id in cell_ids:
            if cell_id in self.cells[cell_id.sheet]:
                del self.cells[cell_id.sheet][cell_id]
            if cell_id in self.cell_meta:
                self.set_raw_code(cell_id, "")
                self.cell_meta[cell_id].compiled_code = ""
                self.cell_meta[cell_id].mime_type = None
                self.cell_meta[cell_id].output = None

    def update_cell_meta_on_value_change(self, address: Address, value: Any) -> None:
        def output_for_data(data: dict[str, Any]) -> Output:
            return Output(
                data=data,
                output_type=OutputType.EXECUTE_RESULT,
                execution_count=None,
                metadata=None,
                name=None,
                text=None,
                ename=None,
                evalue=None,
                traceback=None,
            )

        if isinstance(value, SpreadsheetDateTimeBase):
            meta = self.get_or_create_cell_meta(address)
            if not meta.attributes.get(CellAttribute.NUMBER_FORMAT.value):
                meta.attributes[CellAttribute.NUMBER_FORMAT.value] = "date-MM/dd/yyyy"

        source_info = (
            json.dumps(value.source_info)
            if isinstance(value, WithSourceMixin)
            else None
        )
        self.update_cell_attribute(
            address,
            CellAttribute.SOURCE.value,
            source_info,
            True,
        )
        data: dict[str, Any] = {}

        if isinstance(value, decimal.Decimal):
            data = {DECIMAL_MIME_KEY: str(value)}
        elif isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
            data = {NUMBER_MIME_KEY: str(value)}
        elif isinstance(value, datetime.datetime | datetime.time | datetime.date):
            data = datetime_bundle(value)
            meta = self.get_or_create_cell_meta(address)
            if not meta.attributes.get(CellAttribute.NUMBER_FORMAT.value):
                if isinstance(value, datetime.date):
                    dt_format = "date-MM/dd/yyyy"
                else:
                    dt_format = "date-hh:mm:ss"
                meta.attributes[CellAttribute.NUMBER_FORMAT.value] = dt_format
        if data:
            self.get_or_create_cell_meta(address).output = output_for_data(
                data=data,
            )
            return

        if (
            isinstance(value, int | float | str)
            or value is Empty.MakeItSo
            or value is None
        ):
            if pmeta := self.cell_meta.get(address):
                if value is Empty.MakeItSo:
                    value = None
                pmeta.output = value
            return

        format_data = InteractiveShell.instance().displayhook.compute_format_data
        if render_inline := isinstance(value, InlineWrapper):
            value = value.value

        with self.use_cell_id(address):
            if isinstance(value, bytes):
                data = {BYTES_MIME_KEY: base64.b64encode(value).decode()}
            elif _HAS_SHAPELY and isinstance(value, ShapelyBaseGeometry):
                render_inline = True

                old_svg_method = value.svg

                def zoomed_svg(
                    obj: ShapelyBaseGeometry, scale_factor: float = 1.0, **kwargs: Any
                ) -> str:
                    svg = old_svg_method(scale_factor * 3.5, **kwargs)

                    def move_left(match: re.Match) -> str:
                        return f'cx="{float(match.group(1)) - 1.5}"'

                    svg = re.sub(r'cx="([-\d.]+)"', move_left, svg)

                    if _HAS_SHAPELY and isinstance(obj, ShapelyPoint):
                        transform = "scale(1,-1)"
                        svg += (
                            f'<text x="{obj.x - 0.6}" y="{-obj.y + 0.3}"'
                            f' font-size="{52 * scale_factor}px"'
                            f' transform="{transform}"'
                            f' fill="black">'
                            f"({obj.x:.1f},{obj.y:.1f})</text>"
                        )
                    return svg

                def expand_box(svg: str) -> str:
                    def expand_viewbox(match: re.Match) -> str:
                        min_x, min_y, width, height = map(float, match.groups())
                        new_width = width * 4
                        new_min_x = min_x - (new_width - width) / 2
                        return f'viewBox="{new_min_x} {min_y} {new_width} {height}"'

                    header, svg = svg.split(">", 1)
                    header = header.replace('width="100.0"', 'width="400"')
                    header = re.sub(
                        r'viewBox="([-\d.]+) ([-\d.]+) ([\d.]+) ([\d.]+)"',
                        expand_viewbox,
                        header,
                    )
                    svg = header + ">" + svg
                    return svg

                with patch.object(value.__class__, "svg", zoomed_svg):
                    data = {
                        WELL_KNOWN_TEXT_KEY: value.wkt,
                        SVG_MIME_KEY: expand_box(value._repr_svg_()),
                    }

            elif html := maybe_format_bokeh(value):
                data = {"text/html": html}
            else:
                data, _md_dict = format_data(value)

            if data is None:
                data = {}
            assert isinstance(data, dict)

            if render_inline:
                data["__neptyne_meta__"] = {"inline": True}
            im_types = (Image.Image, SvgImage) if SvgImage is not None else Image.Image
            if isinstance(value, im_types):
                width = value.width
                height = value.height
                if width > 800 or height > 600:
                    scale = min(800 / width, 600 / height)
                    width = int(width * scale)
                    height = int(height * scale)
                self.update_cell_attribute(
                    address,
                    CellAttribute.RENDER_HEIGHT.value,
                    height,
                    False,
                )
                self.update_cell_attribute(
                    address,
                    CellAttribute.RENDER_WIDTH.value,
                    width,
                    False,
                )

            if isinstance(value, BaseWidget):
                data["attributes"] = {
                    CellAttribute.WIDGET_NAME.value: value.__class__.__name__,
                }
            try:
                rendered_widget = maybe_render_widget(value, data)
            except Exception:
                data, _md_dict = format_data(self.stack_trace())
                rendered_widget = None
            if rendered_widget:
                self.update_cell_attribute(
                    address,
                    CellAttribute.RENDER_HEIGHT.value,
                    DEFAULT_OUTPUT_WIDGET_HEIGHT,
                    False,
                )
                self.update_cell_attribute(
                    address,
                    CellAttribute.RENDER_WIDTH.value,
                    DEFAULT_OUTPUT_WIDGET_WIDTH,
                    False,
                )
                widget_changes_dict, _md_dict = format_data(rendered_widget)
                data.update(widget_changes_dict)
                data = as_json(data)
            elif [*data.keys()] == ["text/plain"]:
                # It doesn't have its own mimetype nor is it a simple value, let's try pickling:
                try:
                    data = {
                        **data,
                        MIMETypes.APPLICATION_VND_POPO_V1_JSON.value: encode_pickled_b64(
                            value
                        ),
                    }
                except pickle.PickleError:
                    pass
        try:
            json_encode(data)
        except (TypeError, ValueError):
            data = json_clean(data)

        self.get_or_create_cell_meta(address).output = output_for_data(data)

    def update_cell(
        self,
        address: Address,
        value: Any,
    ) -> None:
        if isinstance(value, BaseWidget):
            errors = value.validate_all_fields()
            if errors:
                # TODO: Custom string format to make the error look nicer.
                value = PYTHON_ERROR.with_message(str(errors))

        self.update_cell_meta_on_value_change(address, value)
        if value is Empty.MakeItSo or value is None:
            if address in self.cells[address.sheet]:
                del self.cells[address.sheet][address]
        else:
            self.cells[address.sheet][address] = value
        self.notify_client_cells_have_changed(address)

    def flush_dirty_cells_now(self) -> None:
        with self.dirty_cell_flush_lock:
            self.resized_sheets, resized_sheets = set(), self.resized_sheets
            for sheet_id in resized_sheets:
                self.send_sheet_attribute_and_grid_size_update(sheet_id, {})

            if self.dirty_cells:
                dirty_cells = self.dirty_cells.copy()
                self.dirty_cells.clear()
                try:
                    cell_updates = [
                        self.sheet_cell_for_address(addr).export(compact=True)
                        for addr in dirty_cells
                    ]

                    self.reply_to_client(
                        MessageTypes.SHEET_UPDATE,
                        SheetUpdateContent(cell_updates=cell_updates).to_dict(),
                        undo_msg=self.scheduled_undo,
                    )
                    self.scheduled_undo = None

                except Exception as e:
                    print("Server error: ", e, file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)

    def flush_loop(self) -> None:
        while True:
            time.sleep(0.1)
            self.flush_dirty_cells_now()

    def tick(self) -> None:
        ready_cells = self.tick_cell_queue.pop_ready()
        if ready_cells:
            addresses = [
                item.item.address.to_float_coord()
                for item in ready_cells
                if isinstance(item.item, TickCell)
            ]
            expressions = [
                item.item.expression
                for item in ready_cells
                if isinstance(item.item, Cron)
            ]
            self.reply_to_client(
                MessageTypes.TICK_REPLY,
                TickReplyContent(
                    addresses=addresses, expressions=expressions
                ).to_dict(),
            )

    def tick_loop(self) -> None:
        while True:
            try:
                if self.initialized:
                    self.tick()
            except Exception as e:
                sys.stderr.write(f"Error in tick() thread: {e}")
            time.sleep(1)

    def reply_to_client(
        self,
        message_type: MessageTypes | str,
        content: dict,
        *,
        broadcast: bool = True,
        undo_msg: dict[str, Any] | None = None,
    ) -> None:
        if self.silent:
            return

        metadata = {}
        if undo_msg:
            metadata = {"undo": undo_msg}

        self.kernel.send_response(
            self.kernel.iopub_socket if broadcast else self.kernel.shell_stream,
            message_type if isinstance(message_type, str) else message_type.value,
            content,
            metadata=metadata,
        )

    def notify_client_cells_have_changed(
        self,
        changed: Iterable[Address] | Address,
        *,
        undo: tuple[MessageTypes, dict] | None = None,
    ) -> None:
        if not self.silent and not self.in_gs_mode:
            if isinstance(changed, Address):
                changed = [changed]
            self.dirty_cells.update(changed)
            if undo:
                if self.scheduled_undo is not None:
                    self.flush_dirty_cells_now()
                self.scheduled_undo = self.undo_msg(undo[0], undo[1])
            for address in changed:
                sheet = self.sheets[address.sheet]
                if address.column >= sheet.n_cols or address.row >= sheet.n_rows:
                    sheet.n_cols = max(address.column + 1, sheet.n_cols)
                    sheet.n_rows = max(address.row + 1, sheet.n_rows)
                    self.resized_sheets.add(address.sheet)

    def find_qualified_name_in_user_space(
        self, qualified_name: str
    ) -> tuple[bool, Any]:
        ns = self.shell.user_ns
        keep = True
        for name in qualified_name.split("."):
            keep = (
                not (module_all := inspect.getattr_static(ns, "__all__", None))
                or name in module_all
            )
            ns = (
                ns.get(name)
                if isinstance(ns, dict)
                else inspect.getattr_static(ns, name, None)
            )
            if ns is None:
                return True, None
        return keep, ns

    @client_callable
    def available_functions(
        self, prefix: str, skip_formulas: bool = False
    ) -> list[tuple[str, str | None, list[str]]]:
        if "." in prefix:
            root_name, prefix = prefix.rsplit(".", 1)
            _, root = self.find_qualified_name_in_user_space(root_name)
            if root is None:
                return []
            if isinstance(root, dict):
                namespace = root
            else:
                namespace = root.__dict__
        else:
            namespace = dict(self.shell.user_global_ns)

        def keep(name: str, candidate: Any) -> bool:
            if not name.startswith(prefix):
                return False
            if inspect.isclass(candidate) and issubclass(candidate, BaseWidget):
                return True
            if not inspect.isfunction(candidate) and not inspect.isbuiltin(candidate):
                return False
            if candidate.__module__ == "__main__":
                return True
            return not skip_formulas

        def params(fun: Callable) -> list[str]:
            try:
                sig = inspect.signature(fun)
            except ValueError:
                return []

            def as_str(p: inspect.Parameter) -> str:
                name = p.name
                if p.kind == p.VAR_POSITIONAL:
                    name = f"*{name}"
                elif p.kind == p.VAR_KEYWORD:
                    name = f"**{name}"
                if p.default is inspect.Parameter.empty:
                    return name
                return f"{name}={p.default!r}"

            return [as_str(p) for p in sig.parameters.values()]

        res = [
            (name, inspect.getdoc(fun), params(fun))
            for name, fun in namespace.items()
            if keep(name, fun)
        ]
        if len(res) > 1:
            res = [
                (name, doc.split("\n", 1)[0] if doc else doc, params)
                for name, doc, params in res
            ]
        return res

    def set_item(
        self, address: Address, value: Any, dynamic_unroll: bool = False
    ) -> Address:
        value = unproxy_val(value)
        value = maybe_format_common_values(value)

        if isinstance(value, types.GeneratorType | tuple):
            value = [*value]
        elif isinstance(value, dict):
            value = [CellRange([k, v]) for k, v in value.items()]
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, pd.Series):
            pd_header = [("", value.name)] if value.name else []
            value = [*pd_header, *value.items()]
        elif isinstance(value, pd.DataFrame):
            value = dataframe_to_grid(value)
        elif (
            hasattr(value, "__iter__")
            and not hasattr(value, "_repr_mimebundle_")
            and not isinstance(value, str | type)
        ):
            value = [*value]

        # Treat none and empty list as clearing a cell.
        if not self.gsheet_service and (
            value is None or (isinstance(value, list) and not value)
        ):
            self.update_cell(address, Empty.MakeItSo)
            return address

        def strip_cell_range(value: Any, depth: int) -> Any:
            for i in range(depth):
                if isinstance(value, list | CellRange) and len(value) == 1:
                    value = value[0]
            return unproxy_val(value)

        def maybe_resolve_cell_ranges(row: Any) -> Any:
            """If we have a cell range that has a ref, we need to make a copy to avoid write conflicts"""
            if isinstance(row, CellRange) and row.ref:
                return [*row]
            if not hasattr(row, "__iter__"):
                row = [row]
            return row

        def handle_meta(
            cell: Any, cell_address: Address, source_address: Address, dx: int, dy: int
        ) -> None:
            def maybe_set_raw_code() -> None:
                if meta := self.cell_meta.get(cell_address):
                    meta.raw_code = str(cell)

            self.side_effect_cells.add(cell_address)
            if dy > 0 or dx > 0:
                if dynamic_unroll:
                    self.set_unroll_source(source_address, cell_address)
                else:
                    maybe_set_raw_code()
                    calculated_by = self.graph.calculated_by.pop(cell_address, None)
                    if calculated_by in self.graph.feeds_into:
                        self.graph.feeds_into[calculated_by].remove(cell_address)
            elif not dynamic_unroll:
                maybe_set_raw_code()

        def assign_cell(cell: Any, address: Address, dx: int = 0, dy: int = 0) -> None:
            cell_address = replace(
                address, column=address.column + dx, row=address.row + dy
            )
            handle_meta(cell, cell_address, address, dx, dy)
            self.update_cell(
                cell_address,
                unproxy_val(cell),
            )

        if self.gsheet_service:
            gsheets_api.set_item(
                self.gsheet_service, self.gsheets_spreadsheet_id, address, value
            )
        else:
            h, w = shape(value)
            if w == 1 and h == 1:
                value = strip_cell_range(value, 2)
                assign_cell(value, address)
            else:
                value = [maybe_resolve_cell_ranges(row) for row in value]
                for dy, row in enumerate(value):
                    if not hasattr(row, "__iter__") or isinstance(row, str):
                        assign_cell(row, address, 0, dy)
                        continue
                    for dx, cell in enumerate(row):
                        assign_cell(cell, address, dx, dy)

        return address

    def initialize_interactive(self) -> None:
        api_key = input("Enter API key: ")
        self.initialize_local_kernel(api_key, "https://app.neptyne.com")

    def get_api_token(self) -> tuple[str, str]:
        if not self.api_key:
            raise ValueError("API key not set. Use `do_repl_init` to set it.")
        res = requests.get(
            f"{self.api_host}/api/v1/token", params={"apiKey": self.api_key}
        )
        res.raise_for_status()
        payload = res.json()
        api_token = payload["token"]
        gsheet_id = payload["gsheet_id"]
        return api_token, gsheet_id

    def initialize_local_kernel(self, api_key: str, api_host: str) -> None:
        if self.initialized:
            print(
                "Already initialized. If you need to connect to a different sheet, restart the kernel."
            )
            print(
                f"https://docs.google.com/spreadsheets/d/{self.gsheets_spreadsheet_id}"
            )
            return
        self.api_key = api_key
        self.api_host = api_host

        api_host_host = urlparse(api_host).hostname
        if api_host_host != "localhost":
            os.environ["API_PROXY_HOST_PORT"] = f"api-proxy.{api_host_host}"

        api_token, gsheet_id = self.get_api_token()

        os.environ["NEPTYNE_API_TOKEN"] = api_token
        self.initialize_phase_1_decoded(
            InitPhase1Payload(
                in_gs_mode=True,
                gsheets_sheet_id=gsheet_id,
                requirements="",
                sheets=[],
                time_zone="UTC",
                streamlit_base_url_path="",
                env={},
            )
        )
        self.initialize_phase_2_decoded(
            InitPhase2Payload(
                sheets=TyneSheets(),
                requires_recompile=False,
            )
        )
        self.initialized = True

        print("Connected to Google Sheet:")
        print(f"https://docs.google.com/spreadsheets/d/{self.gsheets_spreadsheet_id}")

    def __getitem__(
        self, item: CoordinateTuple | Address | Range
    ) -> float | str | int | SpreadsheetError | CellRange | CellApiMixin | None:
        from .dash_ref import DashRef

        if not self.initialized and os.getenv("NEPTYNE_LOCAL_REPL"):
            self.initialize_interactive()
            return None

        if self.gsheet_service:
            return gsheets_api.get_item(
                self.gsheet_service, self.gsheets_spreadsheet_id, item
            )

        item = self.from_coordinate(item)

        if isinstance(item, Address):
            check_max_col_for_address(item)
            return proxy_val(self.cells[item.sheet].get(item), DashRef(self, item))

        assert isinstance(item, Range)
        check_max_col_for_range(item)
        if item.min_row == 0 and item.max_row == -1:
            return self.sheets[item.sheet].cols[item.min_col : item.max_col + 1]
        if item.min_col == 0 and item.max_col == -1:
            return self.sheets[item.sheet].rows[item.min_row : item.max_row + 1]
        return CellRange(DashRef(self, item))

    def __setitem__(self, coord: AddressTuple, value: Any) -> None:
        address = self.from_coordinate(coord)

        if isinstance(value, str | int | float | bool | Empty) or value is None:
            assert isinstance(address, Address)

            if self.gsheet_service:
                gsheets_api.set_item(
                    self.gsheet_service, self.gsheets_spreadsheet_id, address, value
                )
            else:
                check_max_col_for_address(address)
                self.update_cell(address, value)
                self.update_graph_set_item(address, value)
                self.side_effect_cells.add(address)
        else:
            if isinstance(address, Range):
                if self.gsheet_service:
                    address = address.origin()
                else:
                    check_max_col_for_range(address)
                    value, _ = assure_compatible_shape(value, address.shape())
                    address = address.origin()

                    self.update_graph_set_item(address, value)

            self.set_item(address, value)

    def update_graph_set_item(self, address: Address, value: Any) -> None:
        value_code = str(value)
        if isinstance(value, Empty) or value is None:
            value_code = ""
        self.set_raw_code(address, value_code)
        if address in self.graph.depends_on:
            del self.graph.depends_on[address]
        if address in self.graph.calculated_by:
            del self.graph.calculated_by[address]

        cell_meta = self.cell_meta.get(address)
        if cell_meta:
            cell_meta.compiled_code = ""
            cell_meta.mime_type = None

    def stack_trace(self) -> SpreadsheetError:
        tb_offset = 2
        etype, evalue, tb = sys.exc_info()
        ip = get_ipython()  # type: ignore # noqa: F821
        stack_trace = ip.InteractiveTB.structured_traceback(
            etype, evalue, tb, tb_offset=tb_offset
        )
        if len(stack_trace) < 3:
            # Has no real contents:
            stack_trace = stack_trace[-1:]
        self.maybe_notify_error(stack_trace)
        return SpreadsheetError.from_python_exception(
            etype,
            evalue,
            stack_trace,
        )

    async def exec(
        self, graph: CellExecutionGraph, *, undo_content: dict | None = None
    ) -> set[Address]:
        self.side_effect_cells = set()
        changed: set[Address] = set()
        awaitables = set()
        while graph.is_active():
            statements = graph.ready_statements()
            if statements is None:
                break
            for statement in statements:
                if isinstance(statement, ExecOp):
                    address = statement.address
                    while True:
                        skip_value_set = False
                        try:
                            with self.use_cell_id(address):
                                value = eval(
                                    statement.expression,
                                    self.shell.user_global_ns,
                                    self.shell.user_ns,
                                )
                            if (
                                value is None
                                and self._pending_display_msg
                                and (
                                    content := self._pending_display_msg.get("content")
                                )
                            ):
                                value = output_to_value(content.get("data", {}))
                            break
                        except DashRecursionError:
                            skip_value_set = True
                            value = None  # to satisfy the linter, mostly
                            break
                        except NameError as e:
                            if not is_cell(cell_addr_upper := e.name.upper()):
                                value = self.stack_trace()
                                break
                            cell = self.get_or_create_cell_meta(address)
                            new_raw_code = rename_variable_in_code(
                                cell.compiled_code,
                                cell.raw_code,
                                e.name,
                                cell_addr_upper,
                            )
                            if new_raw_code is None:
                                value = self.stack_trace()
                                break

                            cell.raw_code = new_raw_code

                            self.compile_and_update_cell_meta(address)
                            statement.expression = cell.compiled_code
                        except SyntaxError as e:
                            # Try parse a:b as A:B and a1:b1 as A1:B1
                            if (
                                e.offset is not None
                                and e.end_offset is not None
                                and e.text
                                and e.text[
                                    (start_offset := e.offset - 1) : (
                                        end_offset := e.end_offset - 1
                                    )
                                ]
                                == ":"
                            ):
                                cell = self.get_or_create_cell_meta(address)
                                positions: dict[tuple[int, int], str] = {}
                                replaced_code = replace_n_with_a1(
                                    cell.compiled_code,
                                    func=replace_n_with_a1_match,
                                    positions=positions,
                                )
                                modified_code = try_parse_capitalized_range(
                                    replaced_code,
                                    positions,
                                    start_offset,
                                    end_offset,
                                )
                                if modified_code:
                                    cell.raw_code = "=" + modified_code
                                    self.compile_and_update_cell_meta(address)
                                    statement.expression = cell.compiled_code
                                else:
                                    value = self.stack_trace()
                                    break
                            else:
                                value = self.stack_trace()
                                break

                        except Exception:
                            value = self.stack_trace()
                            break
                    is_awaitable = inspect.isawaitable(value)
                    if not skip_value_set:
                        if is_awaitable:
                            task = asyncio.create_task(value)
                            setattr(task, "neptyne_cell_id", address)
                            awaitables.add(task)
                        else:
                            self.set_item(address, value, dynamic_unroll=True)
                            changed.add(address)
                    if not is_awaitable:
                        graph.done(address)
                elif isinstance(statement, ClearOp):
                    to_clear_addrs = statement.to_clear
                    if to_clear_addrs:
                        changed.update(to_clear_addrs)
                        self.clear_cells_internal(to_clear_addrs)
                    graph.done(*statement.to_clear)

            while awaitables:
                done, awaitables = await asyncio.wait(
                    awaitables, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    addr = getattr(task, "neptyne_cell_id")
                    graph.done(addr)
                    self.set_item(addr, task.result(), dynamic_unroll=True)
                    changed.add(addr)

        self.notify_client_cells_have_changed(
            changed,
            undo=(MessageTypes.RUN_CELLS, undo_content) if undo_content else None,
        )
        return changed

    def maybe_notify_error(self, traceback: list[str]) -> None:
        for line in traceback:
            if service := get_api_error_service(line):
                self.reply_to_client(
                    MessageTypes.API_QUOTA_EXCEEDED, {"service": service}
                )
                return

    @client_callable
    def widget_registry(self) -> dict[str, WidgetDefinition]:
        return widget_registry.to_dict()

    # Returns a list of param -> error message.
    @client_callable
    def widget_validate_params(
        self, param_values: dict[str, Any], code: str
    ) -> dict[str, str]:
        widget_name = widget_name_from_code(code)
        if not widget_name:
            return {"": "Couldn't match widget type from code string"}

        # Evaluate all params (to convert cell ranges and variables to their values).
        eval_errors = {}
        evaluated_params = {}
        for param_name, value in param_values.items():
            try:
                evaluated_params[param_name] = eval(
                    value, self.shell.user_global_ns, self.shell.user_ns
                )
            except Exception as e:
                eval_errors[param_name] = str(e)

        param_errors = validate_widget_params(
            widget_name, {**param_values, **evaluated_params}
        )

        if eval_errors or param_errors:
            return param_errors | eval_errors

        # Execute the widget code
        try:
            widget = eval(
                code.lstrip("="), self.shell.user_global_ns, self.shell.user_ns
            )
            return widget.validate_all_fields()
        except Exception as e:
            return {"": str(e)}

    @client_callable
    def get_widget_state(self, address_list: list[float]) -> dict[str, Any]:
        address = Address.from_list(address_list)
        cell = self.cells[address.sheet].get(address)
        if not isinstance(cell, BaseWidget):
            return {}

        def get_parsed_expression() -> dict:
            code = self.get_raw_code(address)

            parsed_expression = {}
            if code:
                parsed_expression = parse_widget_code(code, widget_registry)

            # Deserialize string and color params
            widget_name = widget_name_from_code(code)
            if widget_name in widget_registry.widgets:
                widget_param_types = {
                    p.name: p.type for p in widget_registry.widgets[widget_name].params
                }

                def update_param(name: str, value: str) -> str:
                    if (
                        widget_param_types[name] == WidgetParamType.STRING
                        or widget_param_types[name] == WidgetParamType.COLOR
                    ) and value:
                        return eval(value)
                    return value

                parsed_expression = {
                    k: update_param(k, v) for k, v in parsed_expression.items()
                }
            return parsed_expression

        def handle_callable_field(field_name: str) -> str:
            value = getattr(cell, field_name)
            if isinstance(value, str):
                value = decode_callable(value)
            if hasattr(value, "__name__"):
                return value.__name__
            return repr(value)

        def handle_field(f: dataclasses.Field, parsed_expression: dict) -> str:
            if f.name in parsed_expression:
                return parsed_expression[f.name]
            elif get_widget_param_type(f.type) == WidgetParamType.FUNCTION:
                return handle_callable_field(f.name)
            elif get_widget_param_type(f.type) == WidgetParamType.COLOR:
                return getattr(cell, f.name)._webcolor
            return getattr(cell, f.name)

        parsed_expr = get_parsed_expression()

        state_dict = {
            f.name: handle_field(f, parsed_expr)
            for f in fields(cell)
            if getattr(cell, f.name) is not None
        }

        return state_dict

    @client_callable
    def call_autocomplete_handler(self, val: str, **kwargs: Any) -> list[str]:
        address = Address(kwargs.get("col"), kwargs.get("row"), kwargs.get("sheetId"))  # type: ignore
        cell = self.cells[address.sheet].get(address)
        if not isinstance(cell, Autocomplete):
            return []
        try:
            return cell.get_choices(val)
        except Exception as exception:
            self.shell.parent_header["header"]["cellId"] = None
            self.reply_to_client(
                "error",
                {
                    "ename": exception.__class__.__name__,
                    "evalue": str(exception),
                    "traceback": self.shell.SyntaxTB.structured_traceback(
                        *sys.exc_info(), tb_offset=0
                    ),
                },
            )

        return []

    async def do_events(self, sleep_time: float = 0) -> None:
        while sleep_time >= 0:
            time_to_sleep = min(sleep_time, 0.05)
            await asyncio.sleep(time_to_sleep)
            if time_to_sleep > 0:
                sleep_time -= time_to_sleep
            else:
                sleep_time = -1
            if not isinstance(self.kernel, InProcessKernel):
                while True:
                    res = await self.kernel.process_one(wait=False)
                    if res is None:
                        break
            self.flush_side_effects()

    def initiate_download(
        self, value: Any, filename: str | None = None
    ) -> Download | None:
        payload: str | bytes
        if isinstance(value, CellRange | list | dict | set):
            default_filename = ""
            ext = filename.rsplit(".", 1)[-1] if filename else ""
            if ext == "csv":
                mimetype = CSV_MIME_KEY
            elif ext == "json":
                mimetype = JSON_MIME_KEY
            elif ext == "txt":
                mimetype = TEXT_MIME_KEY
            elif isinstance(value, CellRange):
                sh = value.shape
                if len(sh) == 1:
                    mimetype = TEXT_MIME_KEY
                    default_filename = "list.txt"
                else:
                    mimetype = CSV_MIME_KEY
                    default_filename = "table.csv"
            elif isinstance(value, set):
                mimetype = TEXT_MIME_KEY
                default_filename = "set.txt"
            else:
                mimetype = JSON_MIME_KEY
                default_filename = "data.json"

            if isinstance(value, CellRange):
                value = value.to_list()

            if isinstance(value, set):
                value = [*value]

            if mimetype == CSV_MIME_KEY:
                ft = io.StringIO()
                w = csv.writer(ft, lineterminator="\n")
                for row in value:
                    if isinstance(row, list):
                        w.writerow(row)
                    else:
                        w.writerow([row])
                payload = ft.getvalue()
            elif mimetype == JSON_MIME_KEY:
                payload = json.dumps(value)
            elif mimetype == TEXT_MIME_KEY:
                payload = "\n".join(str(x) for x in value)
            else:
                raise ValueError(f"Unsupported extension: {ext}")
        elif isinstance(value, list | dict):
            payload = json.dumps(value)
            default_filename = "data.json"
            mimetype = JSON_MIME_KEY
        elif isinstance(value, str):
            payload = value
            default_filename = "note.txt"
            mimetype = TEXT_MIME_KEY
        elif isinstance(value, Image.Image):
            with BytesIO() as fb:
                value.save(fb, format="jpeg")
                payload = fb.getvalue()
            mimetype = "image/jpeg"
            default_filename = "image.jpg"
        else:
            default_filename = "data.bin"
            mimetype = "application/octet-stream"
            payload = str(value)

        if isinstance(payload, str):
            payload = payload.encode("utf-8")

        assert isinstance(payload, bytes)

        self.reply_to_client(
            MessageTypes.START_DOWNLOAD,
            DownloadRequest(
                payload=base64.b64encode(payload).decode("utf8"),
                filename=filename or default_filename,
                mimetype=mimetype,
            ).to_dict(),
        )

    def initiate_upload(
        self, prompt: str = "Upload a file", accept: str = "*"
    ) -> tuple[bytes, str]:
        response = send_sync_request(
            MessageTypes.UPLOAD_FILE.value,
            {
                "prompt": prompt,
                "accept": accept,
            },
        )
        if not response:
            raise RequestCancelledError()
        return base64.b64decode(response["bytes"]), response["name"]

    def url_for_image(self, img: BaseFigure | Image.Image) -> str:
        # TODO: Support matplotlib
        if isinstance(img, BaseFigure):
            content_type, value = encode_plotly_figure(img)
        elif isinstance(img, Image.Image):
            content_type, value = encode_pil_image(img)
        else:
            raise ValueError(
                "Invalid image type. Must be a plotly figure or PIL image."
            )

        encoded = json_encode(value).decode("utf-8")
        response = send_sync_request(
            MessageTypes.UPLOAD_FILE_TO_GCP.value,
            {
                "content_type": content_type,
                "content": encoded,
            },
        )

        if not response:
            raise RequestCancelledError()
        return response  # type: ignore

    def clear_sheet(self, sheet_id: int) -> set[Address]:
        if sheet_id in self.cells:
            del self.cells[sheet_id]
        cells_to_reevaluate = set()
        for addr in [a for a in self.cell_meta.keys() if a.sheet == sheet_id]:
            for feeds_into_id in self.graph.feeds_into.get(addr, ()):
                cells_to_reevaluate.add(feeds_into_id)
            self.disconnect_cell(addr)
            del self.cell_meta[addr]
        return cells_to_reevaluate

    def rename_sheet_reference(self, old_name: str, new_name: str) -> None:
        changed: set[Address] = set()
        ref_name = repr(new_name) if not new_name.isidentifier() else new_name
        for cell_id, cell_meta in self.cell_meta.items():
            raw_code = self.get_raw_code(cell_id)
            if self.has_formula(cell_id) and old_name in raw_code:
                tokens = tokenize_with_ranges(raw_code)
                result = process_sheet_rename(tokens, old_name, ref_name)
                if result:
                    # No need to recompile -- compilation replaces sheet name with ID, which has
                    # not changed.
                    self.set_raw_code(cell_id, result)
                    changed.add(cell_id)
        self.notify_client_cells_have_changed(
            changed,
        )

    @client_callable
    def upload_csv_to_new_sheet(self, prompt: str = "Upload a .csv") -> int | None:
        sheet = self.upload_csv_to_new_sheet_and_return(prompt=prompt)
        if sheet is not None:
            return sheet.sheet_id

    def upload_csv_to_new_sheet_and_return(
        self, prompt: str = "Upload a .csv"
    ) -> Optional["NeptyneSheet"]:
        try:
            content, name = self.initiate_upload(prompt=prompt, accept=".csv,.tsv")
        except RequestCancelledError:
            return None

        return self.sheets.sheet_from_dataframe(
            pd.read_csv(BytesIO(content)), os.path.splitext(name)[0]
        )

    def update_execution_policy(self, address: Address, policy: int) -> None:
        if address not in self.cell_meta and policy == -1:
            return
        metadata = self.get_or_create_cell_meta(address)
        metadata.execution_policy = policy
        if metadata.execution_policy > 0:
            metadata.next_execution_time = time.time() + policy
            self.tick_cell_queue.put(
                address,
                metadata.execution_policy,
                metadata.next_execution_time,
            )
        else:
            self.tick_cell_queue.drop(address)

    def update_cell_attribute(
        self, address: Address, attribute: str, value: Any, overwrite: bool = True
    ) -> None:
        """Only updates the state in kernel without broadcast to the client"""
        if attribute == CellAttribute.EXECUTION_POLICY.value:
            self.update_execution_policy(address, policy=int(value) if value else -1)
        else:
            if value is not None:
                metadata = self.get_or_create_cell_meta(address)
                if attribute not in metadata.attributes or overwrite:
                    metadata.attributes[attribute] = value
            elif (
                maybe_metadata := self.cell_meta.get(address)
            ) and attribute in maybe_metadata.attributes:
                del maybe_metadata.attributes[attribute]

    def update_cells_attributes(self, content: dict[str, Any]) -> None:
        update = CellAttributesUpdate.from_dict(content)
        undo_update = copy.deepcopy(update)
        undo_update.updates = []
        changed_addresses: set[Address] = set()
        for ud in update.updates:
            addr = Address.from_list(ud.cell_id)
            changed_addresses.add(addr)
            undo_change = copy.deepcopy(ud)
            attributes = (
                self.cell_meta[addr].attributes if addr in self.cell_meta else None
            )
            undo_change.value = (
                attributes.get(ud.attribute) if attributes is not None else None
            )
            self.update_cell_attribute(addr, ud.attribute, ud.value)
            undo_update.updates.append(undo_change)
        self.notify_client_cells_have_changed(
            changed_addresses,
            undo=(MessageTypes.CHANGE_CELL_ATTRIBUTE, undo_update.to_dict()),
        )

    def parse_mime_typed_expression(self, expression: str, mime_type: str) -> Any:
        return output_to_value({mime_type: expression})

    def update_sheet_attributes_internal(
        self, updates: list[tuple[int, str, Any]]
    ) -> None:
        if self.in_gs_mode:
            print("warning: update_sheet_attributes_internal called in gs mode")
            return
        for sheet_id, attribute, value in updates:
            sheet = self.sheets[sheet_id]
            if value is not None:
                sheet.attributes[attribute] = value
            elif attribute in self.sheets[sheet_id].attributes:
                del sheet.attributes[attribute]

    def update_sheet_attributes(self, content: dict, undo: bool = True) -> None:
        if self.in_gs_mode:
            raise NotImplementedError(
                "Update sheet attributes not implemented for google sheets. Ping the Discord channel if you need it."
            )
        update = SheetAttributeUpdate.from_dict(content)
        copy.deepcopy(update)
        sheet_id = int(update.sheet_id)
        if sheet_id not in self.sheets:
            return
        prev_value = self.sheets[sheet_id].attributes.get(update.attribute)
        self.update_sheet_attributes_internal(
            [(sheet_id, update.attribute, update.value)]
        )

        undo_msg = None
        if undo:
            undo_content = SheetAttributeUpdate(
                update.attribute, update.sheet_id, prev_value
            ).to_dict()
            undo_msg = self.undo_msg(MessageTypes.CHANGE_SHEET_ATTRIBUTE, undo_content)
        self.reply_to_client(
            MessageTypes.CHANGE_SHEET_ATTRIBUTE_REPLY, content, undo_msg=undo_msg
        )

    def sheet_cell_for_address(self, address: Address) -> SheetCell:
        value = self.cells[address.sheet].get(address)
        meta = self.cell_meta.get(address)
        if not meta:
            meta = CellMetadata()
        return SheetCell(
            cell_id=address,
            raw_code=self.get_raw_code(address),
            compiled_code=meta.compiled_code,
            attributes=meta.attributes,
            mime_type=meta.mime_type,
            depends_on=self.graph.depends_on.get(address, set()),
            feeds_into=self.graph.feeds_into.get(address, set()),
            calculated_by=self.graph.calculated_by.get(address),
            execution_policy=meta.execution_policy,
            next_execution_time=meta.next_execution_time,
            output=meta.output if isinstance(meta.output, Output) else value,
        )

    def export_sheet_cell_compact(self, address: Address) -> dict[str, Any] | list:
        if address in self.cell_meta:
            return self.sheet_cell_for_address(address).export(compact=True)
        else:
            return [
                address.to_coord(),
                self.cells[address.sheet].get(address),
                self.get_raw_code(address),
            ]

    def save_state(self, for_client: str | None = None) -> None:
        if not self.initialized:
            return
        msg = V1DashSaveMessage.from_dash_state(
            None if self.in_gs_mode else self.sheets,
            self.cells,
            self.cell_meta,
            self.graph,
            self.tick_cell_queue.next_tick(),
        )
        self.reply_to_client(
            MessageTypes.SAVE_KERNEL_STATE,
            {
                "bytes": msg.to_bytes(),
                "for_client": for_client,
                "version": V1DashSaveMessage.VERSION,
            },
        )

    def process_function_changes(self, changed_fns: list[str]) -> None:
        def should_rerun_cell(raw_code: str) -> bool:
            for change in changed_fns:
                if change in raw_code:
                    if change in set(re.split(r"[\W]", raw_code)):
                        return True
            return False

        cells_to_update = []
        for cell_id, cell in self.cell_meta.items():
            if (
                self.has_formula(cell_id)
                and should_rerun_cell(self.get_raw_code(cell_id))
                and not cell.is_input_widget()
            ):
                cells_to_update.append(cell_id)
        self.reply_to_client(
            MessageTypes.RERUN_CELLS,
            RerunCellsContent(
                changed_functions=changed_fns,
                addresses=[address.to_float_coord() for address in cells_to_update],
            ).to_dict(),
        )

    def run_cells_with_cascade_coords(self, cell_ids: list[CoordAddr]) -> None:
        self.run_cells_with_cascade(
            cell_ids=[Address.from_coord(cell_id) for cell_id in cell_ids],
        )

    def clear_cells_with_cascade(self, cell_ids: list[Address]) -> None:
        for cell_id in cell_ids:
            self.set_raw_code(cell_id, "")
        self.run_cells_with_cascade(cell_ids=cell_ids)

    def run_cells_with_cascade(
        self,
        *,
        cell_changes: list[dict] | None = None,
        cell_ids: list[Address] | None = None,
        pre_clear: Sequence[Address] | None = None,
        undoable: bool = False,
    ) -> None:
        """Main entry point for cell execution. cell_changes and cell_ids are used mutually exclusively"""
        assert not (cell_changes and cell_ids)
        undo_content = (
            self.compute_run_cells_undo_changes(cell_changes)
            if undoable and cell_changes
            else None
        )

        if cell_changes:
            to_run = self.preprocess_changes(cell_changes)
        else:
            to_run = set(cell_ids) if cell_ids else set()
        expected_changes = self.compile_and_execute_cells(
            to_run,
            pre_clear=pre_clear,
            undo_content=undo_content,
        )
        self.flush_side_effects(expected_changes=expected_changes)

    def copy_cells(self, d: dict) -> None:
        copy_cells_content = CopyCellsContent.from_dict(d)

        def get_address(cell_change: CellChange) -> Address:
            cell_id = cell_change.cell_id
            assert cell_id
            if isinstance(cell_id, str):
                return Address.from_a1_or_str(cell_id)
            return Address.from_list(cell_id)

        adjusted_cells = pre_copy_adjust(
            Address.from_a1_or_str(copy_cells_content.anchor),
            [
                (
                    get_address(tc),
                    tc.content,
                    tc.attributes,
                )
                for tc in copy_cells_content.to_copy
            ],
        )

        self.run_cells_with_cascade(
            cell_changes=[
                CellChange(
                    attributes=attributes,
                    cell_id=[*cell_id.to_coord()],
                    content=content,
                    mime_type=None,
                ).to_dict()
                for cell_id, content, attributes in adjusted_cells
            ],
            undoable=True,
        )

    def get_metadata(self) -> dict:
        if self.cur_streamlit_info is not None:
            info = self.cur_streamlit_info[1].to_dict()
        else:
            info = {}

        return {
            "streamlit": info,
            "initialized": bool(self.initialized),
            "crons": [cron.to_dict() for cron in self.tick_cell_queue.get_crons()],
        }

    def post_execute(self) -> None:
        if Dash.in_post_execute_hook:
            return
        try:
            Dash.in_post_execute_hook = True
            self.flush_dirty_cells_now()
            self.track_function_changes()
            self.shell.run_cell("N_.flush_side_effects()")
            self.maybe_apply_on_value_change_rules()
            self.stop_start_streamlit()
        except Exception as e:
            print(e)
            raise
        finally:
            Dash.in_post_execute_hook = False

    def flush_side_effects(
        self,
        widget_trigger_cell: CoordAddr | None = None,
        expected_changes: set[Address] | None = None,
    ) -> None:
        if expected_changes is None:
            expected_changes = set()
        try:
            for i in range(MAX_CASCADE_COUNT):
                to_run = set()
                if widget_trigger_cell:
                    cell_id = Address.from_coord(widget_trigger_cell)
                    if feeds_into := self.graph.feeds_into.get(cell_id):
                        to_run.update(feeds_into)
                    widget_trigger_cell = None
                for cell_id in self.side_effect_cells:
                    if self.has_formula(cell_id):
                        to_run.add(cell_id)
                    elif feeds_into := self.graph.feeds_into.get(cell_id):
                        to_run.update(feeds_into)

                to_run = to_run.difference(expected_changes)
                if not to_run:
                    break
                expected_changes = self.compile_and_execute_cells(to_run)
            self.apply_on_value_change_rule(
                expected_changes.union(self.side_effect_cells)
            )
        finally:
            self.side_effect_cells = set()

    def compile_and_execute_cells(
        self,
        cells_to_run: set[Address],
        *,
        pre_clear: Sequence[Address] | None = None,
        undo_content: dict | None = None,
    ) -> set[Address]:
        """Warning: Do not use this function standalone. Use flush_side_effects or run_cells_with_cascade.
        Using this standalone won't trigger proper cascading"""
        for cell_id in cells_to_run:
            self.compile_and_update_cell_meta(cell_id)

        execution_graph = self.get_execution_graph(cells_to_run, pre_clear=pre_clear)

        from jupyter_core.utils import run_sync

        return run_sync(self.exec)(execution_graph, undo_content=undo_content)

    def compute_run_cells_undo_changes(
        self, cell_changes: list[dict[str, Any]]
    ) -> dict:
        content_before = []

        for dict_change in cell_changes:
            change = CellChange.from_dict(dict_change)
            assert change.cell_id
            cell_id = Address(*change.cell_id)
            cell_meta = self.cell_meta.get(cell_id) or CellMetadata()
            code_before = self.get_raw_code(cell_id)
            attributes_before = cell_meta.attributes or {}
            mime_type_before = cell_meta.mime_type

            content_before.append(
                CellChange(
                    attributes=attributes_before,
                    cell_id=[*cell_id.to_coord()],
                    content=code_before,
                    mime_type=mime_type_before,
                )
            )

        return RunCellsContent(
            to_run=content_before,
            notebook=False,
            for_ai=False,
            gs_mode=False,
            current_sheet=0,
            ai_tables=[],
            current_sheet_name="",
            sheet_ids_by_name={},
        ).to_dict()

    def preprocess_changes(
        self,
        cell_changes: list[dict],
    ) -> set[Address]:
        """The first call to run_cells with special logic from the changes the client sends."""
        cell_ids: set[Address] = set()
        for dict_change in cell_changes:
            change = CellChange.from_dict(dict_change)
            assert change.cell_id
            cell_id = Address(*change.cell_id)
            cell_ids.add(cell_id)
            code = change.content
            new_attributes = change.attributes
            mime_type = change.mime_type

            if new_attributes is not None:
                attributes_to_clear = (
                    CELL_ATTRIBUTES_TO_CLEAR_ON_VALUE_CHANGE
                    if code
                    else CELL_ATTRIBUTES_TO_CLEAR_ON_CLEAR
                )
                self.get_or_create_cell_meta(cell_id).attributes = {}
                for key, value in new_attributes.items():
                    if key not in attributes_to_clear:
                        self.update_cell_attribute(cell_id, key, value)

            self.set_raw_code(cell_id, code)
            if mime_type:
                self.get_or_create_cell_meta(cell_id).mime_type = mime_type

        return cell_ids

    def compile_and_update_cell_meta(self, cell_id: Address) -> None:
        cell_meta = self.cell_meta.get(cell_id)
        if cell_meta and cell_meta.mime_type:
            compile_results = compile_mime_typed_expression(
                self.get_raw_code(cell_id), cell_meta.mime_type
            )
        elif not cell_meta and not self.has_formula(cell_id):
            self.unlink(cell_id)
            if cell_meta:
                cell_meta.compiled_code = ""
            return
        else:
            assert cell_meta is not None
            if cell_id.sheet in self.sheets:
                grid_size = (
                    self.sheets[cell_id.sheet].n_cols,
                    self.sheets[cell_id.sheet].n_rows,
                )
            else:
                grid_size = DEFAULT_GRID_SIZE

            try:
                compile_results = compile_expression(
                    self.get_raw_code(cell_id),
                    cell_id,
                    self.sheets._get_sheet_name_to_id(),
                    grid_size,
                )
            except ValueError:
                # We failed to compile. Try to run the code in the kernel, which will then send a
                # message back to the user with the SyntaxError
                cell_meta.compiled_code = self.get_raw_code(cell_id)
                return

            cell_meta.compiled_code = compile_results.compiled_code
            if compile_results.raw_code:
                cell_meta.raw_code = compile_results.raw_code

        # Rekey the graph from compilation result.
        self.update_cell_graph(cell_id, compile_results)

    def recompile_everything(self) -> None:
        for cell_id in self.all_keys():
            if self.has_formula(cell_id):
                self.compile_and_update_cell_meta(cell_id)

    def update_cell_graph(
        self, cell_id: Address, compile_result: CompileResult
    ) -> None:
        self.unlink(cell_id)
        for cell_mentioned in compile_result.cells_mentioned:
            self.link(cell_id, cell_mentioned)

    def get_execution_graph(
        self,
        cell_ids: set[Address],
        *,
        pre_clear: Sequence[Address] | None = None,
    ) -> CellExecutionGraph:
        return CellExecutionGraph(self, cell_ids, pre_clear)

    def get_or_create_cell_meta(self, cell_id: Address) -> CellMetadata:
        if cell_id in self.cell_meta:
            return self.cell_meta[cell_id]
        cell_meta = CellMetadata(raw_code=self.get_raw_code(cell_id))
        self.cell_meta[cell_id] = cell_meta
        return cell_meta

    def link(self, source_cell_id: Address, target_cell_id: Address) -> None:
        """source_cell's formula references target_cell"""
        if source_cell_id != target_cell_id:
            self.graph.depends_on.setdefault(source_cell_id, set()).add(target_cell_id)
            self.graph.feeds_into.setdefault(target_cell_id, set()).add(source_cell_id)

    def unlink(self, cell_id: Address) -> None:
        """unlink a cell from the dependency graph of things it depends on"""

        def clear_from_other(other_id: Address | None) -> None:
            if other_id and other_id in self.graph.feeds_into:
                self.graph.feeds_into[other_id].discard(cell_id)

        if calculated_by_id := self.graph.calculated_by.get(cell_id):
            clear_from_other(calculated_by_id)
            del self.graph.calculated_by[cell_id]

        if cell_id in self.graph.depends_on:
            for other_cell_id in self.graph.depends_on[cell_id]:
                clear_from_other(other_cell_id)
            del self.graph.depends_on[cell_id]

    def cells_calculated_by(self, cell_id: Address) -> set[Address]:
        calculated_by = set()
        for other_cell_id in self.graph.feeds_into.get(cell_id, ()):
            if self.graph.calculated_by.get(other_cell_id) == cell_id:
                calculated_by.add(other_cell_id)
        return calculated_by

    def disconnect_cell(self, cell_id: Address) -> None:
        """unlink a cell from the dependency graph completely in both directions."""
        self.unlink(cell_id)
        if cell_id not in self.graph.feeds_into:
            return

        for other_cell_id in self.graph.feeds_into[cell_id]:
            self.get_or_create_cell_meta(other_cell_id)
            if other_cell_id in self.graph.depends_on:
                self.graph.depends_on[other_cell_id].discard(cell_id)
                if not self.graph.depends_on[other_cell_id]:
                    del self.graph.depends_on[other_cell_id]
            if self.graph.calculated_by.get(other_cell_id) == cell_id:
                del self.graph.calculated_by[other_cell_id]

        del self.graph.feeds_into[cell_id]

    def set_unroll_source(
        self, source_cell_id: Address, target_cell_id: Address
    ) -> None:
        """source_cell's value unrolls/spills into target_cell"""
        self.unlink(target_cell_id)
        target_cell = self.cell_meta.get(target_cell_id)
        if target_cell:
            self.set_raw_code(target_cell_id, "")
            target_cell.compiled_code = ""
        self.graph.calculated_by[target_cell_id] = source_cell_id
        self.graph.feeds_into.setdefault(source_cell_id, set()).add(target_cell_id)

    def send_sheet_attribute_and_grid_size_update(
        self,
        sheet_id: int,
        sheet_attribute_updates: dict,
        transform: Transformation | None = None,
    ) -> None:
        # TODO: Maybe InsertDeleteReplyCellType isn't the best name here.
        #   This also updates grid size
        sheet = self.sheets[sheet_id]

        content = InsertDeleteReplyCellType(
            cell_updates=[],
            sheet_id=sheet_id,
            sheet_name=sheet.name,
            n_cols=sheet.n_cols,
            n_rows=sheet.n_rows,
            sheet_attribute_updates=sheet_attribute_updates,
        ).to_dict()

        # Transform is used for tyne_info to modify cell references in code editor
        content["transformation"] = transform.to_dict() if transform else None

        self.reply_to_client(MessageTypes.INSERT_DELETE_CELLS_REPLY, content)

    def compute_cells_to_populate(
        self, transformation: Transformation, include_modified_formulas: bool = False
    ) -> list[dict[str, Any]]:
        def cell_formula_is_modified(cell_id: Address) -> bool:
            raw_code = self.get_raw_code(cell_id)
            if not raw_code.startswith("="):
                return False
            tokens = tokenize_with_ranges(raw_code)
            return (
                process_sheet_transformation(
                    tokens,
                    transformation,
                    self.sheets[transformation.sheet_id].name,
                    transformation.sheet_id != cell_id.sheet,
                )
                is not None
            )

        return [
            self.sheet_cell_for_address(cell_id).to_dict()
            for cell_id in self.all_keys()
            if (
                cell_id.sheet == transformation.sheet_id
                and (not transformation.boundary or cell_id in transformation.boundary)
                and (
                    (
                        transformation.transform(cell_id.column, cell_id.row)
                        == Transformation.REF_ERROR
                    )
                    or (include_modified_formulas and cell_formula_is_modified(cell_id))
                )
            )
        ]

    def compute_inverse_insert_delete_transformation(
        self, transformation: Transformation
    ) -> tuple[Transformation, list[dict[str, Any]]]:
        cells_to_populate: list[dict[str, Any]]

        if transformation.operation == SheetTransform.INSERT_BEFORE:
            return (
                Transformation(
                    transformation.dimension,
                    SheetTransform.DELETE,
                    transformation.index,
                    transformation.amount,
                    transformation.sheet_id,
                    transformation.boundary,
                ),
                [],
            )

        elif transformation.operation == SheetTransform.DELETE:
            cells_to_populate = self.compute_cells_to_populate(transformation, True)

            return (
                Transformation(
                    transformation.dimension,
                    SheetTransform.INSERT_BEFORE,
                    transformation.index,
                    transformation.amount,
                    transformation.sheet_id,
                    transformation.boundary,
                ),
                cells_to_populate,
            )
        else:
            raise ValueError(f"Invalid Operation on transformation: {transformation}")

    def add_ai_table_to_run_cells_content(self, content: dict) -> None:
        run_cells_content = RunCellsContent.from_dict(content)
        run_cells_content.ai_tables = [
            table.to_dict() for table in self.compute_ai_tables()
        ]
        run_cells_content.current_sheet_name = self.sheets[
            int(run_cells_content.current_sheet)
        ].name
        run_cells_content.sheet_ids_by_name = {
            s: float(i) for s, i in self.sheets._get_sheet_name_to_id().items()
        }
        self.reply_to_client(MessageTypes.RUN_CELLS, run_cells_content.to_dict())

    def _get_header(self) -> dict | None:
        if header := self.shell.parent_header:
            return header.get("header")

    def _parent_header_matches_codepanel_cell(self) -> bool:
        return self.local_repl_mode or bool(
            (header := self._get_header())
            and header.get("cellId") == CODEPANEL_CELL_ID
            and threading.current_thread().name == "MainThread"
        )

    def pre_execute(self) -> None:
        if not Dash.in_post_execute_hook:
            if self._parent_header_matches_codepanel_cell():
                self.clear_runtime_scoped_objects()
            self.write_secrets_to_fs()

    def clear_runtime_scoped_objects(self) -> None:
        self.on_value_change_rules.clear()
        self.tick_cell_queue.clear_crons()
        self.prev_streamlit_info = self.cur_streamlit_info
        self.cur_streamlit_info = None
        if self._gsheet_name_registry is not None:
            self._gsheet_name_registry.clear_cache()

    def maybe_apply_on_value_change_rules(self) -> None:
        if self._parent_header_matches_codepanel_cell():
            self.apply_on_value_change_rule()

    def stop_start_streamlit(self) -> None:
        try:
            header = self._get_header()
            if not header or not self._parent_header_matches_codepanel_cell():
                return

            if self.cur_streamlit_info:
                code_panel_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "codepanel.py"
                )
                server_import_path_prefix = "server." if "server." in __name__ else ""
                setup_code = (
                    "\n".join(
                        (
                            f"from {server_import_path_prefix}neptyne_kernel.dash import Dash as _Dash",
                            f"from {server_import_path_prefix}neptyne_kernel.proxied_apis import start_api_proxying as _start_api_proxying",
                            "N_ = _Dash.instance()",
                            "_start_api_proxying()",
                            "if N_.in_gs_mode:",
                            f"    from {server_import_path_prefix}neptyne_kernel.kernel_globals.gsheets import *",
                            "else:",
                            f"    from {server_import_path_prefix}neptyne_kernel.kernel_globals.core import *",
                        )
                    )
                    + "\n\n"
                )

                try:
                    import streamlit as _streamlit_module
                except ImportError:
                    print(
                        "Error: streamlit not found. Install streamlit to use nt.streamlit",
                        file=sys.stderr,
                    )
                    return

                setattr(_streamlit_module, "_neptyne_dash", self)

                self.write_streamlit_config()
                with open(code_panel_path, "w") as f:
                    f.write(
                        setup_code
                        + header.get("code", "")
                        + f"\n\nif __name__ == '__main__':\n    {self.cur_streamlit_info[0].__name__}()"
                    )
                if not streamlit_server.is_server_running():
                    # more ugh
                    self.api_token_override = NeptyneSessionInfo.from_message_header(
                        self.shell.parent_header["header"]
                    ).sheets_api_token
                    streamlit_server.start_server(code_panel_path)
            else:
                # TODO(jack) we can't stop the Server because starting it again will raise
                # due to Server.__init__ creating a Runtime, which must always be a singleton
                pass

        except Exception as e:
            print("Server error: ", e, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    def add_ai_context_to_sheet_autofill_content(self, content: dict) -> None:
        sheet_drag_formula_content = SheetAutofillContent.from_dict(content)
        populate_to_start = Address.from_list(
            sheet_drag_formula_content.populate_to_start
        )
        populate_to_end = Address.from_list(sheet_drag_formula_content.populate_to_end)
        populate_from = [
            (Address.from_list(pf.cell_id), pf.content)
            for pf in sheet_drag_formula_content.populate_from
        ]

        min_x_from = min(pf[0].column for pf in populate_from)
        min_x_to = min(populate_to_start.column, populate_to_end.column)
        if min_x_from != min_x_to:
            min_y_to = min(populate_to_start.row, populate_to_end.row)
            min_x = min(
                *[pf[0].column for pf in populate_from],
                populate_to_start.column,
                populate_to_end.column,
            )
            max_x = max(
                *[pf[0].column for pf in populate_from],
                populate_to_start.column,
                populate_to_end.column,
            )
            context = self.get_autofill_context(
                min_y_to, min_x, max_x, populate_to_end.sheet, transpose=True
            )

        else:
            min_y = min(
                *[pf[0].row for pf in populate_from],
                populate_to_start.row,
                populate_to_end.row,
            )
            max_y = max(
                *[pf[0].row for pf in populate_from],
                populate_to_start.row,
                populate_to_end.row,
            )

            context = self.get_autofill_context(
                min_x_from, min_y, max_y, populate_to_end.sheet, transpose=False
            )
        sheet_drag_formula_content.autofill_context = context

        cells = self.cells[populate_to_end.sheet]
        pop_range = Range.from_addresses(populate_to_start, populate_to_end)

        for table in self.ai_tables_for_sheet(populate_to_end.sheet, pop_range):
            table_cell_ids = {cell_id for row in table for cell_id in row}
            if all(cell_id in table_cell_ids for row in pop_range for cell_id in row):
                sheet_drag_formula_content.to_fill = table.to_fill_in(
                    cells, self.cell_meta, pop_range
                )
                sheet_drag_formula_content.table = table.to_dict()
                break

        assert sheet_drag_formula_content.autofill_context is not None
        self.reply_to_client(
            MessageTypes.SHEET_AUTOFILL, sheet_drag_formula_content.to_dict()
        )

    def compute_ai_tables(self) -> list[TableForAI]:
        return [
            table
            for sheet in self.sheets
            for table in self.ai_tables_for_sheet(sheet.sheet_id)
        ]

    def get_autofill_context(
        self, start_x: int, min_y: int, max_y: int, sheet: int, *, transpose: bool
    ) -> list[str]:
        """This is setup to scan the cells in the column to the left of the autofill range.

        To make it work for rows, set transpose to True and swap x and y."""
        x = start_x - 1
        context = None
        while x >= 0 and not context:

            def get_cell_value(x: int, y: int) -> SheetCell | None:
                if transpose:
                    x, y = y, x
                return self.cells[sheet].get(Address(x, y, sheet))

            context = [get_cell_value(x, y) for y in range(min_y, max_y + 1)]
            if any(cell is None for cell in context):
                context = None
                break
            if not all(cell for cell in context):
                context = None
            x -= 1
        if context is None:
            return []
        else:
            return [str(x) for x in context]

    def ai_tables_for_sheet(
        self, sheet_num: int, assume_filled: Range | None = None
    ) -> Iterator[TableForAI]:
        cells = self.cells.get(sheet_num, {})
        sheet = self.sheets[sheet_num]
        return ai_tables_for_sheet(cells, sheet.name, assume_filled)

    def init_widget_copy_from(self, cell_id: Address) -> BaseWidget | None:
        init_code = self.get_or_create_cell_meta(cell_id).compiled_code
        if not init_code:
            return None
        return eval(init_code, self.shell.user_global_ns, self.shell.user_ns)

    def register_cron(
        self,
        schedule: str,
        func: Callable,
        timezone: str | None,
        alert_email: str | None,
    ) -> None:
        if timezone is not None:
            try:
                zoneinfo = ZoneInfo(timezone)
            except ZoneInfoNotFoundError:
                raise ValueError(f"Invalid timezone: {timezone}")
        else:
            zoneinfo = ZoneInfo(self.time_zone)
        expression = f"{func.__name__}()"
        if inspect.iscoroutinefunction(func):
            expression = f"await {expression}"
        self.tick_cell_queue.put_cron(Cron(expression, schedule, zoneinfo, alert_email))

        setattr(func, "next_execution_time", self.next_exec_for(func.__name__))

    def next_exec_for(self, func_name: str) -> Any:
        dash = self

        class GetNextExecutionTime:
            def __repr__(self) -> str:
                for cron in dash.tick_cell_queue.queue:
                    if (
                        isinstance(cron.item, Cron)
                        and cron.item.expression == f"{func_name}()"
                    ):
                        return cron.item.next_execution_time_datetime(
                            time.time()
                        ).isoformat()
                raise ValueError(f"Function {func_name} is not scheduled")

        return GetNextExecutionTime()

    def append_on_value_change_rule(
        self,
        cell_range: CellRange,
        fn: Callable[[Any], None],
        apply_on_full_range: bool,
    ) -> None:
        self.on_value_change_rules.append(
            OnValueChangeRule(cell_range, fn, apply_on_full_range)
        )

    def apply_on_value_change_rule(
        self, restrict_to_cells: set[Address] | None = None
    ) -> None:
        if restrict_to_cells is not None and not restrict_to_cells:
            return

        def maybe_apply_on_value_change(
            callback: Callable[[Any], None], rule_cell: Any
        ) -> None:
            addr = rule_cell.ref.range.origin()
            if not restrict_to_cells or addr in restrict_to_cells:
                callback(rule_cell)

        for rule in self.on_value_change_rules:
            if rule.apply_on_full_range:
                if not restrict_to_cells:
                    rule.callback(rule.range)
                    continue
                for cell in restrict_to_cells:
                    if rule.range.ref is not None and cell in rule.range.ref.range:
                        rule.callback(rule.range)
                        break
            else:
                for cell_or_row in rule.range:
                    if isinstance(cell_or_row, CellRange):
                        for cell in cell_or_row:
                            maybe_apply_on_value_change(rule.callback, cell)
                    else:
                        maybe_apply_on_value_change(rule.callback, cell_or_row)

    @contextmanager
    def use_cell_id(self, cell_address: Address) -> Iterator[None]:
        def hook(msg: dict[str, Any]) -> dict[str, Any]:
            msg["parent_header"]["cellId"] = cell_address.to_cell_id()
            self._pending_display_msg = msg
            return msg

        if cell_address in self._cell_execution_stack:
            raise DashRecursionError

        self._pending_display_msg = None
        self._cell_execution_stack.append(cell_address)

        try:
            self.shell.display_pub.register_hook(hook)
            yield
        finally:
            # Flush any matplotlib messages
            if matplotlib_backend_inline:
                matplotlib_backend_inline.show(True)
            self._cell_execution_stack.pop()
            self.shell.display_pub.unregister_hook(hook)

    def register_api_function(self, func: APIFunction, route: str | None) -> None:
        self.api_functions[route] = func

    def datetime_from_str(self, date_str: str) -> datetime.datetime:
        dt = dateutil.parser.parse(date_str, ignoretz=False)
        return dt.astimezone(dateutil.tz.tzlocal())

    def from_json(self, string: str) -> Any:
        return json.loads(string)

    def execute_gsheet_request(
        self,
        session_id: str,
        cell: str,
        expression: str,
    ) -> None:
        try:
            try:
                with ExitStack() as stack:
                    if cell:
                        address = Address.from_a1_or_str(cell)
                        self.shell.parent_header["header"]["cellId"] = (
                            address.to_cell_id()
                        )
                        stack.enter_context(self.use_cell_id(address))
                    value = eval(
                        expression, self.shell.user_global_ns, self.shell.user_ns
                    )
                    if callable(value):
                        value = value()
                    if (
                        value is None
                        and self._pending_display_msg
                        and (content := self._pending_display_msg.get("content"))
                    ):
                        value = output_to_value(content.get("data", {}))
            except Exception:
                etype, evalue, tb = sys.exc_info()
                value = gsheet_spreadsheet_error_from_python_exception(
                    etype, evalue, tb
                )

            content_type, encoded = encode_for_gsheets(value)

        except Exception as e:
            content_type = GSHEET_ERROR_KEY
            encoded = json.dumps(
                {
                    "ename": "InternalError",
                    "message": repr(e),
                    "line": -1,
                }
            )
        function_name = expression.split("(")[0].strip()
        function = self.shell.user_global_ns.get(function_name)
        function_caching = getattr(function, "caching", None) if function else None

        self.reply_to_client(
            MessageTypes.USER_API_RESPONSE_STREAM,
            {
                "content": encoded,
                "content_type": content_type,
                "session_id": session_id,
                "source": "formula",
                "caching": function_caching,
            },
        )

    def execute_user_server_method(
        self,
        session_id: str,
        function_name: str | None,
        request_body: bytes,
        request_headers: dict[str, str],
        request_query: dict[str, str],
    ) -> None:
        status = "ok"
        func = self.api_functions.get(function_name)
        try:
            if not func:
                raise ValueError(f"Function {function_name} not found")
            request = APIRequest(
                body=request_body,
                headers=request_headers,
                query=request_query,
            )
            value = func(request)
        except Exception:
            etype, evalue, tb = sys.exc_info()
            value = gsheet_spreadsheet_error_from_python_exception(etype, evalue, tb)
            status = "error"
        _, encoded = encode_for_gsheets(value)
        self.reply_to_client(
            MessageTypes.USER_API_RESPONSE_STREAM,
            {
                "content": json.loads(encoded),
                "status": status,
                "session_id": session_id,
            },
        )

    @property
    def gsheet_service(
        self,
    ) -> Resource | None:
        if not self.in_gs_mode:
            return None
        if self._gsheet_service is None:
            if not self.gsheets_spreadsheet_id:
                raise ValueError("This tyne is not connected to a Google Sheet")
            self._gsheet_service = build(
                "sheets",
                "v4",
                http=gsheets_api.ProxiedHttp(),
                requestBuilder=gsheets_api.request_builder(gsheets_api.Credentials()),
            )

        return self._gsheet_service

    @property
    def named_ranges(self) -> GSheetNamedRanges:
        if self._named_ranges is None:
            self._named_ranges = GSheetNamedRanges()
        return self._named_ranges

    @property
    def gsheet_name_registry(self) -> GSheetNameRegistry:
        if not self.in_gs_mode:
            raise ValueError("This is only available in Google Sheets mode.")
        if self._gsheet_name_registry is None:
            self._gsheet_name_registry = GSheetNameRegistry(
                self.gsheet_service, self.gsheets_spreadsheet_id
            )
        return self._gsheet_name_registry

    def google_sheets_request(
        self, method: str, body: dict, mappable_types: dict
    ) -> None:
        if not self.in_gs_mode:
            raise ValueError("This is only available in Google Sheets mode.")

        body = replace_refs(body, mappable_types, self.gsheet_name_registry)
        service = self.gsheet_service
        assert service
        request = service.spreadsheets().batchUpdate(
            spreadsheetId=self.gsheets_spreadsheet_id,
            body={"requests": [{method: body}]},
        )
        try:
            execute(request)
        except HttpError as e:
            print("Sheets API Error: ", e.error_details, file=sys.stderr)
            raise SheetsAPIError(e.error_details)

    def send_email(self, to: str | list[str], subject: str, body: str) -> None:
        if isinstance(to, str):
            to = [to]
        self.reply_to_client(
            MessageTypes.SEND_EMAIL,
            {
                "to": to,
                "subject": subject,
                "body": body,
            },
        )

    def patch_do_complete(self, kernel: Kernel) -> None:
        old_do_complete = kernel.do_complete

        def do_complete(code: str, cursor_pos: int) -> dict[str, Any]:
            result = old_do_complete(
                code,
                cursor_pos,
            )
            keep = []
            for match, meta in zip(
                result["matches"], result["metadata"]["_jupyter_types_experimental"]
            ):
                identifier = code[: result["cursor_start"]] + match
                keep_this, obj = self.find_qualified_name_in_user_space(identifier)
                keep.append(keep_this)
                if obj is not None and (doc := inspect.getdoc(obj)):
                    meta["docstring"] = doc
            result["matches"] = [m for m, k in zip(result["matches"], keep) if k]
            result["metadata"]["_jupyter_types_experimental"] = [
                m
                for m, k in zip(result["metadata"]["_jupyter_types_experimental"], keep)
                if k
            ]
            if len(result["metadata"]["_jupyter_types_experimental"]) > 1:
                for meta in result["metadata"]["_jupyter_types_experimental"]:
                    if "docstring" in meta:
                        meta["docstring"] = meta["docstring"].split("\n")[0]

            return result

        kernel.do_complete = do_complete  # type: ignore

    def write_secrets_to_fs(self) -> None:
        secrets_path = Path(os.getenv("NEPTYNE_SECRETS_PATH", "/tmp/neptyne-secrets"))
        secrets_path.mkdir(exist_ok=True)
        if header := self._get_header():
            session = NeptyneSessionInfo.from_message_header(header)
            if secrets := session.tyne_secrets:
                for file in secrets_path.iterdir():
                    file.unlink()
                for key, value in secrets.items():
                    with open(secrets_path / key, "w") as f:
                        f.write(value)

    def register_streamlit(self, fn: Callable, metadata: StreamlitAppConfig) -> None:
        if streamlit_server.is_running_in_streamlit():
            return
        if self.cur_streamlit_info:
            raise ValueError("Can only have one streamlit entry point")
        self.cur_streamlit_info = (fn, metadata)

    def set_env(self, env: dict[str, str]) -> None:
        neptyne_env = os.getenv("NEPTYNE_ENV", "").split(",")
        for key in neptyne_env:
            if key:
                try:
                    del os.environ[key]
                except KeyError:
                    pass
        for key, value in env.items():
            os.environ[key] = value
        os.environ["NEPTYNE_ENV"] = ",".join(env.keys())
