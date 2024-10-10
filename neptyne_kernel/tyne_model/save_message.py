import gzip
from binascii import b2a_base64
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ..cell_address import Address
from ..primitives import Empty
from ..sheet_api import NeptyneSheetCollection
from .cell import CellMetadata, output_to_dict, represents_simple_value
from .dash_graph import DashGraph
from .jupyter_notebook import Output
from .sheet import TyneSheets


def neptyne_encode(obj: Any) -> Any:
    if isinstance(obj, Empty):
        return None
    if isinstance(obj, Output):
        return output_to_dict(obj)
    if (obj_type := type(obj)) != float and issubclass(obj_type, float):
        return float(obj)
    if isinstance(obj, bytes):
        return b2a_base64(obj).decode("ascii")

    raise TypeError


def cell_dict_for_address(
    address: Address,
    cells: dict[int, dict[Address, Any]],
    cell_meta: dict[Address, CellMetadata],
    graph: DashGraph,
) -> dict:
    value = sheet.get(address) if (sheet := cells.get(address.sheet)) else None
    meta = cell_meta.get(address)
    if meta:
        d = {
            "cell_id": address.to_coord(),
            "raw_code": meta.raw_code,
            "compiled_code": meta.compiled_code,
            "attributes": meta.attributes,
            "execution_policy": meta.execution_policy,
            "next_execution_time": meta.next_execution_time,
            "outputs": [output_to_dict(meta.output)]
            if isinstance(meta.output, Output)
            else [
                {
                    "data": {"application/json": value},
                    "output_type": "execute_result",
                }
            ],
        }
    else:
        d = {
            "cell_id": address.to_coord(),
            "outputs": value,
        }
    if address in graph.depends_on:
        d["depends_on"] = [ad.to_coord() for ad in graph.depends_on[address]]
    if address in graph.feeds_into:
        d["feeds_into"] = [ad.to_coord() for ad in graph.feeds_into[address]]
    if address in graph.calculated_by:
        d["calculated_by"] = graph.calculated_by[address].to_coord()
    return d


def tyne_content_dict(
    tyne_sheets: TyneSheets,
    cells: dict[int, dict[Address, Any]],
    cell_meta: dict[Address, CellMetadata],
    graph: DashGraph,
) -> dict[str, Any]:
    """Return a dict that can be used to create a fully-hydrated TyneSheets"""
    as_dict = tyne_sheets.to_dict()
    all_keys: set[Address] = set()
    all_keys.update()
    for sheet, sheet_cells in cells.items():
        all_keys.update(sheet_cells)
    all_keys.update(cell_meta)
    all_keys.update(graph.depends_on)
    all_keys.update(graph.feeds_into)
    all_keys.update(graph.calculated_by)

    cell_dicts: dict[int, list] = defaultdict(list)
    for cell_id in all_keys:
        cell_dicts[cell_id.sheet].append(
            cell_dict_for_address(cell_id, cells, cell_meta, graph)
        )

    for sh in as_dict["sheets"]:
        sh["cells"] = cell_dicts[sh["id"]]

    return as_dict


def json_encode(d: dict[str, Any]) -> bytes:
    try:
        import orjson

        return orjson.dumps(
            d, default=neptyne_encode, option=orjson.OPT_SERIALIZE_NUMPY
        )
    except ImportError:
        import json

        return json.dumps(d, default=neptyne_encode).encode("utf-8")


@dataclass
class V1DashSaveMessage:
    VERSION = 1

    sheets_without_cells: TyneSheets
    cells: dict[int, dict[Address, Any]]
    cell_meta: dict[Address, CellMetadata]
    graph: DashGraph
    next_tick: float | None = None

    @classmethod
    def from_dash_state(
        cls,
        sheet_collection: NeptyneSheetCollection | None,
        cells: dict[int, dict[Address, Any]],
        cell_meta: dict[Address, CellMetadata],
        graph: DashGraph,
        next_tick: float | None = None,
    ) -> "V1DashSaveMessage":
        tyne_sheets = TyneSheets()
        if sheet_collection is not None:
            for sheet in sheet_collection:
                tyne_sheets.sheets[sheet.sheet_id] = sheet.to_serializable()

        tyne_sheets.next_sheet_id = (
            0
            if sheet_collection is None
            else (max((sheet.sheet_id for sheet in sheet_collection), default=0) + 1)
        )
        return cls(
            sheets_without_cells=tyne_sheets,
            cells=cells,
            cell_meta=cell_meta,
            graph=graph,
            next_tick=next_tick,
        )

    def to_dict(self) -> dict[str, Any]:
        cell_meta = []
        omit_value_for_keys = set()
        for key, value in self.cell_meta.items():
            if not represents_simple_value(value):
                omit_value_for_keys.add(key)

            cell_meta.append((key.to_coord(), value))

        sheet_cells = [
            (
                sheet,
                [
                    (key.to_coord(), value)
                    for key, value in cells.items()
                    if key not in omit_value_for_keys
                ],
            )
            for sheet, cells in self.cells.items()
        ]

        return {
            "sheets_without_cells": self.sheets_without_cells.to_dict(),
            "cells": sheet_cells,
            "cell_meta": cell_meta,
            "graph": self.graph.to_dict(),
            "next_tick": self.next_tick,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "V1DashSaveMessage":
        cells = {
            int(sheet_id): {
                Address.from_coord(cell_id): cell for cell_id, cell in sheet_cells
            }
            for sheet_id, sheet_cells in data["cells"]
        }
        meta = {
            Address.from_coord(key): CellMetadata.from_dict(value)
            for key, value in data["cell_meta"]
        }

        return cls(
            sheets_without_cells=TyneSheets.from_dict(data["sheets_without_cells"]),
            cells=cells,
            cell_meta=meta,
            graph=DashGraph.from_dict(data["graph"]),
            next_tick=data.get("next_tick"),
        )

    def to_bytes(self) -> bytes:
        as_dict = self.to_dict()

        return gzip.compress(
            json_encode(as_dict),
            compresslevel=1,
        )
