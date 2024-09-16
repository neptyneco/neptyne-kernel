from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

from ..cell_address import Address, format_cell
from ..expression_compiler import is_cell_formula
from ..mime_handling import (
    JSONPrimitive,
    output_to_value,
)
from ..mime_types import JSON_MIME_KEY
from ..neptyne_protocol import CellAttribute, MIMETypes
from .jupyter_notebook import Output, OutputType

CODEPANEL_CELL_ID = "00"


def coerce_address(s: str | tuple | Address) -> Address:
    if isinstance(s, str):
        return Address.from_a1_or_str(s)
    if isinstance(s, Address):
        return s
    return Address(*s)


@dataclass
class NotebookCell:
    cell_id: str
    outputs: list[Output] | None = None
    metadata: dict[str, Any] | None = None
    raw_code: str = ""
    compiled_code: str = ""
    execution_count: int = -1

    def output_dict(self) -> list[dict] | None:
        if self.outputs is None:
            return None
        return [output_to_dict(output) for output in self.outputs]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "raw_code": self.raw_code,
            "compiled_code": self.compiled_code,
            "outputs": self.output_dict(),
            "attributes": self.metadata,
            "execution_count": self.execution_count,
        }

    def __repr__(self) -> str:
        return f"{self.cell_id}={self.raw_code}"

    def export(self, omit_outputs: bool = False) -> dict[str, Any]:
        exported: dict[str, Any] = {
            "cell_id": self.cell_id,
            "source": self.raw_code,
            "metadata": self.metadata or {},
            "cell_type": "code",
            "execution_count": self.execution_count,
        }
        if not omit_outputs:
            exported["outputs"] = self.output_dict() or []
        return exported

    @classmethod
    def from_dict(cls, value: dict) -> "NotebookCell":
        value = deepcopy(value)
        if "output" in value and "outputs" not in value:
            del value["output"]
            value["outputs"] = None
        value["outputs"] = (
            [output_from_dict(o) for o in value["outputs"]]
            if value["outputs"]
            else None
        )
        if "attributes" in value:
            value["metadata"] = value.pop("attributes")
        for k in [
            "depends_on",
            "feeds_into",
            "calculated_by",
            "execution_policy",
            "next_execution_time",
            "execute_count",
            "cell_type",
            "mime_type",
        ]:
            if k in value:
                del value[k]
        return cls(**value)


@dataclass
class SheetCell:
    cell_id: Address
    output: Output | JSONPrimitive = None
    raw_code: str = ""
    compiled_code: str = ""
    attributes: dict[str, Any] | None = None
    mime_type: str | None = None
    depends_on: set[Address] = field(default_factory=set)
    feeds_into: set[Address] = field(default_factory=set)
    calculated_by: Address | None = None
    execution_policy: int = -1
    next_execution_time: float = 0

    def output_dict(self) -> list[dict] | None:
        if self.output is None:
            return None
        if not isinstance(self.output, Output):
            return [
                {"data": {JSON_MIME_KEY: self.output}, "output_type": "execute_result"}
            ]
        return [output_to_dict(self.output)]

    def iterate_outputs_data(self) -> Iterator[dict]:
        if isinstance(self.output, Output) and self.output.data:
            yield self.output.data

    def __repr__(self) -> str:
        formatted_cell_id = format_cell(self.cell_id.column, self.cell_id.row)
        if self.raw_code:
            return f"{formatted_cell_id}={self.raw_code}"
        else:
            if isinstance(self.output, Output):
                value = (
                    output_to_value(self.output.data)
                    if self.output.data
                    else repr(self.output)
                )
            else:
                value = repr(self.output)
            return f"{formatted_cell_id}:{value}"

    def editable_in_app_mode(self) -> bool:
        return not is_cell_formula(self.raw_code) and (
            bool(self.feeds_into) or bool(self.calculated_by)
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "cell_id": self.cell_id.to_coord(),
            "raw_code": self.raw_code,
            "compiled_code": self.compiled_code,
            "attributes": self.attributes,
            "outputs": self.output_dict(),
            "depends_on": [a.to_coord() for a in self.depends_on],
            "feeds_into": [a.to_coord() for a in self.feeds_into],
            "execution_policy": self.execution_policy,
            "next_execution_time": self.next_execution_time,
        }
        if self.calculated_by:
            value["calculated_by"] = self.calculated_by.to_coord()
        return value

    def attribute_dict(self) -> dict[str, str]:
        attributes = {**self.attributes} if self.attributes else {}
        attributes[CellAttribute.EXECUTION_POLICY.value] = str(self.execution_policy)
        # in app mode, default protected flips to true:
        if (
            self.editable_in_app_mode()
            and str(attributes.get(CellAttribute.IS_PROTECTED.value)) != "1"
        ):
            attributes[CellAttribute.IS_PROTECTED.value] = "0"
        return attributes

    def export(self, compact: bool = False) -> dict[str, Any] | list:
        if compact and (simple := self.export_simple()):
            return simple
        return self.export_full()

    def export_simple(self) -> list | None:
        if self.execution_policy != -1 or len(self.depends_on) > 1 or self.attributes:
            return None

        if self.editable_in_app_mode():
            return None

        value: Any
        if isinstance(self.output, Output):
            if not represents_simple_value(self.output):
                return None
            assert self.output.data is not None
            value = self.output.data.get(JSON_MIME_KEY)
        else:
            value = self.output

        coord = self.cell_id.to_coord()
        if value == self.raw_code:
            return [coord, value]
        return [
            coord,
            value,
            self.raw_code,
        ]

    def export_full(self) -> dict[str, Any]:
        return {
            "cellId": self.cell_id.to_coord(),
            "code": self.raw_code,
            "outputs": self.output_dict(),
            "attributes": self.attribute_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict, copy_dict: bool = True) -> "SheetCell":
        if copy_dict:
            value = deepcopy(value)

        value["depends_on"] = set(
            coerce_address(s) for s in value.get("depends_on", ())
        )
        value["feeds_into"] = set(
            coerce_address(s) for s in value.get("feeds_into", ())
        )
        if calculated_by := value.get("calculated_by"):
            value["calculated_by"] = coerce_address(calculated_by)

        value.setdefault("compiled_code", "")
        value.setdefault("attributes", {})
        value.setdefault("execution_policy", -1)

        if "output" in value and "outputs" not in value:
            del value["output"]
            value["outputs"] = None

        if isinstance(value["outputs"], list):
            value.setdefault("raw_code", "")
            value["output"] = (
                output_from_dict(value["outputs"][0]) if value["outputs"] else None
            )
        else:
            value["output"] = value["outputs"]
            if value["output"] is None or calculated_by:
                value["raw_code"] = ""
            else:
                value["raw_code"] = str(value["output"])
        if "format" in value:
            value.setdefault("attributes", {})["class"] = value.pop("format")

        if not isinstance(value["cell_id"], Address):
            value["cell_id"] = coerce_address(value["cell_id"])

        for k in ["cell_type", "execute_count", "execution_count", "outputs"]:
            if k in value:
                del value[k]
        return cls(**value)


def output_to_dict(output: Output) -> Any:
    """Output's have a data field that can be arbitrary data if the key ends on .json."""
    d = {
        k: v
        for k, v in output.__dict__.items()
        if not k.startswith("_") and v is not None
    }
    if isinstance(d["output_type"], Enum):
        d["output_type"] = d["output_type"].value
    return d


def output_from_dict(d: dict[str, Any]) -> Output:
    d = d.copy()
    d["output_type"] = OutputType(d["output_type"])
    for k in (
        "metadata",
        "name",
        "text",
        "ename",
        "evalue",
        "traceback",
        "data",
        "execution_count",
    ):
        d.setdefault(k, None)
    return Output(**d)


@dataclass
class CellMetadata:
    attributes: dict[str, Any] = field(default_factory=dict)

    raw_code: str = field(default="")
    compiled_code: str = field(default="")
    mime_type: str | None = field(default=None)

    execution_policy: int = -1
    next_execution_time: float = 0
    output: Output | JSONPrimitive | None = field(default=None)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CellMetadata":
        output = d.pop("output", None)
        if isinstance(output, dict):
            output = output_from_dict(output)
        # TODO(jack): why?
        d = {
            k: v
            for k, v in d.items()
            if k
            in (
                "attributes",
                "raw_code",
                "compiled_code",
                "mime_type",
                "execution_policy",
                "next_execution_time",
            )
        }
        return cls(
            **d,
            output=output,
        )

    def is_input_widget(self) -> bool:
        return bool(
            self.output
            and isinstance(self.output, Output)
            and self.output.data
            and self.output.data.get(
                MIMETypes.APPLICATION_VND_NEPTYNE_WIDGET_V1_JSON.value
            )
        )


def represents_simple_value(outputs_or_meta: Output | CellMetadata) -> bool:
    if isinstance(outputs_or_meta, CellMetadata):
        if not isinstance(outputs_or_meta.output, Output):
            return True
        output = outputs_or_meta.output
    else:
        output = outputs_or_meta
    if output.output_type != OutputType.EXECUTE_RESULT:
        return False
    if not output.data:
        return True
    if len(output.data) > 1:
        return False
    return JSON_MIME_KEY in output.data
