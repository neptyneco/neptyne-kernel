import inspect
import re
from copy import copy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

from .api_ref import ApiRef
from .neptyne_protocol import (
    BorderType,
    CellAttribute,
    LineWrap,
    NumberFormat,
    TextAlign,
    TextStyle,
)
from .widgets.color import Color

if TYPE_CHECKING:
    from server.kernel_runtime import User

UNCONNECTED_MSG = " object not associated directly with a sheet"

if TYPE_CHECKING:
    from .spreadsheet_datetime import SpreadsheetDateTime


@dataclass
class CellEvent:
    cell: "CellApiMixin"
    user: Optional["User"]
    time: "SpreadsheetDateTime"


def camel_to_snake(camel: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel).lower()


ATTRIBUTE_DEFINITIONS = [
    (attr, date_type, camel_to_snake(attr.value))
    for attr, date_type in [
        (CellAttribute.TEXT_ALIGN, TextAlign),
        (CellAttribute.VERTICAL_ALIGN, TextAlign),
        (CellAttribute.LINE_WRAP, LineWrap),
        (CellAttribute.NOTE, str),
        (CellAttribute.LINK, str),
        (CellAttribute.FONT, str),
        (CellAttribute.FONT_SIZE, int),
        (CellAttribute.ROW_SPAN, int),
        (CellAttribute.COL_SPAN, int),
        (CellAttribute.COLOR, Color),
        (CellAttribute.BACKGROUND_COLOR, Color),
        (CellAttribute.TEXT_STYLE, set[TextStyle]),
        (CellAttribute.BORDER, set[BorderType]),
    ]
]


GSHEETS_WORKING_ATTRIBUTES = {
    CellAttribute.COLOR,
    CellAttribute.BACKGROUND_COLOR,
    CellAttribute.NOTE,
    # Font size works, but font color/font_size clear each other.
    # CellAttribute.FONT_SIZE,
}

GSHEETS_NON_WORKING_NAMES = (
    {
        item
        for tup in {
            (f"get_{camel_to_snake(attr.value)}", f"set_{camel_to_snake(attr.value)}")
            for attr in {attr for attr, _, _ in ATTRIBUTE_DEFINITIONS}
            - GSHEETS_WORKING_ATTRIBUTES
        }
        for item in tup
    }.union(
        {
            "get_custom_number_format",
            "set_custom_number_format",
            "get_number_format",
            "set_number_format",
            "get_attributes",
        }
    )
    # Getters don't work in general for gsheets.
    .union(f"get_{camel_to_snake(attr.value)}" for attr in GSHEETS_WORKING_ATTRIBUTES)
)

TOOLTIP_RENAME = {
    CellAttribute.COLOR: "text color",
    CellAttribute.BACKGROUND_COLOR: "background color",
}


def _add_attribute_accessors(cls: type) -> type:
    for attr, data_type, prop_name in ATTRIBUTE_DEFINITIONS:

        def create_accessors(attr: CellAttribute, data_type: type) -> tuple:
            def color_setter(self: Any, *args: Any) -> None:
                self._set_attribute(attr, Color(*args).webcolor)

            def color_getter(self: Any) -> Color | None:
                def as_color(value: str | None) -> Color | None:
                    if value is None:
                        return None
                    return Color.from_webcolor(value)

                return self._get_attribute(attr.value, modifier=as_color)

            def set_setter(self: Any, value: set[str] | str) -> None:
                if isinstance(value, str):
                    value = {value}
                if issubclass(data_type.__args__[0], Enum):  # type: ignore
                    supported = {s.value for s in data_type.__args__[0]}  # type: ignore
                    if not value.issubset(supported):
                        raise ValueError(f"{value} is not a valid {attr.value}")
                self._set_attribute(attr, " ".join(value))

            def set_getter(self: Any) -> set[str]:
                def as_set(value: str) -> set[str]:
                    return set(value.split(" ")) if value else set()

                return self._get_attribute(attr.value, modifier=as_set)

            def setter(self: Any, value: str | int | float) -> None:
                if issubclass(data_type, Enum):
                    supported = {s.value for s in data_type}
                    if value not in supported:
                        raise ValueError(f"{value} is not a valid {attr.value}")
                self._set_attribute(attr, value)

            def getter(self: Any) -> str | int | float:
                return self._get_attribute(attr.value)

            if data_type == Color:
                return color_setter, color_getter
            elif getattr(data_type, "__origin__", None) is set:
                return set_setter, set_getter
            else:
                return setter, getter

        setter, getter = create_accessors(attr, data_type)
        setter_name = f"set_{prop_name}"
        getter_name = f"get_{prop_name}"
        setter.__name__ = setter.__qualname__ = setter_name
        getter.__name__ = getter.__qualname__ = getter_name
        setattr(cls, setter_name, setter)
        setattr(cls, getter_name, getter)

        # Match web documentation to work works in gsheets
        if attr not in GSHEETS_WORKING_ATTRIBUTES:
            setter.__doc__ = "@private: Disabled for web documentation"
            getter.__doc__ = "@private: Disabled for web documentation"
        else:
            attribute = TOOLTIP_RENAME.get(attr, attr.value)
            setter.__doc__ = f"Set the {attribute} of the cell"
            getter.__doc__ = "@private: Disabled for web documentation"

        if data_type == Color:
            color_tooltip = (
                "\n\nColor may be specified in a variety of formats:\n\n"
                + f"1. As three integers representing the red, green, and blue components of the color, e.g. `{setter_name}(255, 0, 0)`\n"
                + f"2. As a list or tuple of three integers, e.g. `{setter_name}([255, 0, 0])`\n"
                + f"3. As a hex color string, e.g. `{setter_name}('#FF0000')`\n"
                + f"4. As a named color e.g. `{setter_name}('red')`"
            )
            setter.__doc__ += color_tooltip
    return cls


InputColor = int | str | list[int] | tuple[int]


@_add_attribute_accessors
class CellApiMixin:
    ref: ApiRef | None
    """@private"""

    def _set_attributes(self, attributes: dict[CellAttribute, str]) -> None:
        if self.ref is None:
            raise ValueError("can't format" + UNCONNECTED_MSG)
        self.ref.set_attributes(attributes)

    def _set_attribute(self, attribute: CellAttribute, attribute_value: str) -> None:
        return self._set_attributes({attribute: attribute_value})

    @staticmethod
    def _webcolor(blue: int, green: int, red: int) -> str:
        return f"#{red:02x}{green:02x}{blue:02x}".upper()

    def set_render_size(self, width: int, height: int) -> None:
        """Change the size of an object anchored in this cell.

        For example, if B2 contains an Image, you can change the size of
        the image by calling `B2.set_render_size(100, 100)`.
        """
        self._set_attributes(
            {
                CellAttribute.RENDER_WIDTH: str(width),
                CellAttribute.RENDER_HEIGHT: str(height),
            }
        )

    def get_render_size(
        self,
    ) -> tuple[int, int] | list[tuple[int, int]] | list[list[tuple[int, int]]]:
        """Get the size of an object anchored in this cell."""
        from .widgets.output_widgets import (
            DEFAULT_OUTPUT_WIDGET_HEIGHT,
            DEFAULT_OUTPUT_WIDGET_WIDTH,
        )

        render_width = self._get_attribute(
            CellAttribute.RENDER_WIDTH.value, default_value=DEFAULT_OUTPUT_WIDGET_WIDTH
        )
        render_height = self._get_attribute(
            CellAttribute.RENDER_HEIGHT.value,
            default_value=DEFAULT_OUTPUT_WIDGET_HEIGHT,
        )
        if isinstance(render_width, list):
            if isinstance(render_width[0], list):
                return [
                    [
                        (int(width), int(height))  # type: ignore
                        for width, height in zip(width_row, height_row)  # type: ignore
                    ]
                    for width_row, height_row in zip(render_width, render_height)  # type: ignore
                ]
            else:
                return [
                    (int(width), int(height))  # type: ignore
                    for width, height in zip(render_width, render_height)  # type: ignore
                ]
        return int(render_width), int(render_height)  # type: ignore

    def set_number_format(
        self, number_type: str, sub_format: str | None = None
    ) -> None:
        """@private"""
        supported = {number_type.value for number_type in NumberFormat}
        if number_type not in supported:
            raise ValueError(f"{number_type} is not a valid format")
        if sub_format:
            number_type += "-" + sub_format
        self._set_attribute(CellAttribute.NUMBER_FORMAT, number_type)

    def get_number_format(
        self, number_type: str | None = None
    ) -> list[list[str | None]] | list[str | None] | str | None:
        """@private"""

        def as_number_format(format: str) -> str:
            assert number_type is not None
            if format and format.startswith(number_type + "-"):
                return format[len(number_type) + 1 :]
            return format

        return self._get_attribute(
            CellAttribute.NUMBER_FORMAT.value,
            modifier=as_number_format if number_type is not None else None,
        )

    def set_custom_number_format(self, format: str) -> None:
        """@private"""
        return self.set_number_format(NumberFormat.CUSTOM.value, format)

    def get_custom_number_format(
        self,
    ) -> list[list[str | None]] | list[str | None] | str | None:
        """@private"""
        return self.get_number_format(NumberFormat.CUSTOM.value)

    def clear(self) -> None:
        """Clear the cell's value"""
        if self.ref is None:
            raise ValueError("can't clear" + UNCONNECTED_MSG)
        self.ref.clear()

    def is_empty(self) -> bool:
        """Check if the cell is empty"""
        return False

    @property
    def xy(self) -> tuple[int, int]:
        """Get the cell's column and row indices respectively as a tuple of integers"""
        if self.ref is None:
            raise ValueError("can't get xy for" + UNCONNECTED_MSG)
        return self.ref.xy()

    def _get_event(self) -> CellEvent:
        from neptyne_kernel.kernel_runtime import get_user

        from .spreadsheet_datetime import SpreadsheetDateTime

        return CellEvent(self, get_user(), SpreadsheetDateTime())

    def get_attributes(self) -> dict[str, Any]:
        """@private Get the cell attributes of a cell. If a range is specified, only the attributes of the origin are returned"""
        if self.ref is None:
            raise ValueError("can't get cell attributes for" + UNCONNECTED_MSG)

        from .dash_ref import DashRef

        if isinstance(self.ref, DashRef):
            address = self.ref.range.origin()
            meta = self.ref.dash.cell_meta.get(address)
            return copy(meta.attributes) if meta else {}

        raise NotImplementedError

    def _get_attribute(
        self,
        attribute: str,
        modifier: Callable[[str], str] | None = None,
        default_value: int | str | None = None,
    ) -> list[list[str | None]] | list[str | None] | str | None:
        if self.ref is None:
            raise ValueError("can't get cell attributes for" + UNCONNECTED_MSG)
        return self.ref.get_attribute(attribute, modifier, default_value)

    def to_datetime(self) -> datetime:
        """Convert the cell's value to a datetime object"""
        raise ValueError(
            "to_datetime is only supported on individual cells containing an int or float"
        )


def set_cell_api_completion_matches(
    msg_content: dict, cell_or_range: str, cursor_pos: int
) -> None:
    members = inspect.getmembers(CellApiMixin)
    if ":" in cell_or_range:
        from .cell_range import CellRange

        members += inspect.getmembers(CellRange)
    type_info = [
        {"start": cursor_pos, "end": cursor_pos, "text": name, "type": "function"}
        for name, member in members
        if not name.startswith("_") and name not in GSHEETS_NON_WORKING_NAMES
    ]
    msg_content["matches"] = [ti["text"] for ti in type_info]
    msg_content["cursor_start"] = cursor_pos
    msg_content["cursor_end"] = cursor_pos
    msg_content["metadata"] = {"_jupyter_types_experimental": type_info}
