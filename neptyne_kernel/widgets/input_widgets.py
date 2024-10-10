import collections.abc
from typing import Any, Callable

from ..spreadsheet_datetime import SpreadsheetDateTime
from ..widgets.register_widget import register_widget
from .base_widget import (
    BaseWidget,
    ColorMixins,
    StrEnum,
    decode_callable,
    widget_field,
)

BUTTON_GSHEET_ATTRIBUTES = ["background_color", "text_color", "width", "height"]


@register_widget(category="Input")
class Button(BaseWidget, ColorMixins):
    """Simple button that calls its action when clicked"""

    caption: str = widget_field("Button's caption text", default="")
    action: Callable | None = widget_field("Function to run on click", default=None)
    disabled: bool = widget_field("Disabled from clicks", default=False)
    is_spinning: bool = widget_field("Show a spinner while working", default=False)

    def __post_init__(self) -> None:
        super().__init__()
        # Support for just supplying an action:
        if callable(self.caption) and self.action is None:
            self.action = self.caption
            self.caption = self.caption.__name__  # type: ignore


@register_widget(category="Input")
class Dropdown(BaseWidget, ColorMixins):
    """Select from a list of choices. The source can be a spreadsheet range"""

    choices: list[str] = widget_field("Option choices", default_factory=list)
    action: Callable[[str], None] | None = widget_field(
        "Function to run on value change", default=None
    )
    default_value: str = widget_field("Initial value", default="")
    disabled: bool = widget_field("Disabled from clicks", default=False)
    multi_select: bool = widget_field("Allow selecting multiple options", default=False)

    def _check_valid_value(self, value: Any) -> bool:
        if self.multi_select:
            return isinstance(value, collections.abc.Sequence) and not isinstance(
                value, str
            )
        return value in self.choices

    def __post_init__(self) -> None:
        if not self._check_valid_value(self.value) and len(self.choices):
            if self.multi_select:
                self.set_value(
                    [self.choices[0]],
                    do_trigger=False,
                )
            else:
                self.set_value(self.choices[0], do_trigger=False)


@register_widget(category="Input")
class Slider(BaseWidget, ColorMixins):
    """Slide between a value of 0 and 100"""

    action: Callable[[float], None] | None = widget_field(
        "Function to run on value change", default=None
    )
    default_value: float = widget_field("Initial value", default=50.0)
    disabled: bool = widget_field("Disabled from clicks", default=False)

    def _check_valid_value(self, value: Any) -> bool:
        return isinstance(value, float | int) and 0 <= value <= 100


@register_widget(category="Input")
class Checkbox(BaseWidget, ColorMixins):
    """Turns a cell into a checkbox"""

    action: Callable[[bool], None] | None = widget_field(
        "Function to run on value change", default=None
    )
    default_value: bool = widget_field("Initial value", default=False)
    disabled: bool = widget_field("Disabled from clicks", default=False)

    def _check_valid_value(self, value: Any) -> bool:
        return isinstance(value, bool)

    def __post_init__(self) -> None:
        super().__init__()
        # Support for just supplying a default_value.
        if isinstance(self.action, bool) and not self.default_value:
            self.default_value = self.action
            self.action = None


@register_widget(category="Input")
class Autocomplete(BaseWidget, ColorMixins):
    """Pick from a long list by typing"""

    choices: list[str] | Callable[[str], list[str]] = widget_field(
        "Choices", default_factory=list
    )
    action: Callable[[str], None] | None = widget_field(
        "Function to run on value change", default=None
    )
    default_value: str = widget_field("Initial value", default="")
    disabled: bool = widget_field("Disabled from clicks", default=False)

    def get_choices(self, query: str) -> list[str]:
        if isinstance(self.choices, str):
            self.choices = decode_callable(self.choices)
        if self.choices is None:
            return []
        if callable(self.choices):
            return self.choices(query)
        return self.choices


@register_widget(category="Input")
class DateTimePicker(BaseWidget):
    """Pick a date or time"""

    class DateTimePickerType(StrEnum):
        DATETIME = "datetime"
        DATE_ONLY = "date"
        TIME_ONLY = "time"

    action: Callable[[SpreadsheetDateTime], None] | None = widget_field(
        "Function to run on value change", default=None
    )
    default_value: SpreadsheetDateTime | None = widget_field(
        "Initial value", default=None
    )
    picker_type: DateTimePickerType = widget_field(
        "Pick date, time or both",
        default=DateTimePickerType.DATETIME,
    )
    disabled: bool = widget_field("Disabled from clicks", default=False)
