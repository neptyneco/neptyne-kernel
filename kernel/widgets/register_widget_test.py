import json
from dataclasses import field
from enum import Enum
from typing import Callable

from ..cell_range import CellRange
from ..neptyne_protocol import WidgetParamType
from ..util import list_like
from .base_widget import BaseWidget
from .color import Color
from .output_widgets import Scatter
from .register_widget import (
    get_widget_param_non_union_type,
    get_widget_param_type,
    is_union_type,
    list_like_type,
    register_widget,
    widget_registry,
)


class DummyEnum(Enum):
    VAL1 = 1
    VAL2 = 2
    VAL3 = 3


def test_widget_type_parse():
    assert get_widget_param_type(float) == WidgetParamType.FLOAT
    assert get_widget_param_type(float | None) == WidgetParamType.FLOAT
    assert get_widget_param_type(int) == WidgetParamType.INT
    assert get_widget_param_type(list[float]) == WidgetParamType.LIST
    assert get_widget_param_type(list[float] | None) == WidgetParamType.LIST
    assert get_widget_param_type(dict[str, str]) == WidgetParamType.DICT
    assert get_widget_param_type(str) == WidgetParamType.STRING
    assert get_widget_param_type(str | None) == WidgetParamType.STRING
    assert get_widget_param_type(bool) == WidgetParamType.BOOLEAN
    assert get_widget_param_type(DummyEnum) == WidgetParamType.ENUM
    assert get_widget_param_type(Callable) == WidgetParamType.FUNCTION
    assert (
        get_widget_param_type(Callable[[float, str], int]) == WidgetParamType.FUNCTION
    )
    assert get_widget_param_type(Callable | None) == WidgetParamType.FUNCTION

    class OtherClass:
        Test = ""

    assert get_widget_param_non_union_type(float) == WidgetParamType.FLOAT
    assert get_widget_param_non_union_type(int) == WidgetParamType.INT
    assert get_widget_param_non_union_type(str) == WidgetParamType.STRING
    assert get_widget_param_non_union_type(bool) == WidgetParamType.BOOLEAN
    assert get_widget_param_non_union_type(DummyEnum) == WidgetParamType.ENUM
    assert (
        get_widget_param_non_union_type(Callable[[int], str])
        == WidgetParamType.FUNCTION
    )
    assert get_widget_param_non_union_type(list[int]) == WidgetParamType.LIST
    assert get_widget_param_non_union_type(Color) == WidgetParamType.COLOR
    assert get_widget_param_non_union_type(dict[str, str]) == WidgetParamType.DICT
    assert get_widget_param_non_union_type(OtherClass) == WidgetParamType.OTHER


@register_widget(category="Test")
class DummyWidget(BaseWidget):
    """Test widget for unit test"""

    test_float: float = field(
        metadata={"description": "test"},
    )
    test_list: list[list[float]] = field(
        metadata={"description": "test"},
    )
    test_callable: Callable[[float, str], int] = field(
        metadata={"description": "test"},
    )
    test_dict: dict[str, str] = field(
        metadata={"description": "test"},
    )

    test_optional_callable: Callable[[str, str], int] | None = field(
        metadata={"description": "test"}, default=None
    )
    test_optional_enum: DummyEnum | None = field(
        metadata={"description": "test"}, default=DummyEnum.VAL1
    )
    test_optional_float: float | None = field(
        metadata={"description": "test"}, default=0
    )


def test_register_widget():
    registered_test_widget = widget_registry.widgets["DummyWidget"]
    assert registered_test_widget.category == "Test"
    assert registered_test_widget.description == "Test widget for unit test"
    assert registered_test_widget.name == "DummyWidget"
    for param in registered_test_widget.params:
        if param.type == WidgetParamType.ENUM:
            assert param.enum_values == {"VAL1": 1, "VAL2": 2, "VAL3": 3}
        if param.name == "test_callable" or param.name == "test_optional_callable":
            assert param.type == WidgetParamType.FUNCTION
        if param.name == "test_optional_enum":
            assert param.type == WidgetParamType.ENUM
        if param.name == "test_dict":
            assert param.type == WidgetParamType.DICT
        if (
            param.name == "test_optional_enum"
            or param.name == "test_optional_float"
            or param.name == "test_optional_callable"
        ):
            assert param.optional is True
        else:
            assert param.optional is False


def test_widgets_with_enums_can_serialize():
    scatter = Scatter(x=[1, 2, 3], y=[4, 5, 6], trendline=Scatter.TrendlineType.POLY)

    s = json.dumps(scatter)
    scatter2 = Scatter(**json.loads(s))

    assert scatter2.trendline == Scatter.TrendlineType.POLY


def test_helpers():
    assert list_like([1, 2, 3])
    assert list_like(CellRange([1, 2, 3, 4]))
    assert list_like_type(CellRange)
    assert list_like_type(list[int])
    assert not list_like({"a": 1, "b": 2})
    assert not list_like("abcd")
    assert not list_like_type(str)
    assert not list_like_type(dict[str, int])
    assert is_union_type(int | float)
    assert is_union_type(int | None)
    assert not is_union_type(list[str])
