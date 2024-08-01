import dataclasses
import inspect
import re
import types
import typing
from dataclasses import dataclass
from enum import Enum
from operator import attrgetter

from ..neptyne_protocol import (
    WidgetDefinition,
    WidgetParamDefinition,
    WidgetParamType,
    WidgetRegistry,
)
from .color import Color

# Single instance of WidgetRegistry which holds the state in the kernel.
widget_registry = WidgetRegistry({})
widget_class_by_name = {}


def widget_name_from_code(code: str) -> str:
    match = re.match(re.compile("(=)([a-zA-Z]*)( *)\("), code)
    if not match:
        return ""
    return match.group(2)


def list_like_type(t: type) -> bool:
    origin = typing.get_origin(t)
    if origin:
        t = origin

    return hasattr(t, "__iter__") and not issubclass(t, str) and not issubclass(t, dict)


def is_union_type(t: type) -> bool:
    origin = typing.get_origin(t)
    return origin is typing.Union or origin is types.UnionType


def get_widget_param_non_union_type(field_type: type) -> WidgetParamType:
    origin = typing.get_origin(field_type)

    if inspect.isclass(field_type) and issubclass(field_type, Color):
        return WidgetParamType.COLOR
    if field_type == float:
        return WidgetParamType.FLOAT
    if field_type == int:
        return WidgetParamType.INT
    if field_type == str:
        return WidgetParamType.STRING
    if field_type == bool:
        return WidgetParamType.BOOLEAN
    if inspect.isclass(field_type) and issubclass(field_type, Enum):
        return WidgetParamType.ENUM
    if typing.get_origin(field_type) == typing.get_origin(typing.Callable):
        return WidgetParamType.FUNCTION
    if list_like_type(field_type):
        return WidgetParamType.LIST
    if inspect.isclass(origin) and issubclass(origin, dict):
        return WidgetParamType.DICT

    return WidgetParamType.OTHER


def get_widget_param_and_resolved_type(
    field_type: type,
) -> tuple[WidgetParamType, type]:
    if is_union_type(field_type):
        for arg in typing.get_args(field_type):
            if arg is not type(None):
                return get_widget_param_non_union_type(arg), arg
    return get_widget_param_non_union_type(field_type), field_type


def get_widget_param_type(field_type: type) -> WidgetParamType:
    return get_widget_param_and_resolved_type(field_type)[0]


def is_optional(field_type: type) -> bool:
    return is_union_type(field_type) and type(None) in typing.get_args(field_type)


def register_widget(
    **kwargs: typing.Any,
) -> typing.Callable[[type[typing.Any]], type[typing.Any]]:
    def register_widget_fn(cls: typing.Any) -> typing.Any:
        widget_class_by_name[cls.__name__.lower()] = cls
        cls = dataclass(cls)
        widget_name = cls.__qualname__
        widget_description = cls.__doc__
        params = []
        for fieldName, field in cls.__dataclass_fields__.items():
            param_type, resolved_type = get_widget_param_and_resolved_type(field.type)
            if param_type == WidgetParamType.ENUM:
                enum_values = {e.name: e.value for e in resolved_type}  # type: ignore
            else:
                enum_values = {}

            default_value = None
            if (
                field.default == dataclasses.MISSING
                and field.default_factory == dataclasses.MISSING
            ):
                optional = False
                if (
                    param_type == WidgetParamType.STRING
                    or param_type == WidgetParamType.BOOLEAN
                    or param_type == WidgetParamType.ENUM
                ):
                    raise ValueError(
                        f"Widget field: {field} cannot be required because it is a String, Bool or Enum. Add a default_value"
                    )
            else:
                optional = True
                default_value = (
                    field.default
                    if field.default != dataclasses.MISSING
                    else field.default_factory()
                )
                if param_type == WidgetParamType.ENUM and hasattr(
                    default_value, "value"
                ):
                    default_value = default_value.value
                elif param_type == WidgetParamType.COLOR and default_value:
                    default_value = default_value.to_dict()
            metadata = field.metadata or {}
            inline = metadata.get("inline", False)
            category = metadata.get("category", "")
            description = metadata.get("description", "")
            param = WidgetParamDefinition(
                category=category,
                default_value=default_value,
                description=description,
                name=field.name,
                inline=inline,
                optional=optional,
                kw_only=field.kw_only,
                type=param_type,
                enum_values=enum_values,
            )
            params.append(param)
        params = sorted(params, key=attrgetter("kw_only"), reverse=False)
        if kwargs.get("enabled", True):
            widget_registry.widgets[widget_name] = WidgetDefinition(
                category=kwargs.get("category", ""),
                description=widget_description,
                name=widget_name,
                params=params,
            )
        return cls

    return register_widget_fn
