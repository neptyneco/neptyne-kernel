from functools import wraps
from typing import Any, Callable

from ..cell_api import CellApiMixin
from ..cell_range import CellRange
from ..dash import APIFunction, Dash
from ..dash_ref import DashRef
from ..primitives import NeptynePrimitive, ProxiedObject

OnValueChangeDecorator = Callable[[Callable[[Any], None]], None]


def _on_value_change(
    apply_on_full_range: bool,
    cell_or_range: CellRange | DashRef | NeptynePrimitive | ProxiedObject,
    *others: CellRange | DashRef | NeptynePrimitive | ProxiedObject,
) -> OnValueChangeDecorator:
    def register_on_value_change_handler(fn: Callable[[Any], None]) -> None:
        @wraps(fn)
        def add_on_value_change_rule(
            arg: CellRange | DashRef | NeptynePrimitive | ProxiedObject,
        ) -> None:
            if isinstance(arg, CellApiMixin):
                assert isinstance(arg.ref, DashRef)
                Dash.instance().append_on_value_change_rule(
                    CellRange(arg.ref), fn, apply_on_full_range
                )
            elif isinstance(arg, DashRef):
                Dash.instance().append_on_value_change_rule(
                    CellRange(arg), fn, apply_on_full_range
                )
            elif isinstance(arg, CellRange):
                Dash.instance().append_on_value_change_rule(
                    arg, fn, apply_on_full_range
                )
            else:
                raise ValueError(
                    f"Invalid format_range: {arg}. Use a cellref or range (e.g. A1 or A1:B2)"
                )

        add_on_value_change_rule(cell_or_range)
        for arg in others:
            add_on_value_change_rule(arg)

    return register_on_value_change_handler


def on_value_change(
    cell_or_range: CellRange | DashRef | NeptynePrimitive,
    *others: CellRange | DashRef | NeptynePrimitive,
) -> OnValueChangeDecorator:
    """Register a function to be called on each value when there is any change in the range"""
    return _on_value_change(False, cell_or_range, *others)


def on_range_change(
    cell_or_range: CellRange | DashRef | NeptynePrimitive,
    *others: CellRange | DashRef | NeptynePrimitive,
) -> OnValueChangeDecorator:
    """Register a function to be called on the full range when there is a change in the range"""
    return _on_value_change(True, cell_or_range, *others)


def api_function(
    func: APIFunction | None = None, route: str | None = None
) -> APIFunction | Callable:
    """Decorator to register a function as an API function"""
    if callable(func):
        Dash.instance().register_api_function(func, None)

        return func
    else:

        def decorator(func: APIFunction) -> APIFunction:
            Dash.instance().register_api_function(func, route)

            return func

        return decorator
