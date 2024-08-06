from typing import Any, overload


class InlineWrapper:
    def __init__(self, value: Any) -> None:
        self.value = value


class WithSourceMixin:
    def __init__(self, value: int | float | str, source_info: dict[str, Any]) -> None:
        self.value = value
        self.source_info = {k: v for k, v in source_info.items() if k != "value"}


class IntWithSource(WithSourceMixin, int):
    def __new__(cls, value: int, source_info: dict[str, Any]) -> "IntWithSource":
        return int.__new__(cls, value)

    def __init__(self, value: int, source_info: dict[str, Any]) -> None:
        WithSourceMixin.__init__(self, value, source_info)


class FloatWithSource(WithSourceMixin, float):
    def __new__(cls, value: float, source_info: dict[str, Any]) -> "FloatWithSource":
        return float.__new__(cls, value)

    def __init__(self, value: float, source_info: dict[str, Any]) -> None:
        WithSourceMixin.__init__(self, value, source_info)


class StrWithSource(WithSourceMixin, str):
    def __new__(cls, value: str, source_info: dict[str, Any]) -> "StrWithSource":
        return str.__new__(cls, value)

    def __init__(self, value: str, source_info: dict[str, Any]) -> None:
        WithSourceMixin.__init__(self, value, source_info)


@overload
def create_with_source(value: int, source_info: dict[str, Any]) -> IntWithSource: ...


@overload
def create_with_source(
    value: float, source_info: dict[str, Any]
) -> FloatWithSource: ...


@overload
def create_with_source(value: str, source_info: dict[str, Any]) -> StrWithSource: ...


def create_with_source(
    value: int | float | str, source_info: dict[str, Any]
) -> IntWithSource | FloatWithSource | StrWithSource:
    if isinstance(value, int):
        return IntWithSource(value, source_info)
    elif isinstance(value, float):
        return FloatWithSource(value, source_info)
    elif isinstance(value, str):
        return StrWithSource(value, source_info)
    else:
        raise TypeError("Value must be int, float, or str")
