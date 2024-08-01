import typing


def list_like(value: typing.Any) -> bool:
    return (
        hasattr(value, "__iter__")
        and not isinstance(value, str)
        and not isinstance(value, dict)
    )
