from typing import Callable

from ..dash import Dash

_CRON_WEEKDAYS = [
    "sunday",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
]


def weekly(
    day_of_week: str,
    hour: int,
    minute: int = 0,
    timezone: str | None = None,
    alert_email: str | None = None,
) -> Callable[[Callable], Callable]:
    """Register a function to be called on a weekly schedule\n
    Use as a decorator, for example:
    ```python
    @weekly("monday", 12, 0)
    def my_function():
        A1 = A1 + 1
    ```
    """

    try:
        dow = _CRON_WEEKDAYS.index(day_of_week.lower())
    except ValueError:
        raise ValueError(
            f"Invalid day of week: {day_of_week}. Should be one of {_CRON_WEEKDAYS}"
        )

    schedule = f"{minute} {hour} * * {dow}"

    def decorator(func: Callable) -> Callable:
        Dash.instance().register_cron(schedule, func, timezone, alert_email)

        return func

    return decorator


def daily(
    hour: int,
    minute: int = 0,
    timezone: str | None = None,
    alert_email: str | None = None,
) -> Callable[[Callable], Callable]:
    """Register a function to be called on a daily schedule\n\n
    Use as a decorator, for example:
    ```python
    @daily(12, 0)
    def my_function():
        A1 = A1 + 1
    ```"""

    schedule = f"{minute} {hour} * * *"

    def decorator(func: Callable) -> Callable:
        Dash.instance().register_cron(schedule, func, timezone, alert_email)

        return func

    return decorator
