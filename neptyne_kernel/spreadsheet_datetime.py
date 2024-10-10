from datetime import date, datetime, time, timedelta
from operator import __add__, __sub__
from typing import TYPE_CHECKING, Any, Callable, Optional

from dateutil.parser import parse
from zoneinfo import ZoneInfo

from .cell_api import CellApiMixin
from .primitives import Empty
from .spreadsheet_error import VALUE_ERROR, SpreadsheetError

if TYPE_CHECKING:
    from .dash_ref import DashRef

SEC_IN_DAY = 86400
MSEC_IN_DAY = SEC_IN_DAY * 1000
MICROSEC_IN_DAY = MSEC_IN_DAY * 1000
SEC_IN_HOUR = 3600
EXCEL_MSECOND = 1 / MSEC_IN_DAY
EXCEL_SECOND = 1 / SEC_IN_DAY

EPOCH_FIRST_YEAR = 1900

INIT_DATE = date(EPOCH_FIRST_YEAR, 1, 1)
FEB_28_1900 = date(EPOCH_FIRST_YEAR, 2, 28)


def excel2time(number: int | float) -> time:
    td = timedelta(milliseconds=number % 1 / EXCEL_MSECOND)
    return time(
        td.seconds // SEC_IN_HOUR,
        td.seconds // 60 % 60,
        td.seconds % 60,
        td.microseconds,
        tzinfo=SpreadsheetTime.TZ_INFO,
    )


def excel2date(number: int | float) -> date:
    days = date(EPOCH_FIRST_YEAR - 1, 12, 31).toordinal() + int(number)
    if number > 59:
        days -= 1
    elif number < 0:
        days += 1
    return date.fromordinal(days)


def excel2datetime(number: int | float) -> datetime:
    dt = excel2date(number)
    if isinstance(dt, SpreadsheetError):
        return dt
    t = excel2time(number)
    return datetime.combine(dt, t, tzinfo=SpreadsheetDateTime.TZ_INFO)


def correct_number_of_days(_date: date) -> int:
    if _date > FEB_28_1900:
        return 1
    elif _date < INIT_DATE:
        return -1
    return 0


class SpreadsheetDateTimeBase(CellApiMixin):
    TZ_INFO = ZoneInfo("America/New_York")

    def __new__(
        cls,
        value: int | float | datetime | date | time | str | None = None,
        ref: Optional["DashRef"] = None,
    ) -> "SpreadsheetDateTimeBase":
        if cls is SpreadsheetDateTimeBase:
            raise TypeError(f"Only children of '{cls.__name__}' may be instantiated")
        if isinstance(value, int | float):
            _value = value
        else:
            if isinstance(value, datetime | date | time):
                dt = value
            elif isinstance(value, str):
                dt = parse(value).replace(tzinfo=cls.TZ_INFO)
            elif value is None:
                dt = datetime.now(tz=cls.TZ_INFO)
            else:
                raise ValueError(f"Unsupported type '{value.__class__.__name__}'")
            _value = cls.serial(dt)  # type: ignore
        x = cls.__bases__[1].__new__(cls, _value)  # type: ignore
        x.ref = ref  # type: ignore
        return x  # type: ignore

    def __copy__(self) -> "SpreadsheetDateTimeBase":
        return self

    def __deepcopy__(self, memo: Any) -> "SpreadsheetDateTimeBase":
        return self

    def apply_operator(self, other: Any, op: Callable) -> Any:
        try:
            num_type = self.__class__.__bases__[1]
            if isinstance(other, Empty):
                other = 0
            left = num_type(self)
            right = num_type(other)
        except (ValueError, TypeError):
            return VALUE_ERROR
        return op(left, right)

    def __add__(self, other: Any) -> Any:
        return self.apply_operator(other, __add__)

    def __radd__(self, other: Any) -> Any:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Any:
        return self.apply_operator(other, __sub__)

    def __rsub__(self, other: Any) -> Any:
        return -self.__sub__(other)


class SpreadsheetDateTime(SpreadsheetDateTimeBase, float):
    @staticmethod
    def serial(dt: datetime | date) -> float:
        if isinstance(dt, datetime) and dt.tzinfo is None:
            dt = dt.replace(tzinfo=SpreadsheetDateTime.TZ_INFO)
        if not isinstance(dt, datetime):
            dt = datetime.combine(dt, time(0, tzinfo=SpreadsheetDateTime.TZ_INFO))

        delta = dt - datetime(
            EPOCH_FIRST_YEAR - 1, 12, 31, tzinfo=SpreadsheetDateTime.TZ_INFO
        )
        days = (
            delta.days
            + (delta.seconds / SEC_IN_DAY)
            + delta.microseconds / MICROSEC_IN_DAY
        )
        return days + correct_number_of_days(dt.date())


class SpreadsheetTime(SpreadsheetDateTimeBase, float):
    @staticmethod
    def serial(dt: time | datetime) -> float:
        if isinstance(dt, datetime):
            dt = dt.time()
        return (
            (dt.hour * SEC_IN_HOUR + dt.minute * 60 + dt.second) * 1e6 + dt.microsecond
        ) / MICROSEC_IN_DAY


class SpreadsheetDate(SpreadsheetDateTimeBase, int):
    @staticmethod
    def serial(dt: time | datetime) -> int:
        if isinstance(dt, datetime):
            dt = dt.date()  # type: ignore
        return (dt - date(EPOCH_FIRST_YEAR - 1, 12, 31)).days + correct_number_of_days(  # type: ignore
            dt  # type: ignore
        )
