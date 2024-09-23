from datetime import date, datetime, time


def datetime_to_serial(dt: datetime | date | time) -> float:
    from .spreadsheet_datetime import EPOCH_FIRST_YEAR, SpreadsheetDateTime

    if isinstance(dt, time):
        dt = datetime.combine(
            datetime(EPOCH_FIRST_YEAR - 1, 12, 31, tzinfo=SpreadsheetDateTime.TZ_INFO),
            dt,
        )

    return SpreadsheetDateTime.serial(dt)
