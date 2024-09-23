from pytest import approx

from ..spreadsheet_error import SpreadsheetError


def approx_or_error(result):
    return result if isinstance(result, SpreadsheetError) else approx(result, rel=1e-3)
