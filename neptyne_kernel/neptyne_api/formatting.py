import json

from pygments import highlight
from pygments.lexers.python import PythonLexer

from ..code_highlight import GSheetsTextRunFormatter
from . import google


def highlight_code_in_cell(cell):
    """Apply code highlighting to a cell"""
    code = str(cell).rstrip()
    text_format_runs = json.loads(
        highlight(code, PythonLexer(), GSheetsTextRunFormatter())
    )
    google.sheets.update_cells(
        cell,
        rows=[{"values": [{"textFormatRuns": text_format_runs}]}],
        fields="textFormatRuns",
    )


__all__ = ["highlight_code_in_cell"]
