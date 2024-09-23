import linecache
import re
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generator, cast

import stack_data
from IPython import get_ipython
from IPython.core.ultratb import (
    AutoFormattedTB,
    ListTB,
    SyntaxTB,
)
from IPython.utils import py3compat
from stack_data.core import RepeatedFrames

from .expression_compiler import is_cell, replace_n_with_a1, replace_n_with_a1_match
from .spreadsheet_error import should_hide_errors_from_file

try:
    from IPython.core.ultratb import _format_filename as format_filename
except ImportError:

    def format_filename(
        filename: str, filename_color: str, normal_color: str, lineno: int | None = None
    ) -> str:
        return filename


if TYPE_CHECKING:
    from stack_data.core import FrameInfo

AddressTuple = tuple[int, int, str | int]
RangeTuple = tuple[int, int, int, int, str | int]
CoordinateTuple = AddressTuple | RangeTuple

CELL_RE = re.compile(r"\$?[A-Za-z]+\s?\$?\d+")
RANGE_RE = re.compile(
    r"(\$?[A-Za-z]+\s?\$?\d+|\$?[A-Za-z]+|\$?\d+)\s?:(\$?[A-Za-z]+\s?\$?\d+|\$?[A-Za-z]+|\$?\d+)"
)


def _pygmented_with_ranges_a1(formatter: Any, code: str, ranges: list[Any]) -> str:
    import pygments
    from pygments.lexers import get_lexer_by_name

    class MyLexer(type(get_lexer_by_name("python3"))):  # type: ignore
        def get_tokens(self, text: str) -> Generator[tuple[Any, str], None, None]:
            text = replace_n_with_a1(text, func=replace_n_with_a1_match)
            length = 0
            for ttype, value in super().get_tokens(text):
                if any(start <= length < end for start, end in ranges):
                    ttype = ttype.ExecutingNode
                length += len(value)
                yield ttype, value

    lexer = MyLexer(stripnl=False)
    return pygments.highlight(code, lexer, formatter).splitlines()


stack_data.core._pygmented_with_ranges = _pygmented_with_ranges_a1


def cell_or_range_case_hint(cell_or_range: str) -> str:
    return (
        f"Did you mean '{cell_or_range}'? (Cell refs need to be upper case in Neptyne)"
    )


def _guess_cell_case_error(line: str) -> tuple[bool, str]:
    m = re.search(RANGE_RE, line)
    if not m:
        m = re.search(CELL_RE, line)
    if m:
        matched = m.group(0)
        if (matched_upper := matched.upper()) != matched:
            return True, "".join(matched_upper.split())
    return False, ""


class DashAutoFormattedTB(AutoFormattedTB):
    def _extract_tb(self, tb: list) -> list:
        tb = ListTB._extract_tb(self, tb)
        for i, frame in enumerate(tb):
            tb[i]._line = replace_n_with_a1(frame.line, func=replace_n_with_a1_match)
        return tb

    def structured_traceback(
        self,
        etype: type | None = None,
        value: Any = None,
        tb: Any = None,
        tb_offset: int | None = None,
        number_of_lines_of_context: int = 5,
    ) -> list[str]:
        offset_delta = 0

        def _replace_and_update_offset(m: re.Match) -> str:
            res = replace_n_with_a1_match(m)
            if hasattr(value, "offset") and value.offset >= m.start():
                nonlocal offset_delta
                offset_delta += m.end() - m.start() - len(res)
            return replace_n_with_a1_match(m)

        if hasattr(value, "text"):
            value.text = replace_n_with_a1(value.text, func=_replace_and_update_offset)
        if hasattr(value, "offset"):
            value.offset -= offset_delta

        if (
            issubclass(etype, NameError)  # type: ignore
            and value.name
            and is_cell(cell_addr_upper := value.name.upper())
        ):
            value.args = (
                "\n".join((value.args[0], cell_or_range_case_hint(cell_addr_upper))),
            )

        return AutoFormattedTB.structured_traceback(
            self, etype, value, tb, tb_offset, number_of_lines_of_context
        )

    def get_records(
        self, etb: TracebackType, number_of_lines_of_context: int, tb_offset: int
    ) -> list["FrameInfo"]:
        return [
            r
            for r in AutoFormattedTB.get_records(
                self, etb, number_of_lines_of_context, tb_offset
            )
            if not isinstance(r, RepeatedFrames)
            and not should_hide_errors_from_file(r.filename)
        ]

    def _format_list(
        self, extracted_list: list[tuple[str, int, str, str]]
    ) -> list[str]:
        Colors = self.Colors
        list = []

        for filename, lineno, name, line in extracted_list[:-1]:
            if not filename.startswith("<"):
                continue
            formatted_filename = format_filename(
                filename, Colors.filename, Colors.Normal, lineno=lineno
            )
            items = [f"  {formatted_filename} in {Colors.name}{name}{Colors.Normal}\n"]
            if line:
                items.append(f"    {line.strip()}\n")
                list.append("".join(items))

        # Emphasize the last entry
        filename, lineno, name, line = extracted_list[-1]
        if not filename.startswith("<"):
            return list[-1:]
        formatted_filename = format_filename(
            filename, Colors.filenameEm, Colors.normalEm, lineno=lineno
        )
        items = [
            f"{Colors.normalEm}  {formatted_filename} in {Colors.nameEm}{name}{Colors.normalEm}{Colors.Normal}\n"
        ]
        if line:
            items.append(f"{Colors.line}    {line.strip()}{Colors.Normal}\n")
            return ["".join(items)]
        return list[-1:]


class DashSyntaxTB(SyntaxTB):
    def _format_exception_only(self, etype: type, value: Exception) -> list:
        offset_delta = 0

        def _replace_and_update_offset(m: re.Match) -> str:
            res = replace_n_with_a1_match(m)
            offset = cast(SyntaxError, value).offset
            if offset is not None and offset >= m.start():
                nonlocal offset_delta
                offset_delta += m.end() - m.start() - len(res)
            return replace_n_with_a1_match(m)

        have_filedata = False
        Colors = self.Colors
        lst = []
        stype = py3compat.cast_unicode(Colors.excName + etype.__name__ + Colors.Normal)
        is_cell_case_error = False
        if value is None:
            lst.append(stype + "\n")
        else:
            if issubclass(etype, SyntaxError):
                have_filedata = True
                value = cast(SyntaxError, value)
                if not value.filename:
                    value.filename = "<string>"
                lineno: int | str
                if value.lineno:
                    lineno = value.lineno
                    textline = linecache.getline(value.filename, value.lineno)
                else:
                    lineno = "unknown"
                    textline = ""
                lst.append(
                    "{}  {}{}\n".format(
                        Colors.normalEm,
                        format_filename(
                            value.filename,
                            Colors.filenameEm,
                            Colors.normalEm,
                            lineno=(None if lineno == "unknown" else lineno),
                        ),
                        Colors.Normal,
                    )
                )
                if textline == "":
                    textline = py3compat.cast_unicode(value.text, "utf-8")

                # Replace _N with A1
                textline = replace_n_with_a1(textline, func=_replace_and_update_offset)
                is_cell_case_error, cell_or_range = _guess_cell_case_error(textline)

                if textline is not None:
                    i = 0
                    while i < len(textline) and textline[i].isspace():
                        i += 1
                    lst.append(f"{Colors.line}    {textline.strip()}{Colors.Normal}\n")
                    if value.offset is not None:
                        value.offset -= offset_delta
                        s = "    "
                        for c in textline[i : value.offset - 1]:
                            s += c if c.isspace() else " "
                        lst.append(f"{Colors.caret}{s}^{Colors.Normal}\n")
                    is_cell_case_error, cell_or_range = _guess_cell_case_error(textline)

            try:
                s = value.msg  # type: ignore
            except Exception:
                s = self._some_str(value)
            if s:
                lst.append(f"{stype}{Colors.excName}:{Colors.Normal} {s}\n")
            else:
                lst.append("%s\n" % stype)

        # sync with user hooks
        if have_filedata:
            value = cast(SyntaxError, value)
            ipinst = get_ipython()
            if ipinst is not None:
                ipinst.hooks.synchronize_with_editor(value.filename, value.lineno, 0)

        if is_cell_case_error:
            lst.append(cell_or_range_case_hint(cell_or_range))

        return lst
