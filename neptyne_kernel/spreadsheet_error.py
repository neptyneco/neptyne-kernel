import os
import re
from dataclasses import dataclass, fields, replace
from typing import Any, Iterable

from googleapiclient.errors import HttpError

from .neptyne_protocol import MIMETypes

UNSUPPORTED_TYPE_RE = re.compile(
    r"unsupported operand type\(s\) for ([^:]+): '(\w+)' and '(\w+)'"
)

CODE_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
FUTURES_THREAD = "concurrent/futures/thread.py"

TRACEBACK_FILE_HEADER_RE = re.compile(r"File \[\d;32m(?P<path>.+)\:(?P<line>\d+)")

SHOW_FULL_TRACEBACK = os.environ.get("KERNEL_NAMESPACE", "") != "default"


def should_hide_errors_from_file(path: str) -> bool:
    if SHOW_FULL_TRACEBACK:
        return False
    path = os.path.expanduser(path)
    if path.endswith(FUTURES_THREAD):
        # remove also futures so the user doesn't get confused when we have bits running in a thread:
        return True
    if os.path.exists(path):
        return path.startswith(CODE_ROOT)
    return os.path.exists(os.path.join(CODE_ROOT, path[1:]))


def cut_string_stack(frames: list[str]) -> list[str]:
    for idx, frame in enumerate(frames):
        m = re.search(TRACEBACK_FILE_HEADER_RE, frame)
        if m and should_hide_errors_from_file(m.group("path")):
            return frames[:idx] + [frames[-1]]
    return frames


class SheetDoesNotExist(KeyError):
    pass


def parse_unsupported_type_exception(msg: str) -> None | tuple[str, str, str]:
    m = UNSUPPORTED_TYPE_RE.match(msg)
    if not m:
        return None
    return m.groups()  # type: ignore


def analyze_type_error(error: TypeError) -> str | None:
    message = str(error)
    patterns = [
        ("missing", "required positional argument"),
        ("takes", "positional argument", "given"),
        ("unexpected keyword argument",),
        ("got multiple values for argument",),
        ("not callable",),
    ]

    for pattern in patterns:
        if all(term in message for term in pattern):
            return message

    return None


@dataclass
class SpreadsheetError:
    ename: str
    msg: str = ""
    traceback: list | None = None
    line_number: int | None = None

    def _repr_mimebundle_(
        self, include: Iterable[str] = (), exclude: Iterable[str] = (), **kwargs: Any
    ) -> dict[str, Any]:
        data = {
            MIMETypes.APPLICATION_VND_NEPTYNE_ERROR_V1_JSON.value: {
                f.name: getattr(self, f.name) for f in fields(self)
            }
        }
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data

    def with_message(
        self, msg: str, traceback: list | None = None, line_number: int | None = None
    ) -> "SpreadsheetError":
        return replace(self, msg=msg, traceback=traceback, line_number=line_number)

    @classmethod
    def from_python_exception(
        cls,
        etype: type | None,
        evalue: Any,
        traceback: list,
        line_number: int | None = None,
    ) -> "SpreadsheetError":
        if etype == TypeError and (m := parse_unsupported_type_exception(str(evalue))):

            def normalize_type(t: str) -> str:
                if t.lower().startswith("neptyne"):
                    t = t[len("neptyne") :].lower()
                return t

            operator, type1, type2 = m
            type1 = normalize_type(type1)
            type2 = normalize_type(type2)
            return VALUE_ERROR.with_message(
                f"You can't apply the {operator} operator on {type1} and {type2}"
            )
        elif etype == TypeError and (msg := analyze_type_error(evalue)):
            return NA_ERROR.with_message(msg)

        if etype == IndexError:
            err = NA_ERROR
        elif etype == ValueError:
            err = VALUE_ERROR
        elif etype == ZeroDivisionError:
            err = ZERO_DIV_ERROR
        elif etype == GSheetNotAuthorized:
            err = GSHEET_NOT_AUTHORIZED
        elif etype == NameError:
            err = NAME_ERROR
        elif etype == SheetDoesNotExist:
            return REF_ERROR.with_message(str(evalue) + " does not exist")
        elif etype == SyntaxError:
            return SYNTAX_ERROR.with_message(
                "Syntax error:" + evalue.msg, traceback, line_number
            )
        else:
            err = PYTHON_ERROR

        return err.with_message(str(evalue), traceback, line_number)

    @classmethod
    def from_mime_type(cls, bundle: dict[str, Any]) -> "SpreadsheetError":
        return cls(**bundle)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, SpreadsheetError) and other.ename == self.ename

    def __add__(self, other: Any) -> "SpreadsheetError":
        return self

    def __radd__(self, other: Any) -> "SpreadsheetError":
        return self

    def __sub__(self, other: Any) -> "SpreadsheetError":
        return self

    def __rsub__(self, other: Any) -> "SpreadsheetError":
        return self

    def __mul__(self, other: Any) -> "SpreadsheetError":
        return self

    def __rmul__(self, other: Any) -> "SpreadsheetError":
        return self

    def __floordiv__(self, other: Any) -> "SpreadsheetError":
        return self

    def __rfloordiv__(self, other: Any) -> "SpreadsheetError":
        return self

    def __truediv__(self, other: Any) -> "SpreadsheetError":
        return self

    def __rtruediv__(self, other: Any) -> "SpreadsheetError":
        return self

    def __str__(self) -> str:
        res = self.ename
        if self.msg:
            res += ": " + self.msg
        return res

    def __hash__(self) -> int:
        return hash(self.ename)


NA_ERROR = SpreadsheetError("#N/A")
VALUE_ERROR = SpreadsheetError("#VALUE!")
REF_ERROR = SpreadsheetError("#REF!")
ZERO_DIV_ERROR = SpreadsheetError("#DIV/0!")
NUM_ERROR = SpreadsheetError("#NUM!")
NAME_ERROR = SpreadsheetError("#NAME?")
NULL_ERROR = SpreadsheetError("#NULL!")
PYTHON_ERROR = SpreadsheetError("#PYTHON")
CALC_ERROR = SpreadsheetError("#CALC!")
GETTING_DATA_ERROR = SpreadsheetError("#GETTING_DATA")
GSHEET_NOT_AUTHORIZED = SpreadsheetError("#UNAUTHORIZED!")
SYNTAX_ERROR = SpreadsheetError("#SYNTAXERROR!")
UNSUPPORTED_ERROR = SpreadsheetError("#UNSUPPORTED!")

SPREADSHEET_ERRORS_STR = {
    NA_ERROR.ename,
    VALUE_ERROR.ename,
    REF_ERROR.ename,
    ZERO_DIV_ERROR.ename,
    NUM_ERROR.ename,
    NAME_ERROR.ename,
    NULL_ERROR.ename,
    PYTHON_ERROR.ename,
    CALC_ERROR.ename,
    GETTING_DATA_ERROR.ename,
    UNSUPPORTED_ERROR.ename,
}


class GSheetError(RuntimeError):
    def __init__(self, error: HttpError):
        self.error = error
        super().__init__(error.reason)


class GSheetNotAuthorized(RuntimeError): ...
