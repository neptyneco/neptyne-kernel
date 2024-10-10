import base64
import binascii
import datetime
import decimal
import inspect
import io
import json
import math
import pickle
import types
from json import JSONDecodeError
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
import plotly.io as pio
from IPython.core.ultratb import FormattedTB
from PIL import Image
from plotly.basedatatypes import BaseFigure

from .cell_address import CoordAddr
from .datetime_conversions import datetime_to_serial
from .download import Download
from .mime_types import (
    BYTES_MIME_KEY,
    DATETIME_KEY,
    DECIMAL_MIME_KEY,
    GSHEET_DOWNLOAD_KEY,
    GSHEET_ERROR_KEY,
    GSHEET_IMAGE_KEY,
    JSON_MIME_KEY,
    NUMBER_MIME_KEY,
    WELL_KNOWN_TEXT_KEY,
)
from .neptyne_protocol import MIMETypes
from .pandas_unrolling import dataframe_to_grid
from .primitives import Empty
from .spreadsheet_error import SpreadsheetError
from .widgets import base_widget
from .widgets.output_widgets import PLOTLY_MIME_TYPE

JSONPrimitive = str | float | int | bool | None
ChangeDict = dict[CoordAddr, dict[str, Any] | JSONPrimitive]
ChangeList = list[tuple[CoordAddr, dict[str, Any] | JSONPrimitive]]
SheetUpdate = tuple[str, CoordAddr | None, ChangeList]

ATTRIBUTE_UPDATE_KEY = "attributes"
REQUEST_ATTRIBUTE_UPDATE_KEY = "request_attributes"


def image_from_b64(encoded: str) -> Image:
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def decode_b64_pickled(encoded: str) -> Any:
    try:
        return pickle.loads(base64.b64decode(encoded.encode("utf-8")))
    except (pickle.PickleError, AttributeError, binascii.Error):
        return None


def encode_pickled_b64(value: Any) -> str:
    return base64.b64encode(pickle.dumps(value)).decode("utf-8")


def datetime_bundle(
    dt: datetime.datetime | datetime.date | datetime.time,
) -> dict[str, Any]:
    if isinstance(dt, datetime.datetime):
        typ = "datetime"
    elif isinstance(dt, datetime.date):
        typ = "date"
    elif isinstance(dt, datetime.time):
        typ = "time"
    else:
        raise TypeError(f"Unsupported type '{dt.__class__.__name__}'")
    return {
        DATETIME_KEY: {
            "type": typ,
            "isoformat": dt.isoformat(),
        },
        NUMBER_MIME_KEY: datetime_to_serial(dt),
    }


def decode_date(
    bundle: dict[str, Any],
) -> datetime.datetime | datetime.date | datetime.time:
    cls: type[datetime.datetime | datetime.date | datetime.time]
    typ = bundle["type"]
    if typ == "datetime":
        cls = datetime.datetime
    elif typ == "date":
        cls = datetime.date
    elif typ == "time":
        cls = datetime.time
    else:
        raise TypeError(f"Unsupported type '{typ}'")
    return cls.fromisoformat(bundle["isoformat"])


MimeBundle = dict[str, Any]

MIME_TYPE_HANDLERS: dict[str, Callable] = {
    DATETIME_KEY: decode_date,
    JSON_MIME_KEY: lambda x: x,
    NUMBER_MIME_KEY: lambda x: float(x),
    DECIMAL_MIME_KEY: lambda x: decimal.Decimal(x),
    BYTES_MIME_KEY: lambda x: base64.b64decode(x.encode("utf-8")),
    MIMETypes.APPLICATION_VND_NEPTYNE_WIDGET_V1_JSON.value: base_widget.from_mime_type,
    MIMETypes.APPLICATION_VND_NEPTYNE_OUTPUT_WIDGET_V1_JSON.value: base_widget.from_mime_type,
    MIMETypes.APPLICATION_VND_NEPTYNE_ERROR_V1_JSON.value: SpreadsheetError.from_mime_type,
    MIMETypes.APPLICATION_VND_POPO_V1_JSON.value: decode_b64_pickled,
    PLOTLY_MIME_TYPE: lambda d: pio.from_json(json.dumps(d)),
    "image/png": image_from_b64,
    "image/gif": image_from_b64,
    "image/jpeg": image_from_b64,
}

try:
    import shapely.wkt

    MIME_TYPE_HANDLERS[WELL_KNOWN_TEXT_KEY] = lambda x: shapely.wkt.loads(x)
except ImportError:
    pass


def maybe_format_common_values(val: Any) -> Any:
    if inspect.isclass(val):
        prefix = "class"
    elif inspect.isfunction(val) or inspect.isbuiltin(val):
        prefix = "function"
    elif inspect.ismodule(val):
        prefix = "module"
    else:
        return val
    return f"<{prefix} {val.__name__}>"


def output_to_value(output: MimeBundle | None) -> Any:
    if not output:
        return None
    for mime_type, handler in MIME_TYPE_HANDLERS.items():
        if mime_type in output:
            try:
                return handler(output[mime_type])
            except Exception:
                # Not catching the error here stops the tyne from loading. We can't return an error instance
                # since that might not work inside the meta this is returned too. So our best bet is to try
                # the next entry in the output or return None if there isn't one.
                continue
    # legacy:
    if "text/plain" in output:
        value = output["text/plain"]
        if value[0] in "\"'":
            value = value[1:-1]
        else:
            try:
                value = json.loads(value)
            except JSONDecodeError:
                pass
        return value
    return None


def outputs_to_value(outputs: list[MimeBundle]) -> Any:
    merged: MimeBundle = {}
    for output in outputs:
        merged = {**merged, **output}
    return output_to_value(merged)


def as_json(value: Any) -> Any:
    """Convert some common types to JSON-compatible types."""
    if isinstance(value, dict):
        return {k: as_json(v) for k, v in value.items()}
    if not isinstance(value, str) and hasattr(value, "__iter__"):
        return [as_json(v) for v in value]
    return value


def is_number_string(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def gsheet_spreadsheet_error_from_python_exception(
    etype: type[BaseException] | None,
    evalue: BaseException | None,
    tb: types.TracebackType | None,
) -> SpreadsheetError:
    traceback = FormattedTB().structured_traceback(etype, evalue, tb, tb_offset=2)
    line_number: int | None = None
    while tb:
        if tb.tb_frame.f_code.co_filename.startswith("<ipython-input-"):
            line_number = tb.tb_lineno
        tb = tb.tb_next
    return SpreadsheetError.from_python_exception(
        etype, evalue, traceback, line_number=line_number
    )


def gsheet_jsonify(value: Any, depth: int = 0) -> Any:
    if isinstance(value, datetime.datetime):
        return {"type": "date", "dateString": value.isoformat()}
    elif isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return str(value)
    elif value is Empty.MakeItSo:
        return None
    elif isinstance(value, int | float | str) or value is None:
        return value
    elif depth >= 2:
        pass
    elif isinstance(value, types.GeneratorType | tuple):
        return [gsheet_jsonify(v, depth + 1) for v in value]
    elif isinstance(value, dict):
        return [
            [gsheet_jsonify(k, depth + 2), gsheet_jsonify(v, depth + 2)]
            for k, v in value.items()
        ]
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, pd.Series):
        pd_header = [("", value.name)] if value.name else []
        return [*pd_header, *gsheet_jsonify(value.items(), depth)]
    elif isinstance(value, pd.DataFrame):
        return gsheet_jsonify(dataframe_to_grid(value), depth)
    elif (
        hasattr(value, "__iter__")
        and not hasattr(value, "_repr_mimebundle_")
        and not isinstance(value, str | type)
    ):
        return [gsheet_jsonify(v, depth + 1) for v in value]
    return str(value)


def encode_plotly_figure(value: BaseFigure) -> tuple[str, Any]:
    value = pio.to_json(value)
    return PLOTLY_MIME_TYPE, value


def im_has_alpha(img: Image) -> bool:
    return img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )


def encode_pil_image(value: Image) -> tuple[str, dict[str, Any]]:
    width = value.width
    height = value.height
    img_format = "png" if im_has_alpha(value) else "jpeg"
    with io.BytesIO() as fb:
        value.save(fb, format=img_format)
        payload = fb.getvalue()

    value = {
        "bytes": base64.b64encode(payload).decode("utf8"),
        "format": img_format,
        "width": width,
        "height": height,
    }
    return GSHEET_IMAGE_KEY, value


def encode_for_gsheets(value: Any) -> tuple[str, str]:
    """Returns a tuple of content_type and encoded value."""
    content_type = "application/json"
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        content_type = NUMBER_MIME_KEY
        value = str(value)
    elif isinstance(value, BaseFigure):
        content_type, value = encode_plotly_figure(value)
    elif isinstance(value, Image.Image):
        content_type, value = encode_pil_image(value)
    elif isinstance(value, SpreadsheetError):
        value = {
            "ename": value.ename,
            "message": value.msg,
            "line": value.line_number,
        }
        content_type = GSHEET_ERROR_KEY
    elif isinstance(value, Download):
        content_type = GSHEET_DOWNLOAD_KEY
        value = {
            "name": value.name,
            "value": base64.b64encode(cast(bytes, value.value)).decode("utf-8"),
            "mimetype": value.mimetype,
        }
    else:
        value = gsheet_jsonify(value)

    from .tyne_model.save_message import json_encode

    encoded = json_encode(value).decode("utf-8")

    return content_type, encoded
