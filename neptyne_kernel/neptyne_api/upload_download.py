import json
from io import BytesIO
from typing import Any, Optional

import pandas as pd
from PIL import Image

from ..dash import Dash
from ..download import Download
from ..sheet_api import NeptyneSheet


def download(value: Any, filename: str | None = None) -> Download | None:
    """Download data as a file with the given filename

    value can refer to any object in Python, including cell ranges. filename is optional
    and determines the name of the file, while the extension determines the file type.
    Supported file types are .csv, .json, .txt, and image files. If no filename is specified,
    Neptyne will automatically generate a filename. If the value is a cell range, this will
    be a text file for 1d data and a csv for 2d. Images will be downloaded as images, anything
    else will default to being a json document.

    """
    return Dash.instance().initiate_download(value, filename)


def upload_csv_to_new_sheet(prompt: str = "Upload a .csv") -> Optional["NeptyneSheet"]:
    """Upload a CSV file to a new sheet and return the sheet object"""
    return Dash.instance().upload_csv_to_new_sheet_and_return(prompt)


def upload(prompt: str = "Upload a file", accept: str = "*") -> Any:
    """Upload a file and return its content"""
    content, filename = Dash.instance().initiate_upload(prompt=prompt, accept=accept)

    if "." in filename:
        extension = filename.rsplit(".", 1)[1]
        if extension == "json":
            return json.loads(content)
        elif extension == "csv":
            return pd.read_csv(BytesIO(content))
        elif extension == "txt":
            return content.decode("utf-8")
        elif extension in ("jpeg", "jpg", "gif", "bmp", "png"):
            return Image.open(BytesIO(content))

    return content
