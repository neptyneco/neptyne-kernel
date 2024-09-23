"""Contains formulas unique to sheets and not found in Excel"""

from io import BytesIO
from typing import Any

import requests
from PIL import Image, UnidentifiedImageError

__all__ = ["IMAGE"]

from xml.etree.ElementTree import XMLParser

from ..mime_types import SVG_MIME_KEY
from ..spreadsheet_error import VALUE_ERROR

XML_OPEN = b"<?xml"
SVG_OPEN = b"<svg"


class SvgXmlTarget:
    width: int
    height: int

    def start(self, tag: str, attributes: dict[str, str]):
        if "}" in tag:
            tag = tag.split("}")[1]
        if tag == "svg":
            self.width = int(attributes.get("width", 400))
            self.height = int(attributes.get("height", 400))


class SvgImage:
    def __init__(self, svg: str):
        target = SvgXmlTarget()
        parser = XMLParser(target=target)
        parser.feed(svg[:4096])
        self.width, self.height = self.size = target.width, target.height
        self.svg = svg

    def _repr_mimebundle_(
        self, include: Any = False, exclude: Any = False, **kwargs: Any
    ) -> dict:
        data = {SVG_MIME_KEY: self.svg}
        if include:
            data = {k: v for (k, v) in data.items() if k in include}
        if exclude:
            data = {k: v for (k, v) in data.items() if k not in exclude}
        return data

    def resize(self, size):
        pass


def IMAGE(
    url: str,
    _mode: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> Image.Image:
    """Inserts an image into a cell."""
    user_agent = "Neptyne/1.0 (https://www.neptyne.com; team@neptyne.com)"
    content = requests.get(url, headers={"User-Agent": user_agent}).content
    if content.startswith(SVG_OPEN) or (
        content.startswith(XML_OPEN) and SVG_OPEN in content
    ):
        if content.startswith(XML_OPEN):
            content = content[content.find(SVG_OPEN) :]
        img = SvgImage(content.decode("utf-8"))
    else:
        try:
            img = Image.open(BytesIO(content))
        except UnidentifiedImageError:
            return VALUE_ERROR.with_message(
                "Could not find an image at this location. Check the url and if you have access"
            )
    if width is not None:
        if height is None:
            w, h = img.size
            height = h * width // w
        img = img.resize((width, height))
    return img
