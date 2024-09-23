from PIL import Image
from plotly.basedatatypes import BaseFigure

from ..dash import Dash


def url_for_image(img: BaseFigure | Image.Image) -> str:
    """Upload an image or plotly figure to Neptyne's image server and return the URL"""
    return Dash.instance().url_for_image(img)


def qr_for_url(
    data: str, size: int = 200, color: str = "black", background_color: str = "white"
) -> Image.Image:
    """Generate a QR code for a URL

    :param data: The URL to encode
    :param size: The size of the QR code in pixels
    :param color: The color of the QR code

    :return: The QR code as a PIL Image"""
    try:
        import qrcode
    except ImportError:
        raise ImportError(
            "The qrcode package is required to generate QR codes. "
            "Please install it by running `pip install qrcode`."
        )
    box_size = size // 21
    border_size = size * 0.01

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=box_size,
        border=border_size,
    )

    qr.add_data(data)
    qr.make(fit=True)
    return qr.make_image(fill_color=color, back_color=background_color).resize(
        (size, size)
    )
