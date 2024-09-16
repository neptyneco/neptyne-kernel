import colorsys
import os
from dataclasses import dataclass
from io import BytesIO

import nest_asyncio
import requests
from ipykernel.ipkernel import IPythonKernel
from PIL import Image

from .get_ipython_mockable import get_ipython_mockable
from .neptyne_protocol import (
    MessageTypes,
)
from .neptyne_secrets import Secrets
from .session_info import NeptyneSessionInfo
from .streamlit_server import is_running_in_streamlit

nest_asyncio.apply()


def in_gs_mode() -> bool:
    from .dash import Dash

    return Dash.instance().in_gs_mode


def send_sync_request(msg_type: str, content: dict) -> dict:
    kernel = get_kernel()
    ident = kernel._parent_ident["shell"]
    parent = kernel.get_parent("shell")
    parent["header"]["neptyne_msg_type"] = msg_type
    parent["header"]["neptyne_msg_content"] = content
    response = kernel._input_request("", ident, parent, False)
    return response


@dataclass
class User:
    name: str
    email: str
    profile_image: str
    secrets: Secrets

    def load_image(self) -> Image.Image | None:
        if not self.profile_image:
            return None
        return Image.open(BytesIO(requests.get(self.profile_image).content))


def email_to_color(user_email: str | None) -> str:
    """Return a color with a hue based on the hash of the email"""
    r, g, b = (
        int(x * 256) for x in colorsys.hls_to_rgb(hash(user_email) / (2**32), 0.8, 0.8)
    )
    return f"#{r:02x}{g:02x}{b:02x}"


def get_session() -> NeptyneSessionInfo | None:
    ip = get_ipython_mockable()
    parent_header = ip.parent_header["header"]
    if parent_header:
        return NeptyneSessionInfo.from_message_header(parent_header)
    return None


def get_user() -> User | None:
    if session := get_session():
        if email := session.user_email:
            user_secrets = session.user_secrets or {}
            tyne_secrets = session.tyne_secrets or {}
            name = session.user_name or ""
            profile_image = session.user_profile_image or ""
            return User(
                name,
                email,
                profile_image,
                Secrets({**tyne_secrets, **user_secrets}, not in_gs_mode()),
            )
    return None


def get_api_token() -> str | None:
    if token := os.getenv("NEPTYNE_API_TOKEN"):
        return token
    if is_running_in_streamlit():
        from .dash import Dash

        return Dash.instance().api_token_override
    if session := get_session():
        return session.user_api_token
    return None


def send_out_of_quota() -> None:
    k = get_kernel()
    k.send_response(
        k.iopub_socket,
        MessageTypes.API_QUOTA_EXCEEDED.value,
        {"service": "neptyne"},
    )


def get_kernel() -> IPythonKernel:
    return get_ipython_mockable().kernel
