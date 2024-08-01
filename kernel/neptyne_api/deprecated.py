import warnings
from typing import Callable

from ..dash import Dash
from ..neptyne_protocol import MessageTypes


def button(
    caption: str, description: str | None = None
) -> Callable[[Callable], Callable]:
    """Register a function to be called when the button is clicked"""

    def decorator(func: Callable) -> Callable:
        Dash.instance().register_button(caption, func, description)

        return func

    return decorator


def send_owner_email(subject: str, body: str) -> None:
    """@private Send an email to the owner of the sheet. Deprecated. Use email.send instead."""
    warnings.warn(
        "send_owner_email is deprecated. Use email.send(to, subject, body) instead.",
        FutureWarning,
    )
    Dash.instance().reply_to_client(
        MessageTypes.NOTIFY_OWNER,
        {
            "subject": subject,
            "body": body,
            "channel": "email",
        },
    )
