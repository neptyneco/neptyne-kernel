from ..dash import Dash


def send(to: list[str], subject: str, body: str):
    """Send an email to the specified recipients.

    Only users who have opened the tyne or GSheet extension will be able to receive emails.
    """
    Dash.instance().send_email(to, subject, body)


__all__ = ["send"]
