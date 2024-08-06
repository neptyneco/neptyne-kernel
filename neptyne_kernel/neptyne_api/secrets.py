import warnings

from ..kernel_runtime import get_session, get_user
from ..neptyne_secrets import EmptySecrets, Secrets


def get_secrets() -> Secrets | EmptySecrets:
    """@private"""
    warnings.warn(
        "get_secrets() is deprecated, use get_secret(<key>) instead",
        category=DeprecationWarning,
    )
    user = get_user()
    if user:
        return user.secrets
    return EmptySecrets()


def get_secret(key: str, help: str = "") -> str | None:
    """Get a secret by **key**. If the secret is not set, it will be prompted for in interactive mode."""
    if session := get_session():
        user_secrets = session.user_secrets or {}
        tyne_secrets = session.tyne_secrets or {}
        return Secrets(
            {**tyne_secrets, **user_secrets}, bool(session.user_email)
        ).maybe_ask_for_secret(key, help)
    return None
