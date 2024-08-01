from dataclasses import dataclass, fields
from typing import Any


@dataclass
class NeptyneSessionInfo:
    session_id: str | None = None
    user_email: str | None = None
    user_name: str | None = None
    user_secrets: dict[str, str] | None = None
    user_profile_image: str | None = None
    user_api_token: str | None = None
    tyne_secrets: dict[str, str] | None = None
    sheets_api_token: str | None = None

    @classmethod
    def from_message_header(cls, msg_header: dict[str, Any]) -> "NeptyneSessionInfo":
        return cls(
            **{
                field.name: msg_header.get(f"neptyne_{field.name}")
                for field in fields(cls)
            }
        )

    @classmethod
    def strip_from_message_header(cls, msg_header: dict[str, Any]) -> None:
        for field in fields(cls):
            msg_header.pop(f"neptyne_{field.name}", None)

    def write_to_header(self, msg_header: dict[str, Any]) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                msg_header[f"neptyne_{field.name}"] = value
