from contextvars import ContextVar

sheet_name_override: ContextVar[str | None] = ContextVar(
    "sheet_name_override", default=None
)
gsheet_id_override: ContextVar[str | None] = ContextVar(
    "gsheet_id_override", default=None
)
