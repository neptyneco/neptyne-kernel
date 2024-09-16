# This code parses date/times, so please
#
#     pip install python-dateutil
#
# To use this code, make sure you
#
#     import json
#
# and then, to convert JSON from a string, do
#
#     result = kernel_protocol_from_dict(json.loads(json_string))
#     result = cell_attribute_from_dict(json.loads(json_string))
#     result = special_users_from_dict(json.loads(json_string))
#     result = tyne_categories_from_dict(json.loads(json_string))
#     result = access_scope_from_dict(json.loads(json_string))
#     result = access_level_from_dict(json.loads(json_string))
#     result = tyne_list_item_from_dict(json.loads(json_string))
#     result = share_record_from_dict(json.loads(json_string))
#     result = tyne_share_response_from_dict(json.loads(json_string))
#     result = line_wrap_from_dict(json.loads(json_string))
#     result = text_style_from_dict(json.loads(json_string))
#     result = text_align_from_dict(json.loads(json_string))
#     result = vertical_align_from_dict(json.loads(json_string))
#     result = border_type_from_dict(json.loads(json_string))
#     result = number_format_from_dict(json.loads(json_string))
#     result = message_types_from_dict(json.loads(json_string))
#     result = kernel_init_state_from_dict(json.loads(json_string))
#     result = mime_types_from_dict(json.loads(json_string))
#     result = dimension_from_dict(json.loads(json_string))
#     result = sheet_transform_from_dict(json.loads(json_string))
#     result = widget_param_type_from_dict(json.loads(json_string))
#     result = sheet_unaware_cell_id_from_dict(json.loads(json_string))
#     result = sheet_cell_id_from_dict(json.loads(json_string))
#     result = notebook_cell_id_from_dict(json.loads(json_string))
#     result = cell_id_from_dict(json.loads(json_string))
#     result = sheet_attribute_from_dict(json.loads(json_string))
#     result = sheet_attribute_update_from_dict(json.loads(json_string))
#     result = cell_attribute_update_from_dict(json.loads(json_string))
#     result = cell_attributes_update_from_dict(json.loads(json_string))
#     result = call_server_content_from_dict(json.loads(json_string))
#     result = cell_change_from_dict(json.loads(json_string))
#     result = run_cells_content_from_dict(json.loads(json_string))
#     result = rerun_cells_content_from_dict(json.loads(json_string))
#     result = sheet_update_content_from_dict(json.loads(json_string))
#     result = tyne_property_update_content_change_from_dict(json.loads(json_string))
#     result = tyne_property_update_content_from_dict(json.loads(json_string))
#     result = copy_cells_content_from_dict(json.loads(json_string))
#     result = sheet_autofill_content_from_dict(json.loads(json_string))
#     result = selection_rect_from_dict(json.loads(json_string))
#     result = insert_delete_content_from_dict(json.loads(json_string))
#     result = drag_row_column_content_from_dict(json.loads(json_string))
#     result = widget_value_content_from_dict(json.loads(json_string))
#     result = widget_param_definition_from_dict(json.loads(json_string))
#     result = widget_definition_from_dict(json.loads(json_string))
#     result = widget_registry_from_dict(json.loads(json_string))
#     result = insert_delete_reply_from_dict(json.loads(json_string))
#     result = delete_sheet_content_from_dict(json.loads(json_string))
#     result = subscriber_from_dict(json.loads(json_string))
#     result = subscribers_updated_content_from_dict(json.loads(json_string))
#     result = rename_sheet_content_from_dict(json.loads(json_string))
#     result = install_requirements_content_from_dict(json.loads(json_string))
#     result = download_request_from_dict(json.loads(json_string))
#     result = traceback_frame_from_dict(json.loads(json_string))
#     result = navigate_to_content_from_dict(json.loads(json_string))
#     result = widget_get_state_content_from_dict(json.loads(json_string))
#     result = widget_validate_params_content_from_dict(json.loads(json_string))
#     result = tyne_event_from_dict(json.loads(json_string))
#     result = stripe_subscription_from_dict(json.loads(json_string))
#     result = rename_tyne_content_from_dict(json.loads(json_string))
#     result = set_secrets_content_from_dict(json.loads(json_string))
#     result = secrets_from_dict(json.loads(json_string))
#     result = tick_reply_content_from_dict(json.loads(json_string))
#     result = organization_create_content_from_dict(json.loads(json_string))
#     result = access_mode_from_dict(json.loads(json_string))
#     result = g_sheets_image_from_dict(json.loads(json_string))
#     result = research_usage_from_dict(json.loads(json_string))
#     result = research_message_from_dict(json.loads(json_string))
#     result = research_error_from_dict(json.loads(json_string))
#     result = research_cell_from_dict(json.loads(json_string))
#     result = research_source_from_dict(json.loads(json_string))
#     result = research_table_from_dict(json.loads(json_string))
#     result = streamlit_app_config_from_dict(json.loads(json_string))
#     result = sheet_data_from_dict(json.loads(json_string))
#     result = user_view_state_from_dict(json.loads(json_string))
#     result = insert_delete_reply_cell_type_from_dict(json.loads(json_string))
#     result = cell_type_from_dict(json.loads(json_string))

from enum import Enum
from typing import List, Optional, Any, Union, Dict, TypeVar, Callable, Type, cast
from datetime import datetime
import dateutil.parser


T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=Enum)


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


def from_datetime(x: Any) -> datetime:
    return dateutil.parser.parse(x)


def to_enum(c: Type[EnumT], x: Any) -> EnumT:
    assert isinstance(x, c)
    return x.value


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def to_float(x: Any) -> float:
    assert isinstance(x, (float, int))
    return x


def from_dict(f: Callable[[Any], T], x: Any) -> Dict[str, T]:
    assert isinstance(x, dict)
    return {k: f(v) for (k, v) in x.items()}


class KernelProtocol(Enum):
    TRITON = "triton"


class CellAttribute(Enum):
    BACKGROUND_COLOR = "backgroundColor"
    BORDER = "border"
    CLASS = "class"
    COLOR = "color"
    COL_SPAN = "colSpan"
    EXECUTION_POLICY = "executionPolicy"
    FONT = "font"
    FONT_SIZE = "fontSize"
    IS_PROTECTED = "isProtected"
    LINE_WRAP = "lineWrap"
    LINK = "link"
    NOTE = "note"
    NUMBER_FORMAT = "numberFormat"
    RENDER_HEIGHT = "renderHeight"
    RENDER_WIDTH = "renderWidth"
    ROW_SPAN = "rowSpan"
    SOURCE = "source"
    TEXT_ALIGN = "textAlign"
    TEXT_STYLE = "textStyle"
    VERTICAL_ALIGN = "verticalAlign"
    WIDGET = "widget"
    WIDGET_NAME = "widgetName"


class SpecialUsers(Enum):
    SYS_MAINTENANCE_NEPTYNE_COM = "sys-maintenance@neptyne.com"
    TRYNEPTYNE_NEPTYNE_DEV = "tryneptyne@neptyne.dev"


class TyneCategories(Enum):
    AUTHORED_BY_ME = "authoredByMe"
    EDITABLE_BY_ME = "editableByMe"
    IN_GALLERY = "inGallery"
    SHARED_WITH_ME = "sharedWithMe"


class TyneListItem:
    access: str
    categories: List[TyneCategories]
    description: Optional[str]
    file_name: str
    gallery_category: Optional[str]
    gallery_screenshot_url: Optional[str]
    """Enables basic storage and retrieval of dates and times."""
    last_modified: datetime
    """Enables basic storage and retrieval of dates and times."""
    last_opened: Optional[datetime]
    name: str
    owner: str
    owner_color: str
    owner_profile_image: Optional[str]

    def __init__(
        self,
        access: str,
        categories: List[TyneCategories],
        description: Optional[str],
        file_name: str,
        gallery_category: Optional[str],
        gallery_screenshot_url: Optional[str],
        last_modified: datetime,
        last_opened: Optional[datetime],
        name: str,
        owner: str,
        owner_color: str,
        owner_profile_image: Optional[str],
    ) -> None:
        self.access = access
        self.categories = categories
        self.description = description
        self.file_name = file_name
        self.gallery_category = gallery_category
        self.gallery_screenshot_url = gallery_screenshot_url
        self.last_modified = last_modified
        self.last_opened = last_opened
        self.name = name
        self.owner = owner
        self.owner_color = owner_color
        self.owner_profile_image = owner_profile_image

    @staticmethod
    def from_dict(obj: Any) -> "TyneListItem":
        assert isinstance(obj, dict)
        access = from_str(obj.get("access"))
        categories = from_list(TyneCategories, obj.get("categories"))
        description = from_union([from_str, from_none], obj.get("description"))
        file_name = from_str(obj.get("fileName"))
        gallery_category = from_union([from_str, from_none], obj.get("galleryCategory"))
        gallery_screenshot_url = from_union(
            [from_str, from_none], obj.get("galleryScreenshotUrl")
        )
        last_modified = from_datetime(obj.get("lastModified"))
        last_opened = from_union([from_datetime, from_none], obj.get("lastOpened"))
        name = from_str(obj.get("name"))
        owner = from_str(obj.get("owner"))
        owner_color = from_str(obj.get("ownerColor"))
        owner_profile_image = from_union(
            [from_str, from_none], obj.get("ownerProfileImage")
        )
        return TyneListItem(
            access,
            categories,
            description,
            file_name,
            gallery_category,
            gallery_screenshot_url,
            last_modified,
            last_opened,
            name,
            owner,
            owner_color,
            owner_profile_image,
        )

    def to_dict(self) -> dict:
        result: dict = {}
        result["access"] = from_str(self.access)
        result["categories"] = from_list(
            lambda x: to_enum(TyneCategories, x), self.categories
        )
        if self.description is not None:
            result["description"] = from_union([from_str, from_none], self.description)
        result["fileName"] = from_str(self.file_name)
        if self.gallery_category is not None:
            result["galleryCategory"] = from_union(
                [from_str, from_none], self.gallery_category
            )
        if self.gallery_screenshot_url is not None:
            result["galleryScreenshotUrl"] = from_union(
                [from_str, from_none], self.gallery_screenshot_url
            )
        result["lastModified"] = self.last_modified.isoformat()
        if self.last_opened is not None:
            result["lastOpened"] = from_union(
                [lambda x: x.isoformat(), from_none], self.last_opened
            )
        result["name"] = from_str(self.name)
        result["owner"] = from_str(self.owner)
        result["ownerColor"] = from_str(self.owner_color)
        if self.owner_profile_image is not None:
            result["ownerProfileImage"] = from_union(
                [from_str, from_none], self.owner_profile_image
            )
        return result


class AccessLevel(Enum):
    EDIT = "EDIT"
    OWNER = "OWNER"
    VIEW = "VIEW"


class AccessScope(Enum):
    ANYONE = "anyone"
    RESTRICTED = "restricted"
    TEAM = "team"


class ShareRecord:
    access_level: AccessLevel
    email: str
    name: Optional[str]

    def __init__(
        self, access_level: AccessLevel, email: str, name: Optional[str]
    ) -> None:
        self.access_level = access_level
        self.email = email
        self.name = name

    @staticmethod
    def from_dict(obj: Any) -> "ShareRecord":
        assert isinstance(obj, dict)
        access_level = AccessLevel(obj.get("access_level"))
        email = from_str(obj.get("email"))
        name = from_union([from_none, from_str], obj.get("name"))
        return ShareRecord(access_level, email, name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["access_level"] = to_enum(AccessLevel, self.access_level)
        result["email"] = from_str(self.email)
        result["name"] = from_union([from_none, from_str], self.name)
        return result


class User:
    email: str
    name: Optional[str]

    def __init__(self, email: str, name: Optional[str]) -> None:
        self.email = email
        self.name = name

    @staticmethod
    def from_dict(obj: Any) -> "User":
        assert isinstance(obj, dict)
        email = from_str(obj.get("email"))
        name = from_union([from_none, from_str], obj.get("name"))
        return User(email, name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["email"] = from_str(self.email)
        result["name"] = from_union([from_none, from_str], self.name)
        return result


class TyneShareResponse:
    description: str
    general_access_level: Optional[AccessLevel]
    general_access_scope: Optional[AccessScope]
    is_app: bool
    share_message: Optional[str]
    shares: List[ShareRecord]
    team_name: Optional[str]
    users: List[User]

    def __init__(
        self,
        description: str,
        general_access_level: Optional[AccessLevel],
        general_access_scope: Optional[AccessScope],
        is_app: bool,
        share_message: Optional[str],
        shares: List[ShareRecord],
        team_name: Optional[str],
        users: List[User],
    ) -> None:
        self.description = description
        self.general_access_level = general_access_level
        self.general_access_scope = general_access_scope
        self.is_app = is_app
        self.share_message = share_message
        self.shares = shares
        self.team_name = team_name
        self.users = users

    @staticmethod
    def from_dict(obj: Any) -> "TyneShareResponse":
        assert isinstance(obj, dict)
        description = from_str(obj.get("description"))
        general_access_level = from_union(
            [AccessLevel, from_none], obj.get("generalAccessLevel")
        )
        general_access_scope = from_union(
            [AccessScope, from_none], obj.get("generalAccessScope")
        )
        is_app = from_bool(obj.get("isApp"))
        share_message = from_union([from_str, from_none], obj.get("shareMessage"))
        shares = from_list(ShareRecord.from_dict, obj.get("shares"))
        team_name = from_union([from_str, from_none], obj.get("teamName"))
        users = from_list(User.from_dict, obj.get("users"))
        return TyneShareResponse(
            description,
            general_access_level,
            general_access_scope,
            is_app,
            share_message,
            shares,
            team_name,
            users,
        )

    def to_dict(self) -> dict:
        result: dict = {}
        result["description"] = from_str(self.description)
        if self.general_access_level is not None:
            result["generalAccessLevel"] = from_union(
                [lambda x: to_enum(AccessLevel, x), from_none],
                self.general_access_level,
            )
        if self.general_access_scope is not None:
            result["generalAccessScope"] = from_union(
                [lambda x: to_enum(AccessScope, x), from_none],
                self.general_access_scope,
            )
        result["isApp"] = from_bool(self.is_app)
        if self.share_message is not None:
            result["shareMessage"] = from_union(
                [from_str, from_none], self.share_message
            )
        result["shares"] = from_list(lambda x: to_class(ShareRecord, x), self.shares)
        if self.team_name is not None:
            result["teamName"] = from_union([from_str, from_none], self.team_name)
        result["users"] = from_list(lambda x: to_class(User, x), self.users)
        return result


class LineWrap(Enum):
    OVERFLOW = "overflow"
    TRUNCATE = "truncate"
    WRAP = "wrap"


class TextStyle(Enum):
    BOLD = "bold"
    ITALIC = "italic"
    UNDERLINE = "underline"


class TextAlign(Enum):
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"


class VerticalAlign(Enum):
    BOTTOM = "bottom"
    MIDDLE = "middle"
    TOP = "top"


class BorderType(Enum):
    BORDER_BOTTOM = "border-bottom"
    BORDER_LEFT = "border-left"
    BORDER_RIGHT = "border-right"
    BORDER_TOP = "border-top"


class NumberFormat(Enum):
    CUSTOM = "custom"
    DATE = "date"
    FLOAT = "float"
    INTEGER = "integer"
    MONEY = "money"
    PERCENTAGE = "percentage"


class MessageTypes(Enum):
    ACK_RUN_CELLS = "ack_run_cells"
    API_QUOTA_EXCEEDED = "api_quota_exceeded"
    AUTH_REPLY = "auth_reply"
    AUTH_REQUIRED = "auth_required"
    CHANGE_CELL_ATTRIBUTE = "change_cell_attribute"
    CHANGE_SHEET_ATTRIBUTE = "change_sheet_attribute"
    CHANGE_SHEET_ATTRIBUTE_REPLY = "change_sheet_attribute_reply"
    CONFETTI = "confetti"
    COPY_CELLS = "copy_cells"
    CREATE_SHEET = "create_sheet"
    DELETE_SHEET = "delete_sheet"
    DRAG_ROW_COLUMN = "drag_row_column"
    GET_SECRETS = "get_secrets"
    INSERT_DELETE_CELLS = "insert_delete_cells"
    INSERT_DELETE_CELLS_REPLY = "insert_delete_cells_reply"
    INSTALL_REQUIREMENTS = "install_requirements"
    INTERRUPT_KERNEL = "interrupt_kernel"
    LINTER = "linter"
    LOG_EVENT = "log_event"
    NAVIGATE_TO = "navigate_to"
    NOTIFY_OWNER = "notify_owner"
    PING = "ping"
    RECONNECT_KERNEL = "reconnect_kernel"
    RELOAD_ENV = "reload_env"
    RENAME_SHEET = "rename_sheet"
    RENAME_TYNE = "rename_tyne"
    RERUN_CELLS = "rerun_cells"
    RPC_REQUEST = "rpc_request"
    RPC_RESULT = "rpc_result"
    RUN_CELLS = "run_cells"
    SAVE_CELL = "save_cell"
    SAVE_KERNEL_STATE = "save_kernel_state"
    SAVE_TYNE = "save_tyne"
    SEND_EMAIL = "send_email"
    SEND_UNDO_MESSAGE = "send_undo_message"
    SET_SECRET = "set_secret"
    SET_SECRETS = "set_secrets"
    SHEET_AUTOFILL = "sheet_autofill"
    SHEET_UPDATE = "sheet_update"
    SHOW_ALERT = "show_alert"
    START_DOWNLOAD = "start_download"
    SUBSCRIBERS_UPDATED = "subscribers_updated"
    TICK_REPLY = "tick_reply"
    TRACEBACK = "traceback"
    TYNE_PROPERTY_UPDATE = "tyne_property_update"
    TYNE_SAVED = "tyne_saved"
    TYNE_SAVING = "tyne_saving"
    UPLOAD_FILE = "upload_file"
    UPLOAD_FILE_TO_GCP = "upload_file_to_gcp"
    USER_API_RESPONSE_STREAM = "user_api_response_stream"
    WIDGET_GET_STATE = "widget_get_state"
    WIDGET_VALIDATE_PARAMS = "widget_validate_params"
    WIDGET_VALUE_UPDATE = "widget_value_update"


class KernelInitState(Enum):
    INSTALLING_REQUIREMENTS = "installing_requirements"
    LOADING_SHEET_VALUES = "loading_sheet_values"
    RUN_CODE_PANEL = "run_code_panel"


class MIMETypes(Enum):
    APPLICATION_VND_NEPTYNE_ERROR_V1_JSON = "application/vnd.neptyne-error.v1+json"
    APPLICATION_VND_NEPTYNE_OUTPUT_WIDGET_V1_JSON = (
        "application/vnd.neptyne-output-widget.v1+json"
    )
    APPLICATION_VND_NEPTYNE_WIDGET_V1_JSON = "application/vnd.neptyne-widget.v1+json"
    APPLICATION_VND_POPO_V1_JSON = "application/vnd.popo.v1+json"


class SheetAttribute(Enum):
    COLS_SIZES = "colsSizes"
    RESEARCH_META_DATA = "researchMetaData"
    ROWS_SIZES = "rowsSizes"


class SheetAttributeUpdate:
    attribute: str
    sheet_id: float
    value: Any

    def __init__(self, attribute: str, sheet_id: float, value: Any) -> None:
        self.attribute = attribute
        self.sheet_id = sheet_id
        self.value = value

    @staticmethod
    def from_dict(obj: Any) -> "SheetAttributeUpdate":
        assert isinstance(obj, dict)
        attribute = from_str(obj.get("attribute"))
        sheet_id = from_float(obj.get("sheetId"))
        value = obj.get("value")
        return SheetAttributeUpdate(attribute, sheet_id, value)

    def to_dict(self) -> dict:
        result: dict = {}
        result["attribute"] = from_str(self.attribute)
        result["sheetId"] = to_float(self.sheet_id)
        result["value"] = self.value
        return result


class CellAttributeUpdate:
    attribute: str
    cell_id: List[float]
    value: Union[float, None, str]

    def __init__(
        self, attribute: str, cell_id: List[float], value: Union[float, None, str]
    ) -> None:
        self.attribute = attribute
        self.cell_id = cell_id
        self.value = value

    @staticmethod
    def from_dict(obj: Any) -> "CellAttributeUpdate":
        assert isinstance(obj, dict)
        attribute = from_str(obj.get("attribute"))
        cell_id = from_list(from_float, obj.get("cellId"))
        value = from_union([from_float, from_str, from_none], obj.get("value"))
        return CellAttributeUpdate(attribute, cell_id, value)

    def to_dict(self) -> dict:
        result: dict = {}
        result["attribute"] = from_str(self.attribute)
        result["cellId"] = from_list(to_float, self.cell_id)
        if self.value is not None:
            result["value"] = from_union([to_float, from_str, from_none], self.value)
        return result


class CellAttributesUpdate:
    updates: List[CellAttributeUpdate]

    def __init__(self, updates: List[CellAttributeUpdate]) -> None:
        self.updates = updates

    @staticmethod
    def from_dict(obj: Any) -> "CellAttributesUpdate":
        assert isinstance(obj, dict)
        updates = from_list(CellAttributeUpdate.from_dict, obj.get("updates"))
        return CellAttributesUpdate(updates)

    def to_dict(self) -> dict:
        result: dict = {}
        result["updates"] = from_list(
            lambda x: to_class(CellAttributeUpdate, x), self.updates
        )
        return result


class CallServerContent:
    args: List[str]
    kwargs: Dict[str, Any]
    method: str

    def __init__(self, args: List[str], kwargs: Dict[str, Any], method: str) -> None:
        self.args = args
        self.kwargs = kwargs
        self.method = method

    @staticmethod
    def from_dict(obj: Any) -> "CallServerContent":
        assert isinstance(obj, dict)
        args = from_list(from_str, obj.get("args"))
        kwargs = from_dict(lambda x: x, obj.get("kwargs"))
        method = from_str(obj.get("method"))
        return CallServerContent(args, kwargs, method)

    def to_dict(self) -> dict:
        result: dict = {}
        result["args"] = from_list(from_str, self.args)
        result["kwargs"] = from_dict(lambda x: x, self.kwargs)
        result["method"] = from_str(self.method)
        return result


class CellChange:
    attributes: Optional[Dict[str, Any]]
    cell_id: Union[List[float], None, str]
    content: str
    mime_type: Optional[str]

    def __init__(
        self,
        attributes: Optional[Dict[str, Any]],
        cell_id: Union[List[float], None, str],
        content: str,
        mime_type: Optional[str],
    ) -> None:
        self.attributes = attributes
        self.cell_id = cell_id
        self.content = content
        self.mime_type = mime_type

    @staticmethod
    def from_dict(obj: Any) -> "CellChange":
        assert isinstance(obj, dict)
        attributes = from_union(
            [lambda x: from_dict(lambda x: x, x), from_none], obj.get("attributes")
        )
        cell_id = from_union(
            [lambda x: from_list(from_float, x), from_str, from_none], obj.get("cellId")
        )
        content = from_str(obj.get("content"))
        mime_type = from_union([from_str, from_none], obj.get("mimeType"))
        return CellChange(attributes, cell_id, content, mime_type)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.attributes is not None:
            result["attributes"] = from_union(
                [lambda x: from_dict(lambda x: x, x), from_none], self.attributes
            )
        if self.cell_id is not None:
            result["cellId"] = from_union(
                [lambda x: from_list(to_float, x), from_str, from_none], self.cell_id
            )
        result["content"] = from_str(self.content)
        if self.mime_type is not None:
            result["mimeType"] = from_union([from_str, from_none], self.mime_type)
        return result


class RunCellsContent:
    ai_tables: Optional[List[Dict[str, Any]]]
    current_sheet: float
    current_sheet_name: Optional[str]
    for_ai: bool
    gs_mode: bool
    notebook: bool
    sheet_ids_by_name: Optional[Dict[str, float]]
    to_run: List[CellChange]

    def __init__(
        self,
        ai_tables: Optional[List[Dict[str, Any]]],
        current_sheet: float,
        current_sheet_name: Optional[str],
        for_ai: bool,
        gs_mode: bool,
        notebook: bool,
        sheet_ids_by_name: Optional[Dict[str, float]],
        to_run: List[CellChange],
    ) -> None:
        self.ai_tables = ai_tables
        self.current_sheet = current_sheet
        self.current_sheet_name = current_sheet_name
        self.for_ai = for_ai
        self.gs_mode = gs_mode
        self.notebook = notebook
        self.sheet_ids_by_name = sheet_ids_by_name
        self.to_run = to_run

    @staticmethod
    def from_dict(obj: Any) -> "RunCellsContent":
        assert isinstance(obj, dict)
        ai_tables = from_union(
            [lambda x: from_list(lambda x: from_dict(lambda x: x, x), x), from_none],
            obj.get("aiTables"),
        )
        current_sheet = from_float(obj.get("currentSheet"))
        current_sheet_name = from_union(
            [from_str, from_none], obj.get("currentSheetName")
        )
        for_ai = from_bool(obj.get("forAI"))
        gs_mode = from_bool(obj.get("gsMode"))
        notebook = from_bool(obj.get("notebook"))
        sheet_ids_by_name = from_union(
            [lambda x: from_dict(from_float, x), from_none], obj.get("sheetIdsByName")
        )
        to_run = from_list(CellChange.from_dict, obj.get("toRun"))
        return RunCellsContent(
            ai_tables,
            current_sheet,
            current_sheet_name,
            for_ai,
            gs_mode,
            notebook,
            sheet_ids_by_name,
            to_run,
        )

    def to_dict(self) -> dict:
        result: dict = {}
        if self.ai_tables is not None:
            result["aiTables"] = from_union(
                [
                    lambda x: from_list(lambda x: from_dict(lambda x: x, x), x),
                    from_none,
                ],
                self.ai_tables,
            )
        result["currentSheet"] = to_float(self.current_sheet)
        if self.current_sheet_name is not None:
            result["currentSheetName"] = from_union(
                [from_str, from_none], self.current_sheet_name
            )
        result["forAI"] = from_bool(self.for_ai)
        result["gsMode"] = from_bool(self.gs_mode)
        result["notebook"] = from_bool(self.notebook)
        if self.sheet_ids_by_name is not None:
            result["sheetIdsByName"] = from_union(
                [lambda x: from_dict(to_float, x), from_none], self.sheet_ids_by_name
            )
        result["toRun"] = from_list(lambda x: to_class(CellChange, x), self.to_run)
        return result


class RerunCellsContent:
    addresses: List[List[float]]
    changed_functions: List[str]

    def __init__(
        self, addresses: List[List[float]], changed_functions: List[str]
    ) -> None:
        self.addresses = addresses
        self.changed_functions = changed_functions

    @staticmethod
    def from_dict(obj: Any) -> "RerunCellsContent":
        assert isinstance(obj, dict)
        addresses = from_list(lambda x: from_list(from_float, x), obj.get("addresses"))
        changed_functions = from_list(from_str, obj.get("changedFunctions"))
        return RerunCellsContent(addresses, changed_functions)

    def to_dict(self) -> dict:
        result: dict = {}
        result["addresses"] = from_list(
            lambda x: from_list(to_float, x), self.addresses
        )
        result["changedFunctions"] = from_list(from_str, self.changed_functions)
        return result


class SheetUpdateContent:
    cell_updates: List[Any]

    def __init__(self, cell_updates: List[Any]) -> None:
        self.cell_updates = cell_updates

    @staticmethod
    def from_dict(obj: Any) -> "SheetUpdateContent":
        assert isinstance(obj, dict)
        cell_updates = from_list(lambda x: x, obj.get("cellUpdates"))
        return SheetUpdateContent(cell_updates)

    def to_dict(self) -> dict:
        result: dict = {}
        result["cellUpdates"] = from_list(lambda x: x, self.cell_updates)
        return result


class TynePropertyUpdateContentChange:
    property: str
    value: Any

    def __init__(self, property: str, value: Any) -> None:
        self.property = property
        self.value = value

    @staticmethod
    def from_dict(obj: Any) -> "TynePropertyUpdateContentChange":
        assert isinstance(obj, dict)
        property = from_str(obj.get("property"))
        value = obj.get("value")
        return TynePropertyUpdateContentChange(property, value)

    def to_dict(self) -> dict:
        result: dict = {}
        result["property"] = from_str(self.property)
        result["value"] = self.value
        return result


class TynePropertyUpdateContent:
    changes: List[TynePropertyUpdateContentChange]

    def __init__(self, changes: List[TynePropertyUpdateContentChange]) -> None:
        self.changes = changes

    @staticmethod
    def from_dict(obj: Any) -> "TynePropertyUpdateContent":
        assert isinstance(obj, dict)
        changes = from_list(
            TynePropertyUpdateContentChange.from_dict, obj.get("changes")
        )
        return TynePropertyUpdateContent(changes)

    def to_dict(self) -> dict:
        result: dict = {}
        result["changes"] = from_list(
            lambda x: to_class(TynePropertyUpdateContentChange, x), self.changes
        )
        return result


class CopyCellsContent:
    anchor: str
    to_copy: List[CellChange]

    def __init__(self, anchor: str, to_copy: List[CellChange]) -> None:
        self.anchor = anchor
        self.to_copy = to_copy

    @staticmethod
    def from_dict(obj: Any) -> "CopyCellsContent":
        assert isinstance(obj, dict)
        anchor = from_str(obj.get("anchor"))
        to_copy = from_list(CellChange.from_dict, obj.get("toCopy"))
        return CopyCellsContent(anchor, to_copy)

    def to_dict(self) -> dict:
        result: dict = {}
        result["anchor"] = from_str(self.anchor)
        result["toCopy"] = from_list(lambda x: to_class(CellChange, x), self.to_copy)
        return result


class PopulateFrom:
    cell_id: List[float]
    content: str

    def __init__(self, cell_id: List[float], content: str) -> None:
        self.cell_id = cell_id
        self.content = content

    @staticmethod
    def from_dict(obj: Any) -> "PopulateFrom":
        assert isinstance(obj, dict)
        cell_id = from_list(from_float, obj.get("cellId"))
        content = from_str(obj.get("content"))
        return PopulateFrom(cell_id, content)

    def to_dict(self) -> dict:
        result: dict = {}
        result["cellId"] = from_list(to_float, self.cell_id)
        result["content"] = from_str(self.content)
        return result


class SheetAutofillContent:
    autofill_context: Optional[List[str]]
    populate_from: List[PopulateFrom]
    populate_to_end: List[float]
    populate_to_start: List[float]
    table: Optional[Dict[str, Any]]
    to_fill: Optional[List[List[str]]]

    def __init__(
        self,
        autofill_context: Optional[List[str]],
        populate_from: List[PopulateFrom],
        populate_to_end: List[float],
        populate_to_start: List[float],
        table: Optional[Dict[str, Any]],
        to_fill: Optional[List[List[str]]],
    ) -> None:
        self.autofill_context = autofill_context
        self.populate_from = populate_from
        self.populate_to_end = populate_to_end
        self.populate_to_start = populate_to_start
        self.table = table
        self.to_fill = to_fill

    @staticmethod
    def from_dict(obj: Any) -> "SheetAutofillContent":
        assert isinstance(obj, dict)
        autofill_context = from_union(
            [lambda x: from_list(from_str, x), from_none], obj.get("autofillContext")
        )
        populate_from = from_list(PopulateFrom.from_dict, obj.get("populateFrom"))
        populate_to_end = from_list(from_float, obj.get("populateToEnd"))
        populate_to_start = from_list(from_float, obj.get("populateToStart"))
        table = from_union(
            [lambda x: from_dict(lambda x: x, x), from_none], obj.get("table")
        )
        to_fill = from_union(
            [lambda x: from_list(lambda x: from_list(from_str, x), x), from_none],
            obj.get("toFill"),
        )
        return SheetAutofillContent(
            autofill_context,
            populate_from,
            populate_to_end,
            populate_to_start,
            table,
            to_fill,
        )

    def to_dict(self) -> dict:
        result: dict = {}
        if self.autofill_context is not None:
            result["autofillContext"] = from_union(
                [lambda x: from_list(from_str, x), from_none], self.autofill_context
            )
        result["populateFrom"] = from_list(
            lambda x: to_class(PopulateFrom, x), self.populate_from
        )
        result["populateToEnd"] = from_list(to_float, self.populate_to_end)
        result["populateToStart"] = from_list(to_float, self.populate_to_start)
        if self.table is not None:
            result["table"] = from_union(
                [lambda x: from_dict(lambda x: x, x), from_none], self.table
            )
        if self.to_fill is not None:
            result["toFill"] = from_union(
                [lambda x: from_list(lambda x: from_list(from_str, x), x), from_none],
                self.to_fill,
            )
        return result


class SelectionRect:
    max_col: float
    max_row: float
    min_col: float
    min_row: float

    def __init__(
        self, max_col: float, max_row: float, min_col: float, min_row: float
    ) -> None:
        self.max_col = max_col
        self.max_row = max_row
        self.min_col = min_col
        self.min_row = min_row

    @staticmethod
    def from_dict(obj: Any) -> "SelectionRect":
        assert isinstance(obj, dict)
        max_col = from_float(obj.get("max_col"))
        max_row = from_float(obj.get("max_row"))
        min_col = from_float(obj.get("min_col"))
        min_row = from_float(obj.get("min_row"))
        return SelectionRect(max_col, max_row, min_col, min_row)

    def to_dict(self) -> dict:
        result: dict = {}
        result["max_col"] = to_float(self.max_col)
        result["max_row"] = to_float(self.max_row)
        result["min_col"] = to_float(self.min_col)
        result["min_row"] = to_float(self.min_row)
        return result


class Dimension(Enum):
    COL = "col"
    ROW = "row"


class SheetTransform(Enum):
    DELETE = "delete"
    INSERT_BEFORE = "insert_before"


class InsertDeleteContent:
    amount: Optional[float]
    boundary: Optional[SelectionRect]
    cells_to_populate: Optional[List[Dict[str, Any]]]
    dimension: Dimension
    selected_index: float
    sheet_id: Optional[float]
    sheet_transform: SheetTransform

    def __init__(
        self,
        amount: Optional[float],
        boundary: Optional[SelectionRect],
        cells_to_populate: Optional[List[Dict[str, Any]]],
        dimension: Dimension,
        selected_index: float,
        sheet_id: Optional[float],
        sheet_transform: SheetTransform,
    ) -> None:
        self.amount = amount
        self.boundary = boundary
        self.cells_to_populate = cells_to_populate
        self.dimension = dimension
        self.selected_index = selected_index
        self.sheet_id = sheet_id
        self.sheet_transform = sheet_transform

    @staticmethod
    def from_dict(obj: Any) -> "InsertDeleteContent":
        assert isinstance(obj, dict)
        amount = from_union([from_float, from_none], obj.get("amount"))
        boundary = from_union([SelectionRect.from_dict, from_none], obj.get("boundary"))
        cells_to_populate = from_union(
            [lambda x: from_list(lambda x: from_dict(lambda x: x, x), x), from_none],
            obj.get("cellsToPopulate"),
        )
        dimension = Dimension(obj.get("dimension"))
        selected_index = from_float(obj.get("selectedIndex"))
        sheet_id = from_union([from_float, from_none], obj.get("sheetId"))
        sheet_transform = SheetTransform(obj.get("sheetTransform"))
        return InsertDeleteContent(
            amount,
            boundary,
            cells_to_populate,
            dimension,
            selected_index,
            sheet_id,
            sheet_transform,
        )

    def to_dict(self) -> dict:
        result: dict = {}
        if self.amount is not None:
            result["amount"] = from_union([to_float, from_none], self.amount)
        if self.boundary is not None:
            result["boundary"] = from_union(
                [lambda x: to_class(SelectionRect, x), from_none], self.boundary
            )
        if self.cells_to_populate is not None:
            result["cellsToPopulate"] = from_union(
                [
                    lambda x: from_list(lambda x: from_dict(lambda x: x, x), x),
                    from_none,
                ],
                self.cells_to_populate,
            )
        result["dimension"] = to_enum(Dimension, self.dimension)
        result["selectedIndex"] = to_float(self.selected_index)
        if self.sheet_id is not None:
            result["sheetId"] = from_union([to_float, from_none], self.sheet_id)
        result["sheetTransform"] = to_enum(SheetTransform, self.sheet_transform)
        return result


class DragRowColumnContent:
    amount: float
    dimension: Dimension
    from_index: float
    sheet_id: float
    to_index: float

    def __init__(
        self,
        amount: float,
        dimension: Dimension,
        from_index: float,
        sheet_id: float,
        to_index: float,
    ) -> None:
        self.amount = amount
        self.dimension = dimension
        self.from_index = from_index
        self.sheet_id = sheet_id
        self.to_index = to_index

    @staticmethod
    def from_dict(obj: Any) -> "DragRowColumnContent":
        assert isinstance(obj, dict)
        amount = from_float(obj.get("amount"))
        dimension = Dimension(obj.get("dimension"))
        from_index = from_float(obj.get("fromIndex"))
        sheet_id = from_float(obj.get("sheetId"))
        to_index = from_float(obj.get("toIndex"))
        return DragRowColumnContent(amount, dimension, from_index, sheet_id, to_index)

    def to_dict(self) -> dict:
        result: dict = {}
        result["amount"] = to_float(self.amount)
        result["dimension"] = to_enum(Dimension, self.dimension)
        result["fromIndex"] = to_float(self.from_index)
        result["sheetId"] = to_float(self.sheet_id)
        result["toIndex"] = to_float(self.to_index)
        return result


class WidgetValueContent:
    cell_id: str
    value: Any

    def __init__(self, cell_id: str, value: Any) -> None:
        self.cell_id = cell_id
        self.value = value

    @staticmethod
    def from_dict(obj: Any) -> "WidgetValueContent":
        assert isinstance(obj, dict)
        cell_id = from_str(obj.get("cellId"))
        value = obj.get("value")
        return WidgetValueContent(cell_id, value)

    def to_dict(self) -> dict:
        result: dict = {}
        result["cellId"] = from_str(self.cell_id)
        result["value"] = self.value
        return result


class WidgetParamType(Enum):
    BOOLEAN = "boolean"
    COLOR = "color"
    DICT = "dict"
    ENUM = "enum"
    FLOAT = "float"
    FUNCTION = "function"
    INT = "int"
    LIST = "list"
    OTHER = "other"
    STRING = "string"


class WidgetParamDefinition:
    category: Optional[str]
    default_value: Any
    description: str
    enum_values: Optional[Dict[str, Any]]
    inline: bool
    kw_only: bool
    name: str
    optional: bool
    type: WidgetParamType

    def __init__(
        self,
        category: Optional[str],
        default_value: Any,
        description: str,
        enum_values: Optional[Dict[str, Any]],
        inline: bool,
        kw_only: bool,
        name: str,
        optional: bool,
        type: WidgetParamType,
    ) -> None:
        self.category = category
        self.default_value = default_value
        self.description = description
        self.enum_values = enum_values
        self.inline = inline
        self.kw_only = kw_only
        self.name = name
        self.optional = optional
        self.type = type

    @staticmethod
    def from_dict(obj: Any) -> "WidgetParamDefinition":
        assert isinstance(obj, dict)
        category = from_union([from_str, from_none], obj.get("category"))
        default_value = obj.get("defaultValue")
        description = from_str(obj.get("description"))
        enum_values = from_union(
            [lambda x: from_dict(lambda x: x, x), from_none], obj.get("enumValues")
        )
        inline = from_bool(obj.get("inline"))
        kw_only = from_bool(obj.get("kwOnly"))
        name = from_str(obj.get("name"))
        optional = from_bool(obj.get("optional"))
        type = WidgetParamType(obj.get("type"))
        return WidgetParamDefinition(
            category,
            default_value,
            description,
            enum_values,
            inline,
            kw_only,
            name,
            optional,
            type,
        )

    def to_dict(self) -> dict:
        result: dict = {}
        if self.category is not None:
            result["category"] = from_union([from_str, from_none], self.category)
        if self.default_value is not None:
            result["defaultValue"] = self.default_value
        result["description"] = from_str(self.description)
        if self.enum_values is not None:
            result["enumValues"] = from_union(
                [lambda x: from_dict(lambda x: x, x), from_none], self.enum_values
            )
        result["inline"] = from_bool(self.inline)
        result["kwOnly"] = from_bool(self.kw_only)
        result["name"] = from_str(self.name)
        result["optional"] = from_bool(self.optional)
        result["type"] = to_enum(WidgetParamType, self.type)
        return result


class WidgetDefinition:
    category: str
    description: str
    name: str
    params: List[WidgetParamDefinition]

    def __init__(
        self,
        category: str,
        description: str,
        name: str,
        params: List[WidgetParamDefinition],
    ) -> None:
        self.category = category
        self.description = description
        self.name = name
        self.params = params

    @staticmethod
    def from_dict(obj: Any) -> "WidgetDefinition":
        assert isinstance(obj, dict)
        category = from_str(obj.get("category"))
        description = from_str(obj.get("description"))
        name = from_str(obj.get("name"))
        params = from_list(WidgetParamDefinition.from_dict, obj.get("params"))
        return WidgetDefinition(category, description, name, params)

    def to_dict(self) -> dict:
        result: dict = {}
        result["category"] = from_str(self.category)
        result["description"] = from_str(self.description)
        result["name"] = from_str(self.name)
        result["params"] = from_list(
            lambda x: to_class(WidgetParamDefinition, x), self.params
        )
        return result


class WidgetRegistry:
    widgets: Dict[str, WidgetDefinition]

    def __init__(self, widgets: Dict[str, WidgetDefinition]) -> None:
        self.widgets = widgets

    @staticmethod
    def from_dict(obj: Any) -> "WidgetRegistry":
        assert isinstance(obj, dict)
        widgets = from_dict(WidgetDefinition.from_dict, obj.get("widgets"))
        return WidgetRegistry(widgets)

    def to_dict(self) -> dict:
        result: dict = {}
        result["widgets"] = from_dict(
            lambda x: to_class(WidgetDefinition, x), self.widgets
        )
        return result


class InsertDeleteReplyCellType:
    cell_updates: List[Dict[str, Any]]
    n_cols: float
    n_rows: float
    sheet_attribute_updates: Dict[str, Any]
    sheet_id: float
    sheet_name: str

    def __init__(
        self,
        cell_updates: List[Dict[str, Any]],
        n_cols: float,
        n_rows: float,
        sheet_attribute_updates: Dict[str, Any],
        sheet_id: float,
        sheet_name: str,
    ) -> None:
        self.cell_updates = cell_updates
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.sheet_attribute_updates = sheet_attribute_updates
        self.sheet_id = sheet_id
        self.sheet_name = sheet_name

    @staticmethod
    def from_dict(obj: Any) -> "InsertDeleteReplyCellType":
        assert isinstance(obj, dict)
        cell_updates = from_list(
            lambda x: from_dict(lambda x: x, x), obj.get("cell_updates")
        )
        n_cols = from_float(obj.get("n_cols"))
        n_rows = from_float(obj.get("n_rows"))
        sheet_attribute_updates = from_dict(
            lambda x: x, obj.get("sheet_attribute_updates")
        )
        sheet_id = from_float(obj.get("sheet_id"))
        sheet_name = from_str(obj.get("sheet_name"))
        return InsertDeleteReplyCellType(
            cell_updates, n_cols, n_rows, sheet_attribute_updates, sheet_id, sheet_name
        )

    def to_dict(self) -> dict:
        result: dict = {}
        result["cell_updates"] = from_list(
            lambda x: from_dict(lambda x: x, x), self.cell_updates
        )
        result["n_cols"] = to_float(self.n_cols)
        result["n_rows"] = to_float(self.n_rows)
        result["sheet_attribute_updates"] = from_dict(
            lambda x: x, self.sheet_attribute_updates
        )
        result["sheet_id"] = to_float(self.sheet_id)
        result["sheet_name"] = from_str(self.sheet_name)
        return result


class DeleteSheetContent:
    sheet_id: float

    def __init__(self, sheet_id: float) -> None:
        self.sheet_id = sheet_id

    @staticmethod
    def from_dict(obj: Any) -> "DeleteSheetContent":
        assert isinstance(obj, dict)
        sheet_id = from_float(obj.get("sheetId"))
        return DeleteSheetContent(sheet_id)

    def to_dict(self) -> dict:
        result: dict = {}
        result["sheetId"] = to_float(self.sheet_id)
        return result


class Subscriber:
    user_color: str
    user_email: str
    user_name: str
    user_profile_image: str

    def __init__(
        self, user_color: str, user_email: str, user_name: str, user_profile_image: str
    ) -> None:
        self.user_color = user_color
        self.user_email = user_email
        self.user_name = user_name
        self.user_profile_image = user_profile_image

    @staticmethod
    def from_dict(obj: Any) -> "Subscriber":
        assert isinstance(obj, dict)
        user_color = from_str(obj.get("user_color"))
        user_email = from_str(obj.get("user_email"))
        user_name = from_str(obj.get("user_name"))
        user_profile_image = from_str(obj.get("user_profile_image"))
        return Subscriber(user_color, user_email, user_name, user_profile_image)

    def to_dict(self) -> dict:
        result: dict = {}
        result["user_color"] = from_str(self.user_color)
        result["user_email"] = from_str(self.user_email)
        result["user_name"] = from_str(self.user_name)
        result["user_profile_image"] = from_str(self.user_profile_image)
        return result


class SubscribersUpdatedContent:
    subscribers: List[Subscriber]

    def __init__(self, subscribers: List[Subscriber]) -> None:
        self.subscribers = subscribers

    @staticmethod
    def from_dict(obj: Any) -> "SubscribersUpdatedContent":
        assert isinstance(obj, dict)
        subscribers = from_list(Subscriber.from_dict, obj.get("subscribers"))
        return SubscribersUpdatedContent(subscribers)

    def to_dict(self) -> dict:
        result: dict = {}
        result["subscribers"] = from_list(
            lambda x: to_class(Subscriber, x), self.subscribers
        )
        return result


class RenameSheetContent:
    name: str
    sheet_id: float

    def __init__(self, name: str, sheet_id: float) -> None:
        self.name = name
        self.sheet_id = sheet_id

    @staticmethod
    def from_dict(obj: Any) -> "RenameSheetContent":
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        sheet_id = from_float(obj.get("sheetId"))
        return RenameSheetContent(name, sheet_id)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        result["sheetId"] = to_float(self.sheet_id)
        return result


class InstallRequirementsContent:
    requirements: str

    def __init__(self, requirements: str) -> None:
        self.requirements = requirements

    @staticmethod
    def from_dict(obj: Any) -> "InstallRequirementsContent":
        assert isinstance(obj, dict)
        requirements = from_str(obj.get("requirements"))
        return InstallRequirementsContent(requirements)

    def to_dict(self) -> dict:
        result: dict = {}
        result["requirements"] = from_str(self.requirements)
        return result


class DownloadRequest:
    filename: str
    mimetype: str
    payload: str

    def __init__(self, filename: str, mimetype: str, payload: str) -> None:
        self.filename = filename
        self.mimetype = mimetype
        self.payload = payload

    @staticmethod
    def from_dict(obj: Any) -> "DownloadRequest":
        assert isinstance(obj, dict)
        filename = from_str(obj.get("filename"))
        mimetype = from_str(obj.get("mimetype"))
        payload = from_str(obj.get("payload"))
        return DownloadRequest(filename, mimetype, payload)

    def to_dict(self) -> dict:
        result: dict = {}
        result["filename"] = from_str(self.filename)
        result["mimetype"] = from_str(self.mimetype)
        result["payload"] = from_str(self.payload)
        return result


class TracebackFrame:
    current_cell: bool
    exec_count: Optional[float]
    line: str
    lineno: float

    def __init__(
        self, current_cell: bool, exec_count: Optional[float], line: str, lineno: float
    ) -> None:
        self.current_cell = current_cell
        self.exec_count = exec_count
        self.line = line
        self.lineno = lineno

    @staticmethod
    def from_dict(obj: Any) -> "TracebackFrame":
        assert isinstance(obj, dict)
        current_cell = from_bool(obj.get("current_cell"))
        exec_count = from_union([from_float, from_none], obj.get("exec_count"))
        line = from_str(obj.get("line"))
        lineno = from_float(obj.get("lineno"))
        return TracebackFrame(current_cell, exec_count, line, lineno)

    def to_dict(self) -> dict:
        result: dict = {}
        result["current_cell"] = from_bool(self.current_cell)
        if self.exec_count is not None:
            result["exec_count"] = from_union([to_float, from_none], self.exec_count)
        result["line"] = from_str(self.line)
        result["lineno"] = to_float(self.lineno)
        return result


class NavigateToContent:
    col: float
    row: float
    sheet: float

    def __init__(self, col: float, row: float, sheet: float) -> None:
        self.col = col
        self.row = row
        self.sheet = sheet

    @staticmethod
    def from_dict(obj: Any) -> "NavigateToContent":
        assert isinstance(obj, dict)
        col = from_float(obj.get("col"))
        row = from_float(obj.get("row"))
        sheet = from_float(obj.get("sheet"))
        return NavigateToContent(col, row, sheet)

    def to_dict(self) -> dict:
        result: dict = {}
        result["col"] = to_float(self.col)
        result["row"] = to_float(self.row)
        result["sheet"] = to_float(self.sheet)
        return result


class WidgetGetStateContent:
    cell_id: List[float]

    def __init__(self, cell_id: List[float]) -> None:
        self.cell_id = cell_id

    @staticmethod
    def from_dict(obj: Any) -> "WidgetGetStateContent":
        assert isinstance(obj, dict)
        cell_id = from_list(from_float, obj.get("cellId"))
        return WidgetGetStateContent(cell_id)

    def to_dict(self) -> dict:
        result: dict = {}
        result["cellId"] = from_list(to_float, self.cell_id)
        return result


class WidgetValidateParamsContent:
    code: str
    params: Dict[str, str]

    def __init__(self, code: str, params: Dict[str, str]) -> None:
        self.code = code
        self.params = params

    @staticmethod
    def from_dict(obj: Any) -> "WidgetValidateParamsContent":
        assert isinstance(obj, dict)
        code = from_str(obj.get("code"))
        params = from_dict(from_str, obj.get("params"))
        return WidgetValidateParamsContent(code, params)

    def to_dict(self) -> dict:
        result: dict = {}
        result["code"] = from_str(self.code)
        result["params"] = from_dict(from_str, self.params)
        return result


class Severity(Enum):
    ERROR = "ERROR"
    INFO = "INFO"
    WARNING = "WARNING"


class TyneEvent:
    date: str
    extra: Dict[str, Any]
    message: str
    severity: Severity

    def __init__(
        self, date: str, extra: Dict[str, Any], message: str, severity: Severity
    ) -> None:
        self.date = date
        self.extra = extra
        self.message = message
        self.severity = severity

    @staticmethod
    def from_dict(obj: Any) -> "TyneEvent":
        assert isinstance(obj, dict)
        date = from_str(obj.get("date"))
        extra = from_dict(lambda x: x, obj.get("extra"))
        message = from_str(obj.get("message"))
        severity = Severity(obj.get("severity"))
        return TyneEvent(date, extra, message, severity)

    def to_dict(self) -> dict:
        result: dict = {}
        result["date"] = from_str(self.date)
        result["extra"] = from_dict(lambda x: x, self.extra)
        result["message"] = from_str(self.message)
        result["severity"] = to_enum(Severity, self.severity)
        return result


class StripeSubscription:
    cancel_at_period_end: bool
    current_period_end: float
    owner_email: str
    portal_url: str
    status: str

    def __init__(
        self,
        cancel_at_period_end: bool,
        current_period_end: float,
        owner_email: str,
        portal_url: str,
        status: str,
    ) -> None:
        self.cancel_at_period_end = cancel_at_period_end
        self.current_period_end = current_period_end
        self.owner_email = owner_email
        self.portal_url = portal_url
        self.status = status

    @staticmethod
    def from_dict(obj: Any) -> "StripeSubscription":
        assert isinstance(obj, dict)
        cancel_at_period_end = from_bool(obj.get("cancelAtPeriodEnd"))
        current_period_end = from_float(obj.get("currentPeriodEnd"))
        owner_email = from_str(obj.get("ownerEmail"))
        portal_url = from_str(obj.get("portalUrl"))
        status = from_str(obj.get("status"))
        return StripeSubscription(
            cancel_at_period_end, current_period_end, owner_email, portal_url, status
        )

    def to_dict(self) -> dict:
        result: dict = {}
        result["cancelAtPeriodEnd"] = from_bool(self.cancel_at_period_end)
        result["currentPeriodEnd"] = to_float(self.current_period_end)
        result["ownerEmail"] = from_str(self.owner_email)
        result["portalUrl"] = from_str(self.portal_url)
        result["status"] = from_str(self.status)
        return result


class RenameTyneContent:
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @staticmethod
    def from_dict(obj: Any) -> "RenameTyneContent":
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        return RenameTyneContent(name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        return result


class SetSecretsContent:
    tyne: Dict[str, str]
    user: Dict[str, str]

    def __init__(self, tyne: Dict[str, str], user: Dict[str, str]) -> None:
        self.tyne = tyne
        self.user = user

    @staticmethod
    def from_dict(obj: Any) -> "SetSecretsContent":
        assert isinstance(obj, dict)
        tyne = from_dict(from_str, obj.get("tyne"))
        user = from_dict(from_str, obj.get("user"))
        return SetSecretsContent(tyne, user)

    def to_dict(self) -> dict:
        result: dict = {}
        result["tyne"] = from_dict(from_str, self.tyne)
        result["user"] = from_dict(from_str, self.user)
        return result


class TickReplyContent:
    addresses: List[List[float]]
    expressions: List[str]

    def __init__(self, addresses: List[List[float]], expressions: List[str]) -> None:
        self.addresses = addresses
        self.expressions = expressions

    @staticmethod
    def from_dict(obj: Any) -> "TickReplyContent":
        assert isinstance(obj, dict)
        addresses = from_list(lambda x: from_list(from_float, x), obj.get("addresses"))
        expressions = from_list(from_str, obj.get("expressions"))
        return TickReplyContent(addresses, expressions)

    def to_dict(self) -> dict:
        result: dict = {}
        result["addresses"] = from_list(
            lambda x: from_list(to_float, x), self.addresses
        )
        result["expressions"] = from_list(from_str, self.expressions)
        return result


class OrganizationCreateContent:
    domain: Optional[str]
    name: str

    def __init__(self, domain: Optional[str], name: str) -> None:
        self.domain = domain
        self.name = name

    @staticmethod
    def from_dict(obj: Any) -> "OrganizationCreateContent":
        assert isinstance(obj, dict)
        domain = from_union([from_str, from_none], obj.get("domain"))
        name = from_str(obj.get("name"))
        return OrganizationCreateContent(domain, name)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.domain is not None:
            result["domain"] = from_union([from_str, from_none], self.domain)
        result["name"] = from_str(self.name)
        return result


class AccessMode(Enum):
    APP = "APP"
    EDIT = "EDIT"
    MAINTENANCE = "MAINTENANCE"
    READ_ONLY_CONNECTED = "READ_ONLY_CONNECTED"
    READ_ONLY_DISCONNECTED = "READ_ONLY_DISCONNECTED"


class GSheetsImage:
    action: Optional[str]
    action_number: Optional[float]
    address: str
    col: float
    object_type: str
    row: float
    sheet: float
    url: str

    def __init__(
        self,
        action: Optional[str],
        action_number: Optional[float],
        address: str,
        col: float,
        object_type: str,
        row: float,
        sheet: float,
        url: str,
    ) -> None:
        self.action = action
        self.action_number = action_number
        self.address = address
        self.col = col
        self.object_type = object_type
        self.row = row
        self.sheet = sheet
        self.url = url

    @staticmethod
    def from_dict(obj: Any) -> "GSheetsImage":
        assert isinstance(obj, dict)
        action = from_union([from_str, from_none], obj.get("action"))
        action_number = from_union([from_float, from_none], obj.get("actionNumber"))
        address = from_str(obj.get("address"))
        col = from_float(obj.get("col"))
        object_type = from_str(obj.get("objectType"))
        row = from_float(obj.get("row"))
        sheet = from_float(obj.get("sheet"))
        url = from_str(obj.get("url"))
        return GSheetsImage(
            action, action_number, address, col, object_type, row, sheet, url
        )

    def to_dict(self) -> dict:
        result: dict = {}
        if self.action is not None:
            result["action"] = from_union([from_str, from_none], self.action)
        if self.action_number is not None:
            result["actionNumber"] = from_union(
                [to_float, from_none], self.action_number
            )
        result["address"] = from_str(self.address)
        result["col"] = to_float(self.col)
        result["objectType"] = from_str(self.object_type)
        result["row"] = to_float(self.row)
        result["sheet"] = to_float(self.sheet)
        result["url"] = from_str(self.url)
        return result


class ResearchMessage:
    msg: str

    def __init__(self, msg: str) -> None:
        self.msg = msg

    @staticmethod
    def from_dict(obj: Any) -> "ResearchMessage":
        assert isinstance(obj, dict)
        msg = from_str(obj.get("msg"))
        return ResearchMessage(msg)

    def to_dict(self) -> dict:
        result: dict = {}
        result["msg"] = from_str(self.msg)
        return result


class ResearchError:
    error: str

    def __init__(self, error: str) -> None:
        self.error = error

    @staticmethod
    def from_dict(obj: Any) -> "ResearchError":
        assert isinstance(obj, dict)
        error = from_str(obj.get("error"))
        return ResearchError(error)

    def to_dict(self) -> dict:
        result: dict = {}
        result["error"] = from_str(self.error)
        return result


class ResearchCell:
    col: float
    row: float

    def __init__(self, col: float, row: float) -> None:
        self.col = col
        self.row = row

    @staticmethod
    def from_dict(obj: Any) -> "ResearchCell":
        assert isinstance(obj, dict)
        col = from_float(obj.get("col"))
        row = from_float(obj.get("row"))
        return ResearchCell(col, row)

    def to_dict(self) -> dict:
        result: dict = {}
        result["col"] = to_float(self.col)
        result["row"] = to_float(self.row)
        return result


class ResearchSource:
    cells: List[ResearchCell]
    title: str
    url: str

    def __init__(self, cells: List[ResearchCell], title: str, url: str) -> None:
        self.cells = cells
        self.title = title
        self.url = url

    @staticmethod
    def from_dict(obj: Any) -> "ResearchSource":
        assert isinstance(obj, dict)
        cells = from_list(ResearchCell.from_dict, obj.get("cells"))
        title = from_str(obj.get("title"))
        url = from_str(obj.get("url"))
        return ResearchSource(cells, title, url)

    def to_dict(self) -> dict:
        result: dict = {}
        result["cells"] = from_list(lambda x: to_class(ResearchCell, x), self.cells)
        result["title"] = from_str(self.title)
        result["url"] = from_str(self.url)
        return result


class ResearchUsage:
    ai_calls: float
    completion_tokens: float
    embeddings_calls: float
    phantom_js_calls: float
    prompt_tokens: float
    running_time: float
    """Enables basic storage and retrieval of dates and times."""
    start_time: datetime
    web_searches: float

    def __init__(
        self,
        ai_calls: float,
        completion_tokens: float,
        embeddings_calls: float,
        phantom_js_calls: float,
        prompt_tokens: float,
        running_time: float,
        start_time: datetime,
        web_searches: float,
    ) -> None:
        self.ai_calls = ai_calls
        self.completion_tokens = completion_tokens
        self.embeddings_calls = embeddings_calls
        self.phantom_js_calls = phantom_js_calls
        self.prompt_tokens = prompt_tokens
        self.running_time = running_time
        self.start_time = start_time
        self.web_searches = web_searches

    @staticmethod
    def from_dict(obj: Any) -> "ResearchUsage":
        assert isinstance(obj, dict)
        ai_calls = from_float(obj.get("AICalls"))
        completion_tokens = from_float(obj.get("completionTokens"))
        embeddings_calls = from_float(obj.get("embeddingsCalls"))
        phantom_js_calls = from_float(obj.get("phantomJsCalls"))
        prompt_tokens = from_float(obj.get("promptTokens"))
        running_time = from_float(obj.get("runningTime"))
        start_time = from_datetime(obj.get("startTime"))
        web_searches = from_float(obj.get("webSearches"))
        return ResearchUsage(
            ai_calls,
            completion_tokens,
            embeddings_calls,
            phantom_js_calls,
            prompt_tokens,
            running_time,
            start_time,
            web_searches,
        )

    def to_dict(self) -> dict:
        result: dict = {}
        result["AICalls"] = to_float(self.ai_calls)
        result["completionTokens"] = to_float(self.completion_tokens)
        result["embeddingsCalls"] = to_float(self.embeddings_calls)
        result["phantomJsCalls"] = to_float(self.phantom_js_calls)
        result["promptTokens"] = to_float(self.prompt_tokens)
        result["runningTime"] = to_float(self.running_time)
        result["startTime"] = self.start_time.isoformat()
        result["webSearches"] = to_float(self.web_searches)
        return result


class ResearchTable:
    sources: List[ResearchSource]
    table: List[List[Union[float, None, str]]]
    usage: Optional[ResearchUsage]

    def __init__(
        self,
        sources: List[ResearchSource],
        table: List[List[Union[float, None, str]]],
        usage: Optional[ResearchUsage],
    ) -> None:
        self.sources = sources
        self.table = table
        self.usage = usage

    @staticmethod
    def from_dict(obj: Any) -> "ResearchTable":
        assert isinstance(obj, dict)
        sources = from_list(ResearchSource.from_dict, obj.get("sources"))
        table = from_list(
            lambda x: from_list(
                lambda x: from_union([from_float, from_str, from_none], x), x
            ),
            obj.get("table"),
        )
        usage = from_union([ResearchUsage.from_dict, from_none], obj.get("usage"))
        return ResearchTable(sources, table, usage)

    def to_dict(self) -> dict:
        result: dict = {}
        result["sources"] = from_list(
            lambda x: to_class(ResearchSource, x), self.sources
        )
        result["table"] = from_list(
            lambda x: from_list(
                lambda x: from_union([to_float, from_str, from_none], x), x
            ),
            self.table,
        )
        if self.usage is not None:
            result["usage"] = from_union(
                [lambda x: to_class(ResearchUsage, x), from_none], self.usage
            )
        return result


class StreamlitAppConfig:
    auto_open: bool
    height: float
    public: bool
    sidebar: bool
    width: float
    window_caption: str

    def __init__(
        self,
        auto_open: bool,
        height: float,
        public: bool,
        sidebar: bool,
        width: float,
        window_caption: str,
    ) -> None:
        self.auto_open = auto_open
        self.height = height
        self.public = public
        self.sidebar = sidebar
        self.width = width
        self.window_caption = window_caption

    @staticmethod
    def from_dict(obj: Any) -> "StreamlitAppConfig":
        assert isinstance(obj, dict)
        auto_open = from_bool(obj.get("auto_open"))
        height = from_float(obj.get("height"))
        public = from_bool(obj.get("public"))
        sidebar = from_bool(obj.get("sidebar"))
        width = from_float(obj.get("width"))
        window_caption = from_str(obj.get("windowCaption"))
        return StreamlitAppConfig(
            auto_open, height, public, sidebar, width, window_caption
        )

    def to_dict(self) -> dict:
        result: dict = {}
        result["auto_open"] = from_bool(self.auto_open)
        result["height"] = to_float(self.height)
        result["public"] = from_bool(self.public)
        result["sidebar"] = from_bool(self.sidebar)
        result["width"] = to_float(self.width)
        result["windowCaption"] = from_str(self.window_caption)
        return result


class SheetData:
    name: str
    values: List[List[Union[float, str]]]

    def __init__(self, name: str, values: List[List[Union[float, str]]]) -> None:
        self.name = name
        self.values = values

    @staticmethod
    def from_dict(obj: Any) -> "SheetData":
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        values = from_list(
            lambda x: from_list(lambda x: from_union([from_float, from_str], x), x),
            obj.get("values"),
        )
        return SheetData(name, values)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        result["values"] = from_list(
            lambda x: from_list(lambda x: from_union([to_float, from_str], x), x),
            self.values,
        )
        return result


class UserViewState:
    latest_release_notes_viewed: Optional[str]
    show_get_started_on_new_sheet: Optional[bool]

    def __init__(
        self,
        latest_release_notes_viewed: Optional[str],
        show_get_started_on_new_sheet: Optional[bool],
    ) -> None:
        self.latest_release_notes_viewed = latest_release_notes_viewed
        self.show_get_started_on_new_sheet = show_get_started_on_new_sheet

    @staticmethod
    def from_dict(obj: Any) -> "UserViewState":
        assert isinstance(obj, dict)
        latest_release_notes_viewed = from_union(
            [from_str, from_none], obj.get("latestReleaseNotesViewed")
        )
        show_get_started_on_new_sheet = from_union(
            [from_bool, from_none], obj.get("showGetStartedOnNewSheet")
        )
        return UserViewState(latest_release_notes_viewed, show_get_started_on_new_sheet)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.latest_release_notes_viewed is not None:
            result["latestReleaseNotesViewed"] = from_union(
                [from_str, from_none], self.latest_release_notes_viewed
            )
        if self.show_get_started_on_new_sheet is not None:
            result["showGetStartedOnNewSheet"] = from_union(
                [from_bool, from_none], self.show_get_started_on_new_sheet
            )
        return result


def kernel_protocol_from_dict(s: Any) -> KernelProtocol:
    return KernelProtocol(s)


def kernel_protocol_to_dict(x: KernelProtocol) -> Any:
    return to_enum(KernelProtocol, x)


def cell_attribute_from_dict(s: Any) -> CellAttribute:
    return CellAttribute(s)


def cell_attribute_to_dict(x: CellAttribute) -> Any:
    return to_enum(CellAttribute, x)


def special_users_from_dict(s: Any) -> SpecialUsers:
    return SpecialUsers(s)


def special_users_to_dict(x: SpecialUsers) -> Any:
    return to_enum(SpecialUsers, x)


def tyne_categories_from_dict(s: Any) -> TyneCategories:
    return TyneCategories(s)


def tyne_categories_to_dict(x: TyneCategories) -> Any:
    return to_enum(TyneCategories, x)


def access_scope_from_dict(s: Any) -> AccessScope:
    return AccessScope(s)


def access_scope_to_dict(x: AccessScope) -> Any:
    return to_enum(AccessScope, x)


def access_level_from_dict(s: Any) -> AccessLevel:
    return AccessLevel(s)


def access_level_to_dict(x: AccessLevel) -> Any:
    return to_enum(AccessLevel, x)


def tyne_list_item_from_dict(s: Any) -> TyneListItem:
    return TyneListItem.from_dict(s)


def tyne_list_item_to_dict(x: TyneListItem) -> Any:
    return to_class(TyneListItem, x)


def share_record_from_dict(s: Any) -> ShareRecord:
    return ShareRecord.from_dict(s)


def share_record_to_dict(x: ShareRecord) -> Any:
    return to_class(ShareRecord, x)


def tyne_share_response_from_dict(s: Any) -> TyneShareResponse:
    return TyneShareResponse.from_dict(s)


def tyne_share_response_to_dict(x: TyneShareResponse) -> Any:
    return to_class(TyneShareResponse, x)


def line_wrap_from_dict(s: Any) -> LineWrap:
    return LineWrap(s)


def line_wrap_to_dict(x: LineWrap) -> Any:
    return to_enum(LineWrap, x)


def text_style_from_dict(s: Any) -> TextStyle:
    return TextStyle(s)


def text_style_to_dict(x: TextStyle) -> Any:
    return to_enum(TextStyle, x)


def text_align_from_dict(s: Any) -> TextAlign:
    return TextAlign(s)


def text_align_to_dict(x: TextAlign) -> Any:
    return to_enum(TextAlign, x)


def vertical_align_from_dict(s: Any) -> VerticalAlign:
    return VerticalAlign(s)


def vertical_align_to_dict(x: VerticalAlign) -> Any:
    return to_enum(VerticalAlign, x)


def border_type_from_dict(s: Any) -> BorderType:
    return BorderType(s)


def border_type_to_dict(x: BorderType) -> Any:
    return to_enum(BorderType, x)


def number_format_from_dict(s: Any) -> NumberFormat:
    return NumberFormat(s)


def number_format_to_dict(x: NumberFormat) -> Any:
    return to_enum(NumberFormat, x)


def message_types_from_dict(s: Any) -> MessageTypes:
    return MessageTypes(s)


def message_types_to_dict(x: MessageTypes) -> Any:
    return to_enum(MessageTypes, x)


def kernel_init_state_from_dict(s: Any) -> KernelInitState:
    return KernelInitState(s)


def kernel_init_state_to_dict(x: KernelInitState) -> Any:
    return to_enum(KernelInitState, x)


def mime_types_from_dict(s: Any) -> MIMETypes:
    return MIMETypes(s)


def mime_types_to_dict(x: MIMETypes) -> Any:
    return to_enum(MIMETypes, x)


def dimension_from_dict(s: Any) -> Dimension:
    return Dimension(s)


def dimension_to_dict(x: Dimension) -> Any:
    return to_enum(Dimension, x)


def sheet_transform_from_dict(s: Any) -> SheetTransform:
    return SheetTransform(s)


def sheet_transform_to_dict(x: SheetTransform) -> Any:
    return to_enum(SheetTransform, x)


def widget_param_type_from_dict(s: Any) -> WidgetParamType:
    return WidgetParamType(s)


def widget_param_type_to_dict(x: WidgetParamType) -> Any:
    return to_enum(WidgetParamType, x)


def sheet_unaware_cell_id_from_dict(s: Any) -> List[float]:
    return from_list(from_float, s)


def sheet_unaware_cell_id_to_dict(x: List[float]) -> Any:
    return from_list(to_float, x)


def sheet_cell_id_from_dict(s: Any) -> List[float]:
    return from_list(from_float, s)


def sheet_cell_id_to_dict(x: List[float]) -> Any:
    return from_list(to_float, x)


def notebook_cell_id_from_dict(s: Any) -> str:
    return from_str(s)


def notebook_cell_id_to_dict(x: str) -> Any:
    return from_str(x)


def cell_id_from_dict(s: Any) -> Union[List[float], str]:
    return from_union([lambda x: from_list(from_float, x), from_str], s)


def cell_id_to_dict(x: Union[List[float], str]) -> Any:
    return from_union([lambda x: from_list(to_float, x), from_str], x)


def sheet_attribute_from_dict(s: Any) -> SheetAttribute:
    return SheetAttribute(s)


def sheet_attribute_to_dict(x: SheetAttribute) -> Any:
    return to_enum(SheetAttribute, x)


def sheet_attribute_update_from_dict(s: Any) -> SheetAttributeUpdate:
    return SheetAttributeUpdate.from_dict(s)


def sheet_attribute_update_to_dict(x: SheetAttributeUpdate) -> Any:
    return to_class(SheetAttributeUpdate, x)


def cell_attribute_update_from_dict(s: Any) -> CellAttributeUpdate:
    return CellAttributeUpdate.from_dict(s)


def cell_attribute_update_to_dict(x: CellAttributeUpdate) -> Any:
    return to_class(CellAttributeUpdate, x)


def cell_attributes_update_from_dict(s: Any) -> CellAttributesUpdate:
    return CellAttributesUpdate.from_dict(s)


def cell_attributes_update_to_dict(x: CellAttributesUpdate) -> Any:
    return to_class(CellAttributesUpdate, x)


def call_server_content_from_dict(s: Any) -> CallServerContent:
    return CallServerContent.from_dict(s)


def call_server_content_to_dict(x: CallServerContent) -> Any:
    return to_class(CallServerContent, x)


def cell_change_from_dict(s: Any) -> CellChange:
    return CellChange.from_dict(s)


def cell_change_to_dict(x: CellChange) -> Any:
    return to_class(CellChange, x)


def run_cells_content_from_dict(s: Any) -> RunCellsContent:
    return RunCellsContent.from_dict(s)


def run_cells_content_to_dict(x: RunCellsContent) -> Any:
    return to_class(RunCellsContent, x)


def rerun_cells_content_from_dict(s: Any) -> RerunCellsContent:
    return RerunCellsContent.from_dict(s)


def rerun_cells_content_to_dict(x: RerunCellsContent) -> Any:
    return to_class(RerunCellsContent, x)


def sheet_update_content_from_dict(s: Any) -> SheetUpdateContent:
    return SheetUpdateContent.from_dict(s)


def sheet_update_content_to_dict(x: SheetUpdateContent) -> Any:
    return to_class(SheetUpdateContent, x)


def tyne_property_update_content_change_from_dict(
    s: Any,
) -> TynePropertyUpdateContentChange:
    return TynePropertyUpdateContentChange.from_dict(s)


def tyne_property_update_content_change_to_dict(
    x: TynePropertyUpdateContentChange,
) -> Any:
    return to_class(TynePropertyUpdateContentChange, x)


def tyne_property_update_content_from_dict(s: Any) -> TynePropertyUpdateContent:
    return TynePropertyUpdateContent.from_dict(s)


def tyne_property_update_content_to_dict(x: TynePropertyUpdateContent) -> Any:
    return to_class(TynePropertyUpdateContent, x)


def copy_cells_content_from_dict(s: Any) -> CopyCellsContent:
    return CopyCellsContent.from_dict(s)


def copy_cells_content_to_dict(x: CopyCellsContent) -> Any:
    return to_class(CopyCellsContent, x)


def sheet_autofill_content_from_dict(s: Any) -> SheetAutofillContent:
    return SheetAutofillContent.from_dict(s)


def sheet_autofill_content_to_dict(x: SheetAutofillContent) -> Any:
    return to_class(SheetAutofillContent, x)


def selection_rect_from_dict(s: Any) -> SelectionRect:
    return SelectionRect.from_dict(s)


def selection_rect_to_dict(x: SelectionRect) -> Any:
    return to_class(SelectionRect, x)


def insert_delete_content_from_dict(s: Any) -> InsertDeleteContent:
    return InsertDeleteContent.from_dict(s)


def insert_delete_content_to_dict(x: InsertDeleteContent) -> Any:
    return to_class(InsertDeleteContent, x)


def drag_row_column_content_from_dict(s: Any) -> DragRowColumnContent:
    return DragRowColumnContent.from_dict(s)


def drag_row_column_content_to_dict(x: DragRowColumnContent) -> Any:
    return to_class(DragRowColumnContent, x)


def widget_value_content_from_dict(s: Any) -> WidgetValueContent:
    return WidgetValueContent.from_dict(s)


def widget_value_content_to_dict(x: WidgetValueContent) -> Any:
    return to_class(WidgetValueContent, x)


def widget_param_definition_from_dict(s: Any) -> WidgetParamDefinition:
    return WidgetParamDefinition.from_dict(s)


def widget_param_definition_to_dict(x: WidgetParamDefinition) -> Any:
    return to_class(WidgetParamDefinition, x)


def widget_definition_from_dict(s: Any) -> WidgetDefinition:
    return WidgetDefinition.from_dict(s)


def widget_definition_to_dict(x: WidgetDefinition) -> Any:
    return to_class(WidgetDefinition, x)


def widget_registry_from_dict(s: Any) -> WidgetRegistry:
    return WidgetRegistry.from_dict(s)


def widget_registry_to_dict(x: WidgetRegistry) -> Any:
    return to_class(WidgetRegistry, x)


def insert_delete_reply_from_dict(s: Any) -> InsertDeleteReplyCellType:
    return InsertDeleteReplyCellType.from_dict(s)


def insert_delete_reply_to_dict(x: InsertDeleteReplyCellType) -> Any:
    return to_class(InsertDeleteReplyCellType, x)


def delete_sheet_content_from_dict(s: Any) -> DeleteSheetContent:
    return DeleteSheetContent.from_dict(s)


def delete_sheet_content_to_dict(x: DeleteSheetContent) -> Any:
    return to_class(DeleteSheetContent, x)


def subscriber_from_dict(s: Any) -> Subscriber:
    return Subscriber.from_dict(s)


def subscriber_to_dict(x: Subscriber) -> Any:
    return to_class(Subscriber, x)


def subscribers_updated_content_from_dict(s: Any) -> SubscribersUpdatedContent:
    return SubscribersUpdatedContent.from_dict(s)


def subscribers_updated_content_to_dict(x: SubscribersUpdatedContent) -> Any:
    return to_class(SubscribersUpdatedContent, x)


def rename_sheet_content_from_dict(s: Any) -> RenameSheetContent:
    return RenameSheetContent.from_dict(s)


def rename_sheet_content_to_dict(x: RenameSheetContent) -> Any:
    return to_class(RenameSheetContent, x)


def install_requirements_content_from_dict(s: Any) -> InstallRequirementsContent:
    return InstallRequirementsContent.from_dict(s)


def install_requirements_content_to_dict(x: InstallRequirementsContent) -> Any:
    return to_class(InstallRequirementsContent, x)


def download_request_from_dict(s: Any) -> DownloadRequest:
    return DownloadRequest.from_dict(s)


def download_request_to_dict(x: DownloadRequest) -> Any:
    return to_class(DownloadRequest, x)


def traceback_frame_from_dict(s: Any) -> TracebackFrame:
    return TracebackFrame.from_dict(s)


def traceback_frame_to_dict(x: TracebackFrame) -> Any:
    return to_class(TracebackFrame, x)


def navigate_to_content_from_dict(s: Any) -> NavigateToContent:
    return NavigateToContent.from_dict(s)


def navigate_to_content_to_dict(x: NavigateToContent) -> Any:
    return to_class(NavigateToContent, x)


def widget_get_state_content_from_dict(s: Any) -> WidgetGetStateContent:
    return WidgetGetStateContent.from_dict(s)


def widget_get_state_content_to_dict(x: WidgetGetStateContent) -> Any:
    return to_class(WidgetGetStateContent, x)


def widget_validate_params_content_from_dict(s: Any) -> WidgetValidateParamsContent:
    return WidgetValidateParamsContent.from_dict(s)


def widget_validate_params_content_to_dict(x: WidgetValidateParamsContent) -> Any:
    return to_class(WidgetValidateParamsContent, x)


def tyne_event_from_dict(s: Any) -> TyneEvent:
    return TyneEvent.from_dict(s)


def tyne_event_to_dict(x: TyneEvent) -> Any:
    return to_class(TyneEvent, x)


def stripe_subscription_from_dict(s: Any) -> StripeSubscription:
    return StripeSubscription.from_dict(s)


def stripe_subscription_to_dict(x: StripeSubscription) -> Any:
    return to_class(StripeSubscription, x)


def rename_tyne_content_from_dict(s: Any) -> RenameTyneContent:
    return RenameTyneContent.from_dict(s)


def rename_tyne_content_to_dict(x: RenameTyneContent) -> Any:
    return to_class(RenameTyneContent, x)


def set_secrets_content_from_dict(s: Any) -> SetSecretsContent:
    return SetSecretsContent.from_dict(s)


def set_secrets_content_to_dict(x: SetSecretsContent) -> Any:
    return to_class(SetSecretsContent, x)


def secrets_from_dict(s: Any) -> Dict[str, str]:
    return from_dict(from_str, s)


def secrets_to_dict(x: Dict[str, str]) -> Any:
    return from_dict(from_str, x)


def tick_reply_content_from_dict(s: Any) -> TickReplyContent:
    return TickReplyContent.from_dict(s)


def tick_reply_content_to_dict(x: TickReplyContent) -> Any:
    return to_class(TickReplyContent, x)


def organization_create_content_from_dict(s: Any) -> OrganizationCreateContent:
    return OrganizationCreateContent.from_dict(s)


def organization_create_content_to_dict(x: OrganizationCreateContent) -> Any:
    return to_class(OrganizationCreateContent, x)


def access_mode_from_dict(s: Any) -> AccessMode:
    return AccessMode(s)


def access_mode_to_dict(x: AccessMode) -> Any:
    return to_enum(AccessMode, x)


def g_sheets_image_from_dict(s: Any) -> GSheetsImage:
    return GSheetsImage.from_dict(s)


def g_sheets_image_to_dict(x: GSheetsImage) -> Any:
    return to_class(GSheetsImage, x)


def research_usage_from_dict(s: Any) -> ResearchUsage:
    return ResearchUsage.from_dict(s)


def research_usage_to_dict(x: ResearchUsage) -> Any:
    return to_class(ResearchUsage, x)


def research_message_from_dict(s: Any) -> ResearchMessage:
    return ResearchMessage.from_dict(s)


def research_message_to_dict(x: ResearchMessage) -> Any:
    return to_class(ResearchMessage, x)


def research_error_from_dict(s: Any) -> ResearchError:
    return ResearchError.from_dict(s)


def research_error_to_dict(x: ResearchError) -> Any:
    return to_class(ResearchError, x)


def research_cell_from_dict(s: Any) -> ResearchCell:
    return ResearchCell.from_dict(s)


def research_cell_to_dict(x: ResearchCell) -> Any:
    return to_class(ResearchCell, x)


def research_source_from_dict(s: Any) -> ResearchSource:
    return ResearchSource.from_dict(s)


def research_source_to_dict(x: ResearchSource) -> Any:
    return to_class(ResearchSource, x)


def research_table_from_dict(s: Any) -> ResearchTable:
    return ResearchTable.from_dict(s)


def research_table_to_dict(x: ResearchTable) -> Any:
    return to_class(ResearchTable, x)


def streamlit_app_config_from_dict(s: Any) -> StreamlitAppConfig:
    return StreamlitAppConfig.from_dict(s)


def streamlit_app_config_to_dict(x: StreamlitAppConfig) -> Any:
    return to_class(StreamlitAppConfig, x)


def sheet_data_from_dict(s: Any) -> SheetData:
    return SheetData.from_dict(s)


def sheet_data_to_dict(x: SheetData) -> Any:
    return to_class(SheetData, x)


def user_view_state_from_dict(s: Any) -> UserViewState:
    return UserViewState.from_dict(s)


def user_view_state_to_dict(x: UserViewState) -> Any:
    return to_class(UserViewState, x)


def insert_delete_reply_cell_type_from_dict(s: Any) -> InsertDeleteReplyCellType:
    return InsertDeleteReplyCellType.from_dict(s)


def insert_delete_reply_cell_type_to_dict(x: InsertDeleteReplyCellType) -> Any:
    return to_class(InsertDeleteReplyCellType, x)


def cell_type_from_dict(s: Any) -> Dict[str, Any]:
    return from_dict(lambda x: x, s)


def cell_type_to_dict(x: Dict[str, Any]) -> Any:
    return from_dict(lambda x: x, x)
