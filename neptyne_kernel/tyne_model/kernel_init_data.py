from dataclasses import dataclass
from typing import Iterator

from ..json_tools import dict_from_bytes, dict_to_bytes
from ..streamlit_config import base_url_path
from .cell import CODEPANEL_CELL_ID
from .sheet import Sheet, TyneSheets


@dataclass
class InitPhase1Payload:
    requirements: str
    sheets: list[Sheet]
    in_gs_mode: bool
    gsheets_sheet_id: str
    time_zone: str
    streamlit_base_url_path: str
    env: dict[str, str]

    def to_bytes(self) -> bytes:
        return dict_to_bytes(
            {
                "requirements": self.requirements,
                "sheets": [sheet.to_dict() for sheet in self.sheets],
                "in_gs_mode": self.in_gs_mode,
                "gsheets_sheet_id": self.gsheets_sheet_id,
                "time_zone": self.time_zone,
                "streamlit_base_url_path": self.streamlit_base_url_path,
                "env": self.env,
            }
        )

    @classmethod
    def from_bytes(cls, data_b: bytes) -> "InitPhase1Payload":
        data = dict_from_bytes(data_b)
        return cls(
            requirements=data["requirements"],
            sheets=[Sheet.from_dict(sheet) for sheet in data["sheets"]],
            in_gs_mode=data["in_gs_mode"],
            gsheets_sheet_id=data.get("gsheets_sheet_id", ""),
            time_zone=data.get("time_zone", "UTC"),
            streamlit_base_url_path=data.get("streamlit_base_url_path", ""),
            env=data.get("env", {}),
        )


@dataclass
class InitPhase2Payload:
    sheets: TyneSheets
    requires_recompile: bool

    def to_bytes(self) -> bytes:
        return dict_to_bytes(
            {
                "sheets": self.sheets.to_dict(),
                "requires_recompile": self.requires_recompile,
            }
        )

    @classmethod
    def from_bytes(cls, data_b: bytes) -> "InitPhase2Payload":
        data = dict_from_bytes(data_b)
        return cls(
            sheets=TyneSheets.from_dict(data["sheets"]),
            requires_recompile=data["requires_recompile"],
        )


@dataclass
class TyneInitializationData:
    sheets: TyneSheets
    code_panel_code: str
    requirements: str
    requires_recompile: bool
    shard_id: int
    tyne_file_name: str
    in_gs_mode: bool
    gsheets_sheet_id: str
    time_zone: str
    env: dict[str, str]

    def get_init_code(self) -> Iterator[tuple[str, str]]:
        phase_1 = InitPhase1Payload(
            self.requirements,
            [sheet.copy(without_cells=True) for sheet in self.sheets.sheets.values()],
            self.in_gs_mode,
            self.gsheets_sheet_id,
            self.time_zone,
            base_url_path(self.shard_id, self.tyne_file_name),
            self.env,
        )
        yield "", f"N_.initialize_phase_1({phase_1.to_bytes()!r})"

        globals_module = "gsheets" if self.in_gs_mode else "core"
        yield (
            "",
            f"from neptyne_kernel.kernel_globals.{globals_module} import *",
        )

        if self.code_panel_code:
            yield CODEPANEL_CELL_ID, self.code_panel_code

        phase_2 = InitPhase2Payload(
            self.sheets,
            self.requires_recompile,
        )

        yield "", f"N_.initialize_phase_2({phase_2.to_bytes()!r})"

        if self.code_panel_code:
            yield CODEPANEL_CELL_ID, self.code_panel_code
