from typing import Any

from ipykernel.ipkernel import IPythonKernel


class Kernel(IPythonKernel):
    def finish_metadata(
        self, parent: dict, metadata: dict, reply_content: dict
    ) -> dict[str, Any]:
        super().finish_metadata(parent, metadata, reply_content)
        try:
            from .dash import Dash

            metadata["neptyne"] = Dash.instance().get_metadata()
        except Exception:
            pass
        return metadata
