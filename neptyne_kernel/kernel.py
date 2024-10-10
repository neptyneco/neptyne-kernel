import os
from pathlib import Path
from typing import Any

from ipykernel.ipkernel import IPythonKernel

kernel_module_path = Path(__file__).parent.parent.absolute()


def pythonpath_with_kernel_module() -> str:
    if pythonpath := os.environ.get("PYTHONPATH", ""):
        return f"{pythonpath}{os.path.pathsep}{kernel_module_path}"
    return str(kernel_module_path)


def compile_jupyter(lines: list[str]) -> list[str]:
    result = compile_src("\n".join(lines))

    return [f"{lines}\n" for lines in result.split("\n")]


def compile_src(src: str) -> str:
    from neptyne_kernel.expression_compiler import compile_expression

    if not src.strip():
        return "\n"
    result = compile_expression(
        src, compute_cells_mentioned=False, reformat_compiled_code=False
    ).compiled_code
    return result


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

    def start(self) -> None:
        super().start()
        if self.shell:
            self.shell.run_cell("from neptyne_kernel.kernel_init import *")

    async def do_execute(
        self,
        code: str,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        if bool(os.getenv("NEPTYNE_LOCAL_REPL")):
            code = "".join(compile_src(code))
        return await super().do_execute(
            code,
            *args,
            **kwargs,
        )


def kernel_main() -> None:
    from ipykernel.kernelapp import IPKernelApp

    IPKernelApp.launch_instance(kernel_class=Kernel)


if __name__ == "__main__":
    kernel_main()
