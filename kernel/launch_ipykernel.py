import argparse
import os
import tempfile

from ipykernel import kernelapp
from jupyter_client.connect import write_connection_file

SHELL_PORT = 53001
IOPUB_PORT = 53002
STDIN_PORT = 53003
CONTROL_PORT = 53005
HB_PORT = 53004


def determine_connection_file() -> str:
    fd, conn_file = tempfile.mkstemp(suffix=".json", prefix="kernel_connection_")
    os.close(fd)

    return conn_file


def get_config_args() -> list[str]:
    return [
        "--Session.packer=kernel.json_tools.json_packer",
        "--IPKernelApp.kernel_class=kernel.neptyne_kernel.Kernel",
        "--HistoryManager.enabled=False",
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--key", type=str)
    parser.add_argument("--connection-file", type=str, default=None)

    parser.add_argument("--shell-port", type=int, default=SHELL_PORT)
    parser.add_argument("--iopub-port", type=int, default=IOPUB_PORT)
    parser.add_argument("--stdin-port", type=int, default=STDIN_PORT)
    parser.add_argument("--hb-port", type=int, default=HB_PORT)
    parser.add_argument("--control-port", type=int, default=CONTROL_PORT)

    arguments = parser.parse_args()
    key = arguments.key
    ip = "0.0.0.0"
    connection_file = arguments.connection_file or determine_connection_file()

    write_connection_file(
        fname=connection_file,
        ip=ip,
        key=key.encode("utf-8"),
        shell_port=arguments.shell_port,
        iopub_port=arguments.iopub_port,
        stdin_port=arguments.stdin_port,
        hb_port=arguments.hb_port,
        control_port=arguments.control_port,
    )

    kernelapp.launch_new_instance(
        [
            *get_config_args(),
        ],
        connection_file=connection_file,
        ip=ip,
        exec_lines=["from kernel.kernel_init import *"],
        log_level="DEBUG",
    )

    try:
        if arguments.connection_file is None:
            os.remove(connection_file)
    except Exception:
        pass
