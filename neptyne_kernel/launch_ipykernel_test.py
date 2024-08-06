import os
import subprocess
import time
from contextlib import ExitStack
from typing import Iterable

import pytest
from jupyter_client import BlockingKernelClient

script_dir = os.path.dirname(__file__)


def choose_available_ports(n: int) -> Iterable[int]:
    import socket

    with ExitStack() as stack:
        for _ in range(n):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            stack.enter_context(s)
            s.bind(("localhost", 0))
            yield s.getsockname()[1]


@pytest.mark.timeout(10)
def test_launch_script(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(script_dir, "..")
    connection_file = str(tmp_path / "kernel_connection.json")

    ports = iter([str(p) for p in choose_available_ports(5)])
    proc = subprocess.Popen(
        [
            "python",
            os.path.join(script_dir, "launch_ipykernel.py"),
            "--key",
            "test_key",
            "--connection-file",
            connection_file,
            "--shell-port",
            next(ports),
            "--iopub-port",
            next(ports),
            "--stdin-port",
            next(ports),
            "--hb-port",
            next(ports),
            "--control-port",
            next(ports),
        ],
        env=env,
    )

    try:
        while not os.path.exists(connection_file):
            time.sleep(0.1)

        client = BlockingKernelClient(connection_file=connection_file)
        client.load_connection_file()
        client.start_channels()
        client.wait_for_ready(timeout=10)

        assert proc.poll() is None
    finally:
        proc.terminate()
        proc.wait()
