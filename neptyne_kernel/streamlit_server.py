# Copy of streamlit/web/bootstrap.py

# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2024)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio


def start_server(
    main_script_path: str,
) -> None:
    from streamlit.web.server import Server

    if getattr(start_server, "server", None) is not None:
        raise RuntimeError("Server is already running")

    server = Server(main_script_path, is_hello=False)
    setattr(start_server, "server", server)

    async def run_server() -> None:
        await server.start()
        await server.stopped

    task = asyncio.get_event_loop().create_task(run_server())
    setattr(start_server, "task", task)


def stop_server() -> None:
    if (server := getattr(start_server, "server", None)) is not None:
        server.stop()
        setattr(start_server, "server", None)


def is_server_running() -> bool:
    return getattr(start_server, "server", None) is not None


def is_running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
    except ImportError:
        return False
    return get_script_run_ctx(suppress_warning=True) is not None
