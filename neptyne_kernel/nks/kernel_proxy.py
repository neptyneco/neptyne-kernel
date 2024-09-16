import inspect
import typing
from collections import defaultdict
from typing import Any

from jupyter_client.session import Session
from tornado.websocket import WebSocketClosedError
from zmq import Context

if typing.TYPE_CHECKING:
    from server.nks_handler import NKSConnectHandler


class Stream:
    def __init__(self, tyne_id: str, ws_connection: "NKSConnectHandler"):
        self.callback = None
        self.channel = ""
        self.ws_connection = ws_connection
        self.tyne_id = tyne_id
        self._closed = False

    def on_recv_stream(self, callback: Any) -> None:
        self.callback = callback

    def on_recv(self, callback: Any) -> None:
        self.callback = callback

    def close(self) -> None:
        self._closed = True

    def send_multipart(self, msg: list[Any], copy: bool = True) -> None:
        try:
            self.ws_connection.write_message(
                {
                    "method": "kernel_message",
                    "tyne_id": self.tyne_id,
                    "channel": self.channel,
                    "payload": [m.decode() for m in msg],
                }
            )
        except WebSocketClosedError:
            self.close()

    def closed(self) -> bool:
        return self._closed


class RemoteNKSKernelManager:
    ws_connection: "NKSConnectHandler"
    channel_handlers: dict[str, list[Stream]]
    session: Session
    tyne_id: str | None
    autorestart: bool
    is_closed: bool

    def __init__(self, ws_connection: "NKSConnectHandler"):
        self.ws_connection = ws_connection
        self.ws_connection.kernel_managers.append(self)  # TODO: clean these up?
        self.channel_handlers = defaultdict(list)
        self.session = Session(key=b"")
        self.tyne_id = None
        self.autorestart = False

    def handle_closed(self) -> None:
        self.is_closed = True

    async def on_message(self, kernel_message: dict[str, Any]) -> None:
        handlers = self.channel_handlers[kernel_message["channel"]]
        for handler in handlers:
            if handler.callback:
                res = handler.callback(
                    handler, [m.encode() for m in kernel_message["payload"]]
                )
                if inspect.isawaitable(res):
                    await res

    async def start_kernel(self, kernel_id: str, **kwargs: Any) -> None:
        self.tyne_id = kernel_id
        await self.ws_connection.write_message(
            {"method": "start_kernel", "tyne_id": kernel_id}
        )

    async def wait_for_ready(self) -> None:
        pass

    def connect_channel(self, channel: str) -> Stream:
        assert self.tyne_id is not None
        s = Stream(self.tyne_id, self.ws_connection)
        s.channel = channel
        self.channel_handlers[channel].append(s)
        return s

    def connect_shell(self, identity: bytes) -> Stream:
        return self.connect_channel("shell")

    def connect_control(self, identity: bytes) -> Stream:
        return self.connect_channel("control")

    def connect_iopub(self, identity: bytes) -> Stream:
        return self.connect_channel("iopub")

    def connect_stdin(self, identity: bytes) -> Stream:
        return self.connect_channel("stdin")

    def client(self, context: Context | None = None) -> "RemoteNKSKernelManager":
        return self

    @property
    def parent(self) -> "RemoteNKSKernelManager":
        return self

    async def is_alive(self) -> bool:
        return all(
            not h.closed()
            for handlers in self.channel_handlers.values()
            for h in handlers
        )

    @property
    def kernel_name(self) -> str:
        return "nks"
