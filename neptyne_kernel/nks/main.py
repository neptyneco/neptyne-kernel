import argparse
import asyncio
import json
from functools import partial
from typing import Any, Awaitable, Callable, cast
from urllib.parse import urlparse

import requests
from jupyter_client import (
    AsyncKernelClient,
    AsyncKernelManager,
    AsyncMultiKernelManager,
)
from jupyter_client.session import Session
from tornado.websocket import WebSocketClientConnection, websocket_connect
from zmq.eventloop.zmqstream import ZMQStream

from server.kernels.spec_manager import NeptyneKernelSpecManager


class KernelHandle:
    kernel_manager: AsyncKernelManager
    kernel_client: AsyncKernelClient
    channels: dict
    message_handler: Callable[[str, list[bytes]], Awaitable[None]]

    def __init__(
        self,
        kernel_manager: AsyncKernelManager,
        message_handler: Callable[[str, list[bytes]], Awaitable[None]],
    ):
        self.kernel_manager = kernel_manager
        self.kernel_client = kernel_manager.client(session=Session(key=b""))
        self.message_handler = message_handler
        self.channels = {}
        for channel in ("shell", "control", "iopub", "stdin"):
            stream_connect = getattr(kernel_manager, f"connect_{channel}")
            self.channels[channel] = stream = stream_connect(
                identity=self.kernel_manager.session.bsession
            )
            stream.on_recv_stream(partial(self.on_kernel_message, channel))

    async def on_kernel_message(
        self, channel: str, stream: ZMQStream, msg: list[bytes]
    ):
        await self.message_handler(channel, msg)

    def forward_message(self, channel: str, msg: Any):
        stream = self.channels[channel]
        self.kernel_client.session.send_raw(stream, msg[2:])  # remove the DELIM


class KernelMessageBroker:
    connection: WebSocketClientConnection | None
    kernel_manager: AsyncMultiKernelManager
    inbox: list
    kernel_handles: dict
    _connection_id: str | None

    def __init__(self, kernel_manager: AsyncMultiKernelManager):
        self.connection = None
        self.kernel_manager = kernel_manager
        self.inbox = []
        self.kernel_handles = {}
        self._connection_id = None

    def on_message(self, msg: None | str | bytes):
        self.inbox.append(msg)

    async def process_messages(self):
        if self.inbox:
            msg = self.inbox.pop(0)
            if msg is None:
                raise RuntimeError("disconnected")
            elif msg == "ping":
                await self.write_message("pong")
            else:
                await self.process_msg(msg)

    def add_kernel_manager(self, tyne_id: str, kernel_manager: AsyncKernelManager):
        async def message_handler(channel: str, payload: list[bytes]):
            await self.write_message(
                {
                    "tyne_id": tyne_id,
                    "channel": channel,
                    "payload": [p.decode() for p in payload],
                }
            )

        self.kernel_handles[tyne_id] = KernelHandle(kernel_manager, message_handler)

    async def process_msg(self, msg: str | bytes):
        message = json.loads(msg)
        if message["method"] == "start_kernel":
            tyne_id = message["tyne_id"]
            print("starting kernel for tyne", tyne_id)
            await self.kernel_manager.start_kernel(
                kernel_name="python_local",
                kernel_id=tyne_id,
            )
            manager = cast(AsyncKernelManager, self.kernel_manager.get_kernel(tyne_id))
            self.add_kernel_manager(tyne_id, manager)
        elif message["method"] == "kernel_message":
            tyne_id = message["tyne_id"]
            kernel_message = message["payload"]
            channel = message["channel"]
            handle = self.kernel_handles[tyne_id]
            handle.forward_message(channel, [m.encode() for m in kernel_message])
        elif message["method"] == "talk_to_user":
            print(f"[{self.connection_id()}]", message["message"])
        else:
            print("unrecognized message", msg)

    async def write_message(self, message: str | bytes | dict[str, Any], binary=False):
        assert self.connection is not None
        await self.connection.write_message(message, binary)

    def connection_id(self):
        if self._connection_id is None:
            assert self.connection is not None
            path = urlparse(self.connection.request.url).path
            shard = path.split("/")[2]
            self._connection_id = f"{self.connection.parsed_hostname}/{shard}"

        return self._connection_id


class InsecureKernelManager(AsyncMultiKernelManager):
    def pre_start_kernel(self, kernel_name: str | None, kwargs):
        km, *rest = super().pre_start_kernel(kernel_name, kwargs)
        km.session = Session(key=b"")
        return km, *rest


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("token")
    parser.add_argument("--host", default="https://app.neptyne.com")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    name = args.name or "neptyne-local"
    host = args.host
    if not host.startswith("http"):
        proto = "http" if host.startswith("localhost") else "https"
        host = f"{proto}://{host}"

    num_shards = requests.get(f"{host}/api/nks/how_many_shards").json()["num_shards"]

    kernel_env = {}
    if "localhost" not in host and "ngrok" not in host:
        domain = host.split("://")[1]
        kernel_env["API_PROXY_HOST_PORT"] = f"api-proxy.{domain}"
        kernel_env["NEPTYNE_LOCAL_NKS_KERNEL"] = "1"

    kernel_manager = InsecureKernelManager(
        kernel_spec_manager=NeptyneKernelSpecManager(
            env=kernel_env,
        )
    )

    brokers = []

    for ix in range(num_shards):
        broker = KernelMessageBroker(kernel_manager)
        addr = f"{host.replace('http', 'ws', 1)}/ws/{ix}/nks/connect/{args.token}"
        print("connecting to", addr)
        connection = await websocket_connect(
            addr,
            on_message_callback=broker.on_message,
            max_message_size=10 * 1024 * 1024 * 1024,
        )
        broker.connection = connection

        await broker.write_message(f"register:{name}")
        brokers.append(broker)

    print("Connected to Neptyne")

    while True:
        for broker in brokers:
            await broker.process_messages()
        await asyncio.sleep(0)


if __name__ == "__main__":
    asyncio.run(main())
