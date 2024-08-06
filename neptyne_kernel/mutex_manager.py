import asyncio
import threading


class AsyncMutexManager:
    def __init__(self) -> None:
        self.mutex_dict: dict[str, asyncio.Lock] = {}
        self.mutex_dict_lock = asyncio.Lock()

    async def get_mutex(self, key: str) -> asyncio.Lock:
        async with self.mutex_dict_lock:
            if key not in self.mutex_dict:
                self.mutex_dict[key] = asyncio.Lock()
            return self.mutex_dict[key]


class MutexManager:
    def __init__(self) -> None:
        self.mutex_dict: dict[str, threading.Lock] = {}
        self.mutex_dict_lock = threading.Lock()

    def get_mutex(self, key: str = "") -> threading.Lock:
        with self.mutex_dict_lock:
            if key not in self.mutex_dict:
                self.mutex_dict[key] = threading.Lock()
            return self.mutex_dict[key]
