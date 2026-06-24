#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import socket
from typing import Any

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            arr = np.ascontiguousarray(obj)
            return {
                "__ndarray__": True,
                "dtype": str(arr.dtype),
                "shape": arr.shape,
                "data": base64.b64encode(arr.tobytes()).decode("ascii"),
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _object_hook(obj: dict[str, Any]) -> Any:
    if obj.get("__ndarray__"):
        raw = base64.b64decode(obj["data"])
        return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    return obj


def dumps(payload: Any) -> bytes:
    return json.dumps(payload, cls=NumpyJSONEncoder).encode("utf-8")


def loads(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"), object_hook=_object_hook)


def recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = nbytes
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("connection closed while receiving payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: socket.socket, payload: Any) -> None:
    data = dumps(payload)
    sock.sendall(len(data).to_bytes(4, "big"))
    sock.sendall(data)


def recv_message(sock: socket.socket) -> Any:
    header = recv_exact(sock, 4)
    size = int.from_bytes(header, "big")
    if size <= 0:
        raise ConnectionError(f"invalid message size: {size}")
    return loads(recv_exact(sock, size))


class RPCClient:
    def __init__(self, host: str, port: int, timeout: float = 120.0):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.sock: socket.socket | None = None
        self.connect()

    def connect(self) -> None:
        self.close()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect((self.host, self.port))
        self.sock = sock

    def call(self, cmd: str, **kwargs: Any) -> Any:
        if self.sock is None:
            self.connect()
        assert self.sock is not None
        send_message(self.sock, {"cmd": cmd, **kwargs})
        response = recv_message(self.sock)
        if not isinstance(response, dict):
            raise RuntimeError(f"malformed RPC response: {type(response)!r}")
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "unknown RPC error"))
        return response.get("result")

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def __enter__(self) -> "RPCClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
