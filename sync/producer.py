"""Getting frames onto whatever the building turns out to be.

We do not yet know how the simulator is delivered on the day: the upstream
SPEC says the row-to-floor mapping and the delivery mechanism are both "to be
confirmed at the hack". So this speaks every shape the repo documents, chosen
by a URL scheme, and falls back to a no-op that still drives the browser
preview:

    ws://host:port/tetris-17x9   protocol v1 over WebSocket
    tcp://host:port              protocol v1 over newline-delimited TCP
    py://module:attr             a legacy Display subclass, send()/makeframe()
    none                         preview only

Protocol v1 shape, from docs/PROTOCOL.md: the first message MUST be `hello`;
`frame` carries 17 rows of 9 [r,g,b] arrays, row-major, row 0 at the top;
`digest` is the lowercase hex SHA-256 of the 459 R,G,B bytes and is optional
for a producer. We send it anyway, because a mismatch is the fastest way to
learn our row order is wrong.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging

from .geometry import COLS, ROWS

log = logging.getLogger("sync.producer")

PROTOCOL = "17x9-tetris-remote"
VERSION = 1


def digest(rows) -> str:
    buf = bytearray()
    for row in rows:
        for px in row:
            buf.extend(bytes(px))
    return hashlib.sha256(bytes(buf)).hexdigest()


def validate(rows) -> None:
    """The receiver is required to reject, not clamp. Catch it on our side."""
    if len(rows) != ROWS:
        raise ValueError(f"expected {ROWS} rows, got {len(rows)}")
    for r, row in enumerate(rows):
        if len(row) != COLS:
            raise ValueError(f"row {r}: expected {COLS} cells, got {len(row)}")
        for c, px in enumerate(row):
            if len(px) != 3 or not all(isinstance(v, int) and 0 <= v <= 255 for v in px):
                raise ValueError(f"cell ({r},{c}) is not 3 ints in 0..255: {px!r}")


def frame_message(frame_no: int, rows, phase: str = "") -> dict:
    msg = {
        "type": "frame",
        "frame_no": frame_no,
        "rows": [[list(px) for px in row] for row in rows],
        "digest": digest(rows),
    }
    if phase:
        msg["phase"] = phase[:32]
    return msg


class Producer:
    """No-op base. The browser preview works with this alone."""

    async def connect(self) -> None:
        return None

    async def send(self, frame_no: int, rows, phase: str = "") -> None:
        return None

    async def close(self) -> None:
        return None

    @property
    def status(self) -> str:
        return "preview only"


class WebSocketProducer(Producer):
    def __init__(self, url: str) -> None:
        self.url = url
        self._ws = None
        self._hello: dict = {}

    async def connect(self) -> None:
        import websockets

        self._ws = await websockets.connect(self.url, max_size=2**20)
        await self._ws.send(
            json.dumps(
                {
                    "type": "hello",
                    "protocol": PROTOCOL,
                    "version": VERSION,
                    "role": "producer",
                    "client": "sync",
                }
            )
        )
        self._hello = json.loads(await self._ws.recv())
        if self._hello.get("type") == "error":
            raise RuntimeError(f"display refused us: {self._hello}")
        # A gatekeeper may demote us; the contract says act on client_role.
        role = self._hello.get("client_role")
        if role and role != "producer":
            raise RuntimeError(f"demoted to {role}; cannot drive the facade")
        log.info("connected to display: %s", self._hello)

    async def send(self, frame_no: int, rows, phase: str = "") -> None:
        if self._ws is None:
            return
        await self._ws.send(json.dumps(frame_message(frame_no, rows, phase)))

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    @property
    def status(self) -> str:
        if self._ws is None:
            return f"disconnected ({self.url})"
        return f"ws {self.url} rows={self._hello.get('rows')} cols={self._hello.get('cols')}"


class TCPProducer(Producer):
    def __init__(self, host: str, port: int) -> None:
        self.host, self.port = host, port
        self._r = self._w = None
        self._hello: dict = {}

    async def connect(self) -> None:
        self._r, self._w = await asyncio.open_connection(self.host, self.port)
        await self._line(
            {
                "type": "hello",
                "protocol": PROTOCOL,
                "version": VERSION,
                "role": "producer",
                "client": "sync",
            }
        )
        raw = await self._r.readline()
        self._hello = json.loads(raw.decode())
        if self._hello.get("type") == "error":
            raise RuntimeError(f"display refused us: {self._hello}")
        log.info("connected to display: %s", self._hello)

    async def _line(self, obj: dict) -> None:
        data = json.dumps(obj).encode() + b"\n"
        if len(data) > 65536:  # the contract's hard ceiling
            raise ValueError(f"message too large: {len(data)} bytes")
        self._w.write(data)
        await self._w.drain()  # back-pressure slows us; it must never drop frames

    async def send(self, frame_no: int, rows, phase: str = "") -> None:
        if self._w is None:
            return
        await self._line(frame_message(frame_no, rows, phase))

    async def close(self) -> None:
        if self._w is not None:
            self._w.close()
            self._w = None

    @property
    def status(self) -> str:
        return f"tcp {self.host}:{self.port}" if self._w else f"disconnected ({self.host}:{self.port})"


class LegacyDisplayProducer(Producer):
    """Drives anything exposing the original `send(frame)` / `makeframe()`
    pair, including `utilities.display.Display` subclasses and, per the
    upstream docs, whatever the building itself exposes."""

    def __init__(self, target: str) -> None:
        self.target = target
        self._display = None

    async def connect(self) -> None:
        import importlib

        mod_name, _, attr = self.target.partition(":")
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, attr)
        self._display = obj() if isinstance(obj, type) else obj

    async def send(self, frame_no: int, rows, phase: str = "") -> None:
        if self._display is None:
            return
        frame = self._display.makeframe()
        arr = frame.asarray()
        for r, row in enumerate(rows):
            for c, (red, green, blue) in enumerate(row):
                px = arr[r][c]
                px.r, px.g, px.b = red, green, blue
        await asyncio.to_thread(self._display.send, frame)

    @property
    def status(self) -> str:
        return f"legacy {self.target}" if self._display else f"unloaded ({self.target})"


def build(spec: str) -> Producer:
    if not spec or spec == "none":
        return Producer()
    if spec.startswith("ws://") or spec.startswith("wss://"):
        return WebSocketProducer(spec)
    if spec.startswith("tcp://"):
        rest = spec[len("tcp://") :]
        host, _, port = rest.partition(":")
        return TCPProducer(host or "127.0.0.1", int(port or 9000))
    if spec.startswith("py://"):
        return LegacyDisplayProducer(spec[len("py://") :])
    raise ValueError(f"unrecognised display spec: {spec!r}")
