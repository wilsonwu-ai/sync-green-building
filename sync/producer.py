"""Getting frames onto whatever the building turns out to be.

We do not yet know how the simulator is delivered on the day: the upstream
SPEC says the row-to-floor mapping and the delivery mechanism are both "to be
confirmed at the hack". So this speaks every shape the repo documents, chosen
by a URL scheme, and falls back to a no-op that still drives the browser
preview:

    ws://host:port/tetris-17x9   protocol v1 over WebSocket
    tcp://host:port              protocol v1 over newline-delimited TCP
    gbsim://instance             the Green Building simulator, one POST a frame
    gbsim+https://host/instance  the same, somewhere other than the default host
    py://module:attr?args=a,b    a legacy Display subclass, send()/makeframe()
    none                         preview only

The gbsim schemes are the one we now know is real: the simulator's own JS
bundle POSTs a frame to /api/i/<instance>/frame as JSON. It is the only
delivery mechanism here that has been read off a running system rather than a
document, so it is the one to bet on. See HttpProducer for what is still a
guess about it.

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
import time
from urllib.parse import unquote

from .geometry import COLS, FPS, ROWS

log = logging.getLogger("sync.producer")

PROTOCOL = "17x9-tetris-remote"
VERSION = 1

# The simulator we were handed. gbsim+https://host/instance overrides it.
GBSIM_ORIGIN = "http://sundai.willsarg.com"


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


class HttpProducer(Producer):
    """The Green Building simulator: one POST per frame, no handshake.

    Three facts about that endpoint shaped this class.

    THE BODY IS A GUESS. The simulator's own JS bundle gave us the route, the
    method and the content type; the API reference behind it is password
    gated. So the first frame goes out as the bare [[[r,g,b] x9] x17] array,
    and if the server calls that a 4xx we send the same frame once more,
    wrapped as {"rows": [...]}. Whichever shape it accepts is remembered for
    the rest of the run. If it refuses both, we stop guessing: a 4xx that
    survives both shapes is about credentials rather than JSON, and doubling
    our request rate forever at a server that is saying no is the wrong answer.

    THE RENDER LOOP MUST NOT WAIT FOR THE NETWORK. send() hands the POST to a
    task and returns. At most one POST is ever in flight, and a frame offered
    while the previous one is still out is dropped rather than queued: a queue
    only buys latency, and on a facade a late frame is worse than a missing
    one - the next one is 33ms behind it anyway.

    A FAILED POST IS NOT AN EXCEPTION. It lands in .status, which the preview
    screen already shows, and the installation carries on.
    """

    def __init__(self, url: str, max_fps: int = FPS, timeout_s: float = 1.0) -> None:
        self.url = url
        self.shape: str | None = None  # "bare" | "rows", once the server tells us
        self.sent = 0
        self.dropped = 0
        self.last_error: str | None = None
        self._min_interval = 1.0 / max_fps
        self._timeout_s = timeout_s  # a POST slower than this is already stale
        self._session = None
        self._task: asyncio.Task | None = None
        self._next_ok = 0.0  # monotonic time at which the next POST is allowed
        self._negotiated = False  # the one body-shape retry has been spent

    async def connect(self) -> None:
        import aiohttp

        # There is no hello on this endpoint, so "connected" only means we
        # hold a session. The first frame is what actually tests the route.
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout_s)
        )
        log.info("posting frames to %s", self.url)

    async def send(self, frame_no: int, rows, phase: str = "", now: float | None = None) -> None:
        if self._session is None:
            return
        if now is None:  # 0.0 is a legitimate timestamp; `or` would discard it
            now = time.monotonic()
        if now < self._next_ok:  # the display contract's 30/s ceiling
            self.dropped += 1
            return
        if self._task is not None and not self._task.done():
            self.dropped += 1  # the previous POST has not come back yet
            return
        self._next_ok = now + self._min_interval
        # Copy out of the renderer's tuples now, on this side of the task, so
        # the body cannot be a view of a frame that has since moved on.
        body = [[list(px) for px in row] for row in rows]
        self._task = asyncio.create_task(self._post(body))

    def _shapes(self) -> tuple[str, ...]:
        if self.shape is not None:
            return (self.shape,)
        if self._negotiated:
            return ("bare",)  # both were refused once; stop doubling the rate
        return ("bare", "rows")

    async def _post(self, body: list) -> None:
        """Runs off the render loop. Nothing in here may raise into it."""
        shapes = self._shapes()
        for i, shape in enumerate(shapes):
            payload = body if shape == "bare" else {"rows": body}
            try:
                async with self._session.post(self.url, json=payload) as resp:
                    status = resp.status
            except Exception as exc:  # noqa: BLE001 - a dropped frame, not a crash
                self.last_error = f"{type(exc).__name__}: {exc}"
                return
            if status < 400:
                if self.shape != shape:
                    log.info("simulator accepted the %s body (HTTP %s)", shape, status)
                self.shape = shape
                self.sent += 1
                self.last_error = None
                return
            self.last_error = f"HTTP {status} on the {shape} body"
            if status >= 500:
                return  # a 5xx says nothing about our JSON; do not reshape it
            if i + 1 < len(shapes):
                self._next_ok += self._min_interval  # the retry is a second POST
        if len(shapes) > 1:
            self._negotiated = True
            log.warning("simulator refused both body shapes: %s", self.last_error)

    async def drain(self) -> None:
        """Wait out the POST in flight. close() needs it, and so does anything
        that wants to know whether a frame actually landed."""
        task, self._task = self._task, None
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    async def close(self) -> None:
        await self.drain()
        if self._session is not None:
            await self._session.close()
            self._session = None

    @property
    def status(self) -> str:
        if self._session is None:
            return f"disconnected ({self.url})"
        body = self.shape or ("bare, refused" if self._negotiated else "bare, unconfirmed")
        out = f"gbsim {self.url} body={body} sent={self.sent} dropped={self.dropped}"
        return out if self.last_error is None else f"{out} last={self.last_error}"


class LegacyDisplayProducer(Producer):
    """Drives anything exposing the original `send(frame)` / `makeframe()`
    pair, including `utilities.display.Display` subclasses and, per the
    upstream docs, whatever the building itself exposes.

    The target may carry constructor arguments, because the display we
    actually have in hand needs two of them:

        py://gbsim:WebDisplay?args=curious-cat,http://host/api

    Without that this called `obj()` on whatever class it found - fine for a
    zero-argument DummyDisplay, an immediate TypeError for
    gbsim.WebDisplay(instance, api_url), which is the display most likely to
    be handed to us on the day.
    """

    def __init__(self, target: str) -> None:
        self.target, _, query = target.partition("?")
        self.args = _ctor_args(query)
        self._display = None
        self._color = None

    async def connect(self) -> None:
        import importlib

        mod_name, _, attr = self.target.partition(":")
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, attr)
        # gbsim writes pixels as `frame[row][col] = Color(r, g, b)` rather than
        # through the legacy asarray() view, so keep that module's own Color.
        self._color = getattr(mod, "Color", None) or (lambda r, g, b: (r, g, b))
        if not isinstance(obj, type):
            self._display = obj  # already an instance
            return
        try:
            self._display = obj(*self.args)
        except TypeError as exc:
            # Name the fix. "missing 2 required positional arguments" in a
            # traceback at the install is not a useful error message.
            raise TypeError(
                f"{self.target} needs constructor arguments; pass them as "
                f"py://{self.target}?args=a,b  ({exc})"
            ) from exc

    async def send(self, frame_no: int, rows, phase: str = "") -> None:
        if self._display is None:
            return
        frame = self._display.makeframe()
        arr = frame.asarray() if hasattr(frame, "asarray") else None
        for r, row in enumerate(rows):
            for c, (red, green, blue) in enumerate(row):
                if arr is not None:
                    px = arr[r][c]
                    px.r, px.g, px.b = red, green, blue
                else:
                    frame[r][c] = self._color(red, green, blue)
        await asyncio.to_thread(self._display.send, frame)

    @property
    def status(self) -> str:
        target = f"{self.target}({', '.join(self.args)})" if self.args else self.target
        return f"legacy {target}" if self._display else f"unloaded ({target})"


def _ctor_args(query: str) -> list[str]:
    """`args=curious-cat,http://host/api` -> ["curious-cat", "http://host/api"].

    Split before unquoting, so an argument that genuinely contains a comma can
    still be written %2C. Everything the simulator wants is a name and a URL.
    """
    raw = ""
    for part in query.split("&"):
        key, _, value = part.partition("=")
        if key == "args":
            raw = value
    return [unquote(a) for a in raw.split(",") if a]


def gbsim_url(spec: str) -> str:
    """Resolve a display spec onto the simulator's frame endpoint.

        gbsim://curious-cat             -> {origin}/api/i/curious-cat/frame
        gbsim+https://host/curious-cat  -> https://host/api/i/curious-cat/frame
        http://host/api/i/x/frame       -> itself
    """
    if spec.startswith("gbsim://"):
        instance = spec[len("gbsim://") :].strip("/")
        if not instance:
            raise ValueError(f"gbsim spec needs an instance name: {spec!r}")
        return f"{GBSIM_ORIGIN}/api/i/{instance}/frame"
    if spec.startswith("gbsim+http://") or spec.startswith("gbsim+https://"):
        scheme, _, rest = spec[len("gbsim+") :].partition("://")
        host, _, instance = rest.partition("/")
        instance = instance.strip("/")
        if not host or not instance:
            raise ValueError(f"gbsim spec needs host and instance: {spec!r}")
        return f"{scheme}://{host}/api/i/{instance}/frame"
    # An explicit URL, for anything the schemes above cannot say. Tolerate the
    # instance URL without the trailing verb, because that is the one you get
    # by copying it out of the simulator's address bar.
    if "/api/i/" in spec and not spec.rstrip("/").endswith("/frame"):
        return spec.rstrip("/") + "/frame"
    return spec


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
    if spec.startswith(("gbsim://", "gbsim+http://", "gbsim+https://", "http://", "https://")):
        return HttpProducer(gbsim_url(spec))
    raise ValueError(f"unrecognised display spec: {spec!r}")
