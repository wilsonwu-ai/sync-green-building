"""The simulator we were actually given, over HTTP.

Every test here runs against a local aiohttp server or a refused local port.
Nothing in this file may touch sundai.willsarg.com: we have no event password,
and a test suite that posts frames at someone else's installation is a bad
citizen twice over.

The two things worth pinning are the two guesses. The body shape is unknown -
the API docs are password gated - so the negotiation is tested from both
sides. And the render loop awaits send() directly at 30fps, so send() is
tested for returning before the network does.
"""

import asyncio
import sys
import time
import types

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer, unused_port

from sync.geometry import COLS, FPS, ROWS
from sync.producer import (
    GBSIM_ORIGIN,
    HttpProducer,
    LegacyDisplayProducer,
    Producer,
    TCPProducer,
    WebSocketProducer,
    build,
    gbsim_url,
    validate,
)
from sync.render import Expression, Sync

FRAME = Sync(Expression(bpm=72, presence=1.0)).render(5)


def run(coro):
    """There is no pytest-asyncio in this environment, and one dependency for
    one decorator is not worth it."""
    return asyncio.run(coro)


class Sim:
    """A stand-in for the Green Building simulator, on localhost.

    `accepts` is the set of body shapes it will take; anything else is a 400,
    which is exactly the signal HttpProducer negotiates against.
    """

    def __init__(self, accepts=("bare", "rows"), status=200, delay=0.0) -> None:
        self.accepts, self.status, self.delay = accepts, status, delay
        self.bodies: list = []
        self.shapes: list[str] = []
        self._server = None

    async def _handle(self, request: web.Request) -> web.Response:
        body = await request.json()
        shape = "rows" if isinstance(body, dict) else "bare"
        self.bodies.append(body)
        self.shapes.append(shape)
        if self.delay:
            await asyncio.sleep(self.delay)
        if shape not in self.accepts:
            return web.json_response({"error": f"unexpected {shape} body"}, status=400)
        return web.json_response({"ok": True}, status=self.status)

    async def start(self) -> str:
        app = web.Application()
        app.router.add_post("/api/i/{instance}/frame", self._handle)
        self._server = TestServer(app)
        await self._server.start_server()
        return str(self._server.make_url("/api/i/curious-cat/frame"))

    async def stop(self) -> None:
        await self._server.close()


async def _drive(sim: Sim, frames: int = 1, times=None, **kwargs) -> HttpProducer:
    """Point a producer at the local sim, send `frames` frames one at a time,
    and hand the producer back once it has been closed cleanly.

    `times` feeds the rate limiter a clock the way policy.py takes `now`:
    real wall-clock time would make the 30/s ceiling flaky under load.
    """
    url = await sim.start()
    p = HttpProducer(url, **kwargs)
    await p.connect()
    try:
        for i in range(frames):
            await p.send(i, FRAME, now=(times[i] if times is not None else float(i)))
            await p.drain()
    finally:
        await p.close()
        await sim.stop()
    return p


# --- url schemes ---------------------------------------------------------

def test_gbsim_scheme_resolves_the_documented_endpoint():
    p = build("gbsim://curious-cat")
    assert isinstance(p, HttpProducer)
    assert p.url == f"{GBSIM_ORIGIN}/api/i/curious-cat/frame"


def test_gbsim_can_be_pointed_at_another_host():
    assert gbsim_url("gbsim+https://sim.example/curious-cat") == (
        "https://sim.example/api/i/curious-cat/frame"
    )
    assert gbsim_url("gbsim+http://127.0.0.1:8000/cat") == (
        "http://127.0.0.1:8000/api/i/cat/frame"
    )


def test_an_explicit_frame_url_is_used_as_given():
    assert gbsim_url("https://host/api/i/x/frame") == "https://host/api/i/x/frame"


def test_an_instance_url_gains_the_missing_verb():
    """The URL you get by copying it out of the simulator has no /frame."""
    assert gbsim_url("http://host/api/i/x") == "http://host/api/i/x/frame"
    assert gbsim_url("http://host/api/i/x/") == "http://host/api/i/x/frame"
    p = build("http://host/api/i/curious-cat")
    assert isinstance(p, HttpProducer)
    assert p.url == "http://host/api/i/curious-cat/frame"


def test_an_http_url_that_is_not_a_frame_endpoint_is_refused():
    """http:// was accepted for any URL at all, so `http://host:9000` typed
    where `ws://host:9000` was meant built a producer that failed on every
    frame - and an install that looks healthy while the facade stays dark is
    worse than one that refuses to start."""
    for spec in (
        "http://host:9000",             # a ws:// spec with the wrong scheme
        "http://host/tetris-17x9",      # ditto, with the path copied over
        "https://host/",
        "http://host/api/i/",           # /api/i/ present, instance missing
    ):
        with pytest.raises(ValueError, match="frame endpoint"):
            gbsim_url(spec)
        with pytest.raises(ValueError, match="frame endpoint"):
            build(spec)


def test_a_gbsim_spec_without_an_instance_is_rejected():
    for spec in ("gbsim://", "gbsim+https://host", "gbsim+https://host/"):
        with pytest.raises(ValueError, match="instance"):
            gbsim_url(spec)


def test_the_other_schemes_still_route_where_they_did():
    assert type(build("none")) is Producer
    assert isinstance(build("ws://host:9000/tetris-17x9"), WebSocketProducer)
    assert isinstance(build("tcp://host:9000"), TCPProducer)
    assert isinstance(build("py://mod:Attr"), LegacyDisplayProducer)
    with pytest.raises(ValueError, match="unrecognised"):
        build("carrier-pigeon://host")


# --- the wire ------------------------------------------------------------

def test_the_bare_array_is_tried_first():
    sim = Sim()
    p = run(_drive(sim))
    assert sim.shapes == ["bare"]
    assert p.shape == "bare"
    assert p.sent == 1
    assert p.dropped == 0
    validate(sim.bodies[0])  # 17 x 9 x 3, in range, straight off the wire
    assert len(sim.bodies[0]) == ROWS
    assert len(sim.bodies[0][0]) == COLS


def test_a_4xx_is_retried_once_with_the_wrapped_body():
    sim = Sim(accepts=("rows",))
    p = run(_drive(sim))
    assert sim.shapes == ["bare", "rows"]
    assert sim.bodies[1]["rows"] == sim.bodies[0]  # the same frame, wrapped
    assert p.shape == "rows"
    assert p.sent == 1
    assert p.last_error is None


def test_the_shape_that_worked_sticks_for_the_rest_of_the_run():
    sim = Sim(accepts=("rows",))
    p = run(_drive(sim, frames=4))
    # One extra POST for the whole run: the single bare guess on frame 0.
    assert sim.shapes == ["bare"] + ["rows"] * 4
    assert len(sim.shapes) == 4 + 1
    assert p.sent == 4


def test_neither_shape_is_retried_more_than_once():
    """If both shapes are refused it is not the JSON - we have no event
    password - and guessing twice a frame forever only doubles the noise."""
    sim = Sim(accepts=())
    p = run(_drive(sim, frames=3))
    assert sim.shapes == ["bare", "rows", "bare", "bare"]
    assert p.sent == 0
    assert "400" in p.last_error
    # Three frames offered, three frames lost. The first cost two POSTs and is
    # still one lost frame.
    assert p.dropped == 3
    assert p.failed == 3


def test_a_post_that_fails_is_counted_as_a_lost_frame():
    """sent + dropped is the number of frames offered, always. A POST that
    went out and came back refused used to be counted nowhere: a live run
    offered 18 frames and read sent=0 dropped=6, with the twelve failures
    missing from the arithmetic entirely."""
    sim = Sim(accepts=())
    offered = 5
    p = run(_drive(sim, frames=offered))
    assert p.sent == 0
    assert p.sent + p.dropped == offered
    assert p.failed == offered  # all five went out on the wire and were refused


def test_the_counters_add_up_across_every_way_of_losing_a_frame():
    """Two kinds of loss in one run: the server refusing a POST, and the rate
    limiter holding the contract. Both are drops; only the first is a failure,
    and an operator reading dropped=4 needs that split to know which."""
    sim = Sim(accepts=())
    # Four frames on a clock that only clears the 30/s ceiling twice.
    p = run(_drive(sim, frames=4, times=[0.0, 0.001, 1.0, 1.001]))
    assert p.sent == 0
    assert p.dropped == 4  # every frame offered is accounted for
    assert p.failed == 2  # two of them were POSTed and refused
    assert p.dropped - p.failed == 2  # the other two never left the process


def test_a_server_error_does_not_burn_the_retry():
    """A 5xx says nothing about our body shape, so it must not reshape it."""
    sim = Sim(status=503)
    p = run(_drive(sim))
    assert sim.shapes == ["bare"]
    assert p.sent == 0
    assert p.shape is None
    assert p.last_error == "HTTP 503 on the bare body"
    assert p.dropped == 1  # the frame still did not land
    assert p.failed == 1


def test_a_refused_connection_becomes_status_not_an_exception():
    """The render loop awaits send(). A dead simulator must cost us a frame,
    not the show."""

    async def go():
        p = HttpProducer(f"http://127.0.0.1:{unused_port()}/api/i/nobody/frame")
        await p.connect()
        await p.send(0, FRAME, now=0.0)  # must not raise
        await p.drain()
        status = p.status
        await p.close()
        return p, status

    p, status = run(go())
    assert p.sent == 0
    assert p.dropped == 1  # a frame we could not even hand to a socket
    assert p.failed == 1
    assert p.last_error is not None
    assert p.url in status and p.last_error in status
    # Read while still connected: the line an operator is actually watching
    # has to say that the frame was refused, not merely that one was dropped.
    assert "failed=1" in status


def test_send_returns_before_the_post_does():
    """The 30fps loop awaits send() directly, so send() may never wait on the
    network. This simulator takes a quarter of a second to answer - seven and
    a half frames - and send() must be back long before it."""
    sim = Sim(delay=0.25)

    async def go():
        url = await sim.start()
        p = HttpProducer(url)
        await p.connect()
        t0 = time.monotonic()
        await p.send(0, FRAME, now=0.0)
        elapsed = time.monotonic() - t0
        await p.close()  # drains, so the slow POST still lands
        await sim.stop()
        return elapsed, p

    elapsed, p = run(go())
    assert elapsed < 0.05
    assert p.sent == 1


def test_a_frame_is_dropped_while_a_post_is_in_flight():
    """Dropped, not queued: the next frame is 33ms away and a stale one is
    worse than a missing one."""
    sim = Sim(delay=0.15)

    async def go():
        url = await sim.start()
        p = HttpProducer(url)
        await p.connect()
        await p.send(0, FRAME, now=0.0)
        await p.send(1, FRAME, now=10.0)  # the clock allows it; the socket does not
        await p.close()
        await sim.stop()
        return p

    p = run(go())
    assert p.sent == 1
    assert p.dropped == 1
    assert len(sim.bodies) == 1


def test_it_never_posts_faster_than_thirty_a_second():
    """The display contract's ceiling. 200 frames offered across one second of
    clock must not become 200 POSTs - nor collapse to a trickle."""
    sim = Sim()
    offered = 200
    times = [i / offered for i in range(offered)]
    p = run(_drive(sim, frames=offered, times=times))
    span = times[-1] - times[0]

    assert len(sim.bodies) <= FPS  # the contract, and the only hard bound here

    # The floor is derived from the limiter, not fitted to it. The limiter
    # re-arms 1/FPS after the frame it let through rather than on a fixed
    # grid, and a frame can only be offered every 1/offered seconds, so it
    # drifts by at most one offer period per POST: consecutive POSTs land at
    # most (1/FPS + 1/offered) apart. `FPS - 2` was not this bound, it was the
    # number this test happened to observe (29), which would have gone on
    # passing for a limiter that had started drifting twice as fast.
    floor = 1 + int(span / (1.0 / FPS + 1.0 / offered))
    assert len(sim.bodies) >= floor

    assert p.sent == len(sim.bodies)
    assert p.sent + p.dropped == offered


def test_status_survives_a_closed_session():
    sim = Sim()
    p = run(_drive(sim))
    assert p.url in p.status  # reports disconnected rather than raising


# --- py:// constructor arguments -----------------------------------------

class _FakeColor:
    """gbsim's Color: three channels, constructed positionally."""

    def __init__(self, r, g, b) -> None:
        self.rgb = (r, g, b)


class _FakeFrame:
    """gbsim's Frame, written through directly: f[0][0] = Color(...)."""

    def __init__(self) -> None:
        self.cells = [[None] * COLS for _ in range(ROWS)]

    def __getitem__(self, row):
        return self.cells[row]


class _FakeWebDisplay:
    """gbsim.WebDisplay(instance, api_url) - the constructor that blew up."""

    def __init__(self, instance, api_url) -> None:
        self.instance, self.api_url = instance, api_url
        self.frames: list = []

    def makeframe(self):
        return _FakeFrame()

    def send(self, frame) -> None:
        self.frames.append(frame)


class _LegacyPixel:
    def __init__(self) -> None:
        self.r = self.g = self.b = 0


class _LegacyFrame:
    """utilities.display.Frame: mutated through an asarray() view."""

    def __init__(self) -> None:
        self.rows = [[_LegacyPixel() for _ in range(COLS)] for _ in range(ROWS)]

    def asarray(self):
        return self.rows


class _DummyDisplay:
    """The zero-argument display the old code could already drive."""

    def __init__(self) -> None:
        self.frames: list = []

    def makeframe(self):
        return _LegacyFrame()

    def send(self, frame) -> None:
        self.frames.append(frame)


class _BrokenDisplay:
    """A display that takes no arguments and is simply broken inside.

    The TypeError out of its __init__ has nothing to do with how it was
    called, which is the case the old `except TypeError` around the
    constructor could not tell apart from a missing argument.
    """

    def __init__(self) -> None:
        raise TypeError("unsupported operand type(s) for +: 'int' and 'str'")


@pytest.fixture
def fake_gbsim(monkeypatch):
    mod = types.ModuleType("fake_gbsim")
    mod.WebDisplay = _FakeWebDisplay
    mod.DummyDisplay = _DummyDisplay
    mod.BrokenDisplay = _BrokenDisplay
    mod.Color = _FakeColor
    monkeypatch.setitem(sys.modules, "fake_gbsim", mod)
    return mod


def test_the_py_scheme_carries_constructor_args():
    p = build("py://fake_gbsim:WebDisplay?args=curious-cat,http://host/api")
    assert p.target == "fake_gbsim:WebDisplay"
    assert p.args == ["curious-cat", "http://host/api"]


def test_an_argument_may_contain_an_escaped_comma():
    p = build("py://m:C?args=a%2Cb,c")
    assert p.args == ["a,b", "c"]


def test_a_display_class_is_constructed_with_its_args(fake_gbsim):
    """The bug: obj() with no arguments. gbsim.WebDisplay(instance, api_url)
    takes two, so every py:// run against the real simulator died on connect."""
    p = build("py://fake_gbsim:WebDisplay?args=curious-cat,http://host/api")
    run(p.connect())
    assert p._display.instance == "curious-cat"
    assert p._display.api_url == "http://host/api"
    assert "curious-cat" in p.status


def test_a_zero_argument_display_still_works(fake_gbsim):
    p = build("py://fake_gbsim:DummyDisplay")
    run(p.connect())
    assert p.args == []
    assert isinstance(p._display, _DummyDisplay)


def test_a_missing_argument_names_the_fix(fake_gbsim):
    p = build("py://fake_gbsim:WebDisplay")
    with pytest.raises(TypeError, match=r"\?args="):
        run(p.connect())


def test_too_many_arguments_also_names_the_fix(fake_gbsim):
    """The other half of an arity error: args given to a display that takes
    none. Same knob to turn, so the same message."""
    p = build("py://fake_gbsim:DummyDisplay?args=cat,http://host/api")
    with pytest.raises(TypeError, match=r"\?args="):
        run(p.connect())


def test_a_typeerror_from_inside_the_constructor_is_not_relabelled(fake_gbsim):
    """The over-broad except: a display that takes no arguments and raises
    TypeError from its own body was reported as needing ?args=, sending the
    operator at the install to fix an argument list that was already right."""
    p = build("py://fake_gbsim:BrokenDisplay")
    with pytest.raises(TypeError) as caught:
        run(p.connect())
    assert "unsupported operand" in str(caught.value)
    assert "?args=" not in str(caught.value)


def test_frames_reach_a_gbsim_style_frame(fake_gbsim):
    """gbsim has no asarray(); its frame is indexed and assigned a Color."""
    p = build("py://fake_gbsim:WebDisplay?args=cat,http://host/api")
    run(p.connect())
    run(p.send(0, FRAME))
    sent = p._display.frames[0]
    assert sent[8][4].rgb == tuple(FRAME[8][4])
    assert sent[0][0].rgb == tuple(FRAME[0][0])


def test_frames_still_reach_a_legacy_asarray_display(fake_gbsim):
    p = build("py://fake_gbsim:DummyDisplay")
    run(p.connect())
    run(p.send(0, FRAME))
    px = p._display.frames[0].asarray()[8][4]
    assert (px.r, px.g, px.b) == tuple(FRAME[8][4])
