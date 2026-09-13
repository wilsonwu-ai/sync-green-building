"""The render loop's clock.

Both things pinned here were caught against the real simulator, not by reading.

  THE PHASE MUST COME OFF THE WALL CLOCK. render() recovers seconds by dividing
  the frame index it is handed by geometry.FPS. Hand it a loop counter and the
  displayed heart rate becomes a function of how fast we manage to push frames,
  which over a congested link is not 30 a second and therefore is not the
  crowd's rate. Measured: 35s of content took 48.7s to send, so a crowd at a
  measured 72 BPM watched itself beat at 52.

  FAILURES MUST REACH THE OPERATOR. HttpProducer is built never to raise - it
  counts its failures into .status instead. A status sampled once before the
  loop therefore reads "healthy" for the rest of the night while nothing at all
  is landing on the building.

And the third question the loop now answers separately from the first: how
often to send, which is not how often to draw.

Everything here runs on a virtual clock. Nothing sleeps, nothing opens a
socket, and nothing touches sundai.willsarg.com.
"""

import asyncio
import time

import pytest

from sync.crowd import Reading
from sync.geometry import FPS
from sync.render import Sync
from sync.server import SEND_FPS, _render_loop, create_app

BPM = 72.0


def run(coro):
    """There is no pytest-asyncio in this environment, and one dependency for
    one decorator is not worth it (same call the producer tests make)."""
    return asyncio.run(coro)


class _Stop(Exception):
    """Ends the loop from inside the injected sleep. The real loop is a
    `while True`; only an exception gets out of it."""


class Clock:
    """A virtual monotonic clock. Time moves when the loop sleeps, and when a
    producer we have told to be slow takes its time - and at no other moment,
    so every assertion below is exact."""

    def __init__(self, frames: int) -> None:
        self.t = 0.0
        self.frames = frames
        self.sleeps = 0

    def now(self) -> float:
        return self.t

    async def sleep(self, delay: float) -> None:
        self.sleeps += 1
        if self.sleeps >= self.frames:
            raise _Stop  # `frames` iterations have now run
        self.t += delay


class FakeProducer:
    """The shape server.py depends on, carrying HttpProducer's defining habit:
    send() never raises, so a failure exists only in .status."""

    def __init__(self, clock: Clock | None = None, cost_s: float = 0.0) -> None:
        self._clock = clock
        self._cost_s = cost_s
        self.sent: list[int] = []
        self.dropped = 0
        self.connected = False
        self.raises: Exception | None = None

    async def connect(self) -> None:
        self.connected = True

    async def send(self, frame_no, rows, phase="") -> None:
        if self._cost_s and self._clock is not None:
            self._clock.t += self._cost_s  # the network, taking its time
        if self.raises is not None:
            self.dropped += 1
            raise self.raises
        self.sent.append(frame_no)

    async def close(self) -> None:
        return None

    @property
    def status(self) -> str:
        return f"fake sent={len(self.sent)} dropped={self.dropped}"


class SilentFailure(FakeProducer):
    """HttpProducer against a refused port, in miniature: send() returns
    normally, the frame is counted as dropped, and .status is the only place
    the operator can ever learn about it."""

    async def send(self, frame_no, rows, phase="") -> None:
        self.dropped += 1

    @property
    def status(self) -> str:
        return (
            "gbsim http://127.0.0.1:1/api/i/nobody/frame body=bare,unconfirmed "
            f"sent=0 dropped={self.dropped} "
            "last=ClientConnectorError: Cannot connect to host"
        )


class RecordingSync(Sync):
    """The real renderer, remembering which index the loop handed it and when.
    That index IS the phase - render() divides it by geometry.FPS to get
    seconds - so recording it is recording the beat."""

    def __init__(self, clock: Clock | None = None) -> None:
        super().__init__()
        self.clock = clock
        self.indices: list[int] = []
        self.at: list[float] = []

    def render(self, t):
        self.indices.append(t)
        self.at.append(self.clock.t if self.clock else 0.0)
        return super().render(t)


def make_app(producer, *, render_fps=FPS, send_fps=SEND_FPS, renderer=None, people=4):
    app = create_app(
        display="none", appraisal=False, render_fps=render_fps, send_fps=send_fps
    )
    app["producer"] = producer
    app["renderer"] = renderer if renderer is not None else Sync()
    now = time.time()
    for i in range(people):  # a real crowd, so the loop takes the real path
        app["crowd"].add(Reading(person=f"p{i}", bpm=BPM, confidence=1.0, ts=now))
    return app


def drive(app, clock) -> None:
    """Run the real loop until the injected sleep stops it."""
    try:
        run(_render_loop(app, clock=clock.now, sleep=clock.sleep))
    except _Stop:
        pass


# --- the phase is wall-clock ---------------------------------------------

def test_the_heartbeat_keeps_the_measured_rate_over_a_slow_link():
    """A producer that eats 50ms a frame drags the loop down to 20fps. The
    content must still advance 30 frames per second of wall clock: the frames
    go missing, the heart rate does not slow down."""
    clock = Clock(frames=200)
    prod = FakeProducer(clock, cost_s=0.05)
    rec = RecordingSync(clock)
    app = make_app(prod, renderer=rec, send_fps=FPS)
    drive(app, clock)

    rendered = len(rec.indices)
    assert rendered == 200
    assert rec.at[-1] > rendered / FPS * 1.4  # the link really was that slow
    assert rec.indices[-1] > rendered  # the clock outran the loop counter

    cycle = FPS * 60.0 / BPM  # frames of content in one cardiac cycle
    minutes = (rec.at[-1] - rec.at[0]) / 60.0
    shown = ((rec.indices[-1] - rec.indices[0]) / cycle) / minutes
    assert abs(shown - BPM) < 1.0

    # And the loop counter - what the loop used to hand the renderer - would
    # have shown this crowd a heart rate nobody in it had.
    from_counter = ((rendered - 1) / cycle) / minutes
    assert from_counter < 60.0


def test_a_healthy_link_still_advances_one_frame_at_a_time():
    """The fix must not cost the ordinary case its smoothness: at 30fps with
    nothing in the way, the clock-derived index is the frame count."""
    clock = Clock(frames=60)
    rec = RecordingSync(clock)
    app = make_app(FakeProducer(clock), renderer=rec, send_fps=FPS)
    drive(app, clock)
    assert rec.indices == list(range(60))


def test_the_phase_survives_a_stall_long_enough_to_trip_the_resync():
    """One 3-second stall is past the loop's `fell far behind` branch. That
    branch drops the backlog, which is right, and it must cost frames only."""
    clock = Clock(frames=20)
    rec = RecordingSync(clock)
    prod = FakeProducer(clock)
    app = make_app(prod, renderer=rec, send_fps=FPS)

    real_send = prod.send

    async def stall_once(frame_no, rows, phase=""):
        if frame_no == 5:
            clock.t += 3.0  # the link goes away for three seconds
        await real_send(frame_no, rows, phase)

    prod.send = stall_once
    drive(app, clock)

    assert rec.indices[-1] == round(rec.at[-1] * FPS)  # still on the clock
    assert rec.indices[-1] > 3.0 * FPS  # the lost seconds were not swallowed


# --- the operator can see failures ---------------------------------------

def test_a_producer_that_fails_silently_still_reaches_the_operator():
    """The defect: display_status was sampled once, before the loop, and the
    only refresh path fired when send() raised - which HttpProducer never
    does. Live, the producer read `sent=0 dropped=6 last=ClientConnectorError`
    while the UI field read `sent=0 dropped=0` with no error at all."""
    clock = Clock(frames=6)
    prod = SilentFailure()
    app = make_app(prod, send_fps=FPS)
    drive(app, clock)

    assert prod.dropped == 6
    assert app["display_status"] == prod.status
    assert "dropped=6" in app["display_status"]
    assert "ClientConnectorError" in app["display_status"]
    assert "dropped=0" not in app["display_status"]  # i.e. not the startup sample
    assert app["status"]["display"] == prod.status  # the field the UI reads


def test_a_producer_that_raises_still_names_itself():
    """WebSocket and TCP producers do raise when the socket dies. Re-reading
    .status every frame must not lose that."""
    clock = Clock(frames=3)
    prod = FakeProducer(clock)
    prod.raises = RuntimeError("socket is gone")
    app = make_app(prod, send_fps=FPS)
    drive(app, clock)

    assert "send failed: RuntimeError: socket is gone" in app["display_status"]
    assert prod.status in app["display_status"]  # and the live counters too


def test_a_recovered_send_clears_its_error():
    """A stale error pinned to the screen is the same lie as a stale success."""
    clock = Clock(frames=6)
    prod = FakeProducer(clock)
    prod.raises = RuntimeError("socket is gone")
    app = make_app(prod, send_fps=FPS)

    real_send = prod.send

    async def recover(frame_no, rows, phase=""):
        if frame_no >= 2:
            prod.raises = None
        await real_send(frame_no, rows, phase)

    prod.send = recover
    drive(app, clock)

    assert "send failed" not in app["display_status"]
    assert app["display_status"] == prod.status


def test_a_display_that_never_connected_says_so():
    """connect() failing is not transient - there is no reconnect path - so
    that message outranks the producer's own idea of its health."""

    class Refuses(FakeProducer):
        async def connect(self):
            raise ConnectionRefusedError("nothing listening")

    clock = Clock(frames=3)
    app = make_app(Refuses(clock), send_fps=FPS)
    drive(app, clock)
    assert app["display_status"].startswith("failed: ConnectionRefusedError")


def test_the_first_frame_is_validated_before_anything_goes_out():
    """Existing behaviour, re-pinned because the loop moved underneath it: a
    malformed frame must kill the process now rather than be rejected by the
    building mid-show."""

    class BadSync(Sync):
        def render(self, t):
            return tuple(super().render(t)[:3])  # 3 rows, not 17

    clock = Clock(frames=5)
    prod = FakeProducer(clock)
    app = make_app(prod, renderer=BadSync(), send_fps=FPS)
    with pytest.raises(ValueError, match="expected 17 rows"):
        drive(app, clock)
    assert prod.sent == []


# --- render rate and send rate are different questions --------------------

def test_the_send_rate_is_settable_independently_of_the_render_rate():
    """15 out of 30: the facade gets every other frame, evenly spaced, and the
    renderer and the browser preview still get all of them."""
    clock = Clock(frames=60)
    rec = RecordingSync(clock)
    prod = FakeProducer(clock)
    app = make_app(prod, renderer=rec, render_fps=FPS, send_fps=15)
    drive(app, clock)

    assert len(rec.indices) == 60
    assert prod.sent == list(range(0, 60, 2))  # every other one, not 30 in a burst


def test_the_two_rates_may_also_be_the_same():
    clock = Clock(frames=40)
    prod = FakeProducer(clock)
    app = make_app(prod, render_fps=FPS, send_fps=FPS)
    drive(app, clock)
    assert prod.sent == list(range(40))


def test_a_send_rate_above_the_render_rate_cannot_invent_frames():
    """You cannot send a frame nobody drew: the effective rate is the lower of
    the two, one send per rendered frame at most."""
    clock = Clock(frames=20)
    prod = FakeProducer(clock)
    app = make_app(prod, render_fps=10, send_fps=FPS)
    drive(app, clock)
    assert prod.sent == list(range(20))


def test_the_default_send_rate_sits_under_the_measured_saturation_point():
    """Measured live against the simulator: 10/s sent -> 9.1 arrived, 20/s ->
    15.2, 28/s -> 13.4. Arrivals peak near 15 and then fall off a cliff, so
    pushing 30 delivers a choppier facade than pushing 15. The default is the
    knee of that curve, not the contract's ceiling."""
    assert SEND_FPS == 15
    assert SEND_FPS < FPS
    app = create_app(appraisal=False)
    assert app["send_fps"] == SEND_FPS
    assert app["render_fps"] == FPS


def test_the_send_rate_is_capped_at_the_contract_ceiling():
    """geometry.FPS is the display contract's hard ceiling: never send faster,
    whatever the operator types."""
    app = create_app(appraisal=False, send_fps=60)
    assert app["send_fps"] == FPS


def test_a_zero_send_rate_does_not_stop_the_show():
    """A fat-fingered 0 must not be a division by zero or a facade that stops
    after one frame."""
    app = create_app(appraisal=False, send_fps=0, render_fps=0)
    assert app["send_fps"] == 1
    assert app["render_fps"] == 1.0
