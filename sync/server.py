"""One process: serves the phone page, collects pulses, renders 153 windows,
and pushes them at the facade.

Phones POST their readings rather than holding a socket open. Conference wifi
drops sockets constantly and a lost reconnect during a live install is a dead
participant; a POST either lands or is retried on the next tap.

TWO RATES, NOT ONE. Rendering and sending are different questions and they get
different answers. We render at the contract's 30fps because it is local, it is
cheap, and the browser preview wants every frame. The facade does not get every
frame: measured against the real simulator, the viewer's arrival rate saturates
near 15fps and gets WORSE above it.

    sent 10/s -> 9.1 arrived      (0.91)
    sent 20/s -> 15.2 arrived     (0.76)
    sent 28/s -> 13.4 arrived     (0.48)

That is congestion collapse, and it means driving the facade at 30 delivers
fewer frames than driving it at 15 - the same building, visibly choppier, for
twice the traffic. See SEND_FPS.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from pathlib import Path

from aiohttp import WSMsgType, web

from .appraisal import Appraiser
from .crowd import Crowd, Reading
from .geometry import COLS, FPS, ROWS
from .policy import Policy
from .producer import build as build_producer
from .producer import validate
from .render import Expression, Sync

log = logging.getLogger("sync.server")
WEB = Path(__file__).resolve().parent.parent / "web"

# What actually goes out at the building, as opposed to what we draw. 15 sits
# just under the knee of the curve in the module docstring, where almost every
# frame we send still arrives, so the facade moves evenly instead of arriving
# in gusts. It also divides FPS exactly: one sent frame every other rendered
# frame, rather than a send cadence beating against the render cadence and
# wobbling the motion. Override with --send-fps if the link on the night is
# better than the one we measured.
SEND_FPS = 15


def _pack(rows) -> str:
    buf = bytearray()
    for row in rows:
        for px in row:
            buf.extend(bytes(px))
    return base64.b64encode(bytes(buf)).decode()


async def _render_loop(
    app: web.Application,
    *,
    clock=time.monotonic,
    sleep=asyncio.sleep,
) -> None:
    """Draw on the clock, send on a slower clock, and never infer either one
    from how many times we have been round this loop.

    `clock` and `sleep` are injectable for the same reason policy.py and
    HttpProducer.send() take `now`: a test that asserts on cadence and uses the
    wall clock is a flake with a schedule.
    """
    crowd: Crowd = app["crowd"]
    policy: Policy = app["policy"]
    renderer: Sync = app["renderer"]
    appraiser: Appraiser = app["appraiser"]
    producer = app["producer"]
    render_fps = app.get("render_fps") or FPS
    send_fps = app.get("send_fps") or SEND_FPS

    connect_error: str | None = None
    try:
        await producer.connect()
    except Exception as exc:  # noqa: BLE001
        connect_error = f"failed: {type(exc).__name__}: {exc}"
        log.warning("display connect failed, preview only: %s", exc)
    app["display_status"] = connect_error or producer.status

    frame_no = 0
    send_error: str | None = None
    last_slot = -1
    period = 1.0 / render_fps
    started = clock()
    next_at = started

    while True:
        now = clock()
        snap = crowd.snapshot()
        policy.set_ready(appraiser.current.lead_ready)
        policy.update(present=snap.n > 0)
        pacing = policy.pacing(snap.bpm)
        appraiser.maybe_refresh(snap, pacing)

        renderer.set_expression(
            Expression(
                bpm=snap.bpm,
                breath_rate=pacing.breath_rate,
                lead=pacing.lead,
                calm=snap.calm,
                coherence=snap.coherence,
                presence=snap.presence,
                palette_bias=appraiser.current.palette_bias,
            )
        )
        # The phase comes off the CLOCK, never off this loop's counter.
        # render() divides the index it is handed by geometry.FPS to recover
        # seconds, so an index that is really an iteration count tells it less
        # time has passed than has actually passed - the instant we cannot keep
        # up, and over a congested link we never can. Measured live: 35s of
        # content took 48.7s to send, so a crowd measured at 72 BPM watched
        # itself beat at 52. The entire falsifiable claim of this installation
        # is that the facade beats at the rate the crowd measured. A slow link
        # is allowed to cost smoothness; it may never cost that.
        t = round((now - started) * FPS)
        rows = renderer.render(t)

        if frame_no == 0:
            validate(rows)  # fail loudly on the first frame, never mid-show

        # The send slot comes off the same integer index, so the two cadences
        # cannot drift apart: every other frame at 15 of 30, every frame when
        # the two rates match, and at most one send per rendered frame however
        # they are set.
        slot = t * send_fps // FPS
        if slot > last_slot:
            last_slot = slot
            try:
                await producer.send(frame_no, rows, pacing.phase)
                send_error = None
            except Exception as exc:  # noqa: BLE001
                send_error = f"send failed: {type(exc).__name__}: {exc}"

        # Re-sampled every frame, not once before the loop. HttpProducer is
        # deliberately built never to raise, so its failures live ONLY in
        # .status - a refused endpoint reads `sent=0 dropped=6
        # last=ClientConnectorError...` there while the exception path here
        # never fires. Sampled once at startup, the operator's screen reported
        # a healthy link forever with nothing landing on the building.
        live = connect_error or producer.status
        app["display_status"] = f"{live} | {send_error}" if send_error else live

        app["last_frame"] = rows
        app["status"] = {
            "phase": pacing.phase,
            "lead": round(pacing.lead, 3),
            "breath_rate": round(pacing.breath_rate, 2),
            "remaining": round(pacing.remaining, 1),
            "bpm": round(snap.bpm, 1),
            "participants": snap.n,
            "coherence": round(snap.coherence, 2),
            "calm": round(snap.calm, 2),
            "narration": appraiser.current.narration,
            "headline": appraiser.current.headline,
            "appraisal_source": appraiser.current.source,
            "appraisal_error": appraiser.last_error,
            "display": app.get("display_status", "preview only"),
            "delta": crowd.delta(),
        }

        if app["preview_sockets"]:
            payload = {"n": frame_no, "px": _pack(rows), "status": app["status"]}
            dead = []
            for ws in app["preview_sockets"]:
                try:
                    await ws.send_json(payload)
                except Exception:  # noqa: BLE001
                    dead.append(ws)
            for ws in dead:
                app["preview_sockets"].discard(ws)

        frame_no += 1
        next_at += period
        delay = next_at - clock()
        if delay < -1.0:  # we fell far behind; resync rather than spiral
            # Free to do now: the phase is on the clock, so skipping the
            # backlog costs frames and not a single beat of accuracy.
            next_at = clock()
            delay = 0.0
        await sleep(max(0.0, delay))


async def post_reading(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        raise web.HTTPBadRequest(reason="expected JSON")

    try:
        bpm = float(body["bpm"])
        confidence = float(body.get("confidence", 0.0))
    except (KeyError, TypeError, ValueError):
        raise web.HTTPBadRequest(reason="bpm and confidence required")
    if not 30.0 <= bpm <= 220.0:
        raise web.HTTPBadRequest(reason=f"bpm {bpm} out of range")

    person = str(body.get("person") or uuid.uuid4())[:64]
    tag = body.get("tag", "live")
    if tag not in ("baseline", "live", "after"):
        tag = "live"

    rmssd = body.get("rmssd_ms")
    crowd: Crowd = request.app["crowd"]
    crowd.add(
        Reading(
            person=person,
            bpm=bpm,
            confidence=confidence,
            ts=time.time(),
            rmssd_ms=float(rmssd) if rmssd is not None else None,
            tag=tag,
        )
    )
    return web.json_response(
        {
            "ok": True,
            "person": person,
            "status": request.app.get("status", {}),
            "you": crowd.person_delta(person),
        }
    )


async def get_state(request: web.Request) -> web.Response:
    return web.json_response(request.app.get("status", {}))


async def preview_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    request.app["preview_sockets"].add(ws)
    # The rate the preview will actually receive frames at, which is the render
    # rate and not the (lower) rate the facade is fed at.
    fps = request.app.get("render_fps", FPS)
    await ws.send_json({"hello": {"rows": ROWS, "cols": COLS, "fps": fps}})
    try:
        async for msg in ws:
            if msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                break
    finally:
        request.app["preview_sockets"].discard(ws)
    return ws


async def _index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(WEB / "index.html")


async def _preview(request: web.Request) -> web.FileResponse:
    return web.FileResponse(WEB / "preview.html")


async def _start(app: web.Application) -> None:
    app["loop_task"] = asyncio.create_task(_render_loop(app))


async def _stop(app: web.Application) -> None:
    task = app.get("loop_task")
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    await app["producer"].close()


def create_app(
    display: str = "none",
    appraisal: bool = True,
    render_fps: float = FPS,
    send_fps: int = SEND_FPS,
) -> web.Application:
    app = web.Application()
    # Two rates, set independently. Rendering is local and costs us nothing
    # anyone can see, so it runs at the contract's 30; sending is metered by a
    # link that measurably falls over above ~15/s.
    app["render_fps"] = max(1.0, float(render_fps))
    if int(send_fps) > FPS:
        # geometry.FPS is the contract's ceiling - "never send faster" - and
        # the measured ceiling is half of it. Cap rather than obey.
        log.warning("send-fps %s is above the display contract's %s/s; capped", send_fps, FPS)
    app["send_fps"] = max(1, min(int(send_fps), FPS))
    app["crowd"] = Crowd()
    app["policy"] = Policy()
    app["renderer"] = Sync()
    app["appraiser"] = Appraiser(enabled=appraisal)
    app["producer"] = build_producer(display)
    app["preview_sockets"] = set()
    app["status"] = {}
    app["last_frame"] = None

    app.router.add_get("/", _index)
    app.router.add_get("/preview", _preview)
    app.router.add_post("/api/reading", post_reading)
    app.router.add_get("/api/state", get_state)
    app.router.add_get("/ws/preview", preview_ws)
    app.router.add_static("/static", WEB)

    app.on_startup.append(_start)
    app.on_cleanup.append(_stop)
    return app
