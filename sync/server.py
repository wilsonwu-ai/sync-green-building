"""One process: serves the phone page, collects pulses, renders 153 windows
30 times a second, and pushes them at the facade.

Phones POST their readings rather than holding a socket open. Conference wifi
drops sockets constantly and a lost reconnect during a live install is a dead
participant; a POST either lands or is retried on the next tap.
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


def _pack(rows) -> str:
    buf = bytearray()
    for row in rows:
        for px in row:
            buf.extend(bytes(px))
    return base64.b64encode(bytes(buf)).decode()


async def _render_loop(app: web.Application) -> None:
    crowd: Crowd = app["crowd"]
    policy: Policy = app["policy"]
    renderer: Sync = app["renderer"]
    appraiser: Appraiser = app["appraiser"]
    producer = app["producer"]

    try:
        await producer.connect()
        app["display_status"] = producer.status
    except Exception as exc:  # noqa: BLE001
        app["display_status"] = f"failed: {type(exc).__name__}: {exc}"
        log.warning("display connect failed, preview only: %s", exc)

    frame_no = 0
    period = 1.0 / FPS
    next_at = time.monotonic()

    while True:
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
        rows = renderer.render(frame_no)

        if frame_no == 0:
            validate(rows)  # fail loudly on the first frame, never mid-show

        try:
            await producer.send(frame_no, rows, pacing.phase)
        except Exception as exc:  # noqa: BLE001
            app["display_status"] = f"send failed: {type(exc).__name__}: {exc}"

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
        delay = next_at - time.monotonic()
        if delay < -1.0:  # we fell far behind; resync rather than spiral
            next_at = time.monotonic()
            delay = 0.0
        await asyncio.sleep(max(0.0, delay))


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
    await ws.send_json({"hello": {"rows": ROWS, "cols": COLS, "fps": FPS}})
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


def create_app(display: str = "none", appraisal: bool = True) -> web.Application:
    app = web.Application()
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
