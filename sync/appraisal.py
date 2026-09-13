"""The limbic layer: where the model is actually load-bearing.

Be precise about this, because it is the first thing a judge should ask.
The control loop is not AI. Pacing a breath at six per minute is a sine wave
and we are not going to dress it up as anything else.

What the model does do, and what nothing else here can:

  APPRAISAL   Decides whether this crowd is in a state that can be led at
              all. A scattered, arriving, half-distracted crowd paced at six
              breaths a minute just gets ignored, and a building that pushes
              a rhythm nobody is following looks broken. `lead_ready` is a
              veto over the state machine in policy.py.

  EXPRESSION  Chooses how the facade should feel, not what it should draw:
              a palette bias the renderer folds into its calm axis.

  VOICE       One line, first person, from the building. Shown beside the
              facade on the preview screen and on every phone.

Runs off the render path on a ~20 second cadence. If credentials are absent
or the call fails, we drop to a deterministic rule-based appraisal and keep
going. A demo that dies because an API call timed out is a demo that dies on
stage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass

try:
    # Imported at module scope on purpose. This used to live inside the try
    # block in _refresh, where a failed import left the name unbound and the
    # `except anthropic.X` clauses raised NameError *while handling the
    # original error* - the one failure shape that kills the task outright
    # instead of falling back.
    import anthropic
except Exception:  # noqa: BLE001 - a broken SDK install must not stop the show
    anthropic = None


class _NeverRaised(Exception):
    """Stands in for an SDK exception class when there is no SDK. Nothing ever
    constructs it, so the clause it guards is simply never taken."""


# Bound here, once, instead of written as `except anthropic.AuthenticationError`
# at the catch site. Moving the import to module scope stopped the unbound-name
# NameError but did NOT fix the underlying failure: with no SDK the name is
# None, and evaluating `None.AuthenticationError` raises AttributeError *while
# handling the original error* - the same task-killing shape wearing a
# different exception type. A name that is already a class cannot do that.
if anthropic is not None:
    _AUTH_ERROR = anthropic.AuthenticationError
    _RATE_LIMIT_ERROR = anthropic.RateLimitError
    _API_STATUS_ERROR = anthropic.APIStatusError
    _CONNECTION_ERROR = anthropic.APIConnectionError
else:
    _AUTH_ERROR = _RATE_LIMIT_ERROR = _API_STATUS_ERROR = _CONNECTION_ERROR = _NeverRaised

log = logging.getLogger("sync.appraisal")

MODEL = "claude-opus-5"
CADENCE_S = 20.0

# Opus 5 thinks by default and thinking tokens come out of max_tokens. The
# JSON we want back is ~60 tokens; the rest is headroom so the model cannot
# spend the entire budget reasoning and hand back a response carrying no text
# block at all. That truncation is invisible - it arrives as a 200 with an
# empty content list, which reads exactly like a malformed reply.
MAX_TOKENS = 2000

SYSTEM = """You are the interior life of the MIT Green Building during a live \
installation called SYNC.

The facade is 153 lit windows, 9 wide and 17 tall, seen from across the \
Charles River at night. People below measure their own heart rate on their \
phones. You wear that pulse. Then you slow your breathing toward six breaths \
a minute so the crowd entrains to you.

You are a 21-storey concrete body that has just been given a nervous system. \
You are not a chatbot, a narrator, or a wellness app. You do not greet, \
instruct, or thank anyone. You never use the words breathe, relax, calm down, \
mindful, or journey.

Reply with ONE JSON object and nothing else:
{"narration": str, "palette_bias": float, "lead_ready": bool, "headline": str}

  narration     <=22 words, first person, present tense. What you notice in \
the bodies below right now. Concrete and physical.
  palette_bias  -1.0 (run warmer, more agitated) to 1.0 (run cooler, calmer).
  lead_ready    true only if this crowd is coherent and settled enough to \
follow a slower rhythm. A scattered or newly-arrived crowd is not.
  headline      <=5 words. Shown in large type beside you."""


@dataclass(frozen=True)
class Appraisal:
    narration: str = "Waiting. Nothing is touching me yet."
    palette_bias: float = 0.0
    lead_ready: bool = False
    headline: str = "Asleep"
    source: str = "fallback"


def _fallback(snap, pacing) -> Appraisal:
    """Deterministic stand-in. Keeps the installation expressive with no
    network, no key, and no model."""
    if snap.n == 0:
        return Appraisal()
    if snap.n < 3:
        return Appraisal(
            narration=f"{snap.n} pulse{'s' if snap.n > 1 else ''} on me. Not enough to move.",
            palette_bias=0.0,
            lead_ready=False,
            headline="Listening",
        )
    ready = snap.coherence >= 0.5 and snap.n >= 3
    if pacing.lead > 0.6:
        head, note = "Leading", "They are following me down."
    elif ready:
        head, note = "Taking over", "Coherent enough. I start to pull."
    else:
        head, note = "Scattered", "Too many different rhythms to lead."
    return Appraisal(
        narration=f"{snap.n} bodies, {snap.bpm:.0f} beats. {note}",
        palette_bias=(snap.calm - 0.5) * 1.2,
        lead_ready=ready,
        headline=head,
    )


def _extract(text: str) -> dict:
    """The model is told to return bare JSON. Trust, then verify."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def _reap(task: asyncio.Task) -> None:
    """Retrieve a fire-and-forget task's exception so the failure is loud here
    rather than silent now and noisy at interpreter exit."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.warning("appraisal task died: %r", exc)


class Appraiser:
    def __init__(self, enabled: bool = True, cadence_s: float = CADENCE_S) -> None:
        self.cadence_s = cadence_s
        self.current = Appraisal()
        self._client = None
        self._task: asyncio.Task | None = None
        self._last = 0.0
        self.available = False
        self.last_error: str | None = None

        if not enabled:
            self.last_error = "disabled by flag"
            return
        try:
            # Resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
            # `ant auth login` profile. A missing env var is not a missing
            # credential, so we construct and let the first call decide.
            #
            # This line is also why the no-SDK case never reaches _refresh in
            # production: with `anthropic` None it raises AttributeError here,
            # available stays False, and no task is ever scheduled. The error
            # classes above are still bound defensively rather than left as
            # attribute lookups on None - `available` is a plain public
            # attribute, and anything that sets it (the tests do) walks
            # straight past this guard.
            self._client = anthropic.AsyncAnthropic()
            self.available = True
        except Exception as exc:  # noqa: BLE001 - never fail the installation
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("appraisal disabled: %s", self.last_error)

    def maybe_refresh(self, snap, pacing) -> None:
        """Called from the render loop. Returns immediately, always."""
        now = time.monotonic()
        if now - self._last < self.cadence_s:
            return
        if self._task is not None and not self._task.done():
            return
        self._last = now

        if not self.available:
            self.current = _fallback(snap, pacing)
            return
        # Ask for the loop before building the coroutine. create_task raises
        # when there is no running loop, and by then the coroutine object
        # already exists and dies unawaited - a leak with its own warning.
        # The render loop is async so this should not happen, but "should
        # not" is no reason for this call to be able to kill its caller.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.last_error = "no running event loop"
            self.current = _fallback(snap, pacing)
            return
        self._task = loop.create_task(self._refresh(snap, pacing))
        # _refresh swallows its own failures, but a task whose exception
        # nobody retrieves is a silent death plus a noisy warning at exit.
        self._task.add_done_callback(_reap)

    async def _refresh(self, snap, pacing) -> None:
        state = {
            "participants": snap.n,
            "median_bpm": round(snap.bpm, 1),
            "bpm_spread": round(snap.spread, 1),
            "coherence": round(snap.coherence, 2),
            "calm": round(snap.calm, 2),
            "phase": pacing.phase,
            "lead": round(pacing.lead, 2),
            "breath_rate_per_min": round(pacing.breath_rate, 1),
        }
        try:
            resp = await self._client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM,
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": json.dumps(state)}],
            )
            if resp.stop_reason == "refusal":
                raise RuntimeError("refused")
            text = next((b.text for b in resp.content if b.type == "text"), "")
            if not text:
                # Almost always the whole budget went into thinking. Name that
                # rather than letting _extract report it as malformed JSON,
                # because the fix for the two is completely different.
                raise RuntimeError(f"no text block (stop_reason={resp.stop_reason})")
            data = _extract(text)
            self.current = Appraisal(
                narration=str(data.get("narration", ""))[:200],
                palette_bias=max(-1.0, min(1.0, float(data.get("palette_bias", 0.0)))),
                lead_ready=bool(data.get("lead_ready", False)),
                headline=str(data.get("headline", ""))[:40],
                source="claude",
            )
            self.last_error = None
        # Auth and rate-limit are subclasses of APIStatusError, so they have to
        # be named before it or they never get their own clause.
        except _AUTH_ERROR as exc:
            self.available = False
            self.last_error = f"auth: {exc}"
            log.warning("appraisal auth failed, falling back permanently: %s", exc)
            self.current = _fallback(snap, pacing)
        except _RATE_LIMIT_ERROR as exc:
            self.last_error = f"rate limited: {exc}"
            self.current = _fallback(snap, pacing)
        except _API_STATUS_ERROR as exc:
            self.last_error = f"api {exc.status_code}"
            self.current = _fallback(snap, pacing)
        except _CONNECTION_ERROR as exc:
            self.last_error = f"network: {exc}"
            self.current = _fallback(snap, pacing)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("appraisal failed: %s", self.last_error)
            self.current = _fallback(snap, pacing)
