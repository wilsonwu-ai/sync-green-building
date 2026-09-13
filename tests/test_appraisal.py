"""The limbic layer is the one part of this system that can fail from the
outside - no key, no network, a rate limit, a model that answers in prose. On
stage none of those may reach the render loop, so the contract under test is
narrow and absolute: the appraisal either improves the facade or it changes
nothing at all.

Every test here is offline. The client is always a fake; a real one is never
constructed, so there is no path from this file to the network and no way for
a green suite to be hiding an unpaid API call."""

import asyncio
import contextlib
import importlib.util
import logging
import sys
import time
from unittest import mock

import anthropic
import httpx
import pytest

import sync.appraisal
from sync.appraisal import (
    MAX_TOKENS,
    MODEL,
    Appraisal,
    Appraiser,
    _extract,
    _fallback,
    _reap,
)


# --- doubles -------------------------------------------------------------

class _Snap:
    """Duck-typed Snapshot. _fallback and _refresh only ever read attributes."""

    def __init__(self, n, bpm=72.0, coherence=1.0, presence=0.5, calm=0.5, spread=0.0):
        self.n, self.bpm, self.coherence = n, bpm, coherence
        self.presence, self.calm, self.spread = presence, calm, spread


class _Pacing:
    def __init__(self, phase="follow", lead=0.0, breath_rate=12.0):
        self.phase, self.lead, self.breath_rate = phase, lead, breath_rate


class _Block:
    def __init__(self, text, type="text"):
        self.type, self.text = type, text


class _Reply:
    def __init__(self, text=None, stop_reason="end_turn", blocks=None):
        self.content = blocks if blocks is not None else [_Block(text)]
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, reply=None, exc=None, delay=0.0):
        self.reply, self.exc, self.delay = reply, exc, delay
        self.calls, self.kwargs = 0, None

    async def create(self, **kw):
        self.calls += 1
        self.kwargs = kw
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        return self.reply


class _FakeClient:
    def __init__(self, **kw):
        self.messages = _FakeMessages(**kw)


GOOD = (
    '{"narration": "Six chests. The spread is closing.", '
    '"palette_bias": 0.4, "lead_ready": true, "headline": "Taking over"}'
)


def _appraiser(client, cadence_s=20.0):
    """Wire an Appraiser to a fake. enabled=False so __init__ never builds a
    real AsyncAnthropic - the credential lookup is exactly what we are not
    testing, and exactly what would make this suite depend on a machine."""
    a = Appraiser(enabled=False, cadence_s=cadence_s)
    a._client = client
    a.available = True
    a.last_error = None
    return a


def _err(cls, status):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx.Response(status, request=req), body=None)


# --- _fallback: the installation with no model at all ---------------------

def test_fallback_on_an_empty_crowd_is_asleep():
    a = _fallback(_Snap(n=0), _Pacing())
    assert a.source == "fallback"
    assert a.lead_ready is False
    assert a.headline == "Asleep"


def test_fallback_names_a_single_pulse_in_the_singular():
    a = _fallback(_Snap(n=1), _Pacing())
    assert "1 pulse on me" in a.narration
    assert "pulses" not in a.narration
    assert a.headline == "Listening"
    assert a.lead_ready is False


def test_fallback_pluralises_two_pulses():
    assert "2 pulses" in _fallback(_Snap(n=2), _Pacing()).narration


def test_fallback_takes_over_a_coherent_crowd():
    a = _fallback(_Snap(n=6, coherence=0.9), _Pacing())
    assert a.lead_ready is True
    assert a.headline == "Taking over"


def test_fallback_refuses_to_lead_a_scattered_crowd():
    """The veto has to survive the model being absent, or a scattered crowd
    gets paced at anyway - the exact failure the appraisal layer exists for."""
    a = _fallback(_Snap(n=6, coherence=0.2), _Pacing())
    assert a.lead_ready is False
    assert a.headline == "Scattered"


def test_fallback_reports_leading_once_the_building_has_the_room():
    a = _fallback(_Snap(n=6, coherence=0.9), _Pacing(phase="lead", lead=0.9))
    assert a.headline == "Leading"


def test_fallback_needs_three_bodies_before_it_will_lead():
    """Two coherent people are coherent by arithmetic, not by agreement."""
    assert _fallback(_Snap(n=2, coherence=1.0), _Pacing()).lead_ready is False


@pytest.mark.parametrize("calm", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_fallback_palette_bias_stays_in_range(calm):
    a = _fallback(_Snap(n=6, coherence=0.9, calm=calm), _Pacing())
    assert -1.0 <= a.palette_bias <= 1.0


@pytest.mark.parametrize("n", [0, 1, 2, 3, 40])
@pytest.mark.parametrize("coherence", [0.0, 0.49, 0.5, 1.0])
@pytest.mark.parametrize("lead", [0.0, 0.6, 0.61, 1.0])
def test_fallback_never_raises_whatever_the_crowd_does(n, coherence, lead):
    a = _fallback(_Snap(n=n, coherence=coherence, calm=0.3), _Pacing(lead=lead))
    assert isinstance(a, Appraisal)
    assert a.source == "fallback"


# --- _extract: trust, then verify ----------------------------------------

def test_extract_reads_clean_json():
    d = _extract(GOOD)
    assert d["narration"].startswith("Six chests")
    assert d["palette_bias"] == 0.4
    assert d["lead_ready"] is True
    assert d["headline"] == "Taking over"


def test_extract_survives_surrounding_prose():
    """Told to return bare JSON, a model will still occasionally introduce it."""
    assert _extract(f"Here is the appraisal you asked for:\n{GOOD}\nHope that helps.")["headline"] == "Taking over"


def test_extract_survives_a_markdown_fence():
    assert _extract(f"```json\n{GOOD}\n```")["lead_ready"] is True


def test_extract_raises_on_prose_with_no_object():
    with pytest.raises(ValueError, match="no JSON object"):
        _extract("I am a building, not a chatbot.")


def test_extract_raises_on_empty_text():
    with pytest.raises(ValueError, match="no JSON object"):
        _extract("")


def test_extract_raises_on_a_closing_brace_before_the_opening_one():
    with pytest.raises(ValueError, match="no JSON object"):
        _extract("} something {")


def test_extract_raises_on_malformed_json_between_braces():
    """json.JSONDecodeError is a ValueError, so the caller's one except clause
    covers both shapes of garbage."""
    with pytest.raises(ValueError):
        _extract("{not actually json}")


# --- the request we put on the wire --------------------------------------

def test_the_request_is_shaped_for_the_current_api():
    """Pins the three things that would silently disable this path if the API
    moved under us: the model id, effort nested inside output_config rather
    than sent top-level, and a max_tokens with room for Opus 5's thinking.

    All three are pinned against literals, the same way the model id already
    was. `kw["max_tokens"] == MAX_TOKENS` on its own asserts nothing - both
    sides are the same constant, so it holds for any value the module happens
    to define, including a budget of 1 that guarantees every reply is thinking
    truncated with no text block. A test that passes for every possible value
    of the thing it names is worse than no test, because it reads as coverage.
    The literal is the part that has to be updated deliberately."""
    c = _FakeClient(reply=_Reply(GOOD))
    a = _appraiser(c)
    asyncio.run(a._refresh(_Snap(n=6, coherence=0.9), _Pacing()))

    kw = c.messages.kwargs
    assert kw["model"] == MODEL == "claude-opus-5"
    assert kw["output_config"] == {"effort": "low"}
    assert kw["max_tokens"] == MAX_TOKENS == 2000
    assert kw["messages"][0]["role"] == "user"
    assert kw["system"]


def test_a_good_reply_becomes_a_claude_appraisal():
    a = _appraiser(_FakeClient(reply=_Reply(GOOD)))
    asyncio.run(a._refresh(_Snap(n=6, coherence=0.9), _Pacing()))
    assert a.current.source == "claude"
    assert a.current.narration == "Six chests. The spread is closing."
    assert a.current.palette_bias == pytest.approx(0.4)
    assert a.current.lead_ready is True
    assert a.current.headline == "Taking over"
    assert a.last_error is None


@pytest.mark.parametrize("sent,want", [(9.0, 1.0), (-9.0, -1.0), (0.25, 0.25)])
def test_palette_bias_from_the_model_is_clamped(sent, want):
    reply = _Reply('{"narration": "x", "palette_bias": %s, "lead_ready": false, "headline": "h"}' % sent)
    a = _appraiser(_FakeClient(reply=reply))
    asyncio.run(a._refresh(_Snap(n=6), _Pacing()))
    assert a.current.palette_bias == pytest.approx(want)


def test_an_overlong_narration_is_truncated_rather_than_shown_whole():
    reply = _Reply('{"narration": "%s", "palette_bias": 0, "lead_ready": false, "headline": "%s"}'
                   % ("x" * 500, "y" * 100))
    a = _appraiser(_FakeClient(reply=reply))
    asyncio.run(a._refresh(_Snap(n=6), _Pacing()))
    assert len(a.current.narration) == 200
    assert len(a.current.headline) == 40


# --- degradation: every failure lands on _fallback ------------------------

def test_a_client_error_degrades_to_the_fallback():
    a = _appraiser(_FakeClient(exc=RuntimeError("boom")))
    asyncio.run(a._refresh(_Snap(n=6, coherence=0.9), _Pacing()))
    assert a.current.source == "fallback"
    assert a.current.headline == "Taking over"  # still expressive, not blank
    assert "RuntimeError" in a.last_error


def test_prose_instead_of_json_degrades_to_the_fallback():
    a = _appraiser(_FakeClient(reply=_Reply("I am a building, not a chatbot.")))
    asyncio.run(a._refresh(_Snap(n=6, coherence=0.9), _Pacing()))
    assert a.current.source == "fallback"
    assert "no JSON object" in a.last_error


def test_a_refusal_degrades_to_the_fallback():
    a = _appraiser(_FakeClient(reply=_Reply(GOOD, stop_reason="refusal")))
    asyncio.run(a._refresh(_Snap(n=6), _Pacing()))
    assert a.current.source == "fallback"
    assert "refused" in a.last_error


def test_a_response_with_no_text_block_is_reported_as_truncation():
    """Opus 5 thinks by default and thinking spends max_tokens. When the budget
    runs out the reply arrives as a 200 with no text block, which would
    otherwise be reported as malformed JSON and send someone hunting the
    prompt instead of the token budget."""
    a = _appraiser(_FakeClient(reply=_Reply(blocks=[], stop_reason="max_tokens")))
    asyncio.run(a._refresh(_Snap(n=6), _Pacing()))
    assert a.current.source == "fallback"
    assert "no text block" in a.last_error
    assert "max_tokens" in a.last_error


def test_auth_failure_disables_the_path_permanently():
    """A bad key will still be a bad key in twenty seconds. Retrying it every
    cadence for the length of the show is just latency and spend."""
    a = _appraiser(_FakeClient(exc=_err(anthropic.AuthenticationError, 401)))
    asyncio.run(a._refresh(_Snap(n=6, coherence=0.9), _Pacing()))
    assert a.available is False
    assert a.current.source == "fallback"
    assert a.last_error.startswith("auth:")


def test_a_rate_limit_degrades_but_keeps_the_path_alive():
    """Unlike a bad key, this one is worth trying again."""
    a = _appraiser(_FakeClient(exc=_err(anthropic.RateLimitError, 429)))
    asyncio.run(a._refresh(_Snap(n=6), _Pacing()))
    assert a.available is True
    assert a.current.source == "fallback"
    assert "rate limited" in a.last_error


def test_a_server_error_degrades_but_keeps_the_path_alive():
    a = _appraiser(_FakeClient(exc=_err(anthropic.APIStatusError, 500)))
    asyncio.run(a._refresh(_Snap(n=6), _Pacing()))
    assert a.available is True
    assert a.current.source == "fallback"
    assert "500" in a.last_error


def test_a_network_error_degrades_but_keeps_the_path_alive():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    a = _appraiser(_FakeClient(exc=anthropic.APIConnectionError(request=req)))
    asyncio.run(a._refresh(_Snap(n=6), _Pacing()))
    assert a.available is True
    assert a.current.source == "fallback"
    assert a.last_error.startswith("network:")


def test_a_disabled_appraiser_still_produces_an_appraisal():
    """--no-appraisal must not mean a dead facade."""
    a = Appraiser(enabled=False)
    a.maybe_refresh(_Snap(n=6, coherence=0.9), _Pacing())
    assert a.current.source == "fallback"
    assert a.current.lead_ready is True


# --- ordering: maybe_refresh is called 30 times a second ------------------

def test_maybe_refresh_returns_immediately_even_when_the_call_is_slow():
    """It is called from the render loop. If it ever waits on the API, the
    facade stutters at 30fps for as long as the model is thinking."""
    async def scenario():
        c = _FakeClient(reply=_Reply(GOOD), delay=5.0)
        a = _appraiser(c, cadence_s=0.0)
        t0 = time.monotonic()
        a.maybe_refresh(_Snap(n=6), _Pacing())
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5, f"blocked the render loop for {elapsed:.2f}s"
        assert a._task is not None and not a._task.done()
        a._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await a._task

    asyncio.run(scenario())


def test_maybe_refresh_never_starts_a_second_call_while_one_is_in_flight():
    """cadence_s=0 removes the rate gate, so the only thing that can stop the
    second call is the in-flight check. Without it a slow response would fan
    out one request per frame."""
    async def scenario():
        c = _FakeClient(reply=_Reply(GOOD), delay=5.0)
        a = _appraiser(c, cadence_s=0.0)
        a.maybe_refresh(_Snap(n=6), _Pacing())
        await asyncio.sleep(0.01)  # let the task reach its await
        for _ in range(10):
            a.maybe_refresh(_Snap(n=6), _Pacing())
        await asyncio.sleep(0.01)
        assert c.messages.calls == 1
        a._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await a._task

    asyncio.run(scenario())


def test_maybe_refresh_honours_the_cadence_once_a_call_has_landed():
    """The cadence gate is the spend gate. Without it the render loop bills one
    paid request per frame - thirty a second, for the length of the show.

    The `await asyncio.sleep(0)` below is load-bearing and must not be tidied
    away. A task that has been created but not yet scheduled has not touched the
    client, so asserting the call count the instant maybe_refresh() returns
    passes whether or not the gate exists - the assertion cannot fail. The yield
    lets a task that should never have been created get as far as the API and be
    counted."""
    async def scenario():
        c = _FakeClient(reply=_Reply(GOOD))
        a = _appraiser(c, cadence_s=100.0)
        a.maybe_refresh(_Snap(n=6), _Pacing())
        first = a._task
        await first

        a.maybe_refresh(_Snap(n=6), _Pacing())
        await asyncio.sleep(0)
        assert a._task is first, "the cadence gate let a second task through"
        assert c.messages.calls == 1

    asyncio.run(scenario())


def test_maybe_refresh_starts_again_once_the_previous_call_has_landed():
    """The in-flight guard must not latch - it has to release on completion."""
    async def scenario():
        c = _FakeClient(reply=_Reply(GOOD))
        a = _appraiser(c, cadence_s=0.0)
        a.maybe_refresh(_Snap(n=6), _Pacing())
        await a._task
        a.maybe_refresh(_Snap(n=6), _Pacing())
        await a._task
        assert c.messages.calls == 2

    asyncio.run(scenario())


def test_maybe_refresh_outside_an_event_loop_falls_back_instead_of_raising():
    """create_task raises RuntimeError with no running loop. The render loop
    is async so this should not happen in production - but 'should not' is not
    a reason for the call to be able to kill its caller."""
    a = _appraiser(_FakeClient(reply=_Reply(GOOD)), cadence_s=0.0)
    a.maybe_refresh(_Snap(n=6, coherence=0.9), _Pacing())  # must not raise
    assert a.current.source == "fallback"
    assert a.last_error == "no running event loop"


# --- _reap: the fire-and-forget task's only listener ----------------------

def test_a_task_that_dies_is_reaped_and_logged(caplog):
    """_refresh swallows its own failures, so a task that raises anyway is a
    shape nobody predicted. Without the done-callback its exception is never
    retrieved: silent while it matters, then dumped by the interpreter at exit,
    long after the show, attached to nothing anyone can act on."""
    async def scenario():
        a = _appraiser(_FakeClient(reply=_Reply(GOOD)), cadence_s=0.0)

        async def detonate(snap, pacing):
            raise ZeroDivisionError("limbic")

        a._refresh = detonate  # set on the instance, so create_task picks it up

        with caplog.at_level(logging.WARNING, logger="sync.appraisal"):
            a.maybe_refresh(_Snap(n=6), _Pacing())
            # _reap was attached before this await added its own callback, and
            # done-callbacks run in the order they were added, so by the time
            # this resumes _reap has already had its turn.
            with pytest.raises(ZeroDivisionError):
                await a._task

        assert "appraisal task died" in caplog.text
        assert "ZeroDivisionError" in caplog.text

    asyncio.run(scenario())


def test_reap_stays_quiet_about_a_cancelled_task(caplog):
    """task.exception() re-raises CancelledError rather than returning it, so
    the cancelled check is not tidiness. Dropping it turns an ordinary shutdown
    into an exception raised inside a done-callback, where there is no caller
    left to catch it."""
    async def scenario():
        task = asyncio.get_running_loop().create_task(asyncio.sleep(3600))
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        with caplog.at_level(logging.WARNING, logger="sync.appraisal"):
            _reap(task)  # must not raise

        assert "appraisal task died" not in caplog.text

    asyncio.run(scenario())


def test_reap_stays_quiet_about_a_task_that_finished_cleanly(caplog):
    """The ordinary path runs through _reap on every single refresh. If it
    logged there, twenty seconds of show would fill the console with warnings
    about nothing being wrong."""
    async def scenario():
        a = _appraiser(_FakeClient(reply=_Reply(GOOD)), cadence_s=0.0)
        with caplog.at_level(logging.WARNING, logger="sync.appraisal"):
            a.maybe_refresh(_Snap(n=6, coherence=0.9), _Pacing())
            await a._task

        assert a.current.source == "claude"
        assert "appraisal task died" not in caplog.text

    asyncio.run(scenario())


# --- the SDK itself being missing ----------------------------------------

def _appraisal_module_without_the_sdk():
    """A second, private copy of sync/appraisal.py loaded with `import
    anthropic` failing, the way it fails on a machine where the SDK was never
    installed or the install is broken.

    Loaded under its own name and never left behind in sys.modules, so the real
    module that every other test in this file imported is untouched."""
    spec = importlib.util.spec_from_file_location(
        "sync_appraisal_without_sdk", sync.appraisal.__file__
    )
    mod = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"anthropic": None}):
        # dataclasses resolves cls.__module__ through sys.modules while the
        # module body is still executing, so the module has to be registered
        # for @dataclass to run at all. patch.dict takes the entry back out.
        sys.modules["sync_appraisal_without_sdk"] = mod
        spec.loader.exec_module(mod)
    return mod


def test_the_error_clauses_survive_the_sdk_being_absent():
    """Written as `except anthropic.AuthenticationError`, every one of these
    clauses raises AttributeError on a None module *while handling the original
    error* - so an ordinary API failure stops being a fallback and becomes a
    dead task. Moving the import to module scope changed the exception type
    from NameError to AttributeError and nothing else; binding the classes up
    front is what actually fixes it."""
    mod = _appraisal_module_without_the_sdk()
    assert mod.anthropic is None
    assert "sync_appraisal_without_sdk" not in sys.modules

    a = mod.Appraiser(enabled=False)
    a._client = _FakeClient(exc=RuntimeError("boom"))
    a.available = True
    asyncio.run(a._refresh(_Snap(n=6, coherence=0.9), _Pacing()))

    assert a.current.source == "fallback"
    assert a.current.headline == "Taking over"  # still expressive, not blank
    assert "RuntimeError" in a.last_error


def test_with_no_sdk_the_constructor_is_what_actually_disables_the_path():
    """Why the clauses above are belt-and-braces rather than a live bug: with
    no SDK the client is never built, available stays False, and maybe_refresh
    takes the fallback branch without scheduling anything. enabled=True is safe
    here for exactly that reason - `anthropic` is None, so there is nothing to
    construct and no credential lookup to reach the network."""
    mod = _appraisal_module_without_the_sdk()
    a = mod.Appraiser(enabled=True)
    assert a.available is False
    assert "AttributeError" in a.last_error

    a.maybe_refresh(_Snap(n=6, coherence=0.9), _Pacing())
    assert a.current.source == "fallback"
