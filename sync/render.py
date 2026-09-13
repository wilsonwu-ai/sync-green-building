"""The frame producer: 153 windows, 30 times a second.

Two layers, composited.

  HEART  A lub-dub that originates at the chest floors and propagates out
         toward the head and the feet. This is the crowd's real measured
         heart rate. It is the layer that says "the building is wearing a
         body".

  BREATH A lung. Light fills from the base to the crown on the inhale and
         empties on the exhale, at whatever rate the policy is pacing. This
         is the layer that leads.

As `lead` goes 0 -> 1 the weights cross over: the heartbeat recedes and the
breath takes the facade. That crossover IS the piece.

Everything here is deliberately low-frequency and single-hued. At 153 pixels
seen from across the Charles at night, detail is not available and contrast
is everything. Anything that reads as blocky or multi-coloured reads as
Tetris, which is exactly what this must not look like.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import COLS, FPS, ROWS, chest_distance

Rows = tuple[tuple[tuple[int, int, int], ...], ...]

# Palettes. Heart stays blood-coloured; breath carries the emotional state.
#
# Breath needs THREE stops, not two. Interpolating hot orange straight to cyan
# in RGB runs the midpoint through grey - they are near-complementary, so the
# channels cross over and cancel - and a facade that desaturates to concrete
# colour exactly during the transition is the worst possible time to lose it.
# The violet stop keeps the whole arc saturated.
# Breath's hot end is AMBER, not orange-red. It used to sit at (255, 96, 40),
# which is close enough to the heart's crimson that the two layers merged into
# one red wash and the beat stopped reading as a discrete event. Separating the
# hues gives the heartbeat figure against the breath's ground.
BREATH_AGITATED = (255, 146, 32)
BREATH_MID = (186, 72, 196)
BREATH_CALM = (26, 176, 196)
HEART_AGITATED = (255, 28, 44)
HEART_CALM = (255, 96, 122)

CENTER_COL = (COLS - 1) / 2

PULSE_TRAVEL = 0.16  # fraction of a cardiac cycle for the wave to reach the crown
FILL_SOFTNESS = 1.6  # rows over which the lung's waterline ramps


@dataclass(frozen=True)
class Expression:
    """Everything the renderer is allowed to know."""

    bpm: float = 68.0
    breath_rate: float = 14.0
    lead: float = 0.0
    calm: float = 0.5
    coherence: float = 1.0
    presence: float = 0.0
    palette_bias: float = 0.0  # appraisal nudge, -1 warmer .. +1 cooler


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _mix(c0: tuple[int, int, int], c1: tuple[int, int, int], t: float):
    t = max(0.0, min(1.0, t))
    return tuple(_lerp(c0[i], c1[i], t) for i in range(3))


def _ramp(stops, t: float):
    """Piecewise mix across an arbitrary number of colour stops."""
    t = max(0.0, min(1.0, t)) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    return _mix(stops[i], stops[i + 1], t - i)


def _heart_profile(col: int) -> float:
    """The pulse runs strongest up the tower's centre line and falls away
    toward the outer bays.

    Without this every column in a row is identical and the facade reads as
    seventeen horizontal stripes - a barcode, not a body - which throws away
    eight ninths of the horizontal resolution we actually have.
    """
    d = abs(col - CENTER_COL) / CENTER_COL
    return 1.0 - 0.5 * d**1.4


def _breath_profile(col: int) -> float:
    """A lung fills fairly evenly; a little edge falloff gives it volume."""
    d = abs(col - CENTER_COL) / CENTER_COL
    return 1.0 - 0.20 * d * d


def _circular_gauss(p: float, mu: float, sigma: float) -> float:
    d = abs(p - mu)
    d = min(d, 1.0 - d)  # the cycle wraps
    return math.exp(-0.5 * (d / sigma) ** 2)


def lubdub(p: float) -> float:
    """Cardiac envelope over one normalised cycle. First sound, then the
    quieter second sound about a third of the way through.

    Deliberately wider than a real heart sound. At 72 BPM one cycle is 25
    frames, so a physiologically honest sigma puts the whole `lub` inside a
    single 33ms frame - one frame of light at a quarter mile is not a
    heartbeat, it is a glitch.
    """
    return _circular_gauss(p, 0.0, 0.060) + 0.62 * _circular_gauss(p, 0.30, 0.070)


def breath_level(q: float) -> float:
    """0 = fully exhaled, 1 = fully inhaled, over one normalised cycle.

    Asymmetric on purpose: a 40/60 inhale/exhale split. The long exhale is
    the half that does the calming, and it is also the half that looks
    better on a building.
    """
    if q < 0.40:
        x = q / 0.40
    else:
        x = 1.0 - (q - 0.40) / 0.60
    return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, x)))


def _noise(r: int, c: int, k: int) -> float:
    h = (r * 73856093) ^ (c * 19349663) ^ (k * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    h &= 0xFFFFFFFF
    return (h % 10007) / 10007.0 - 0.5


def _smooth_noise(r: int, c: int, t: int, period: int = 5) -> float:
    k, frac = divmod(t, period)
    f = frac / period
    f = f * f * (3 - 2 * f)  # smoothstep, so agitation shimmers instead of strobing
    return _lerp(_noise(r, c, k), _noise(r, c, k + 1), f)


class Sync:
    """A frame producer in the shape the upstream simulator already expects:
    init() / tick(state) / render(t). Frames are tuples of 17 rows of 9 RGB
    triples, which is exactly what `Display.send()` and the wire protocol's
    `frame` message both take."""

    name = "sync"

    def __init__(self, expression: Expression | None = None) -> None:
        self.expression = expression or Expression()

    def init(self) -> int:
        return 0

    def tick(self, state: int, events=()) -> int:
        return state + 1

    def set_expression(self, expression: Expression) -> None:
        self.expression = expression

    def render(self, t: int) -> Rows:
        e = self.expression

        # Nobody here yet: the building idles, barely awake.
        if e.presence <= 0.0:
            return self._idle(t)

        heart_cycle = max(1.0, FPS * 60.0 / max(30.0, e.bpm))
        breath_cycle = max(1.0, FPS * 60.0 / max(3.0, e.breath_rate))

        hp = (t % heart_cycle) / heart_cycle
        bq = (t % breath_cycle) / breath_cycle
        level = breath_level(bq)
        waterline = (ROWS - 1) * (1.0 - level)

        calm = max(0.0, min(1.0, e.calm + 0.25 * e.palette_bias))
        breath_rgb = _ramp((BREATH_AGITATED, BREATH_MID, BREATH_CALM), calm)
        heart_rgb = _mix(HEART_AGITATED, HEART_CALM, calm)

        # The crossover. As the facade takes the lead, the borrowed heartbeat
        # fades and the breath it is teaching takes over.
        #
        # The weights are deliberately lopsided at each end rather than a
        # gentle blend. While following, the breath is held down to a dim bed
        # so the beat spikes clearly above it; while leading, the lung owns the
        # tower. A symmetric crossover looked like neither thing.
        w_heart = 1.10 - 0.75 * e.lead
        w_breath = 0.26 + 0.69 * e.lead

        agitation = 0.16 * (1.0 - calm) * (1.0 - 0.5 * e.coherence)
        gain = 0.35 + 0.65 * e.presence

        out = []
        for r in range(ROWS):
            dist = chest_distance(r)

            # Heart: delayed by distance from the chest, damped as it travels.
            pr = (hp - dist * PULSE_TRAVEL) % 1.0
            heart = lubdub(pr) * (1.0 - 0.58 * dist)

            # Breath: a soft waterline rising and falling through the tower.
            fill = max(0.0, min(1.0, (waterline - r) / FILL_SOFTNESS + 0.5))
            edge = math.exp(-0.5 * ((r - waterline) / 1.25) ** 2)
            breath = 0.10 + 0.75 * fill + 0.45 * edge

            row = []
            for c in range(COLS):
                h_col = _heart_profile(c)
                b_col = _breath_profile(c)
                jitter = 1.0 + agitation * _smooth_noise(r, c, t)
                k = gain * jitter
                px = tuple(
                    heart_rgb[i] * heart * w_heart * h_col
                    + breath_rgb[i] * breath * w_breath * b_col
                    for i in range(3)
                )
                row.append(tuple(max(0, min(255, int(v * k))) for v in px))
            out.append(tuple(row))
        return tuple(out)

    def _idle(self, t: int) -> Rows:
        """Asleep. A very slow, very dim swell so the facade is never dead."""
        period = FPS * 11
        phase = 0.5 - 0.5 * math.cos(2 * math.pi * (t % period) / period)
        rows = []
        for r in range(ROWS):
            falloff = 1.0 - 0.6 * chest_distance(r)
            k = 0.05 + 0.09 * phase * falloff
            rows.append(
                tuple(
                    tuple(
                        max(0, min(255, int(BREATH_CALM[i] * k * _breath_profile(c))))
                        for i in range(3)
                    )
                    for c in range(COLS)
                )
            )
        return tuple(rows)
