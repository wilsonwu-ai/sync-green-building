"""When the building stops mirroring you and starts leading you.

The whole idea lives in this state machine. First the facade shows the crowd
its own pulse, truthfully. Then it slides its breathing away from the crowd's
natural rate and down toward roughly six breaths a minute, the resonance band
where paced breathing reliably raises HRV and drops sympathetic arousal.
Then it holds there and the crowd follows it, because that is what bodies do
in the presence of a slower rhythm. Then it shows the delta.

`lead` is the single number that carries the narrative: 0.0 = the building is
wearing you, 1.0 = you are wearing the building.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

TARGET_BREATH_RATE = 6.0  # breaths per minute; the resonance-frequency band

FOLLOW_S = 30.0
ENTRAIN_S = 60.0
LEAD_S = 150.0
REVEAL_S = 30.0

PHASES = ("idle", "follow", "entrain", "lead", "reveal")


def natural_breath_rate(bpm: float) -> float:
    """A rough resting breath rate implied by a heart rate.

    Healthy adults sit near a 4.5:1 cardiac-to-respiratory ratio at rest.
    Clamped hard because this is an expressive default, not a measurement.
    """
    return max(9.0, min(20.0, bpm / 4.5))


@dataclass(frozen=True)
class Pacing:
    phase: str
    lead: float  # 0..1
    breath_rate: float  # breaths per minute the facade is pacing
    elapsed: float
    remaining: float


class Policy:
    """Advances only while someone is actually present."""

    def __init__(self) -> None:
        self.phase = "idle"
        self._entered = time.monotonic()
        self._ready = False

    def set_ready(self, ready: bool) -> None:
        """The appraisal layer's veto. A crowd that is not settled enough to be
        led is left alone rather than paced at."""
        self._ready = ready

    def _enter(self, phase: str, now: float) -> None:
        self.phase = phase
        self._entered = now

    def update(self, present: bool, now: float | None = None) -> None:
        if now is None:  # 0.0 is a legitimate timestamp; `or` would discard it
            now = time.monotonic()
        elapsed = now - self._entered

        if not present:
            if self.phase != "idle":
                self._enter("idle", now)
            return

        if self.phase == "idle":
            self._enter("follow", now)
        elif self.phase == "follow" and elapsed >= FOLLOW_S and self._ready:
            self._enter("entrain", now)
        elif self.phase == "entrain" and elapsed >= ENTRAIN_S:
            self._enter("lead", now)
        elif self.phase == "lead" and elapsed >= LEAD_S:
            self._enter("reveal", now)
        elif self.phase == "reveal" and elapsed >= REVEAL_S:
            self._enter("follow", now)

    def pacing(self, bpm: float, now: float | None = None) -> Pacing:
        if now is None:  # 0.0 is a legitimate timestamp; `or` would discard it
            now = time.monotonic()
        elapsed = now - self._entered
        natural = natural_breath_rate(bpm)

        if self.phase == "idle":
            lead, total = 0.0, 0.0
        elif self.phase == "follow":
            lead, total = 0.0, FOLLOW_S
        elif self.phase == "entrain":
            lead, total = min(1.0, elapsed / ENTRAIN_S), ENTRAIN_S
        elif self.phase == "lead":
            lead, total = 1.0, LEAD_S
        else:  # reveal
            lead, total = 1.0, REVEAL_S

        rate = natural + (TARGET_BREATH_RATE - natural) * lead
        return Pacing(
            phase=self.phase,
            lead=lead,
            breath_rate=rate,
            elapsed=elapsed,
            remaining=max(0.0, total - elapsed),
        )
