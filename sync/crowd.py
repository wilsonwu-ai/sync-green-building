"""Aggregate many phones into one body the building can wear."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

WINDOW_S = 120.0  # a reading older than this no longer describes the crowd
RESTING_BPM = 68.0  # anchor for the calm scale


@dataclass(frozen=True)
class Reading:
    person: str
    bpm: float
    confidence: float
    ts: float
    rmssd_ms: float | None = None
    tag: str = "live"  # "baseline" | "live" | "after"


@dataclass(frozen=True)
class Snapshot:
    n: int
    bpm: float
    coherence: float  # 0 scattered .. 1 beating together
    presence: float  # 0 nobody .. 1 a crowd
    calm: float  # 0 agitated .. 1 calm
    spread: float  # IQR of the crowd's BPM

    @property
    def empty(self) -> bool:
        return self.n == 0


class Crowd:
    def __init__(self, window_s: float = WINDOW_S) -> None:
        self.window_s = window_s
        self._readings: list[Reading] = []
        self._baseline: dict[str, float] = {}
        self._after: dict[str, float] = {}

    def add(self, r: Reading) -> None:
        self._readings.append(r)
        if r.tag == "baseline":
            self._baseline.setdefault(r.person, r.bpm)
        elif r.tag == "after":
            self._after[r.person] = r.bpm
        self._prune()

    def _prune(self, now: float | None = None) -> None:
        if now is None:  # 0.0 is a legitimate timestamp; `or` would discard it
            now = time.time()
        cutoff = now - self.window_s
        self._readings = [r for r in self._readings if r.ts >= cutoff]

    def snapshot(self, now: float | None = None) -> Snapshot:
        if now is None:  # 0.0 is a legitimate timestamp; `or` would discard it
            now = time.time()
        self._prune(now)
        live = [r for r in self._readings if r.confidence >= 0.35]
        if not live:
            return Snapshot(0, RESTING_BPM, 0.0, 0.0, 0.5, 0.0)

        # One vote per person: their most recent usable reading.
        latest: dict[str, Reading] = {}
        for r in live:
            prev = latest.get(r.person)
            if prev is None or r.ts > prev.ts:
                latest[r.person] = r
        bpms = sorted(r.bpm for r in latest.values())
        n = len(bpms)

        bpm = statistics.median(bpms)
        spread = (bpms[int(n * 0.75)] - bpms[int(n * 0.25)]) if n >= 4 else 0.0

        # Coherence collapses as the crowd's rates scatter. 25 BPM of spread
        # is "everyone is somewhere different"; 0 is a single shared pulse.
        coherence = 1.0 if n < 2 else max(0.0, 1.0 - spread / 25.0)
        presence = min(1.0, n / 12.0)

        # Calm: how far below a fast-and-anxious 100 BPM the crowd sits.
        calm = max(0.0, min(1.0, (100.0 - bpm) / (100.0 - RESTING_BPM)))
        return Snapshot(n, bpm, coherence, presence, calm, spread)

    def delta(self) -> dict:
        """Before/after, for the reveal. Only people with both readings count."""
        pairs = [
            (p, self._baseline[p], self._after[p])
            for p in self._baseline
            if p in self._after
        ]
        if not pairs:
            return {"n": 0, "before": None, "after": None, "drop": None}
        before = statistics.median(b for _, b, _ in pairs)
        after = statistics.median(a for _, _, a in pairs)
        return {
            "n": len(pairs),
            "before": round(before, 1),
            "after": round(after, 1),
            "drop": round(before - after, 1),
        }

    def person_delta(self, person: str) -> dict | None:
        if person not in self._baseline or person not in self._after:
            return None
        before, after = self._baseline[person], self._after[person]
        return {
            "before": round(before, 1),
            "after": round(after, 1),
            "drop": round(before - after, 1),
        }
