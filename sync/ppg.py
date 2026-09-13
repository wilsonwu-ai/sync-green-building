"""Reference photoplethysmography estimator.

This is the Python mirror of `web/ppg.js`. The phone runs the JS version;
this one exists so the algorithm is testable in CI against synthetic
signals, and so we can re-analyse a captured trace offline.

Method: resample to a uniform rate, detrend, then normalised
autocorrelation over the lags that correspond to 40-180 BPM. Autocorrelation
beats peak counting on the noisy, motion-corrupted signal you get from a
finger held against a phone camera, and beats an FFT at these very short
capture lengths because it degrades gracefully instead of smearing across
bins.

Honesty note carried through the whole project: BPM from a phone camera is
solid. HRV from a phone camera is not. `rmssd_ms` below is reported as a
COARSE calm proxy and is never presented to a participant as a clinical
number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

BPM_MIN = 40.0
BPM_MAX = 180.0
TARGET_HZ = 30.0


@dataclass(frozen=True)
class Estimate:
    bpm: float
    confidence: float  # 0..1
    rmssd_ms: float | None  # coarse; None when too few beats were found
    n_beats: int

    @property
    def usable(self) -> bool:
        return self.confidence >= 0.35 and BPM_MIN <= self.bpm <= BPM_MAX


def resample(times: list[float], values: list[float], hz: float = TARGET_HZ) -> list[float]:
    """Linear-interpolate irregular (t, v) samples onto a uniform grid.

    `times` are seconds, ascending. requestAnimationFrame does not tick
    evenly, so this runs before any frequency analysis.
    """
    if len(times) < 4:
        return []
    t0, t1 = times[0], times[-1]
    n = int((t1 - t0) * hz)
    if n < 8:
        return []
    out = []
    j = 0
    for i in range(n):
        t = t0 + i / hz
        while j + 2 < len(times) and times[j + 1] < t:
            j += 1
        span = times[j + 1] - times[j]
        frac = 0.0 if span <= 0 else (t - times[j]) / span
        out.append(values[j] + (values[j + 1] - values[j]) * frac)
    return out


def detrend(xs: list[float], window: int = 31) -> list[float]:
    """Subtract a centred moving average. Kills the slow baseline drift from
    the finger warming the sensor and from auto-exposure hunting."""
    n = len(xs)
    if n == 0:
        return []
    half = window // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(xs[i] - sum(xs[lo:hi]) / (hi - lo))
    return out


def _autocorr(xs: list[float], lag: int) -> float:
    """Biased estimator: divide by the full length, not by the overlap.

    Dividing by the overlap (len - lag) is "unbiased" in the textbook sense
    and is exactly wrong here - it inflates long lags, where fewer sample
    pairs contribute, and so systematically pulls the winner toward half the
    true heart rate.
    """
    n = len(xs) - lag
    if n <= 0:
        return 0.0
    return sum(xs[i] * xs[i + lag] for i in range(n)) / len(xs)


def estimate(times: list[float], values: list[float], hz: float = TARGET_HZ) -> Estimate:
    """Estimate heart rate from a raw camera-channel trace."""
    xs = detrend(resample(times, values, hz))
    if len(xs) < int(hz * 4):  # need at least ~4 seconds
        return Estimate(0.0, 0.0, None, 0)

    energy = _autocorr(xs, 0)
    if energy <= 1e-12:
        return Estimate(0.0, 0.0, None, 0)

    lag_min = max(1, int(hz * 60.0 / BPM_MAX))
    lag_max = min(len(xs) - 2, int(hz * 60.0 / BPM_MIN))
    if lag_max <= lag_min:
        return Estimate(0.0, 0.0, None, 0)

    corrs = [(lag, _autocorr(xs, lag) / energy) for lag in range(lag_min, lag_max + 1)]

    global_best = max(c for _, c in corrs)
    if global_best <= 0.0:
        return Estimate(0.0, 0.0, None, 0)

    # Octave guard. For a clean periodic signal the correlation at TWICE the
    # true period is just as high as at the period itself, so a plain argmax
    # reports half the real heart rate roughly as often as not. Take instead
    # the SHORTEST local peak still scoring within 85% of the best: that one
    # is the fundamental, and the equally tall candidates past it are its
    # harmonics. A genuinely slow pulse is unaffected, because its half-lag
    # sits in antiphase and scores near -1.
    peaks_idx = [
        i
        for i in range(1, len(corrs) - 1)
        if corrs[i][1] >= corrs[i - 1][1] and corrs[i][1] >= corrs[i + 1][1]
    ]
    threshold = 0.85 * global_best
    idx = next((i for i in peaks_idx if corrs[i][1] >= threshold), None)
    if idx is None:
        idx = max(range(len(corrs)), key=lambda i: corrs[i][1])

    best = corrs[idx][1]
    best_lag = float(corrs[idx][0])

    # Parabolic refinement around the chosen lag for sub-sample accuracy.
    if 0 < idx < len(corrs) - 1:
        y0, y1, y2 = corrs[idx - 1][1], corrs[idx][1], corrs[idx + 1][1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            best_lag += 0.5 * (y0 - y2) / denom

    bpm = 60.0 * hz / best_lag

    # Confidence: how much the winning lag stands out from the rest of the band.
    others = [c for lag, c in corrs if abs(lag - best_lag) > lag_min * 0.4]
    baseline = sum(abs(c) for c in others) / len(others) if others else 0.0
    confidence = max(0.0, min(1.0, (best - baseline) / 0.5)) if best > 0 else 0.0

    beats = _peaks(xs, int(hz * 60.0 / bpm * 0.6))
    rmssd = _rmssd_ms(beats, hz)
    return Estimate(bpm, confidence, rmssd, len(beats))


def _peaks(xs: list[float], min_gap: int) -> list[int]:
    thresh = 0.4 * max((abs(x) for x in xs), default=0.0)
    out: list[int] = []
    for i in range(1, len(xs) - 1):
        if xs[i] > thresh and xs[i] >= xs[i - 1] and xs[i] > xs[i + 1]:
            if not out or i - out[-1] >= min_gap:
                out.append(i)
    return out


def _rmssd_ms(peaks: list[int], hz: float) -> float | None:
    """Coarse RMSSD. Reported as a calm proxy only; see module docstring."""
    if len(peaks) < 4:
        return None
    ibis = [(peaks[i + 1] - peaks[i]) * 1000.0 / hz for i in range(len(peaks) - 1)]
    diffs = [ibis[i + 1] - ibis[i] for i in range(len(ibis) - 1)]
    if not diffs:
        return None
    return math.sqrt(sum(d * d for d in diffs) / len(diffs))
