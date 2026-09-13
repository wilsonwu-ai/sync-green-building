"""The pulse estimator, against synthetic signals with known answers.

This is the anchor for the whole installation: if BPM recovery is wrong, the
building is beating a lie and every downstream number is theatre.
"""

import math
import random

import pytest

from sync.ppg import detrend, estimate, resample


def synth(bpm, seconds=25.0, hz=30.0, noise=0.0, drift=0.0, jitter=0.0, seed=7):
    """A plausible fingertip trace: a pulse, sensor noise, baseline drift from
    the finger warming the lens, and irregular frame timing from rAF."""
    rng = random.Random(seed)
    times, values = [], []
    t = 0.0
    while t < seconds:
        phase = 2 * math.pi * bpm / 60.0 * t
        # Sharper than a sine: real PPG has a steep systolic upstroke.
        pulse = math.sin(phase) + 0.35 * math.sin(2 * phase)
        v = 128.0 + 6.0 * pulse + drift * t + rng.gauss(0, noise)
        times.append(t)
        values.append(v)
        t += 1.0 / hz + (rng.uniform(-jitter, jitter) if jitter else 0.0)
    return times, values


@pytest.mark.parametrize("bpm", [48, 58, 72, 88, 104, 132])
def test_recovers_clean_rate(bpm):
    est = estimate(*synth(bpm))
    assert est.usable
    assert est.bpm == pytest.approx(bpm, abs=2.0)


@pytest.mark.parametrize("bpm", [55, 72, 96])
def test_survives_noise_and_drift(bpm):
    est = estimate(*synth(bpm, noise=2.5, drift=1.8, jitter=0.004))
    assert est.usable
    assert est.bpm == pytest.approx(bpm, abs=3.0)


def test_rejects_pure_noise():
    rng = random.Random(3)
    times = [i / 30.0 for i in range(750)]
    values = [128 + rng.gauss(0, 5) for _ in times]
    assert not estimate(times, values).usable


def test_rejects_a_finger_that_never_moved():
    """A flat signal is a phone on a table, not a person."""
    times = [i / 30.0 for i in range(750)]
    assert not estimate(times, [128.0] * len(times)).usable


def test_rejects_too_short_a_capture():
    assert not estimate(*synth(72, seconds=2.0)).usable


def test_confidence_tracks_signal_quality():
    clean = estimate(*synth(72, noise=0.0))
    dirty = estimate(*synth(72, noise=6.0))
    assert clean.confidence > dirty.confidence


def test_resample_puts_samples_on_a_uniform_grid():
    times = [0.0, 0.05, 0.11, 0.19, 0.25, 0.30]
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    out = resample(times, values, hz=30.0)
    assert len(out) == pytest.approx(9, abs=1)
    assert all(isinstance(v, float) for v in out)


def test_detrend_removes_a_linear_baseline():
    xs = [float(i) for i in range(200)]
    out = detrend(xs, window=31)
    assert abs(sum(out[40:160]) / 120) < 0.5


def test_rmssd_is_reported_when_beats_are_found():
    est = estimate(*synth(72, seconds=25.0))
    assert est.n_beats >= 20
    assert est.rmssd_ms is not None
