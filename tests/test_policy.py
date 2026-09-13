"""The state machine that carries the whole narrative, plus crowd aggregation."""

import time

import pytest

from sync.crowd import Crowd, Reading
from sync.policy import (
    ENTRAIN_S,
    FOLLOW_S,
    TARGET_BREATH_RATE,
    Policy,
    natural_breath_rate,
)


def _reading(person, bpm, tag="live", ts=None, conf=0.9):
    return Reading(person=person, bpm=bpm, confidence=conf, ts=ts or time.time(), tag=tag)


# --- crowd ---------------------------------------------------------------

def test_empty_crowd_is_empty():
    assert Crowd().snapshot().empty


def test_low_confidence_readings_are_ignored():
    c = Crowd()
    c.add(_reading("a", 70, conf=0.1))
    assert c.snapshot().empty


def test_one_person_votes_once():
    """A person hammering the button must not become a crowd."""
    c = Crowd()
    for bpm in (70, 72, 74, 76):
        c.add(_reading("a", bpm))
    assert c.snapshot().n == 1


def test_median_resists_an_outlier():
    c = Crowd()
    for i, bpm in enumerate([68, 70, 72, 200]):
        c.add(_reading(f"p{i}", bpm))
    assert c.snapshot().bpm == pytest.approx(71.0, abs=1.5)


def test_stale_readings_fall_out_of_the_window():
    c = Crowd(window_s=60)
    c.add(_reading("a", 70, ts=time.time() - 120))
    assert c.snapshot().empty


def test_coherence_falls_as_the_crowd_scatters():
    tight = Crowd()
    loose = Crowd()
    for i, bpm in enumerate([70, 71, 72, 73]):
        tight.add(_reading(f"p{i}", bpm))
    for i, bpm in enumerate([50, 70, 95, 130]):
        loose.add(_reading(f"p{i}", bpm))
    assert tight.snapshot().coherence > loose.snapshot().coherence


def test_calm_falls_as_the_crowd_speeds_up():
    slow, fast = Crowd(), Crowd()
    slow.add(_reading("a", 62))
    fast.add(_reading("a", 105))
    assert slow.snapshot().calm > fast.snapshot().calm


def test_delta_pairs_before_and_after():
    c = Crowd()
    c.add(_reading("a", 84, tag="baseline"))
    c.add(_reading("b", 80, tag="baseline"))
    c.add(_reading("a", 74, tag="after"))
    c.add(_reading("b", 72, tag="after"))
    d = c.delta()
    assert d["n"] == 2
    assert d["drop"] == pytest.approx(9.0, abs=0.1)


def test_delta_ignores_people_with_only_one_reading():
    c = Crowd()
    c.add(_reading("a", 84, tag="baseline"))
    assert c.delta()["n"] == 0


# --- policy --------------------------------------------------------------

def test_natural_breath_rate_is_clamped():
    assert natural_breath_rate(70) == pytest.approx(70 / 4.5, abs=0.01)
    assert natural_breath_rate(30) == 9.0
    assert natural_breath_rate(200) == 20.0


def test_idle_until_someone_arrives():
    p = Policy()
    p.update(present=False, now=100.0)
    assert p.phase == "idle"
    p.update(present=True, now=101.0)
    assert p.phase == "follow"


def test_an_empty_crowd_resets_to_idle():
    p = Policy()
    p.update(present=True, now=0.0)
    p.update(present=False, now=5.0)
    assert p.phase == "idle"


def test_follow_does_not_advance_without_the_appraisal_veto():
    """A scattered crowd is left alone rather than paced at."""
    p = Policy()
    p.update(present=True, now=0.0)
    p.set_ready(False)
    p.update(present=True, now=FOLLOW_S + 10)
    assert p.phase == "follow"


def test_ready_crowd_advances_into_entrainment():
    p = Policy()
    p.update(present=True, now=0.0)
    p.set_ready(True)
    p.update(present=True, now=FOLLOW_S + 1)
    assert p.phase == "entrain"


def test_following_mirrors_the_crowds_own_rate():
    p = Policy()
    p.update(present=True, now=0.0)
    pacing = p.pacing(bpm=90.0, now=1.0)
    assert pacing.lead == 0.0
    assert pacing.breath_rate == pytest.approx(natural_breath_rate(90.0), abs=0.01)


def test_leading_paces_the_resonance_rate():
    p = Policy()
    p.update(present=True, now=0.0)
    p.set_ready(True)
    p.update(present=True, now=FOLLOW_S + 1)
    p.update(present=True, now=FOLLOW_S + ENTRAIN_S + 2)
    assert p.phase == "lead"
    pacing = p.pacing(bpm=90.0, now=FOLLOW_S + ENTRAIN_S + 3)
    assert pacing.lead == 1.0
    assert pacing.breath_rate == pytest.approx(TARGET_BREATH_RATE, abs=0.01)


def test_entrainment_moves_monotonically_toward_the_target():
    p = Policy()
    p.update(present=True, now=0.0)
    p.set_ready(True)
    p.update(present=True, now=FOLLOW_S + 1)
    t0 = FOLLOW_S + 1
    rates = [p.pacing(90.0, now=t0 + f * ENTRAIN_S).breath_rate for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert rates == sorted(rates, reverse=True)
    assert rates[-1] == pytest.approx(TARGET_BREATH_RATE, abs=0.1)
