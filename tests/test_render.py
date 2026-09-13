"""The renderer's output is a wire message. If it is malformed the display
rejects the frame outright - the contract says reject, never clamp - so these
tests guard the exact shape that goes on the wire."""

import math

import pytest

from sync.geometry import COLS, N_WINDOWS, ROWS, bay_of, chest_distance, floor_of
from sync.producer import digest, frame_message, validate
from sync.render import Expression, Sync, breath_level, lubdub


def test_facade_is_153_windows():
    assert ROWS * COLS == N_WINDOWS == 153


def test_row_zero_is_the_top_floor():
    assert floor_of(0) == 20
    assert floor_of(ROWS - 1) == 4
    assert bay_of(0) == 1
    assert bay_of(COLS - 1) == 9


def test_chest_distance_is_zero_at_the_heart():
    assert chest_distance(8) == 0.0
    assert chest_distance(0) == 1.0
    assert chest_distance(16) == 1.0


@pytest.mark.parametrize("presence", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("lead", [0.0, 0.5, 1.0])
def test_every_frame_is_a_valid_frame(presence, lead):
    s = Sync(Expression(bpm=72, breath_rate=9.0, lead=lead, presence=presence, calm=0.4))
    for t in range(0, 400, 7):
        validate(s.render(t))


def test_frames_are_stable_across_extreme_inputs():
    """Nothing the crowd can do should produce an out-of-range pixel."""
    for bpm in (30, 45, 100, 220):
        for rate in (3.0, 6.0, 25.0):
            for calm in (0.0, 1.0):
                for bias in (-1.0, 1.0):
                    s = Sync(
                        Expression(
                            bpm=bpm, breath_rate=rate, lead=0.5, calm=calm,
                            coherence=0.0, presence=1.0, palette_bias=bias,
                        )
                    )
                    validate(s.render(123))


def test_idle_is_dark_but_never_black():
    s = Sync(Expression(presence=0.0))
    peaks = []
    for t in range(0, 400, 5):
        rows = s.render(t)
        peaks.append(max(max(px) for row in rows for px in row))
    assert max(peaks) < 60, "idle should be dim"
    assert max(peaks) > 0, "the facade should never be fully dead"


def test_lubdub_has_two_sounds():
    first = lubdub(0.0)
    second = lubdub(0.30)
    trough = lubdub(0.60)
    assert first > second > trough
    assert trough < 0.05


def test_breath_level_spans_the_full_range():
    assert breath_level(0.0) == pytest.approx(0.0, abs=1e-6)
    assert breath_level(0.40) == pytest.approx(1.0, abs=1e-6)
    assert breath_level(0.999) == pytest.approx(0.0, abs=0.01)


def test_exhale_is_longer_than_inhale():
    """The long exhale is the half that does the calming."""
    rising = sum(1 for i in range(1000) if breath_level(i / 1000) < breath_level((i + 1) / 1000))
    assert rising / 1000 == pytest.approx(0.40, abs=0.02)


def test_heartbeat_reaches_the_crown_after_the_chest():
    """The pulse must propagate, not flash. Chest peaks before the roof."""
    s = Sync(Expression(bpm=60, presence=1.0, lead=0.0, calm=0.5, coherence=1.0))
    chest, head = [], []
    for t in range(60):  # one full cardiac cycle at 60bpm / 30fps
        rows = s.render(t)
        chest.append(sum(rows[8][4]))
        head.append(sum(rows[0][4]))
    assert chest.index(max(chest)) < head.index(max(head))


def test_the_beat_reads_against_the_breath_bed():
    """While following, the heartbeat must be a visible thump rather than a
    wash. This is the defect that only showed up by rendering frames and
    looking at them: with the breath layer sitting too bright underneath, and
    in too similar a hue, the beat vanished into it. Every other test still
    passed while the facade was unreadable, so the contrast is pinned here."""
    s = Sync(
        Expression(bpm=72, breath_rate=18.0, lead=0.0, calm=0.3,
                   coherence=0.9, presence=0.9)
    )
    chest = [sum(s.render(t)[8][4]) for t in range(25)]  # one cardiac cycle
    assert max(chest) > 2.2 * max(1, min(chest))


def test_following_favours_the_heart_and_leading_favours_the_breath():
    """The two ends of the arc should look like different things, not like a
    50/50 blend of the same thing at slightly different weights."""
    def layers(lead):
        s = Sync(Expression(bpm=72, breath_rate=6.0, lead=lead, calm=0.5,
                            coherence=1.0, presence=1.0))
        vals = [sum(s.render(t)[8][4]) for t in range(25)]
        return max(vals) - min(vals)  # temporal swing = the beat's visibility

    # 2.0x is a floor, not a fit. The measured separation is ~2.7x; a
    # symmetric blend of the two layers lands near 1.2x and must fail here.
    assert layers(0.0) > 2.0 * layers(1.0)


def test_lead_crosses_the_layers_over():
    """As the facade takes the lead the borrowed heartbeat must recede."""
    def heart_amplitude(lead):
        s = Sync(Expression(bpm=60, breath_rate=6.0, lead=lead, presence=1.0, calm=0.5))
        # Sample the chest row across one cardiac cycle; the peak-to-trough
        # swing is the heartbeat's visible contribution.
        vals = [sum(s.render(t)[8][4]) for t in range(60)]
        return max(vals) - min(vals)

    assert heart_amplitude(0.0) > heart_amplitude(1.0)


def test_digest_matches_the_spec_length():
    s = Sync(Expression(presence=1.0))
    rows = s.render(42)
    msg = frame_message(7, rows, phase="lead")
    assert msg["type"] == "frame"
    assert msg["frame_no"] == 7
    assert len(msg["rows"]) == ROWS
    assert all(len(r) == COLS for r in msg["rows"])
    assert len(msg["digest"]) == 64
    assert msg["digest"] == digest(rows)
    assert msg["phase"] == "lead"


def test_validate_rejects_out_of_range():
    s = Sync(Expression(presence=1.0))
    rows = [list(r) for r in s.render(1)]
    rows[3][3] = (300, 0, 0)
    with pytest.raises(ValueError, match="not 3 ints"):
        validate(rows)


def test_validate_rejects_wrong_shape():
    with pytest.raises(ValueError, match="expected 17 rows"):
        validate([[(0, 0, 0)] * COLS] * 5)
