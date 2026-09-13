"""The renderer's output is a wire message. If it is malformed the display
rejects the frame outright - the contract says reject, never clamp - so these
tests guard the exact shape that goes on the wire."""

import math

import pytest

from sync.geometry import (
    COLS,
    N_WINDOWS,
    OCCLUDED_ROWS,
    ROWS,
    VISIBLE_ROWS,
    bay_of,
    chest_distance,
    floor_of,
    occluded_rows,
)
from sync.producer import digest, frame_message, validate
from sync.render import (
    PULSE_TRAVEL,
    Expression,
    Sync,
    breath_at,
    breath_level,
    lubdub,
    waterline_of,
)


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
    """The pulse must propagate, not flash. Chest peaks before the roof.

    The window starts in mid-diastole and runs exactly one cardiac cycle, so
    the single lub inside it is unambiguous. This test used to start at the
    lub and run 60 frames - two cycles at 60bpm, despite the comment - which
    compares the heights of two different beats, and the slowly moving breath
    bed underneath then decides which of the two is taller. That measures the
    breath, not the propagation, which is why it flipped the moment the lung's
    direction was corrected.
    """
    cycle = 30  # 60 bpm at 30 fps
    s = Sync(Expression(bpm=60, presence=1.0, lead=0.0, calm=0.5, coherence=1.0))

    def peak_frame(row):
        start = int(0.6 * cycle)  # diastole, so the lub lands mid-window
        vals = [sum(s.render(t)[row][4]) for t in range(start, start + cycle)]
        return start + vals.index(max(vals))

    chest, crown = peak_frame(8), peak_frame(0)
    assert chest < crown
    # It also has to arrive when the model says: chest_distance is 1.0 at the
    # crown, so the lag is PULSE_TRAVEL of a cycle. Measured 5 frames against
    # a modelled 4.8, and 0 frames with propagation removed.
    assert crown - chest == pytest.approx(PULSE_TRAVEL * cycle, abs=2)


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


# --- the tree line -------------------------------------------------------
# Trees on the river bank hide the bottom two rows from the across-the-river
# viewpoint the piece is composed for (geometry.OCCLUDED_ROWS). The breath is
# the layer that cares, because it is the layer with a moving edge. These pin
# the opt-in remap that keeps that edge where the audience is.


def test_tree_line_is_off_by_default_and_changes_nothing():
    """The remap is opt-in, so that every test above is still measuring the
    behaviour it was written against."""
    assert Expression().tree_line is False
    # The unoccluded waterline is exactly the formula it has always been.
    assert waterline_of(0.25) == pytest.approx((ROWS - 1) * 0.75, abs=1e-12)

    kw = dict(bpm=72, breath_rate=6.0, lead=0.5, calm=0.4, presence=1.0)
    default = Sync(Expression(**kw))
    explicit = Sync(Expression(**kw, tree_line=False))
    for t in range(0, 600, 3):
        assert default.render(t) == explicit.render(t)


def test_tree_line_keeps_the_whole_breath_above_the_trees():
    """The defect, stated geometrically. Unoccluded, the lung's surface spends
    part of every cycle down at rows 15-16, which is behind the trees. With
    the flag on it bottoms out on the last visible row instead, so both
    turnarounds happen in view."""
    assert occluded_rows() == (15, 16)
    assert VISIBLE_ROWS == ROWS - OCCLUDED_ROWS == 15

    def excursion(tree_line):
        return [waterline_of(breath_level(i / 900), tree_line) for i in range(900)]

    off, on = excursion(False), excursion(True)
    assert max(off) == pytest.approx(ROWS - 1)  # row 16, i.e. behind trees
    assert any(w > VISIBLE_ROWS - 1 for w in off)

    assert not any(w > VISIBLE_ROWS - 1 for w in on)
    assert min(on) == pytest.approx(0.0, abs=1e-9)
    assert max(on) == pytest.approx(VISIBLE_ROWS - 1)


def test_breath_reaches_both_extremes_inside_the_visible_rows():
    """Rows 0..14 alone must carry the full dynamic range of the breath layer:
    the darkest and the brightest the unoccluded facade ever gets are both
    still reached, and now both reached where they can be seen."""

    def extremes(tree_line, rows):
        v = [
            breath_at(r, waterline_of(breath_level(i / 900), tree_line))
            for i in range(900)
            for r in rows
        ]
        return min(v), max(v)

    lo_all, hi_all = extremes(False, range(ROWS))
    lo_vis, hi_vis = extremes(True, range(VISIBLE_ROWS))
    # Measured: 0.10000 / 1.21663 unoccluded across all 17 rows, against
    # 0.10000 / 1.21660 occlusion-aware across the 15 visible ones.
    assert lo_vis == pytest.approx(lo_all, abs=1e-3)
    assert hi_vis == pytest.approx(hi_all, abs=1e-3)


def test_the_start_of_the_breath_is_legible_from_the_river():
    """The motivating defect. The first OCCLUDED_ROWS rows of the surface's
    travel are an eighth of the excursion, and unoccluded they happen entirely
    behind the trees - so from the river the breath reads as a dead hold at
    the bottom followed by a late start. The turnaround at the bottom is the
    moment a breathing exercise is actually teaching."""

    step = OCCLUDED_ROWS / (ROWS - 1)  # two rows of the surface's travel

    def visible_mean(level, tree_line):
        w = waterline_of(level, tree_line)
        return sum(breath_at(r, w) for r in range(VISIBLE_ROWS)) / VISIBLE_ROWS

    def move(level, tree_line):
        """What `step` of travel starting from `level` does to the visible rows."""
        return abs(visible_mean(level + step, tree_line) - visible_mean(level, tree_line))

    # Unoccluded, the opening move is the LEAST visible move of the whole
    # cycle - the surface is down at rows 15-16 behind the trees and only its
    # glow leaks into view. Remapped, it is the MOST visible one. That sign
    # change is the defect and the fix stated together, and it is pinned in
    # place of the tuned ratio this test used to carry: a ratio is only as
    # honest as the geometry it was measured against, and the old 2.30x was
    # measured against a lung that filled downward from the crown.
    # Measured now: 0.0768 opening against 0.1000 mid-travel unoccluded
    # (0.77x), and 0.1196 against 0.0906 occlusion-aware (1.32x).
    assert move(0.0, False) < move(0.5, False)
    assert move(0.0, True) > move(0.5, True)

    # And in absolute terms the audience gets more of the turnaround.
    # Measured 1.56x. 1.25 is a floor, not a fit.
    assert move(0.0, True) > 1.25 * move(0.0, False)


def test_visible_contrast_survives_the_tree_line():
    """Folding the breath into fifteen rows must not cost it brightness range
    on real frames: what rows 0..14 swing through with the flag on should
    match what the whole facade swings through with it off."""

    def swing(tree_line, rows):
        s = Sync(
            Expression(bpm=72, breath_rate=6.0, lead=1.0, calm=0.5,
                       coherence=1.0, presence=1.0, tree_line=tree_line)
        )
        series = []
        for t in range(300):  # one full breath cycle at 6/min, 30fps
            f = s.render(t)
            series.append(
                sum(sum(f[r][c]) for r in rows for c in range(COLS)) / (len(rows) * COLS)
            )
        return max(series) - min(series)

    unoccluded = swing(False, range(ROWS))
    occluded = swing(True, range(VISIBLE_ROWS))
    # Measured 333.82 against 335.58, a ratio of 1.005.
    assert occluded > 0.90 * unoccluded
    assert occluded < 1.15 * unoccluded


# --- which way the lung runs ---------------------------------------------
# Every test above this line measures CONTRAST or SWING - max minus min, or a
# ratio of two of those. An absolute value cannot see a sign, so the breath
# layer shipped inverted (brightest when EMPTY, the lit region descending from
# the crown instead of filling from the base) and all 236 tests stayed green
# through it for the whole life of the project. The facade measured 5.8x
# brighter fully exhaled than fully inhaled and nothing in the suite objected.
#
# These assert DIRECTION, which is the one thing |max - min| throws away.


def _facade_total(frame, rows):
    return sum(sum(frame[r][c]) for r in rows for c in range(COLS))


def _breath_phase_mean(s, rows, lo, hi, cycle=300):
    """Mean facade brightness over the frames of one breath cycle whose level
    falls in [lo, hi]. `cycle` is 300 frames: 6 breaths a minute at 30fps.

    Averaged over the band rather than sampled at one frame because the
    heartbeat is still running underneath, and a single frame hands the answer
    to whatever cardiac phase it happens to land on.
    """
    vals = [
        _facade_total(s.render(t), rows)
        for t in range(cycle)
        if lo <= breath_level(t / cycle) <= hi
    ]
    assert vals, "no frames in that part of the breath"
    return sum(vals) / len(vals)


def test_the_lung_body_sits_below_its_surface():
    """breath_at's sign, on its own, where the inversion actually lived.

    Row 0 is the crown and row 16 is the base, so the filled body is the rows
    with the HIGHER index. Written the other way round the facade drains from
    the roof - and every contrast metric in this file reads identically."""
    mid = waterline_of(0.5)  # surface at row 8
    assert breath_at(12, mid) > breath_at(4, mid)
    assert breath_at(ROWS - 1, mid) > 4 * breath_at(0, mid)

    # Fully inhaled, every row is body; fully exhaled, every row but the one
    # the surface rests on is air. Measured 0.85 min against 0.43 max.
    inhaled = waterline_of(1.0)
    exhaled = waterline_of(0.0)
    assert min(breath_at(r, inhaled) for r in range(ROWS)) > max(
        breath_at(r, exhaled) for r in range(ROWS - 1)
    )


@pytest.mark.parametrize("lead", [0.0, 0.5, 1.0])
def test_the_facade_is_brighter_inhaled_than_exhaled(lead):
    """A lung is brightest when it is FULL. This is the test the project did
    not have.

    Inverted, the facade measured 59108 fully exhaled against 10471 fully
    inhaled - backwards by 5.6x - and the suite was green. Measured now, the
    right way round at every point of the crossover: 2.17x at lead=0.0, 3.06x
    at 0.5, 3.72x at 1.0. 1.5x is a floor, not a fit."""
    s = Sync(Expression(bpm=72, breath_rate=6.0, lead=lead, calm=0.5,
                        coherence=1.0, presence=1.0))
    exhaled = _breath_phase_mean(s, range(ROWS), 0.0, 0.1)
    inhaled = _breath_phase_mean(s, range(ROWS), 0.9, 1.0)
    assert inhaled > 1.5 * exhaled


def test_the_facade_is_brighter_inhaled_than_exhaled_above_the_tree_line():
    """The same claim for the fifteen rows the river audience actually has.
    The remap folds the excursion into fewer rows and must not smuggle the
    inversion back in on the way. Measured 3.57x."""
    s = Sync(Expression(bpm=72, breath_rate=6.0, lead=1.0, calm=0.5,
                        coherence=1.0, presence=1.0, tree_line=True))
    rows = range(VISIBLE_ROWS)
    assert _breath_phase_mean(s, rows, 0.9, 1.0) > 1.5 * _breath_phase_mean(
        s, rows, 0.0, 0.1
    )


def test_the_lit_region_grows_upward_from_the_base_on_an_inhale():
    """Not just brighter at the top of the breath - brighter by filling
    UPWARD out of the base, rather than downward from the crown.

    Measured with mirrored bands: rows 12-14 and rows 4-2 sit at identical
    chest distances, so the heartbeat underneath contributes equally to both
    and cancels in the difference. What is left is the breath alone, and the
    SIGN of that difference is the entire defect - at half fill it runs to
    +2666 filling from the base and -2664 filling from the crown.

    t=0 is full exhale and t=120 full inhale at 6 breaths a minute on 30fps.
    """
    base_band, crown_band = (12, 13, 14), (4, 3, 2)
    assert [chest_distance(r) for r in base_band] == [
        chest_distance(r) for r in crown_band
    ], "the bands must mirror, or the heartbeat does not cancel"

    s = Sync(Expression(bpm=72, breath_rate=6.0, lead=1.0, calm=0.5,
                        coherence=1.0, presence=1.0))

    def base_lead(t):
        f = s.render(t)

        def band(rows):
            return sum(sum(f[r][c]) for r in rows for c in range(COLS)) / len(rows)

        return band(base_band) - band(crown_band)

    half_full = min(range(121), key=lambda t: abs(breath_level(t / 300) - 0.5))
    assert half_full == 60

    # Through the whole first half of the fill the base stays ahead of the
    # crown. Measured minimum +181; the inverted lung has already crossed to
    # -98 by t=35, which is what makes this the check that catches it.
    early = [base_lead(t) for t in range(0, half_full + 1, 5)]
    assert all(g > 0 for g in early), early
    # At half fill it leads by a wide margin. Measured +2666 against -2664
    # inverted. 1000 is a floor, not a fit.
    assert early[-1] > 1000, early[-1]

    # By the top of the inhale the crown has caught up, so the fill reaches
    # the roof instead of stalling below it. Measured -195 against a peak of
    # +3293 mid-fill.
    assert base_lead(120) < 0.2 * max(early)

    # The surface itself sweeps the tower monotonically, base to crown. This
    # pins the waterline's travel and is deliberately kept as a SEPARATE
    # claim: on its own it cannot detect the inversion, because waterline_of
    # was never the broken half. The surface travels base-to-crown either way;
    # it was which side of it lit up that was backwards. A test built on this
    # measurement alone passes against the defect.
    def brightest_row(t):
        f = s.render(t)
        return max(range(ROWS), key=lambda r: sum(sum(px) for px in f[r]))

    track = [brightest_row(t) for t in range(0, 121, 5)]
    assert track[0] >= ROWS - 2, track  # starts at the base
    assert track[-1] <= 2, track  # ends at the crown
    assert all(later <= earlier for earlier, later in zip(track, track[1:])), track


def test_the_base_lights_before_the_crown_on_an_inhale():
    """The fill arrives from below, floor by floor.

    A row counts as lit when it first crosses the midpoint of its own range
    over the inhale - a per-row threshold, so this measures the order of
    arrival and not whatever absolute brightness the palette emits."""
    s = Sync(Expression(bpm=72, breath_rate=6.0, lead=1.0, calm=0.5,
                        coherence=1.0, presence=1.0))

    def lit_at(row):
        series = [sum(sum(px) for px in s.render(t)[row]) for t in range(121)]
        half = (min(series) + max(series)) / 2
        return next(t for t, v in enumerate(series) if v >= half)

    ladder = [lit_at(r) for r in (14, 11, 8, 5, 2)]  # base -> crown
    # Measured 25, 44, 57, 73, 90: every three floors up the tower lights
    # about half a second later. A 5-frame gap is a floor, not a fit.
    assert ladder == sorted(ladder), ladder
    assert all(later - earlier > 5 for earlier, later in zip(ladder, ladder[1:])), ladder
