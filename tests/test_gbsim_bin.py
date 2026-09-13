"""The .bin codec, against three real files from the simulator's own server.

This is the best-anchored thing in the repo: we are not asserting that our
decoder agrees with our encoder, we are asserting that both agree with bytes
somebody else wrote. The round-trip test is the load-bearing one - if
encode(*decode(f)) is not byte-identical to f, we have the format wrong and
every demo we hand the organisers would be subtly corrupt.

The anchors live in /tmp, which is volatile. If they are gone these tests FAIL
rather than skip, on purpose: a green suite that silently stopped checking the
only ground truth we have is worse than a red one.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from sync.gbsim_bin import (
    FRAME_BYTES,
    HEADER_BYTES,
    MAGIC,
    BinFormatError,
    decode,
    encode,
    export_sync_run,
    preview,
    read_header,
)
from sync.geometry import COLS, FPS, ROWS
from sync.producer import validate
from sync.render import Expression

REPO_ROOT = Path(__file__).resolve().parents[1]

# (path, fps, frames, size) exactly as published in /tmp/demos.json and as
# measured on disk. Both halves must agree with the file or the anchor is not
# an anchor.
ANCHORS = [
    ("/tmp/sundai-float.bin", 30, 120, 55084),
    ("/tmp/rainbow.bin", 30, 60, 27544),
    ("/tmp/sundai-critters.bin", 4, 34, 15610),
]


def test_the_anchor_files_are_present():
    missing = [path for path, *_ in ANCHORS if not Path(path).exists()]
    assert not missing, (
        f"anchor .bin files are gone: {missing}. These are the only ground "
        "truth for the format; re-download them from /demos/{slug}.bin before "
        "trusting this module."
    )


@pytest.mark.parametrize("path,fps,frames,size", ANCHORS)
def test_the_header_arithmetic_holds_on_disk(path, fps, frames, size):
    """4 + frames * 459, to the byte, on files we did not write."""
    blob = Path(path).read_bytes()
    assert len(blob) == size
    assert len(blob) == HEADER_BYTES + frames * FRAME_BYTES
    assert blob[0] == MAGIC
    assert read_header(blob) == (fps, frames)


@pytest.mark.parametrize("path,fps,frames,size", ANCHORS)
def test_decode_matches_the_published_metadata(path, fps, frames, size):
    got_fps, got_frames = decode(path)
    assert got_fps == fps
    assert len(got_frames) == frames
    for frame in got_frames:
        validate(frame)  # same legality bar as anything we put on the wire


@pytest.mark.parametrize("path,fps,frames,size", ANCHORS)
def test_round_trip_is_byte_identical(path, fps, frames, size):
    """The whole lane in one assertion. Not 'close enough' - identical.

    Both call shapes are pinned: the canonical encode(frames, fps), and the
    encode(*decode(blob)) one-liner, which arrives swapped because decode
    returns the header's own (fps, frames) order.
    """
    original = Path(path).read_bytes()
    got_fps, got_frames = decode(original)
    assert encode(got_frames, got_fps) == original
    assert encode(*decode(original)) == original


def test_rainbow_confirms_the_nine_pixel_row_stride():
    """Orientation, checked against content instead of assumed.

    rainbow's first frame is a diagonal hue ramp: each pixel is one step
    further along than the one above-right of it. That identity only holds if
    a row is 9 pixels wide, so it is independent evidence that their row-major
    17x9 is our row-major 17x9.
    """
    fps, frames = decode("/tmp/rainbow.bin")
    first = frames[0]
    assert fps == 30
    assert first[0][0] == (255, 0, 0)  # hue 0, the start of the ramp
    for r in range(ROWS - 1):
        for c in range(COLS - 1):
            assert first[r + 1][c] == first[r][c + 1]


def test_critters_confirm_that_row_zero_is_the_top_floor():
    """The other axis, and the one that would silently ship a mirrored demo.

    demos.json publishes sundai-critters as "drifting up the tower". Its lit
    rows walk from high row index toward low at one row per frame, so
    decreasing row index is upward and their row 0 is the top - the same way
    geometry.py maps it. Everything export_sync_run writes depends on this.
    """
    _, frames = decode("/tmp/sundai-critters.bin")

    def mean_lit_row(frame):
        lit = [r for r, row in enumerate(frame) for px in row if any(px)]
        return sum(lit) / len(lit)

    # Frames 0..5 are one clean ascent, before the critters wrap around the
    # top and re-enter at the bottom.
    means = [mean_lit_row(f) for f in frames[:6]]
    for a, b in zip(means, means[1:]):
        assert b - a == pytest.approx(-1.0, abs=0.01), "one row up per frame"


def test_decode_takes_bytes_as_well_as_a_path():
    blob = Path("/tmp/rainbow.bin").read_bytes()
    assert decode(blob) == decode("/tmp/rainbow.bin")


def test_decode_of_a_zero_frame_file():
    """The header alone is a legal, if pointless, file."""
    blob = bytes([MAGIC, 30, 0, 0])
    assert decode(blob) == (30, [])
    assert encode([], 30) == blob


# --- the ways a file can be wrong ---------------------------------------


def test_rejects_bad_magic():
    blob = bytearray(Path("/tmp/rainbow.bin").read_bytes())
    blob[0] = 0x44
    with pytest.raises(BinFormatError, match="bad magic"):
        decode(bytes(blob))


def test_rejects_a_header_that_is_not_even_a_header():
    with pytest.raises(BinFormatError, match="too short for a header"):
        decode(b"C\x1e")


def test_rejects_a_truncated_file():
    blob = Path("/tmp/rainbow.bin").read_bytes()[:-1]
    with pytest.raises(BinFormatError, match="truncated"):
        decode(blob)


def test_rejects_trailing_bytes():
    """A byte we cannot account for means we do not understand the format."""
    blob = Path("/tmp/rainbow.bin").read_bytes() + b"\x00"
    with pytest.raises(BinFormatError, match="trailing"):
        decode(blob)


def test_rejects_a_frame_count_that_disagrees_with_the_body():
    blob = bytearray(Path("/tmp/rainbow.bin").read_bytes())
    blob[2] = 61  # claim one more frame than the body carries
    with pytest.raises(BinFormatError, match="truncated"):
        decode(bytes(blob))


def test_encode_rejects_an_illegal_frame():
    fps, frames = decode("/tmp/sundai-critters.bin")
    bad = [list(map(list, f)) for f in frames]
    bad[2][3][4] = (300, 0, 0)
    with pytest.raises(BinFormatError, match="frame 2"):
        encode(bad, fps)


def test_encode_rejects_a_wrong_shaped_frame():
    with pytest.raises(BinFormatError, match="expected 17 rows"):
        encode([[[(0, 0, 0)] * COLS] * 5], 30)


@pytest.mark.parametrize("fps", [0, 256, -1, 30.0])
def test_encode_rejects_an_unwritable_fps(fps):
    """fps is one byte, and a 0 fps demo is an unplayable one."""
    blank = [tuple(tuple((0, 0, 0) for _ in range(COLS)) for _ in range(ROWS))]
    with pytest.raises(BinFormatError, match="fps must be"):
        encode(blank, fps)


# --- our own run, in their format ---------------------------------------


def test_export_sync_run_writes_a_file_their_decoder_would_accept(tmp_path):
    out = tmp_path / "sync.bin"
    blob = export_sync_run(
        Expression(bpm=72, breath_rate=8.0, lead=0.3, calm=0.4, presence=1.0),
        frames=40,
        fps=30,
        path=out,
    )
    assert out.read_bytes() == blob
    assert len(blob) == HEADER_BYTES + 40 * FRAME_BYTES
    assert blob[0] == MAGIC

    fps, frames = decode(out)
    assert (fps, len(frames)) == (30, 40)
    for frame in frames:
        validate(frame)
    # And it round-trips through their format the same way their demos do.
    assert encode(frames, fps) == blob


def test_export_sync_run_is_not_a_black_rectangle():
    """A demo file that decodes perfectly and shows nothing is a silent
    failure, and the one this lane could most easily ship."""
    blob = export_sync_run(
        Expression(bpm=72, breath_rate=8.0, lead=0.0, calm=0.3, presence=1.0),
        frames=30,
        fps=30,
    )
    _, frames = decode(blob)
    peak = max(max(px) for frame in frames for row in frame for px in row)
    assert peak > 40, "the facade should be visibly lit"


def test_export_samples_the_renderers_30fps_timeline():
    """A 15 fps export must be the same performance at half the frame rate,
    not the same motion played at half speed. Frame i at 15 fps is therefore
    frame 2i at 30 fps."""
    e = Expression(bpm=72, breath_rate=8.0, lead=0.3, calm=0.4, presence=1.0)
    _, slow = decode(export_sync_run(e, frames=10, fps=15))
    _, fast = decode(export_sync_run(e, frames=20, fps=FPS))
    for i in range(10):
        assert slow[i] == fast[2 * i]


def test_export_rejects_an_empty_run():
    with pytest.raises(BinFormatError, match="at least one frame"):
        export_sync_run(Expression(presence=1.0), frames=0, fps=30)


# --- the eyeball path ----------------------------------------------------


def test_preview_draws_one_line_per_floor():
    _, frames = decode("/tmp/rainbow.bin")
    art = preview(frames[0])
    lines = art.splitlines()
    assert len(lines) == ROWS
    assert art.count("\x1b[48;2;") == ROWS * COLS
    assert "fl 20" in lines[0] and "fl  4" in lines[-1]


def test_cli_prints_the_header_and_a_frame():
    """Run it the way a human would, so the entry point is covered too."""
    proc = subprocess.run(
        [sys.executable, "-m", "sync.gbsim_bin", "/tmp/sundai-critters.bin"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "4 fps, 34 frames" in proc.stdout
    assert "sundai-critters.bin" in proc.stdout
    assert "\x1b[48;2;" in proc.stdout


def test_cli_fails_loudly_on_a_file_that_is_not_a_demo(tmp_path):
    junk = tmp_path / "notademo.bin"
    junk.write_bytes(b"PK\x03\x04 not a demo")
    proc = subprocess.run(
        [sys.executable, "-m", "sync.gbsim_bin", str(junk)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "bad magic" in proc.stderr
