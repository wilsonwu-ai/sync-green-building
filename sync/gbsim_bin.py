"""The Green Building simulator's own .bin demo format, read and written.

Why this module exists: the simulator's live API is behind an event password
we do not have. Its `/demos/{slug}.bin` files are not. A .bin is the one
artefact we can hand the organisers that drops straight into their player
without a credential, a deploy, or anyone trusting our code - so SYNC has to
be able to *write* their format, not just speak their wire protocol.

The format, derived from three real files rather than from documentation:

    byte 0      magic 0x43, ASCII 'C'
    byte 1      fps, uint8
    bytes 2-3   frame count, uint16 LITTLE-endian
    then        frame_count x 459 bytes, raw RGB, row-major, 17 rows x 9 cols

459 = 17 * 9 * 3, the same 153 windows every other module here talks about.
The header is 4 bytes, so a well-formed file is exactly 4 + frames * 459 and
nothing else. That arithmetic is checked against all three real files in
tests/test_gbsim_bin.py:

    sundai-float    30 fps  120 frames  55084 bytes
    rainbow         30 fps   60 frames  27544 bytes
    sundai-critters  4 fps   34 frames  15610 bytes

Row order is theirs, not ours, and both axes were confirmed from content
rather than assumed - a demo we export upside down would decode perfectly and
still be wrong on the building.

  WIDTH   In rainbow's first frame every pixel is one step further along the
          hue ramp than the pixel above-right of it. That diagonal is only
          coherent when read with a 9-pixel row stride.
  UP      sundai-critters is published as "drifting up the tower", and its
          lit rows walk from high row index to low at one row per frame. So
          decreasing row index is upward, and their row 0 is the top floor -
          which is exactly geometry.py's mapping, carried over unchanged.

Strict in what we write, tolerant in what we read - except where tolerance
would hide the fact that we have the format wrong. A length that disagrees
with the header by even one byte is an error here, because the round trip
being byte-identical is the only evidence we have that we understood the
format at all.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

from .geometry import COLS, FPS, ROWS, floor_of
from .producer import validate
from .render import Expression, Rows, Sync

__all__ = [
    "BinFormatError",
    "decode",
    "encode",
    "export_sync_run",
    "preview",
    "read_header",
]

MAGIC = 0x43  # 'C'
HEADER_BYTES = 4
FRAME_BYTES = ROWS * COLS * 3  # 459
MAX_FPS = 0xFF  # fps is a single byte
MAX_FRAMES = 0xFFFF  # frame count is a uint16


class BinFormatError(ValueError):
    """A .bin that is not a .bin, or not one we can account for byte by byte.

    Subclasses ValueError so it lands in the same handlers as `validate()`'s
    complaints; a malformed frame and a malformed file are the same class of
    problem to a caller.
    """


def _as_bytes(src) -> bytes:
    """Accept a path or an already-loaded blob. Callers have both: the CLI has
    a filename, the tests have bytes they just built."""
    if isinstance(src, (bytes, bytearray, memoryview)):
        return bytes(src)
    return Path(src).read_bytes()


def read_header(src) -> tuple[int, int]:
    """(fps, frame_count), without paying to decode 153 pixels a frame."""
    blob = _as_bytes(src)
    if len(blob) < HEADER_BYTES:
        raise BinFormatError(
            f"too short for a header: {len(blob)} bytes, need at least {HEADER_BYTES}"
        )
    if blob[0] != MAGIC:
        raise BinFormatError(
            f"bad magic 0x{blob[0]:02x}, expected 0x{MAGIC:02x} ('C'): "
            "this is not a Green Building demo file"
        )
    fps = blob[1]
    (count,) = struct.unpack_from("<H", blob, 2)  # little-endian, verified on disk
    return fps, count


def decode(src) -> tuple[int, list[Rows]]:
    """A .bin -> (fps, [frame, ...]).

    Each frame comes back in exactly the shape `Sync.render()` produces and
    `producer.validate()` accepts: 17 tuples of 9 (r, g, b) tuples. That is
    deliberate - a decoded demo and a rendered frame must be interchangeable
    everywhere downstream, or this module is just a file reader.
    """
    blob = _as_bytes(src)
    fps, count = read_header(blob)

    expected = HEADER_BYTES + count * FRAME_BYTES
    if len(blob) < expected:
        raise BinFormatError(
            f"truncated: header claims {count} frames ({expected} bytes), "
            f"file is {len(blob)}"
        )
    if len(blob) > expected:
        # Not pedantry. Trailing bytes mean either the header lied or our
        # frame size is wrong, and both of those are us misreading the format.
        raise BinFormatError(
            f"{len(blob) - expected} trailing bytes after {count} frames "
            f"(expected exactly {expected})"
        )

    frames: list[Rows] = []
    for i in range(count):
        base = HEADER_BYTES + i * FRAME_BYTES
        frames.append(
            tuple(
                tuple(
                    tuple(blob[base + (r * COLS + c) * 3 + k] for k in range(3))
                    for c in range(COLS)
                )
                for r in range(ROWS)
            )
        )
    return fps, frames


def encode(frames, fps: int) -> bytes:
    """The exact inverse of `decode`. encode(*decode(blob)) == blob.

    Every frame goes through `producer.validate()` first, so there is one
    definition of a legal frame in this repo rather than two that can drift.

    The argument order is tolerated both ways, which needs justifying. decode
    returns (fps, frames) because that is the order the header itself is in,
    and reading a file should look like the file. But that makes the natural
    round-trip expression `encode(*decode(blob))` arrive with the arguments
    reversed, so a caller writing the single most obvious line gets a
    TypeError. The two are distinguishable with certainty - fps is an int and
    a frame list never is - so we accept either rather than leave a trap in
    the one call everybody will make. Anything genuinely ambiguous still
    raises below.
    """
    if isinstance(frames, int) and not isinstance(fps, int):
        frames, fps = fps, frames
    frames = list(frames)
    if not isinstance(fps, int) or not 1 <= fps <= MAX_FPS:
        # fps 0 would encode cleanly and then be unplayable on their side;
        # refusing to write one is the whole point of checking.
        raise BinFormatError(f"fps must be an int in 1..{MAX_FPS}, got {fps!r}")
    if len(frames) > MAX_FRAMES:
        raise BinFormatError(
            f"{len(frames)} frames exceeds the header's uint16 ceiling ({MAX_FRAMES})"
        )

    out = bytearray(struct.pack("<BBH", MAGIC, fps, len(frames)))
    for i, frame in enumerate(frames):
        try:
            validate(frame)
        except ValueError as exc:
            raise BinFormatError(f"frame {i}: {exc}") from exc
        for row in frame:
            for px in row:
                out.extend(bytes(px))
    return bytes(out)


def export_sync_run(
    expression: Expression,
    frames: int,
    fps: int = FPS,
    path=None,
) -> bytes:
    """Render SYNC through `sync.render.Sync` and write it as one of their
    demos. This is the deliverable that needs no event password.

    The fps argument is a real sampling decision, not a header field. The
    renderer derives its cardiac and breath periods from `geometry.FPS` (30),
    so `render(t)` is indexed in 30ths of a second whatever we write in the
    header. Exporting at 12 fps by calling render(0..n) would hand the
    organisers a heartbeat playing at 40% speed. So we sample the renderer's
    30 fps timeline at the export rate instead, and a 12 fps export is the
    same performance with fewer frames of it.
    """
    if frames < 1:
        raise BinFormatError(f"a demo needs at least one frame, got {frames}")
    if not isinstance(fps, int) or not 1 <= fps <= MAX_FPS:
        raise BinFormatError(f"fps must be an int in 1..{MAX_FPS}, got {fps!r}")

    sync = Sync(expression)
    step = FPS / fps
    rendered = [sync.render(round(i * step)) for i in range(frames)]

    blob = encode(rendered, fps)
    if path is not None:
        Path(path).write_bytes(blob)
    return blob


def preview(frame: Rows, label: bool = True) -> str:
    """One frame as 24-bit ANSI blocks, so their demos can be eyeballed in a
    terminal instead of guessed at.

    Two character cells per window, because a single cell is roughly half as
    wide as it is tall and the facade would come out looking squashed - which
    matters when the thing you are checking by eye IS the geometry.
    """
    lines = []
    for r, row in enumerate(frame):
        cells = "".join(f"\x1b[48;2;{px[0]};{px[1]};{px[2]}m  " for px in row)
        prefix = f"  fl {floor_of(r):>2}  " if label else "  "
        lines.append(f"{prefix}{cells}\x1b[0m")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python3 -m sync.gbsim_bin",
        description="Inspect a Green Building .bin demo: header, and one frame in ANSI.",
    )
    p.add_argument("path", help="a .bin demo file, e.g. /tmp/rainbow.bin")
    p.add_argument("-f", "--frame", type=int, default=0, help="frame to preview (default 0)")
    args = p.parse_args(argv)

    try:
        fps, frames = decode(args.path)
    except (BinFormatError, OSError) as exc:
        print(f"  cannot read {args.path}: {exc}", file=sys.stderr)
        return 2

    seconds = len(frames) / fps if fps else 0.0
    print()
    print(f"  {Path(args.path).name}")
    print(f"  {fps} fps, {len(frames)} frames, {seconds:.2f}s, {ROWS}x{COLS}")

    if not frames:
        print("  (no frames)")
        print()
        return 0

    i = args.frame
    if not 0 <= i < len(frames):
        print(f"  no frame {i}: file has 0..{len(frames) - 1}", file=sys.stderr)
        return 2

    print(f"  frame {i}")
    print()
    print(preview(frames[i]))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
