"""The facade as an anatomy, not a screen.

The MIT Green Building (Building 54) has 153 individually lit windows.
The display contract is a Frame of 17 rows x 9 columns of RGB.

Row 0 is the TOP lit floor (floor 20); row 16 is the bottom (floor 4).
Column c is bay c+1, counted left to right as the viewer sees it.
That mapping is PROVISIONAL in the upstream SPEC (§10.4) until it is
confirmed on the building; every consumer here goes through this module
so a single edit re-maps the whole installation.

Nine wide and seventeen tall is a torso. We name it like one.
"""

ROWS = 17
COLS = 9
N_WINDOWS = ROWS * COLS  # 153

STORIES = 21
TOP_FLOOR = 20

FPS = 30  # the display contract's ceiling; never send faster

# --- anatomy -------------------------------------------------------------
# Row bands, [start, end) in display-row space.
HEAD = (0, 3)
THROAT = (3, 6)
CHEST = (6, 11)
BELLY = (11, 14)
LEGS = (14, 17)

CHEST_ROW = 8  # where the heart sits, and where every pulse originates
MAX_CHEST_DIST = max(CHEST_ROW, ROWS - 1 - CHEST_ROW)  # 8


def floor_of(row: int) -> int:
    """Display row -> physical floor number."""
    if not 0 <= row < ROWS:
        raise IndexError(row)
    return TOP_FLOOR - row


def bay_of(col: int) -> int:
    """Display column -> physical bay, left to right as seen by the viewer."""
    if not 0 <= col < COLS:
        raise IndexError(col)
    return col + 1


def chest_distance(row: int) -> float:
    """0.0 at the heart, 1.0 at the furthest floor from it."""
    return abs(row - CHEST_ROW) / MAX_CHEST_DIST
