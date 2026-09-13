# SYNC

**The MIT Green Building takes the crowd's pulse and gives it back slower.**

Built for [Sundai Hack 140 — *Beyond Tetris: Building-Scale Physical AI*](https://www.sundai.club/events/boston/beyond-tetris-building-scale-physical-ai-for-mit-green-building), 13 September 2026. Targets the live install on the Green Building facade, 29 September.

---

## The idea in one paragraph

You open a web page and hold a fingertip over your phone's camera for 25 seconds. That's a real photoplethysmogram — each heartbeat dims the red light reaching the sensor — and it gives your heart rate with no hardware at all. The facade then beats *your* pulse across 21 storeys. Once enough people are on it, the building stops mirroring the crowd and starts **leading** it: it slides its breathing down to roughly six breaths a minute, the resonance band where paced breathing reliably raises HRV and drops sympathetic arousal. Seventeen floors is an extraordinary breathing pacer. Two minutes later everyone measures again, and we put the delta on the screen.

The claim is falsifiable, which is the point: **a building that measurably slows down the people watching it.**

## Why the facade is a torso, not a screen

The display contract is 17 rows × 9 columns of RGB — 153 cells, one per lit window, at 30 FPS. Nine wide and seventeen tall, seen from across the Charles at night. Text dies at that resolution. Faces die. Video dies. What survives is **colour, vertical motion, and rhythm**, which happens to be exactly the vocabulary a body has.

So the grid is addressed as an anatomy (`sync/geometry.py`): head at the crown, chest at row 8, legs at the base. Two layers composite:

| Layer | What it is | What it means |
|---|---|---|
| **Heart** | A lub-dub originating at the chest floors and propagating out toward head and feet, damped as it travels. Driven by the crowd's real measured BPM. | The building is *wearing* a body. |
| **Breath** | A lung. Light fills from the base to the crown on the inhale, empties on the exhale, at whatever rate the policy is pacing. 40/60 inhale/exhale split — the long exhale is the half that does the calming. | The building is *leading* a body. |

As `lead` runs 0 → 1 the weights cross over: the borrowed heartbeat recedes and the breath takes the facade. **That crossover is the piece.**

## Where the AI is actually load-bearing

Being precise about this, because it's the first thing worth asking. The control loop is **not** AI — pacing a breath at six per minute is a sine wave and we won't dress it up. The model (`sync/appraisal.py`, Claude Opus 5) does three things nothing else here can:

- **Appraisal** — decides whether this crowd *can* be led at all. A scattered, half-arrived crowd paced at six breaths a minute just ignores you, and a building pushing a rhythm nobody follows looks broken. `lead_ready` is a hard veto over the state machine.
- **Expression** — chooses how the facade should *feel*, as a palette bias folded into the renderer's calm axis. Not what to draw; how to be.
- **Voice** — one line, first person, from the building. Shown beside the facade and on every phone.

It runs off the render path on a 20-second cadence. **With no credentials, no network, or a failed call it drops to a deterministic rule-based appraisal and keeps going** — a demo that dies because an API call timed out is a demo that dies on stage.

## Honest limits

- **Heart rate from a phone camera is solid. HRV is not.** `rmssd_ms` is computed and carried as a *coarse* calm proxy. It is never shown to anyone as a clinical number, and the phone UI says so.
- The row→floor and column→bay mapping is **provisional** in the upstream SPEC (§10.4) until confirmed on the building. Every consumer goes through `sync/geometry.py`, so it's a one-line re-map.
- The wire protocol allows **one producer at a time**. The server aggregates; only the server talks to the display.

## Run it

```bash
pip install -r requirements.txt
python -m sync                      # preview only, port 8080
```

It prints your LAN address — phones cannot reach `localhost`, so put that one on the QR code.

- `http://<lan-ip>:8080/` — the phone page
- `http://<lan-ip>:8080/preview` — the facade, for the projector

Camera access requires a secure context. On a laptop `localhost` counts; for phones on the LAN, front it with a tunnel (`cloudflared tunnel --url http://localhost:8080`) or any HTTPS terminator.

### Driving a real display

We don't yet know how the simulator is delivered on the day, so SYNC speaks every shape the upstream repo documents, chosen by URL scheme:

```bash
python -m sync --display ws://HOST:9000/tetris-17x9   # protocol v1 over WebSocket
python -m sync --display tcp://HOST:9000              # protocol v1, newline-delimited
python -m sync --display py://utilities.dummy:DummyDisplay   # legacy Display subclass
python -m sync --display none                         # preview only
python -m sync --no-appraisal                         # zero model calls
```

Frames carry the SPEC §9.4 `digest` even though a producer may omit it — a mismatch is the fastest possible way to learn our row order is upside down.

### Tests

```bash
python -m pytest -q
```

The suite anchors the things that can silently lie: BPM recovery against synthetic signals with known answers (noise, baseline drift, irregular frame timing), every frame being a valid 17×9 wire message across the full input range, the heartbeat propagating rather than flashing, the lead crossover actually crossing, and the appraisal veto genuinely blocking the state machine.

## Layout

```
sync/geometry.py    the facade as an anatomy; the one place the mapping lives
sync/ppg.py         reference pulse estimator (mirrors web/ppg.js, and is what's tested)
sync/crowd.py       many phones -> one body; median, coherence, before/after delta
sync/policy.py      idle -> follow -> entrain -> lead -> reveal
sync/render.py      the frame producer: heart layer + breath layer, composited
sync/appraisal.py   the limbic layer (Claude), with a deterministic fallback
sync/producer.py    ws:// | tcp:// | py:// | none
sync/server.py      one process: phone page, pulse intake, 30 FPS render loop
web/                phone capture UI + the facade preview screen
```

## Credits

Starting point: [Nevin-Thinagar/17x9-Tetris](https://github.com/Nevin-Thinagar/17x9-Tetris) (the display contract) and [aygp-dr/17x9-Tetris](https://github.com/aygp-dr/17x9-Tetris) (SPEC + remote protocol v1). The 2026 Green Building Tetris installation is the work of the MIT team who put 153 LED modules in 153 windows; SYNC only borrows their display.
