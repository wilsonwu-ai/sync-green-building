// The facade, drawn the way it reads from across the river: 153 soft glows on
// a dark tower, not 153 crisp squares. The bloom is the point. A hard-edged
// grid looks like a spreadsheet; the real windows scatter light through glass.

const ROWS = 17;
const COLS = 9;

const canvas = document.getElementById("facade");
const ctx = canvas.getContext("2d");
const $ = (id) => document.getElementById(id);

const CELL = 52;
const GAP = 8;
const PAD = 26;
canvas.width = COLS * CELL + (COLS - 1) * GAP + PAD * 2;
canvas.height = ROWS * CELL + (ROWS - 1) * GAP + PAD * 2;

function draw(bytes) {
  ctx.fillStyle = "#080d16";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Tower shell, so the grid reads as a building rather than a matrix.
  ctx.fillStyle = "#0d1524";
  ctx.strokeStyle = "#18233a";
  ctx.lineWidth = 2;
  const r = 10;
  ctx.beginPath();
  ctx.roundRect(PAD * 0.4, PAD * 0.4, canvas.width - PAD * 0.8, canvas.height - PAD * 0.8, r);
  ctx.fill();
  ctx.stroke();

  ctx.globalCompositeOperation = "lighter";
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLS; col++) {
      const i = (row * COLS + col) * 3;
      const red = bytes[i];
      const green = bytes[i + 1];
      const blue = bytes[i + 2];
      const x = PAD + col * (CELL + GAP);
      const y = PAD + row * (CELL + GAP);
      const lum = (red + green + blue) / 765;

      if (lum > 0.02) {
        const g = ctx.createRadialGradient(
          x + CELL / 2, y + CELL / 2, CELL * 0.16,
          x + CELL / 2, y + CELL / 2, CELL * 1.5
        );
        g.addColorStop(0, `rgba(${red},${green},${blue},${0.5 * lum})`);
        g.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = g;
        ctx.fillRect(x - CELL, y - CELL, CELL * 3, CELL * 3);
      }

      ctx.fillStyle = `rgb(${red},${green},${blue})`;
      ctx.fillRect(x, y, CELL, CELL);
    }
  }
  ctx.globalCompositeOperation = "source-over";
}

function paintStatus(s) {
  if (!s) return;
  if (s.headline) $("headline").textContent = s.headline;
  if (s.narration) $("narration").textContent = s.narration;
  $("m-bpm").textContent = s.bpm ? Math.round(s.bpm) : "--";
  $("m-n").textContent = s.participants ?? 0;
  $("m-rate").textContent = s.breath_rate ? s.breath_rate.toFixed(1) : "--";
  $("m-coh").textContent = s.coherence != null ? s.coherence.toFixed(2) : "--";
  $("f-display").textContent = s.display || "--";
  $("f-appraisal").textContent = s.appraisal_error
    ? `${s.appraisal_source} (${s.appraisal_error})`
    : s.appraisal_source || "--";

  for (const el of document.querySelectorAll(".ph")) {
    el.classList.toggle("on", el.dataset.p === s.phase);
  }

  const d = s.delta;
  if (d && d.n > 0 && d.drop != null) {
    $("delta").hidden = false;
    $("d-drop").textContent = `${d.drop > 0 ? "-" : "+"}${Math.abs(d.drop).toFixed(1)}`;
    $("d-cap").textContent =
      `BPM across ${d.n} ${d.n === 1 ? "person" : "people"} · ${d.before} → ${d.after}`;
  } else {
    $("delta").hidden = true;
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/preview`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (!msg.px) return;
    const bin = atob(msg.px);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    draw(bytes);
    paintStatus(msg.status);
  };
  ws.onclose = () => {
    $("headline").textContent = "Reconnecting";
    setTimeout(connect, 1200);
  };
}

connect();
