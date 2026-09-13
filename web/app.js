import { openCamera, closeCamera, capture, estimate } from "./ppg.js";

const CAPTURE_S = 25;
const RING = 2 * Math.PI * 52;

const $ = (id) => document.getElementById(id);
const stages = {
  intro: $("stage-intro"),
  measure: $("stage-measure"),
  result: $("stage-result"),
  live: $("stage-live"),
};

function show(name) {
  for (const [k, el] of Object.entries(stages)) el.hidden = k !== name;
}

// A stable per-person id so before/after can be paired without any account.
let person = null;
try {
  person = localStorage.getItem("sync.person");
  if (!person) {
    person = crypto.randomUUID();
    localStorage.setItem("sync.person", person);
  }
} catch {
  person = crypto.randomUUID(); // private window; pairing just won't persist
}

let pending = null; // the estimate awaiting send
let hasBaseline = false;
try {
  hasBaseline = localStorage.getItem("sync.baseline") === "1";
} catch {
  /* ignore */
}

// --- trace drawing -------------------------------------------------------

const traceCanvas = $("trace");
const tctx = traceCanvas.getContext("2d");
const recent = [];

function drawTrace() {
  const w = traceCanvas.width;
  const h = traceCanvas.height;
  tctx.clearRect(0, 0, w, h);
  if (recent.length < 3) return;

  let lo = Infinity;
  let hi = -Infinity;
  for (const v of recent) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  const span = hi - lo || 1;

  tctx.beginPath();
  recent.forEach((v, i) => {
    const x = (i / (recent.length - 1)) * w;
    const y = h - ((v - lo) / span) * (h - 16) - 8;
    i ? tctx.lineTo(x, y) : tctx.moveTo(x, y);
  });
  tctx.strokeStyle = "#ff4d63";
  tctx.lineWidth = 2.5;
  tctx.lineJoin = "round";
  tctx.shadowColor = "rgba(255,77,99,.6)";
  tctx.shadowBlur = 10;
  tctx.stroke();
}

// --- measurement ---------------------------------------------------------

async function measure() {
  show("measure");
  recent.length = 0;
  $("ring-fg").style.strokeDashoffset = RING;
  $("countdown").textContent = CAPTURE_S;
  $("measure-hint").textContent = "Hold still. Keep the lens fully covered.";
  $("measure-hint").classList.remove("bad");

  let cam;
  try {
    cam = await openCamera($("cam"));
  } catch (err) {
    show("intro");
    alert(
      "Camera access was refused. SYNC needs the rear camera to read your pulse.\n\n" +
        err.message
    );
    return;
  }
  if (!cam.torch) {
    $("measure-hint").textContent =
      "No flash available. Hold your finger against the lens in bright light.";
    $("measure-hint").classList.add("bad");
  }

  const { times, values } = await capture(
    $("cam"),
    $("work"),
    CAPTURE_S,
    (frac, mean, t) => {
      $("ring-fg").style.strokeDashoffset = String(RING * (1 - frac));
      $("countdown").textContent = Math.max(0, Math.ceil(CAPTURE_S - t));
      recent.push(mean);
      if (recent.length > 260) recent.shift();
      drawTrace();
    }
  );

  closeCamera(cam.stream);

  const est = estimate(times, values);
  pending = est;
  show("result");
  $("bpm").textContent = est.usable ? Math.round(est.bpm) : "--";

  if (est.usable) {
    const pct = Math.round(est.confidence * 100);
    $("quality").textContent = `Signal quality ${pct}%. ${est.nBeats} beats seen.`;
    $("quality").classList.remove("bad");
    $("send").disabled = false;
  } else {
    $("quality").textContent =
      "Could not find a clean pulse. Press a little more gently and keep the lens fully covered.";
    $("quality").classList.add("bad");
    $("send").disabled = true;
  }
}

// --- sending -------------------------------------------------------------

async function send() {
  if (!pending || !pending.usable) return;
  const tag = hasBaseline ? "after" : "baseline";
  $("send").disabled = true;

  try {
    const res = await fetch("/api/reading", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        person,
        bpm: pending.bpm,
        confidence: pending.confidence,
        rmssd_ms: pending.rmssdMs,
        tag,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    if (!hasBaseline) {
      hasBaseline = true;
      try {
        localStorage.setItem("sync.baseline", "1");
      } catch {
        /* ignore */
      }
    }
    renderDelta(data.you);
    show("live");
  } catch (err) {
    $("send").disabled = false;
    alert(`Could not reach the building.\n\n${err.message}`);
  }
}

function renderDelta(you) {
  const el = $("delta");
  if (!you) {
    el.hidden = true;
    return;
  }
  const drop = you.drop;
  const better = drop > 0;
  el.hidden = false;
  el.innerHTML = `
    <div class="big">${better ? "-" : "+"}${Math.abs(drop).toFixed(1)}</div>
    <div class="cap">BPM ${better ? "slower" : "faster"} than when you started
      &nbsp;·&nbsp; ${you.before} &rarr; ${you.after}</div>`;
}

// --- live state ----------------------------------------------------------

let pacerRaf = null;

function startPacer(ratePerMin) {
  if (pacerRaf) cancelAnimationFrame(pacerRaf);
  const period = (60 / ratePerMin) * 1000;
  const t0 = performance.now();

  function step(now) {
    const q = ((now - t0) % period) / period;
    // Same 40/60 inhale/exhale split the facade uses, so the phone in your
    // hand and the tower in front of you are the same breath.
    const x = q < 0.4 ? q / 0.4 : 1 - (q - 0.4) / 0.6;
    const level = 0.5 - 0.5 * Math.cos(Math.PI * Math.max(0, Math.min(1, x)));
    $("pacer-fill").style.width = `${level * 100}%`;
    $("pacer-word").textContent = q < 0.4 ? "in" : "out";
    pacerRaf = requestAnimationFrame(step);
  }
  pacerRaf = requestAnimationFrame(step);
}

async function poll() {
  try {
    const res = await fetch("/api/state");
    const s = await res.json();
    if (s.headline) $("headline").textContent = s.headline;
    if (s.narration) $("narration").textContent = s.narration;
    $("live-bpm").textContent = s.bpm ? Math.round(s.bpm) : "--";
    $("live-n").textContent = s.participants ?? 0;
    $("live-rate").textContent = s.breath_rate ? s.breath_rate.toFixed(1) : "--";

    const leading = s.lead > 0.15;
    $("pacer-wrap").hidden = !leading;
    if (leading && !pacerRaf) startPacer(s.breath_rate);
    if (!leading && pacerRaf) {
      cancelAnimationFrame(pacerRaf);
      pacerRaf = null;
    }
  } catch {
    // Polling is best-effort; the next tick will pick it back up.
  }
}

setInterval(poll, 2000);

$("begin").addEventListener("click", measure);
$("retry").addEventListener("click", measure);
$("remeasure").addEventListener("click", measure);
$("send").addEventListener("click", send);
