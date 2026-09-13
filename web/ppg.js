// Photoplethysmography from a phone camera.
//
// Finger over the lens with the torch on. Each heartbeat pushes blood through
// the fingertip and the amount of red light reaching the sensor dips. Average
// the red channel per frame and you have a pulse waveform.
//
// Mirrors sync/ppg.py exactly, which is where the algorithm is tested.
// Autocorrelation, not FFT: at 25 seconds of noisy, motion-corrupted signal it
// degrades gracefully instead of smearing across bins.

const BPM_MIN = 40;
const BPM_MAX = 180;
const HZ = 30;

export function resample(times, values, hz = HZ) {
  if (times.length < 4) return [];
  const t0 = times[0];
  const t1 = times[times.length - 1];
  const n = Math.floor((t1 - t0) * hz);
  if (n < 8) return [];
  const out = new Array(n);
  let j = 0;
  for (let i = 0; i < n; i++) {
    const t = t0 + i / hz;
    while (j + 2 < times.length && times[j + 1] < t) j++;
    const span = times[j + 1] - times[j];
    const frac = span <= 0 ? 0 : (t - times[j]) / span;
    out[i] = values[j] + (values[j + 1] - values[j]) * frac;
  }
  return out;
}

export function detrend(xs, window = 31) {
  const n = xs.length;
  if (!n) return [];
  const half = window >> 1;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - half);
    const hi = Math.min(n, i + half + 1);
    let sum = 0;
    for (let k = lo; k < hi; k++) sum += xs[k];
    out[i] = xs[i] - sum / (hi - lo);
  }
  return out;
}

// Biased estimator: divide by the full length, not by the overlap. Dividing
// by the overlap inflates long lags, where fewer sample pairs contribute, and
// so systematically pulls the winner toward half the true heart rate.
function autocorr(xs, lag) {
  const n = xs.length - lag;
  if (n <= 0) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) s += xs[i] * xs[i + lag];
  return s / xs.length;
}

function findPeaks(xs, minGap) {
  let peak = 0;
  for (const x of xs) peak = Math.max(peak, Math.abs(x));
  const thresh = 0.4 * peak;
  const out = [];
  for (let i = 1; i < xs.length - 1; i++) {
    if (xs[i] > thresh && xs[i] >= xs[i - 1] && xs[i] > xs[i + 1]) {
      if (!out.length || i - out[out.length - 1] >= minGap) out.push(i);
    }
  }
  return out;
}

function rmssdMs(peaks, hz) {
  if (peaks.length < 4) return null;
  const ibis = [];
  for (let i = 0; i < peaks.length - 1; i++) {
    ibis.push(((peaks[i + 1] - peaks[i]) * 1000) / hz);
  }
  const diffs = [];
  for (let i = 0; i < ibis.length - 1; i++) diffs.push(ibis[i + 1] - ibis[i]);
  if (!diffs.length) return null;
  const mean = diffs.reduce((a, d) => a + d * d, 0) / diffs.length;
  return Math.sqrt(mean);
}

export function estimate(times, values, hz = HZ) {
  const xs = detrend(resample(times, values, hz));
  const empty = { bpm: 0, confidence: 0, rmssdMs: null, nBeats: 0, usable: false };
  if (xs.length < hz * 4) return empty;

  const energy = autocorr(xs, 0);
  if (energy <= 1e-12) return empty;

  const lagMin = Math.max(1, Math.floor((hz * 60) / BPM_MAX));
  const lagMax = Math.min(xs.length - 2, Math.floor((hz * 60) / BPM_MIN));
  if (lagMax <= lagMin) return empty;

  const corrs = [];
  for (let lag = lagMin; lag <= lagMax; lag++) corrs.push([lag, autocorr(xs, lag) / energy]);

  let globalBest = -Infinity;
  for (const [, c] of corrs) globalBest = Math.max(globalBest, c);
  if (globalBest <= 0) return empty;

  // Octave guard. For a clean periodic signal the correlation at TWICE the
  // true period is just as high as at the period itself, so a plain argmax
  // reports half the real heart rate roughly as often as not. Take instead
  // the SHORTEST local peak still scoring within 85% of the best: that one is
  // the fundamental, and the equally tall candidates past it are its
  // harmonics. A genuinely slow pulse is unaffected, because its half-lag
  // sits in antiphase and scores near -1.
  const threshold = 0.85 * globalBest;
  let bestIdx = -1;
  for (let i = 1; i < corrs.length - 1; i++) {
    if (corrs[i][1] >= corrs[i - 1][1] && corrs[i][1] >= corrs[i + 1][1] && corrs[i][1] >= threshold) {
      bestIdx = i;
      break;
    }
  }
  if (bestIdx < 0) {
    bestIdx = 0;
    for (let i = 1; i < corrs.length; i++) if (corrs[i][1] > corrs[bestIdx][1]) bestIdx = i;
  }
  let bestLag = corrs[bestIdx][0];
  const best = corrs[bestIdx][1];

  // Parabolic refinement for sub-sample accuracy.
  if (bestIdx > 0 && bestIdx < corrs.length - 1) {
    const y0 = corrs[bestIdx - 1][1];
    const y1 = corrs[bestIdx][1];
    const y2 = corrs[bestIdx + 1][1];
    const denom = y0 - 2 * y1 + y2;
    if (Math.abs(denom) > 1e-12) bestLag += (0.5 * (y0 - y2)) / denom;
  }

  const bpm = (60 * hz) / bestLag;

  const others = corrs.filter(([lag]) => Math.abs(lag - bestLag) > lagMin * 0.4);
  const baseline = others.length
    ? others.reduce((a, [, c]) => a + Math.abs(c), 0) / others.length
    : 0;
  const confidence = best > 0 ? Math.max(0, Math.min(1, (best - baseline) / 0.5)) : 0;

  const beats = findPeaks(xs, Math.floor(((hz * 60) / bpm) * 0.6));
  return {
    bpm,
    confidence,
    rmssdMs: rmssdMs(beats, hz),
    nBeats: beats.length,
    usable: confidence >= 0.35 && bpm >= BPM_MIN && bpm <= BPM_MAX,
  };
}

// --- capture -------------------------------------------------------------

export async function openCamera(videoEl) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: "environment" },
      width: { ideal: 320 },
      height: { ideal: 240 },
      frameRate: { ideal: 30 },
    },
    audio: false,
  });
  videoEl.srcObject = stream;
  await videoEl.play();

  const track = stream.getVideoTracks()[0];
  let torch = false;
  try {
    await track.applyConstraints({ advanced: [{ torch: true }] });
    torch = true;
  } catch {
    // No torch on this device or the browser refuses. Ambient light still
    // works if the finger is lit from behind; we surface this in the UI.
  }
  return { stream, track, torch };
}

export function closeCamera(stream) {
  if (!stream) return;
  for (const t of stream.getTracks()) {
    try {
      t.stop();
    } catch {
      /* already gone */
    }
  }
}

export function capture(videoEl, canvasEl, seconds, onProgress) {
  return new Promise((resolve) => {
    const ctx = canvasEl.getContext("2d", { willReadFrequently: true });
    const W = (canvasEl.width = 48);
    const H = (canvasEl.height = 48);
    const times = [];
    const values = [];
    let start = null;
    let stopped = false;

    function frame(ts) {
      if (stopped) return;
      if (start === null) start = ts;
      const t = (ts - start) / 1000;

      ctx.drawImage(videoEl, 0, 0, W, H);
      // Centre crop: the edges of the frame catch stray light around the finger.
      const d = ctx.getImageData(W / 4, H / 4, W / 2, H / 2).data;
      let red = 0;
      for (let i = 0; i < d.length; i += 4) red += d[i];
      const mean = red / (d.length / 4);

      times.push(t);
      values.push(mean);
      if (onProgress) onProgress(Math.min(1, t / seconds), mean, t);

      if (t >= seconds) {
        stopped = true;
        resolve({ times, values });
        return;
      }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  });
}
