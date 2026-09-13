// Photoplethysmography from a phone camera.
//
// Finger over a lit lens. Each heartbeat pushes blood through the fingertip
// and the amount of light reaching the sensor dips. Average one colour channel
// per frame and you have a pulse waveform.
//
// Which lens, which lamp and which channel depend on the phone - see the
// capture section at the bottom. The estimator below does not care: it gets a
// single scalar per frame either way, and is identical in both modes.
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

// Three ways to light a fingertip. The phone decides which one it can do, and
// the caller has to say so in the UI, because the finger goes somewhere
// different in each.
//
//   torch    Rear camera, flash on, fingertip over both. Best signal by a
//            distance, and the RED channel carries it: red is the wavelength
//            that makes it through a fingertip at all.
//   screen   Front camera, page driven to white, the display itself as the
//            lamp. iOS Safari implements no torch constraint, so this is the
//            path on every iPhone - and at a night demo the old ambient
//            fallback was close to unusable. Screen light is white and weak,
//            so the GREEN channel wins: haemoglobin absorbs green hard and
//            the sensor is most sensitive there.
//   ambient  No torch, no front camera. Rear lens in whatever room light there
//            happens to be. Geometrically this is torch mode with a worse lamp
//            - the light still reaches the sensor THROUGH the fingertip - so it
//            reads RED, for the same reason torch does. A fingertip passes red
//            and swallows green, and with no lamp of our own the few green
//            photons that survive are mostly sensor noise. Last resort only.
export const MODE_TORCH = "torch";
export const MODE_SCREEN = "screen";
export const MODE_AMBIENT = "ambient";

// Byte offset of the channel we average, within each RGBA quad.
const CHANNEL_RED = 0;
const CHANNEL_GREEN = 1;

function videoConstraints(facing) {
  return {
    video: {
      facingMode: { ideal: facing },
      width: { ideal: 320 },
      height: { ideal: 240 },
      frameRate: { ideal: 30 },
    },
    audio: false,
  };
}

// Feature-detect the torch, never sniff the user agent. Chrome on Android
// advertises it here; Safari implements getCapabilities but has never listed
// it. Returns true/false when the device is explicit and null when it says
// nothing at all - some older Android builds honour the constraint without
// advertising it, so silence is still worth one attempt.
function torchCapability(track) {
  let caps = null;
  try {
    caps = track.getCapabilities ? track.getCapabilities() : null;
  } catch {
    caps = null; // implemented but throwing; same as not knowing
  }
  if (caps && "torch" in caps) return !!caps.torch;
  return null;
}

async function attach(videoEl, stream) {
  videoEl.srcObject = stream;
  await videoEl.play();
}

export async function openCamera(videoEl) {
  // Preferred path: rear camera, flash on, finger over both.
  const rear = await navigator.mediaDevices.getUserMedia(videoConstraints("environment"));
  let track = rear.getVideoTracks()[0];
  await attach(videoEl, rear);

  let torch = false;
  if (torchCapability(track) !== false) {
    try {
      await track.applyConstraints({ advanced: [{ torch: true }] });
      torch = true;
    } catch {
      // A throw means no torch. There is nothing else to read from it.
    }
  }
  if (torch) {
    return { stream: rear, track, torch: true, mode: MODE_TORCH, channel: CHANNEL_RED };
  }

  // No flash: switch to the front camera and let the caller turn the page
  // white, so the screen is the lamp. Phones will not reliably hand out two
  // camera streams at once, so the rear one has to be released first - which
  // means that if the front camera then refuses we have to reopen the rear.
  closeCamera(rear);
  try {
    const front = await navigator.mediaDevices.getUserMedia(videoConstraints("user"));
    track = front.getVideoTracks()[0];
    await attach(videoEl, front);
    return { stream: front, track, torch: false, mode: MODE_SCREEN, channel: CHANNEL_GREEN };
  } catch {
    const again = await navigator.mediaDevices.getUserMedia(videoConstraints("environment"));
    track = again.getVideoTracks()[0];
    await attach(videoEl, again);
    return { stream: again, track, torch: false, mode: MODE_AMBIENT, channel: CHANNEL_RED };
  }
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

export function capture(videoEl, canvasEl, seconds, onProgress, channel = CHANNEL_RED) {
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
      // Red whenever the light crosses the fingertip to reach the sensor
      // (torch, ambient), green when the screen is the lamp. Only the offset
      // into the RGBA quad changes; the maths downstream is identical.
      let sum = 0;
      for (let i = channel; i < d.length; i += 4) sum += d[i];
      const mean = sum / (d.length / 4);

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
