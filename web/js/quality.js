// Photo quality gate: blur (variance of Laplacian) and exposure checks.
// Thresholds are deliberately loose; tune them with real shop photos.

export const QUALITY = {
  size: 256,
  blurSevere: 25,
  blurWarn: 60,
  darkSevere: 40,
  darkWarn: 70,
  brightWarn: 220,
};

/** @param {Uint8ClampedArray} rgba pixels of a QUALITY.size x QUALITY.size image */
export function assessQuality(rgba, size = QUALITY.size) {
  const n = size * size;
  const gray = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const g = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
    gray[i] = g;
    sum += g;
  }
  const brightness = sum / n;

  let lapSum = 0;
  let lapSq = 0;
  let count = 0;
  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      const i = y * size + x;
      const lap = gray[i - size] + gray[i + size] + gray[i - 1] + gray[i + 1] - 4 * gray[i];
      lapSum += lap;
      lapSq += lap * lap;
      count++;
    }
  }
  const mean = lapSum / count;
  const sharpness = lapSq / count - mean * mean;

  const issues = [];
  if (sharpness < QUALITY.blurSevere) {
    issues.push({ code: 'blur', severe: true, message: 'Photo is very blurry. Hold the camera steady and tap to focus.' });
  } else if (sharpness < QUALITY.blurWarn) {
    issues.push({ code: 'blur', severe: false, message: 'Photo is slightly blurry.' });
  }
  if (brightness < QUALITY.darkSevere) {
    issues.push({ code: 'dark', severe: true, message: 'Photo is too dark. Use more light or the flash.' });
  } else if (brightness < QUALITY.darkWarn) {
    issues.push({ code: 'dark', severe: false, message: 'Photo is quite dark.' });
  } else if (brightness > QUALITY.brightWarn) {
    issues.push({ code: 'bright', severe: false, message: 'Photo is over-exposed. Avoid direct sunlight or glare.' });
  }
  return { sharpness: Math.round(sharpness), brightness: Math.round(brightness), issues };
}
