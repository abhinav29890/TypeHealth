// Class activation map -> translucent heatmap drawn over the analysed photo.

/**
 * @param {HTMLCanvasElement} canvas overlay canvas (same box as the photo)
 * @param {Float32Array} cam model output, shape [1, K, h, w]
 * @param {number[]} dims [1, K, h, w]
 * @param {number} k class index to visualise
 */
export function drawCam(canvas, cam, dims, k) {
  const [, , h, w] = dims;
  const offset = k * h * w;
  let max = 0;
  for (let i = 0; i < h * w; i++) max = Math.max(max, cam[offset + i]);

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (max <= 0) return false;

  const small = document.createElement('canvas');
  small.width = w;
  small.height = h;
  const sctx = small.getContext('2d');
  const img = sctx.createImageData(w, h);
  for (let i = 0; i < h * w; i++) {
    const v = Math.max(0, cam[offset + i]) / max;
    const [r, g, b] = heatColor(v);
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = Math.round(255 * Math.max(0, v - 0.25) / 0.75 * 0.7);
  }
  sctx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(small, 0, 0, canvas.width, canvas.height);
  return true;
}

export function clearCam(canvas) {
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

// yellow -> orange -> red
function heatColor(v) {
  return [255, Math.round(230 * (1 - v)), Math.round(60 * (1 - v))];
}
