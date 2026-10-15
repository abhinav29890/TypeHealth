// Image -> model input. Mirrors training/src/dataset.py:eval_tensor exactly:
// centre square crop -> resize to size x size -> /255 -> ImageNet normalise -> NCHW float32.

export function centerSquare(width, height) {
  const side = Math.min(width, height);
  return { sx: Math.floor((width - side) / 2), sy: Math.floor((height - side) / 2), side };
}

/** Draw the centre square of `source` into a size x size canvas (what the model "sees"). */
export function squareCanvas(source, size) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  const { sx, sy, side } = centerSquare(source.width, source.height);
  ctx.drawImage(source, sx, sy, side, side, 0, 0, size, size);
  return canvas;
}

/** RGBA pixels -> normalised NCHW Float32Array. */
export function toTensorData(rgba, size, mean, std) {
  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      out[c * plane + i] = (rgba[i * 4 + c] / 255 - mean[c]) / std[c];
    }
  }
  return out;
}
