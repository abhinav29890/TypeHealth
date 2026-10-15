// Run with: node --test tests/scoring.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeHealth, softmax } from '../web/js/scoring.js';
import { toTensorData, centerSquare } from '../web/js/preprocess.js';
import { assessQuality } from '../web/js/quality.js';

const cond = (good, defective, notTyre) => ({ labels: ['good', 'defective', 'not_tyre'], probs: [good, defective, notTyre] });
const tread = (good, moderate, worn) => ({ labels: ['good', 'moderate', 'worn'], probs: [good, moderate, worn] });

test('healthy tyre scores green', () => {
  const h = computeHealth(cond(0.95, 0.04, 0.01), tread(0.9, 0.08, 0.02));
  assert.equal(h.valid, true);
  assert.equal(h.status, 'ok');
  assert.ok(h.score >= 90);
  assert.equal(h.uncertain, false);
});

test('likely defect caps score in the red zone', () => {
  const h = computeHealth(cond(0.2, 0.78, 0.02), tread(0.9, 0.1, 0));
  assert.equal(h.status, 'replace');
  assert.ok(h.score <= 40);
  assert.ok(h.recommendations.some((r) => r.includes('bulge')));
});

test('worn tread caps score even with a clean condition result', () => {
  const h = computeHealth(cond(0.97, 0.02, 0.01), tread(0.05, 0.15, 0.8));
  assert.equal(h.status, 'replace');
  assert.ok(h.score <= 35);
});

test('moderate wear gives amber', () => {
  const h = computeHealth(cond(0.9, 0.08, 0.02), tread(0.1, 0.85, 0.05));
  assert.equal(h.status, 'monitor');
});

test('not a tyre is rejected', () => {
  const h = computeHealth(cond(0.1, 0.1, 0.8), tread(0.3, 0.3, 0.4));
  assert.equal(h.valid, false);
  assert.equal(h.reason, 'not_tyre');
});

test('works without a tread model and with a 2-class tread model', () => {
  assert.equal(computeHealth(cond(0.9, 0.1, 0), null).findings.length, 1);
  const h = computeHealth(cond(0.9, 0.1, 0), { labels: ['good', 'worn'], probs: [0.9, 0.1] });
  assert.equal(h.findings.length, 2);
  assert.equal(h.status, 'ok');
});

test('low confidence and bad photo quality are flagged', () => {
  const h = computeHealth(cond(0.5, 0.45, 0.05), tread(0.4, 0.35, 0.25), { issues: [{ code: 'blur', severe: true, message: '' }] });
  assert.equal(h.uncertain, true);
  assert.equal(h.lowQuality, true);
});

test('softmax sums to one and is stable for large logits', () => {
  const p = softmax([1000, 1001, 999]);
  assert.ok(Math.abs(p.reduce((a, b) => a + b) - 1) < 1e-9);
  assert.ok(p[1] > p[0] && p[0] > p[2]);
});

test('preprocessing matches the Python formula', () => {
  assert.deepEqual(centerSquare(400, 300), { sx: 50, sy: 0, side: 300 });
  const rgba = new Uint8ClampedArray([255, 0, 128, 255, 0, 255, 64, 255, 10, 20, 30, 255, 200, 100, 50, 255]);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const t = toTensorData(rgba, 2, mean, std);
  assert.equal(t.length, 12);
  // channel-first layout: t[c*4 + pixel]
  assert.ok(Math.abs(t[0] - (1 - 0.485) / 0.229) < 1e-6);
  assert.ok(Math.abs(t[4 + 1] - (1 - 0.456) / 0.224) < 1e-6);
  assert.ok(Math.abs(t[8 + 3] - (50 / 255 - 0.406) / 0.225) < 1e-6);
});

test('quality check flags flat dark images and passes sharp ones', () => {
  const size = 64;
  const dark = new Uint8ClampedArray(size * size * 4).fill(10);
  const q1 = assessQuality(dark, size);
  assert.ok(q1.issues.some((i) => i.code === 'blur' && i.severe));
  assert.ok(q1.issues.some((i) => i.code === 'dark' && i.severe));

  const sharp = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    const v = ((i % size) + Math.floor(i / size)) % 2 ? 230 : 30;
    sharp.set([v, v, v, 255], i * 4);
  }
  assert.equal(assessQuality(sharp, size).issues.length, 0);
});
