// Loads the ONNX models described by models/<task>.json and runs them in the browser
// with ONNX Runtime Web (WebAssembly). Nothing leaves the device.
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs';
import { softmax } from './scoring.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
// Multi-threading needs cross-origin isolation, which static hosts like GitHub Pages don't provide.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

export const TASKS = ['condition', 'tread'];

/** @returns {Promise<Record<string, {meta: object, session: ort.InferenceSession}>>} */
export async function loadModels(onProgress = () => {}) {
  const models = {};
  for (const task of TASKS) {
    const res = await fetch(`models/${task}.json`, { cache: 'no-cache' });
    if (!res.ok) continue; // the tread model is optional
    const meta = await res.json();
    onProgress(`Loading ${task} model…`);
    const modelRes = await fetch(`models/${meta.file}?v=${meta.sha256 ?? ''}`);
    if (!modelRes.ok) throw new Error(`models/${meta.file} not found`);
    models[task] = {
      meta,
      session: await ort.InferenceSession.create(await modelRes.arrayBuffer(),
        { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }),
    };
  }
  if (!models.condition) throw new Error('models/condition.json not found');
  return models;
}

/** Run one model on preprocessed NCHW data. */
export async function classify(model, tensorData) {
  const { input_size: size, labels } = model.meta;
  const input = new ort.Tensor('float32', tensorData, [1, 3, size, size]);
  const out = await model.session.run({ input });
  return {
    labels,
    probs: softmax(Array.from(out.logits.data)),
    cam: { data: out.cam.data, dims: out.cam.dims },
  };
}
