import { loadModels, classify } from './inference.js';
import { squareCanvas, toTensorData } from './preprocess.js';
import { assessQuality, QUALITY } from './quality.js';
import { computeHealth } from './scoring.js';
import { drawCam, clearCam } from './cam.js';
import { loadHistory, saveCheck, clearHistory } from './history.js';

const $ = (id) => document.getElementById(id);
const el = {
  status: $('model-status'),
  demo: $('demo-banner'),
  error: $('error-banner'),
  dropzone: $('dropzone'),
  camera: $('camera-input'),
  file: $('file-input'),
  result: $('result'),
  photo: $('photo'),
  overlay: $('overlay'),
  busy: $('busy'),
  toggle: $('view-toggle'),
  quality: $('quality'),
  report: $('report'),
  history: $('history'),
};

let modelsPromise;
let last = null; // { outputs, models } of the most recent analysis, for the heatmap toggle

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function showError(msg) {
  el.error.textContent = msg;
  el.error.hidden = false;
}

async function init() {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').catch(() => {});
  }
  renderHistory(loadHistory());

  modelsPromise = loadModels((msg) => { el.status.textContent = msg; });
  try {
    const models = await modelsPromise;
    el.status.textContent = 'Ready · on-device';
    el.status.className = 'pill pill-ready';
    el.demo.hidden = !Object.values(models).some((m) => m.meta.placeholder);
  } catch (err) {
    console.error(err);
    el.status.textContent = 'Model error';
    el.status.className = 'pill pill-error';
    showError(`Could not load the AI models: ${err.message}. Check your connection and reload.`);
  }
}

async function analyse(file) {
  if (!file || !file.type.startsWith('image/')) {
    showError('Please choose an image file.');
    return;
  }
  el.error.hidden = true;
  el.result.hidden = false;
  el.busy.hidden = false;
  el.report.innerHTML = '';
  el.quality.hidden = true;
  setView('none');
  el.result.scrollIntoView({ behavior: 'smooth', block: 'start' });

  let bitmap;
  try {
    bitmap = await createImageBitmap(file);
    const pctx = el.photo.getContext('2d');
    pctx.drawImage(squareCanvas(bitmap, el.photo.width), 0, 0);
    clearCam(el.overlay);

    const qCanvas = squareCanvas(bitmap, QUALITY.size);
    const quality = assessQuality(qCanvas.getContext('2d').getImageData(0, 0, QUALITY.size, QUALITY.size).data);

    const models = await modelsPromise;
    const outputs = {};
    for (const [task, model] of Object.entries(models)) {
      const { input_size: size, mean, std } = model.meta;
      const rgba = squareCanvas(bitmap, size).getContext('2d').getImageData(0, 0, size, size).data;
      outputs[task] = await classify(model, toTensorData(rgba, size, mean, std));
    }

    const health = computeHealth(outputs.condition, outputs.tread ?? null, quality,
      { treadBeta: Boolean(models.tread?.meta.beta) });
    last = { outputs, models, health };
    renderQuality(quality);
    renderReport(health);
    updateToggle(health);

    if (health.valid) {
      renderHistory(saveCheck({
        id: Date.now(),
        date: new Date().toISOString(),
        thumb: squareCanvas(bitmap, 96).toDataURL('image/jpeg', 0.7),
        score: health.score,
        status: health.status,
        headline: health.headline,
        ...jobDetails(),
      }));
    }
  } catch (err) {
    console.error(err);
    showError(`Analysis failed: ${err.message}`);
  } finally {
    bitmap?.close();
    el.busy.hidden = true;
  }
}

function jobDetails() {
  return {
    vehicle: $('vehicle').value.trim(),
    position: $('position').value,
    notes: $('notes').value.trim(),
  };
}

function renderQuality(quality) {
  if (!quality.issues.length) {
    el.quality.hidden = true;
    return;
  }
  const severe = quality.issues.some((i) => i.severe);
  el.quality.className = `quality${severe ? ' severe' : ''}`;
  el.quality.innerHTML = `<strong>${severe ? 'Retake recommended' : 'Photo quality'}</strong>
    <ul>${quality.issues.map((i) => `<li>${escapeHtml(i.message)}</li>`).join('')}</ul>`;
  el.quality.hidden = false;
}

function gauge(score, status) {
  const r = 46;
  const circ = 2 * Math.PI * r;
  const color = { ok: 'var(--ok)', monitor: 'var(--warn)', replace: 'var(--bad)' }[status];
  return `<svg class="gauge" viewBox="0 0 112 112" role="img" aria-label="Health score ${score} out of 100">
    <circle class="track" cx="56" cy="56" r="${r}" fill="none" stroke-width="10"/>
    <circle class="value" cx="56" cy="56" r="${r}" fill="none" stroke="${color}" stroke-width="10"
      stroke-linecap="round" stroke-dasharray="${circ}" stroke-dashoffset="${circ * (1 - score / 100)}"
      transform="rotate(-90 56 56)"/>
    <text x="56" y="56" text-anchor="middle" dominant-baseline="central" font-size="30">${score}</text>
  </svg>`;
}

function renderReport(health) {
  const job = jobDetails();
  const jobLine = [job.vehicle, job.position, job.notes].filter(Boolean).map(escapeHtml).join(' · ');
  const when = new Date().toLocaleString();

  if (!health.valid) {
    el.report.innerHTML = `<p class="headline status-replace">Not a tyre?</p>
      <p>${escapeHtml(health.message)}</p>`;
    return;
  }

  const findings = health.findings.map((f) => `
    <div class="finding">
      <div class="finding-top">
        <div>
          <div class="finding-aspect">${escapeHtml(f.aspect)}${f.beta
            ? ' <span class="badge" title="Low accuracy. Confirm with a tread-depth gauge.">Beta</span>' : ''}</div>
          <div class="finding-text">${escapeHtml(f.text)}</div>
        </div>
        <span class="muted small">${Math.round(f.confidence * 100)}% confident</span>
      </div>
      <div class="bar"><span class="sev-${f.severity}" style="width:${Math.round(f.confidence * 100)}%"></span></div>
    </div>`).join('');

  el.report.innerHTML = `
    <div class="score-head">
      ${gauge(health.score, health.status)}
      <div>
        <div class="muted small">Tyre health score</div>
        <p class="headline status-${health.status}">${health.headline}</p>
        ${jobLine ? `<div class="small">${jobLine}</div>` : ''}
        <div class="muted small">${escapeHtml(when)}</div>
      </div>
    </div>
    ${findings}
    <h2 style="margin-top:16px">Recommendations</h2>
    <ul class="recs">${health.recommendations.map((r) => `<li>${escapeHtml(r)}</li>`).join('')}</ul>`;
}

// ---- heatmap toggle -------------------------------------------------------

function camClassIndex(task) {
  const out = last?.outputs[task];
  if (!out) return -1;
  const preferred = task === 'condition' ? 'defective' : 'worn';
  const i = out.labels.indexOf(preferred);
  return i >= 0 ? i : out.probs.indexOf(Math.max(...out.probs));
}

function updateToggle(health) {
  for (const btn of el.toggle.querySelectorAll('button')) {
    const view = btn.dataset.view;
    btn.disabled = view !== 'none' && (!health.valid || camClassIndex(view) < 0);
  }
}

function setView(view) {
  for (const btn of el.toggle.querySelectorAll('button')) {
    btn.setAttribute('aria-checked', String(btn.dataset.view === view));
  }
  if (view === 'none' || !last) {
    clearCam(el.overlay);
    return;
  }
  const { cam } = last.outputs[view];
  drawCam(el.overlay, cam.data, cam.dims, camClassIndex(view));
}

el.toggle.addEventListener('click', (e) => {
  const btn = e.target.closest('button');
  if (btn && !btn.disabled) setView(btn.dataset.view);
});

// ---- history --------------------------------------------------------------

function renderHistory(list) {
  if (!list.length) {
    el.history.innerHTML = '<li class="muted small">No checks yet.</li>';
    return;
  }
  el.history.innerHTML = list.map((h) => {
    const title = [h.vehicle, h.position].filter(Boolean).join(' · ') || h.headline;
    return `<li>
      <img src="${h.thumb}" alt="">
      <div class="h-main">
        <div class="h-title">${escapeHtml(title)}</div>
        <div class="muted small">${escapeHtml(new Date(h.date).toLocaleString())}${h.notes ? ` · ${escapeHtml(h.notes)}` : ''}</div>
      </div>
      <span class="h-score status-${h.status}">${h.score}</span>
    </li>`;
  }).join('');
}

$('clear-history').addEventListener('click', () => {
  clearHistory();
  renderHistory([]);
});

// ---- inputs ---------------------------------------------------------------

for (const input of [el.camera, el.file]) {
  input.addEventListener('change', () => {
    analyse(input.files[0]);
    input.value = '';
  });
}

el.dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  el.dropzone.classList.add('dragover');
});
el.dropzone.addEventListener('dragleave', () => el.dropzone.classList.remove('dragover'));
el.dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  el.dropzone.classList.remove('dragover');
  analyse(e.dataTransfer.files[0]);
});

$('print-btn').addEventListener('click', () => window.print());
$('new-btn').addEventListener('click', () => {
  el.result.hidden = true;
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

init();
