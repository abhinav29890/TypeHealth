// Turns raw model probabilities into a tyre health score, status and advice.
// Pure functions (no DOM) so they can be unit-tested with `node --test`.

export const TREAD_VALUE = { good: 1, moderate: 0.5, worn: 0 };

export const THRESHOLDS = {
  notTyre: 0.5,      // P(not_tyre) at or above this -> ask for a retake
  defect: 0.5,       // P(defective) at or above this caps the score at DEFECT_CAP
  worn: 0.5,         // P(worn) at or above this caps the score at WORN_CAP
  moderate: 0.5,     // P(moderate) at or above this caps the score at MODERATE_CAP (amber)
  uncertain: 0.6,    // top-1 probability below this is flagged as uncertain
  ok: 70,            // score >= ok -> green
  monitor: 45,       // score >= monitor -> amber, else red
};
const DEFECT_CAP = 40;
const WORN_CAP = 35;
const MODERATE_CAP = 65;
const CONDITION_WEIGHT = 0.6;
// A model marked beta (test macro-F1 below the export threshold) counts for less.
const CONDITION_WEIGHT_TREAD_BETA = 0.85;

export function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export function toMap({ labels, probs }) {
  return Object.fromEntries(labels.map((l, i) => [l, probs[i]]));
}

export function top1({ labels, probs }) {
  let best = 0;
  probs.forEach((p, i) => { if (p > probs[best]) best = i; });
  return { label: labels[best], prob: probs[best], index: best };
}

const CONDITION_TEXT = { good: 'No visible defects', defective: 'Possible defect (crack / cut / bulge)' };
const TREAD_TEXT = { good: 'Good tread depth', moderate: 'Partially worn tread', worn: 'Worn tread' };

/**
 * @param {{labels:string[], probs:number[]}} condition  output of the condition model
 * @param {{labels:string[], probs:number[]}|null} tread  output of the tread model
 * @param {{issues:{code:string, message:string, severe:boolean}[]}} [quality]
 * @param {{treadBeta?: boolean}} [options]  treadBeta: the tread model is low-accuracy (from tread.json)
 */
export function computeHealth(condition, tread, quality = { issues: [] }, { treadBeta = false } = {}) {
  const c = toMap(condition);
  const notTyre = c.not_tyre ?? 0;
  if (notTyre >= THRESHOLDS.notTyre) {
    return {
      valid: false,
      reason: 'not_tyre',
      message: "This doesn't look like a tyre. Retake the photo with the tyre filling most of the frame.",
    };
  }

  const tyreMass = Math.max(1e-6, 1 - notTyre);
  const pGood = (c.good ?? 0) / tyreMass;
  const pDefect = (c.defective ?? 0) / tyreMass;
  const condTop = pDefect > pGood ? 'defective' : 'good';
  const findings = [{
    aspect: 'Condition',
    key: condTop,
    text: CONDITION_TEXT[condTop],
    confidence: Math.max(pGood, pDefect),
    severity: condTop === 'good' ? 'ok' : 'bad',
  }];

  let treadScore = null;
  let pWorn = 0;
  let pModerate = 0;
  if (tread) {
    const t = toMap(tread);
    let num = 0;
    let den = 0;
    for (const [label, p] of Object.entries(t)) {
      if (label in TREAD_VALUE) { num += p * TREAD_VALUE[label]; den += p; }
    }
    treadScore = den > 0 ? num / den : null;
    pWorn = t.worn ?? 0;
    pModerate = t.moderate ?? 0;
    const tt = top1(tread);
    findings.push({
      aspect: 'Tread wear',
      key: tt.label,
      text: TREAD_TEXT[tt.label] ?? tt.label,
      confidence: tt.prob,
      severity: { good: 'ok', moderate: 'warn', worn: 'bad' }[tt.label] ?? 'warn',
      beta: treadBeta,
    });
  }

  const wc = treadBeta ? CONDITION_WEIGHT_TREAD_BETA : CONDITION_WEIGHT;
  let score = treadScore === null
    ? 100 * pGood
    : 100 * (wc * pGood + (1 - wc) * treadScore);
  if (pDefect >= THRESHOLDS.defect) score = Math.min(score, DEFECT_CAP);
  if (pWorn >= THRESHOLDS.worn) score = Math.min(score, WORN_CAP);
  if (pModerate >= THRESHOLDS.moderate) score = Math.min(score, MODERATE_CAP);
  score = Math.round(score);

  const status = score >= THRESHOLDS.ok ? 'ok' : score >= THRESHOLDS.monitor ? 'monitor' : 'replace';
  const uncertain = findings.some((f) => f.confidence < THRESHOLDS.uncertain);
  const lowQuality = quality.issues.length > 0;

  return {
    valid: true,
    score,
    status,
    headline: { ok: 'Good to go', monitor: 'Monitor / inspect soon', replace: 'Repair or replace' }[status],
    findings,
    uncertain,
    lowQuality,
    recommendations: recommend(findings, status, uncertain, lowQuality, treadBeta && tread !== null),
  };
}

function recommend(findings, status, uncertain, lowQuality, treadBeta) {
  const recs = [];
  const has = (aspect, key) => findings.some((f) => f.aspect === aspect && f.key === key);

  if (has('Condition', 'defective')) {
    recs.push('Inspect the highlighted area closely for cracks, cuts, bulges or exposed cords.');
    recs.push('A sidewall bulge or visible cords means the tyre is unsafe: replace it immediately.');
  }
  if (has('Tread wear', 'worn')) {
    recs.push('Measure the tread with a depth gauge. Replace if below the legal minimum (1.6 mm in many countries) and ideally below 3 mm.');
  } else if (has('Tread wear', 'moderate')) {
    recs.push('Plan for replacement soon and re-check tread depth with a gauge at the next service.');
    recs.push('Check pressure and wheel alignment if wear looks uneven across the tread.');
  }
  if (treadBeta) {
    recs.push('The tread-wear estimate is in beta and can miss worn tyres. Always confirm tread depth with a gauge.');
  }
  if (status === 'ok') {
    recs.push('No action needed. Check pressure monthly and rotate tyres every 8,000–10,000 km.');
  }
  if (uncertain) {
    recs.push('The model is not confident. Take another photo from a different angle to confirm.');
  }
  if (lowQuality) {
    recs.push('Photo quality is low, so this result may be unreliable. Retake in good light, holding the camera steady.');
  }
  return recs;
}
