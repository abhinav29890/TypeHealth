// Past checks, stored only in this browser (localStorage). Best-effort: storage can be
// unavailable (private mode, quota), in which case history silently stays empty.

const KEY = 'tyrecheck.history.v1';
const MAX_ENTRIES = 30;

export function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(KEY)) ?? [];
  } catch {
    return [];
  }
}

export function saveCheck(entry) {
  const list = [entry, ...loadHistory()].slice(0, MAX_ENTRIES);
  try {
    localStorage.setItem(KEY, JSON.stringify(list));
  } catch {
    // quota exceeded or storage blocked: keep going without history
  }
  return list;
}

export function clearHistory() {
  try {
    localStorage.removeItem(KEY);
  } catch {
    // ignore
  }
}
