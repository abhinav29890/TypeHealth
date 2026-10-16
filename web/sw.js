// Offline support. Strategies:
//  - app shell (html/css/js/icons): stale-while-revalidate
//  - models/<task>.json: network-first (so retrained models are noticed)
//  - models/*.onnx?v=<sha>: cache-first; older versions of the same file are evicted
//  - onnxruntime-web from the CDN: cache-first (version-pinned URL)
const CACHE = 'tyrecheck-v3';
const SHELL = [
  './',
  'index.html',
  'style.css',
  'manifest.json',
  'icons/icon.svg',
  'icons/icon-192.png',
  'icons/icon-512.png',
  'js/app.js',
  'js/inference.js',
  'js/preprocess.js',
  'js/quality.js',
  'js/scoring.js',
  'js/cam.js',
  'js/history.js',
];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) {
      if (key !== CACHE) await caches.delete(key);
    }
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  if (url.hostname === 'cdn.jsdelivr.net') {
    event.respondWith(cacheFirst(req));
  } else if (url.origin === self.location.origin) {
    if (url.pathname.includes('/models/') && url.pathname.endsWith('.json')) event.respondWith(networkFirst(req));
    else if (url.pathname.endsWith('.onnx')) event.respondWith(modelCacheFirst(req, url));
    else event.respondWith(staleWhileRevalidate(req, event));
  }
});

async function cacheFirst(req) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(req);
  if (hit) return hit;
  const res = await fetch(req);
  if (res.ok) cache.put(req, res.clone());
  return res;
}

async function modelCacheFirst(req, url) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(req);
  if (hit) return hit;
  const res = await fetch(req);
  if (res.ok) {
    for (const old of await cache.keys()) {
      const oldUrl = new URL(old.url);
      if (oldUrl.pathname === url.pathname && oldUrl.search !== url.search) await cache.delete(old);
    }
    await cache.put(req, res.clone());
  }
  return res;
}

async function networkFirst(req) {
  const cache = await caches.open(CACHE);
  try {
    const res = await fetch(req, { cache: 'no-cache' });
    if (res.ok) cache.put(req, res.clone());
    return res;
  } catch {
    return (await cache.match(req)) ?? Response.error();
  }
}

async function staleWhileRevalidate(req, event) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(req, { ignoreSearch: true });
  const update = fetch(req)
    .then((res) => {
      if (res.ok) cache.put(req, res.clone());
      return res;
    })
    .catch(() => null);
  if (hit) {
    event.waitUntil(update);
    return hit;
  }
  return (await update) ?? Response.error();
}
