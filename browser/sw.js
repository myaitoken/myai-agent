// MyAI Browser Agent — Service Worker
// Strategy:
//   • App shell (HTML, manifest, icons): cache-first, background revalidate
//   • Google Fonts CSS + CDN JS bundles: cache-first (long TTL)
//   • API calls (api.myaitoken.io, coordinator endpoints): network-only — never cache live job data
//   • HuggingFace model weights (large binary blobs): let the browser's own cache handle them
//   • Offline: serve cached shell with an offline overlay

const CACHE_VERSION = 'myai-agent-v2';
const API_ORIGINS = [
  'api.myaitoken.io',
  'api.infinihash.com',
  'huggingface.co',
  'cdn-lfs.hf.co',
  'cdn-lfs-us-1.hf.co',
];

// App shell — fetch + cache on install
const SHELL_URLS = [
  '/',
  '/manifest.json',
  '/icons/icon.svg',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
  '/icons/apple-touch-icon.png',
];

// CDN bundles to cache on first use
const CDN_ORIGINS = [
  'cdn.jsdelivr.net',
  'fonts.googleapis.com',
  'fonts.gstatic.com',
];

// ── Install: pre-cache the shell ─────────────────────────────────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then(cache => {
      // Non-fatal — if any shell resource fails, still install
      return Promise.allSettled(SHELL_URLS.map(url => cache.add(url)));
    }).then(() => self.skipWaiting())
  );
});

// ── Activate: purge old caches ───────────────────────────────────────────────
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys.filter(k => k !== CACHE_VERSION).map(k => caches.delete(k))
      ))
      .then(() => self.clients.claim())
  );
});

// ── Fetch routing ────────────────────────────────────────────────────────────
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // 1. Non-GET requests: always network (POST job submissions, heartbeats, etc.)
  if (request.method !== 'GET') return;

  // 2. API / model weight origins: network-only
  //    These are live data — caching would serve stale jobs or wrong balances
  if (API_ORIGINS.some(o => url.hostname.includes(o))) return;

  // 3. CDN bundles (Transformers.js, Coinbase SDK, Google Fonts): cache-first
  if (CDN_ORIGINS.some(o => url.hostname.includes(o))) {
    event.respondWith(cacheFirst(request));
    return;
  }

  // 4a. HTML navigations: network-first so a new deploy reaches returning
  //     visitors on their very next load (cache is offline fallback only).
  //     Avoids the stale-shell problem where users were one deploy behind.
  const isHtmlNav = request.mode === 'navigate' ||
    (request.headers.get('accept') || '').includes('text/html');
  if (url.origin === self.location.origin && isHtmlNav) {
    event.respondWith(networkFirst(request));
    return;
  }

  // 4b. Other same-origin assets (manifest, icons, css): stale-while-revalidate
  if (url.origin === self.location.origin) {
    event.respondWith(staleWhileRevalidate(request));
    return;
  }

  // 5. Everything else: network with cache fallback
  event.respondWith(networkWithFallback(request));
});

// ── Strategies ───────────────────────────────────────────────────────────────

// Cache-first: serve from cache; if miss, fetch and cache
async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_VERSION);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    return offlineResponse(request);
  }
}

// Stale-while-revalidate: serve from cache immediately, update in background
async function staleWhileRevalidate(request) {
  const cache = await caches.open(CACHE_VERSION);
  const cached = await cache.match(request);
  const fetchPromise = fetch(request).then(response => {
    if (response.ok) cache.put(request, response.clone());
    return response;
  }).catch(() => cached || offlineResponse(request));

  return cached || fetchPromise;
}

// Network-first: always try the network (freshest deploy); fall back to cache
// (then offline shell) only when the network is unavailable.
async function networkFirst(request) {
  const cache = await caches.open(CACHE_VERSION);
  try {
    const response = await fetch(request);
    if (response.ok) cache.put(request, response.clone());
    return response;
  } catch {
    const cached = await cache.match(request);
    return cached || offlineResponse(request);
  }
}

// Network with cache fallback
async function networkWithFallback(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_VERSION);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await caches.match(request);
    return cached || offlineResponse(request);
  }
}

// Offline response: for navigation requests serve the cached shell with an offline notice
function offlineResponse(request) {
  const url = new URL(request.url);
  const isNavigation = request.mode === 'navigate' ||
    (request.method === 'GET' && request.headers.get('accept')?.includes('text/html'));

  if (isNavigation) {
    // Try to serve cached shell first
    return caches.match('/').then(cached => cached || offlinePage());
  }

  return new Response('', { status: 503, statusText: 'Offline' });
}

function offlinePage() {
  return new Response(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>MyAI Agent — Offline</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #020617; color: #e2e8f0;
           display: flex; align-items: center; justify-content: center;
           min-height: 100vh; margin: 0; text-align: center; padding: 2rem; }
    .card { max-width: 400px; }
    .logo { width: 64px; height: 64px; border-radius: 16px;
            background: linear-gradient(135deg, #8b5cf6, #22d3ee);
            display: inline-grid; place-items: center;
            font-size: 28px; font-weight: 800; color: white; margin-bottom: 1.5rem; }
    h1 { font-size: 1.5rem; font-weight: 700; margin: 0 0 0.75rem; }
    p  { color: #94a3b8; line-height: 1.6; }
    button { margin-top: 1.5rem; padding: 0.75rem 1.5rem;
             background: linear-gradient(135deg, #8b5cf6, #22d3ee);
             border: none; border-radius: 8px; color: white;
             font-weight: 600; cursor: pointer; font-size: 1rem; }
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">M</div>
    <h1>You're offline</h1>
    <p>The MyAI Browser Agent needs an internet connection to poll for jobs and run inference.</p>
    <button onclick="location.reload()">Try again</button>
  </div>
</body>
</html>`, {
    status: 200,
    headers: { 'Content-Type': 'text/html; charset=utf-8' }
  });
}

// ── Background sync: re-register heartbeat on reconnect ─────────────────────
self.addEventListener('message', event => {
  if (event.data?.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
