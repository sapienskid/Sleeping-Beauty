// Sleeping Beauty - Inert Pass-Through Service Worker
// Clears stale Workbox precache storages cleanly without intercepting fetch requests.
self.addEventListener('install', () => {
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) => Promise.all(keys.map((key) => caches.delete(key))))
    );
});
