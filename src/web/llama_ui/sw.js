// Sleeping Beauty - Service Worker Cache Buster & Uninstaller
self.addEventListener('install', (event) => {
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(keys.map((key) => caches.delete(key)));
        }).then(() => self.clients.claim()).then(() => {
            return self.registration.unregister();
        })
    );
});

// Always bypass cache and fetch directly from network
self.addEventListener('fetch', (event) => {
    event.respondWith(fetch(event.request));
});
