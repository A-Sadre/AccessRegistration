// src/api.ts
const API_BASE = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

/**
 * In dev:
 * - if VITE_API_BASE is empty, we keep "/api" so Vite proxy works
 * In prod (Vercel):
 * - you must set VITE_API_BASE to your tunnel URL, so it becomes "https://xxx.trycloudflare.com/api"
 */
export function apiUrl(path: string) {
  const p = path.startsWith("/") ? path : `/${path}`;
  if (!API_BASE) return `/api${p}`;         // dev -> Vite proxy
  return `${API_BASE}/api${p}`;            // prod -> tunnel backend
}
