// Single source of truth for the backend base URL.
//
// Reads NEXT_PUBLIC_API_URL (inlined at build time). Falls back to localhost in
// dev and the deployed Render service in production, so the app never silently
// calls the visitor's own machine the way the old hardcoded `localhost:8001`
// calls did.
const FALLBACK =
  process.env.NODE_ENV === "development"
    ? "http://localhost:8000"
    : "https://fraud-detection-api.onrender.com";

export const API_BASE = (process.env.NEXT_PUBLIC_API_URL || FALLBACK).replace(/\/+$/, "");

export const apiUrl = (path: string) =>
  `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const OFFLINE_MSG =
  "Can't reach the DataLink API. On the free tier it sleeps after ~15 min of inactivity - give it ~30s to wake up, then retry.";

/** fetch + JSON parse with friendly errors. Body objects are auto-JSON-encoded; pass FormData as-is. */
export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const isForm = init.body instanceof FormData;
  let res: Response;
  try {
    res = await fetch(apiUrl(path), {
      ...init,
      headers: {
        ...(init.body && !isForm ? { "Content-Type": "application/json" } : {}),
        ...init.headers,
      },
    });
  } catch {
    throw new ApiError(OFFLINE_MSG, 0);
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      if (data?.detail) detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

/** Lightweight liveness probe used by the BackendStatus banner. */
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(apiUrl("/api/health"), { cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
}
