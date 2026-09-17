import type {
  HealthStatus,
  ImportanceEntry,
  Prediction,
  RefreshResult,
  Sneaker,
  WatchlistItem,
} from "@/lib/types"

const BASE = "/api"

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: init?.body ? { "Content-Type": "application/json" } : undefined,
      ...init,
    })
  } catch {
    throw new Error("Network error: could not reach the API")
  }

  if (res.status === 204) {
    return undefined as T
  }

  let body: unknown = null
  const text = await res.text()
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      // non-JSON body, leave as null
    }
  }

  if (!res.ok) {
    const message =
      body && typeof body === "object" && "error" in body && typeof (body as { error: unknown }).error === "string"
        ? (body as { error: string }).error
        : `Request failed with status ${res.status}`
    throw new Error(message)
  }

  return body as T
}

export function getHealth(): Promise<HealthStatus> {
  return request<HealthStatus>("/health")
}

export function searchSneakers(q: string, limit = 8): Promise<Sneaker[]> {
  const params = new URLSearchParams({ q, limit: String(limit) })
  return request<Sneaker[]>(`/sneakers?${params.toString()}`)
}

export function predict(sneakerId: string, size: number, asOf?: string): Promise<Prediction> {
  return request<Prediction>("/predict", {
    method: "POST",
    body: JSON.stringify({
      sneaker_id: sneakerId,
      size,
      ...(asOf ? { as_of: asOf } : {}),
    }),
  })
}

export function getImportance(): Promise<ImportanceEntry[]> {
  return request<ImportanceEntry[]>("/importance")
}

export function getWatchlist(): Promise<WatchlistItem[]> {
  return request<WatchlistItem[]>("/watchlist")
}

export function addToWatchlist(sneakerId: string, size: number): Promise<WatchlistItem> {
  return request<WatchlistItem>("/watchlist", {
    method: "POST",
    body: JSON.stringify({ sneaker_id: sneakerId, size }),
  })
}

export function removeFromWatchlist(id: number): Promise<void> {
  return request<void>(`/watchlist/${id}`, { method: "DELETE" })
}

export function refreshWatchlist(asOf?: string, advanceDays?: number): Promise<RefreshResult[]> {
  const body: Record<string, unknown> = {}
  if (asOf) body.as_of = asOf
  if (advanceDays !== undefined) body.advance_days = advanceDays
  return request<RefreshResult[]>("/watchlist/refresh", {
    method: "POST",
    body: JSON.stringify(body),
  })
}
