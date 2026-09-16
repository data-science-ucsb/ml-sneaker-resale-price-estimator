export interface Sneaker {
  id: string
  sku: string
  sneaker_name: string
  display_name: string
  brand: string
  silhouette: string
  colorway: string
  retail_price: number
  release_date: string
  n_sales: number
  median_sale: number
  sizes_seen: number[]
}

export interface Contribution {
  feature: string
  value: unknown
  effect_pct: number
}

export interface Prediction {
  sneaker: Sneaker
  size: number
  as_of: string
  low: number
  mid: number
  high: number
  crossed: boolean
  retail_price: number
  premium_pct: number | null
  extrapolated: boolean
  contributions: Contribution[]
  model_version: string
}

export interface Snapshot {
  as_of: string
  mid: number
}

export interface WatchlistLatest {
  as_of: string
  low: number
  mid: number
  high: number
  extrapolated: boolean
}

export interface WatchlistItem {
  id: number
  sneaker_id: string
  display_name: string
  size: number
  created_at: string
  latest: WatchlistLatest | null
  history: Snapshot[]
}

export interface RefreshResult {
  sneaker_id: string
  as_of: string
  mid: number
  extrapolated: boolean
}

export interface ImportanceEntry {
  feature: string
  importance: number
}

export interface HealthStatus {
  status: string
  model_version: string
  n_catalog: number
}
