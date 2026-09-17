import { useCallback, useEffect, useState } from "react"
import { addToWatchlist, getWatchlist, predict, refreshWatchlist, removeFromWatchlist } from "@/lib/api"
import type { Prediction, Sneaker, WatchlistItem } from "@/lib/types"
import { SearchBar } from "@/components/SearchBar"
import { SizePicker } from "@/components/SizePicker"
import { ResultCard } from "@/components/ResultCard"
import { Watchlist } from "@/components/Watchlist"
import { ImportancePanel } from "@/components/ImportancePanel"
import { Skeleton } from "@/components/ui/skeleton"

function App() {
  const [selectedSneaker, setSelectedSneaker] = useState<Sneaker | null>(null)
  const [selectedSize, setSelectedSize] = useState<number | null>(null)

  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [predictLoading, setPredictLoading] = useState(false)
  const [predictError, setPredictError] = useState<string | null>(null)

  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([])
  const [watchlistLoading, setWatchlistLoading] = useState(true)
  const [watchlistError, setWatchlistError] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [removingId, setRemovingId] = useState<number | null>(null)
  const [adding, setAdding] = useState(false)

  const fetchWatchlist = useCallback(() => {
    setWatchlistLoading(true)
    setWatchlistError(null)
    getWatchlist()
      .then((items) => setWatchlist(items))
      .catch((err: Error) => setWatchlistError(err.message))
      .finally(() => setWatchlistLoading(false))
  }, [])

  useEffect(() => {
    fetchWatchlist()
  }, [fetchWatchlist])

  function handleSelectSneaker(sneaker: Sneaker) {
    setSelectedSneaker(sneaker)
    setSelectedSize(null)
    setPrediction(null)
    setPredictError(null)
  }

  function handleSelectSize(size: number) {
    setSelectedSize(size)
    if (!selectedSneaker) return
    setPredictLoading(true)
    setPredictError(null)
    predict(selectedSneaker.id, size)
      .then((p) => setPrediction(p))
      .catch((err: Error) => setPredictError(err.message))
      .finally(() => setPredictLoading(false))
  }

  function handleAddToWatchlist() {
    if (!selectedSneaker || selectedSize === null) return
    setAdding(true)
    addToWatchlist(selectedSneaker.id, selectedSize)
      .then(() => fetchWatchlist())
      .catch((err: Error) => setPredictError(err.message))
      .finally(() => setAdding(false))
  }

  function handleRemove(id: number) {
    setRemovingId(id)
    removeFromWatchlist(id)
      .then(() => fetchWatchlist())
      .catch((err: Error) => setWatchlistError(err.message))
      .finally(() => setRemovingId(null))
  }

  function handleRefresh() {
    setRefreshing(true)
    setWatchlistError(null)
    refreshWatchlist()
      .then(() => fetchWatchlist())
      .catch((err: Error) => setWatchlistError(err.message))
      .finally(() => setRefreshing(false))
  }

  const alreadyOnWatchlist =
    !!selectedSneaker &&
    selectedSize !== null &&
    watchlist.some((w) => w.sneaker_id === selectedSneaker.id && w.size === selectedSize)

  return (
    <div className="min-h-svh bg-background text-foreground">
      <div className="mx-auto max-w-3xl px-4 py-8 space-y-8">
        <header>
          <h1 className="text-2xl font-bold tracking-tight">Sneaker Resale Price Estimator</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Search a sneaker, pick a size, and see an estimated resale price with an explanation of what drove it.
          </p>
        </header>

        <section className="flex flex-wrap items-center gap-3">
          <SearchBar onSelect={handleSelectSneaker} />
          {selectedSneaker && (
            <SizePicker
              sizes={selectedSneaker.sizes_seen}
              value={selectedSize}
              onChange={handleSelectSize}
              disabled={predictLoading}
            />
          )}
        </section>

        <section>
          {!selectedSneaker && (
            <p className="text-sm text-muted-foreground">Search for a sneaker to get started.</p>
          )}
          {selectedSneaker && !selectedSize && !predictLoading && (
            <p className="text-sm text-muted-foreground">Pick a size to see an estimate.</p>
          )}
          {predictLoading && (
            <div className="space-y-3 rounded-xl bg-neutral-900 p-6">
              <Skeleton className="h-6 w-1/2 bg-neutral-700" />
              <Skeleton className="h-10 w-1/3 bg-neutral-700" />
              <Skeleton className="h-4 w-full bg-neutral-700" />
              <Skeleton className="h-4 w-full bg-neutral-700" />
            </div>
          )}
          {!predictLoading && predictError && <p className="text-sm text-destructive">{predictError}</p>}
          {!predictLoading && !predictError && prediction && (
            <ResultCard
              prediction={prediction}
              onAddToWatchlist={handleAddToWatchlist}
              adding={adding}
              alreadyOnWatchlist={alreadyOnWatchlist}
            />
          )}
        </section>

        <section>
          <ImportancePanel />
        </section>

        <section>
          <Watchlist
            items={watchlist}
            loading={watchlistLoading}
            error={watchlistError}
            refreshing={refreshing}
            onRefresh={handleRefresh}
            onRemove={handleRemove}
            removingId={removingId}
          />
        </section>
      </div>
    </div>
  )
}

export default App
