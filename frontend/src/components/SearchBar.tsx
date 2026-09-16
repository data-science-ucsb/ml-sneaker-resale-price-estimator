import { useEffect, useRef, useState } from "react"
import { searchSneakers } from "@/lib/api"
import type { Sneaker } from "@/lib/types"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"

interface SearchBarProps {
  onSelect: (sneaker: Sneaker) => void
}

const DEBOUNCE_MS = 250
const MIN_CHARS = 2

export function SearchBar({ onSelect }: SearchBarProps) {
  const [query, setQuery] = useState("")
  const [results, setResults] = useState<Sneaker[]>([])
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const requestId = useRef(0)

  useEffect(() => {
    const trimmed = query.trim()
    if (trimmed.length < MIN_CHARS) {
      setResults([])
      setLoading(false)
      setError(null)
      setOpen(false)
      return
    }

    const currentRequest = ++requestId.current
    setLoading(true)
    setError(null)

    const timer = window.setTimeout(() => {
      searchSneakers(trimmed, 8)
        .then((sneakers) => {
          if (requestId.current !== currentRequest) return
          setResults(sneakers)
          setOpen(true)
          setLoading(false)
        })
        .catch((err: Error) => {
          if (requestId.current !== currentRequest) return
          setError(err.message)
          setResults([])
          setOpen(true)
          setLoading(false)
        })
    }, DEBOUNCE_MS)

    return () => window.clearTimeout(timer)
  }, [query])

  function handleSelect(sneaker: Sneaker) {
    onSelect(sneaker)
    setQuery(sneaker.display_name)
    setOpen(false)
    setResults([])
  }

  return (
    <div className="relative w-full max-w-xl">
      <Input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => {
          if (results.length > 0 || error) setOpen(true)
        }}
        placeholder="Search sneakers, e.g. 'beluga' or 'jordan 1 chicago'"
        aria-label="Search sneakers"
      />
      {open && (
        <div className="absolute z-20 mt-1 w-full rounded-md border bg-popover text-popover-foreground shadow-lg overflow-hidden">
          {loading && (
            <div className="p-3 space-y-2">
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-1/2" />
            </div>
          )}
          {!loading && error && (
            <div className="p-3 text-sm text-destructive">{error}</div>
          )}
          {!loading && !error && results.length === 0 && (
            <div className="p-3 text-sm text-muted-foreground">No matches found.</div>
          )}
          {!loading &&
            !error &&
            results.map((sneaker) => (
              <button
                key={sneaker.id}
                type="button"
                onClick={() => handleSelect(sneaker)}
                className="flex w-full flex-col items-start gap-1 px-3 py-2 text-left hover:bg-accent hover:text-accent-foreground border-b last:border-b-0"
              >
                <div className="flex w-full items-center justify-between gap-2">
                  <span className="font-medium truncate">{sneaker.display_name}</span>
                  <Badge variant="secondary" className="shrink-0">
                    {sneaker.brand}
                  </Badge>
                </div>
                <span className="text-xs text-muted-foreground">{sneaker.sku}</span>
              </button>
            ))}
        </div>
      )}
    </div>
  )
}
