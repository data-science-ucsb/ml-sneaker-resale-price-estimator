import { useEffect, useState } from "react"
import { getImportance } from "@/lib/api"
import type { ImportanceEntry } from "@/lib/types"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"

/**
 * Global "what drives price overall" view: mean |SHAP| importance per
 * feature, fetched once when the panel is expanded. Bars are all one
 * color (unlike ContributionBars) because global importance has no
 * direction -- only magnitude. Bar length is normalized to the max
 * importance in the list (0-100% relative length).
 */
export function ImportancePanel() {
  const [open, setOpen] = useState(false)
  const [entries, setEntries] = useState<ImportanceEntry[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open || entries !== null || loading) return
    setLoading(true)
    setError(null)
    getImportance()
      .then((data) => setEntries(data))
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false))
  }, [open, entries, loading])

  const maxImportance = entries && entries.length > 0 ? Math.max(...entries.map((e) => e.importance)) : 1

  return (
    <div className="rounded-lg border p-4">
      <Button
        variant="ghost"
        className="flex w-full items-center justify-between p-0 h-auto font-semibold text-base"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span>What drives price overall?</span>
        <span className="text-muted-foreground text-sm">{open ? "Hide" : "Show"}</span>
      </Button>

      {open && (
        <div className="mt-4">
          {loading && (
            <div className="space-y-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-5/6" />
              <Skeleton className="h-4 w-2/3" />
            </div>
          )}
          {!loading && error && <p className="text-sm text-destructive">{error}</p>}
          {!loading && !error && entries !== null && entries.length === 0 && (
            <p className="text-sm text-muted-foreground">No importance data available yet.</p>
          )}
          {!loading && !error && entries !== null && entries.length > 0 && (
            <ul className="space-y-2">
              {entries.map((entry) => (
                <li key={entry.feature} className="grid grid-cols-[9rem_1fr_4rem] items-center gap-2 text-sm">
                  <span className="truncate text-muted-foreground" title={entry.feature}>
                    {entry.feature}
                  </span>
                  <div className="h-3 rounded-sm bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-sm bg-primary"
                      style={{ width: `${(entry.importance / maxImportance) * 100}%` }}
                    />
                  </div>
                  <span className="tabular-nums text-right text-muted-foreground">
                    {entry.importance.toFixed(3)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
