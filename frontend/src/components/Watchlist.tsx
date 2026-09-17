import type { WatchlistItem } from "@/lib/types"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Sparkline } from "@/components/Sparkline"

interface WatchlistProps {
  items: WatchlistItem[]
  loading: boolean
  error: string | null
  refreshing: boolean
  onRefresh: () => void
  onRemove: (id: number) => void
  removingId: number | null
}

const currency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
})

function formatSize(size: number): string {
  return Number.isInteger(size) ? String(size) : size.toFixed(1)
}

function formatDelta(delta: number): string {
  const sign = delta > 0 ? "+" : delta < 0 ? "-" : "±"
  return `${sign}${currency.format(Math.abs(delta))}`
}

export function Watchlist({
  items,
  loading,
  error,
  refreshing,
  onRefresh,
  onRemove,
  removingId,
}: WatchlistProps) {
  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-semibold">Watchlist</h2>
        <Button size="sm" variant="outline" onClick={onRefresh} disabled={refreshing || items.length === 0}>
          {refreshing ? "Refreshing..." : "Refresh values"}
        </Button>
      </div>

      {loading && (
        <div className="space-y-2">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
        </div>
      )}

      {!loading && error && <p className="text-sm text-destructive">{error}</p>}

      {!loading && !error && items.length === 0 && (
        <p className="text-sm text-muted-foreground">Your watchlist is empty. Add a sneaker above to track it.</p>
      )}

      {!loading && !error && items.length > 0 && (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Shoe</TableHead>
              <TableHead>Size</TableHead>
              <TableHead>Latest mid</TableHead>
              <TableHead>Trend</TableHead>
              <TableHead>Since first</TableHead>
              <TableHead className="text-right">Remove</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.map((item) => {
              const first = item.history[0]?.mid
              const latestMid = item.latest?.mid ?? item.history[item.history.length - 1]?.mid
              const delta = first !== undefined && latestMid !== undefined ? latestMid - first : null
              return (
                <TableRow key={item.id}>
                  <TableCell className="max-w-[16rem] truncate font-medium" title={item.display_name}>
                    {item.display_name}
                  </TableCell>
                  <TableCell className="tabular-nums">{formatSize(item.size)}</TableCell>
                  <TableCell className="tabular-nums">
                    {latestMid !== undefined ? currency.format(latestMid) : "—"}
                    {item.latest?.extrapolated && (
                      <span className="ml-1 text-xs text-warning-foreground bg-warning rounded px-1 py-0.5">
                        extrap.
                      </span>
                    )}
                  </TableCell>
                  <TableCell>
                    <Sparkline history={item.history} />
                  </TableCell>
                  <TableCell
                    className={
                      "tabular-nums " +
                      (delta === null ? "text-muted-foreground" : delta >= 0 ? "text-positive" : "text-negative")
                    }
                  >
                    {delta === null ? "—" : formatDelta(delta)}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => onRemove(item.id)}
                      disabled={removingId === item.id}
                    >
                      {removingId === item.id ? "Removing..." : "Remove"}
                    </Button>
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      )}
    </div>
  )
}
