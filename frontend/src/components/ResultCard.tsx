import type { CSSProperties } from "react"
import type { Prediction } from "@/lib/types"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ContributionBars } from "@/components/ContributionBars"

interface ResultCardProps {
  prediction: Prediction
  onAddToWatchlist: () => void
  adding: boolean
  alreadyOnWatchlist: boolean
}

const currency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
})

export function ResultCard({ prediction, onAddToWatchlist, adding, alreadyOnWatchlist }: ResultCardProps) {
  const { sneaker, low, mid, high, crossed, retail_price, premium_pct, extrapolated, as_of, contributions } =
    prediction

  return (
    <div className="rounded-xl bg-neutral-900 text-neutral-50 dark:bg-neutral-950 p-6 shadow-lg">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <h2 className="text-xl font-semibold truncate">{sneaker.display_name}</h2>
          <div className="mt-2 flex flex-wrap gap-1.5">
            <Badge variant="secondary">{sneaker.brand}</Badge>
            {sneaker.silhouette && <Badge variant="secondary">{sneaker.silhouette}</Badge>}
            {sneaker.colorway && <Badge variant="secondary">{sneaker.colorway}</Badge>}
          </div>
          <p className="mt-1 text-xs text-neutral-400">
            {sneaker.sku} &middot; size {prediction.size} &middot; as of {as_of}
          </p>
        </div>
        {extrapolated && (
          <Badge className="bg-warning text-warning-foreground border-transparent shrink-0">
            Extrapolated
          </Badge>
        )}
      </div>

      <div className="mt-6">
        <div className="tabular-nums text-4xl font-bold">{currency.format(mid)}</div>
        <div className="tabular-nums mt-1 text-sm text-neutral-400">
          range {currency.format(low)} &ndash; {currency.format(high)}
        </div>
        <div className="mt-2 text-sm text-neutral-300">
          retail {currency.format(retail_price)}
          {premium_pct !== null && (
            <span className={premium_pct >= 0 ? "text-positive" : "text-negative"}>
              {" "}
              &middot; {premium_pct >= 0 ? "+" : ""}
              {premium_pct.toFixed(0)}% over retail
            </span>
          )}
        </div>
        {crossed && (
          <p className="mt-2 text-xs text-neutral-500 italic">
            Note: the quantile models briefly disagreed here (low/high crossed slightly) -- the range shown has
            been corrected for display.
          </p>
        )}
      </div>

      <div className="mt-6">
        <h3 className="mb-2 text-sm font-medium text-neutral-300">What drove this estimate</h3>
        <ContributionBarsDark contributions={contributions} />
      </div>

      <div className="mt-6">
        <Button onClick={onAddToWatchlist} disabled={adding || alreadyOnWatchlist} variant="secondary">
          {alreadyOnWatchlist ? "On watchlist" : adding ? "Adding..." : "Add to watchlist"}
        </Button>
      </div>
    </div>
  )
}

// The shared ContributionBars component reads the `--muted` /
// `--muted-foreground` theme tokens via Tailwind's bg-muted/text-muted-foreground
// utilities. Overriding those two CSS custom properties locally (rather than
// forking the bar-rendering logic) makes the bars legible against this
// card's dark background without changing them anywhere else in the app.
function ContributionBarsDark({ contributions }: { contributions: Prediction["contributions"] }) {
  return (
    <div
      style={
        {
          "--muted": "oklch(0.32 0 0)",
          "--muted-foreground": "oklch(0.75 0 0)",
        } as CSSProperties
      }
    >
      <ContributionBars contributions={contributions} />
    </div>
  )
}
