import type { Contribution } from "@/lib/types"

interface ContributionBarsProps {
  contributions: Contribution[]
}

function formatFeatureValue(value: unknown): string {
  if (value === null || value === undefined) return ""
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(2)
  }
  return String(value)
}

function formatSigned(pct: number): string {
  const sign = pct > 0 ? "+" : pct < 0 ? "" : "±"
  return `${sign}${pct.toFixed(0)}%`
}

/**
 * Horizontal bars showing each feature's contribution to a single
 * prediction. Order is preserved exactly as the API returns it (already
 * sorted by |effect_pct| descending) -- we only normalize bar length
 * against the max |effect_pct| in this list.
 */
export function ContributionBars({ contributions }: ContributionBarsProps) {
  if (contributions.length === 0) {
    return <p className="text-sm text-muted-foreground">No explanation available for this prediction.</p>
  }

  const maxAbs = Math.max(...contributions.map((c) => Math.abs(c.effect_pct)), 1e-9)

  return (
    <ul className="space-y-2">
      {contributions.map((c) => {
        const isPositive = c.effect_pct > 0
        const isZero = c.effect_pct === 0
        const widthPct = (Math.abs(c.effect_pct) / maxAbs) * 100
        return (
          <li key={c.feature} className="grid grid-cols-[9rem_1fr_3.5rem] items-center gap-2 text-sm">
            <span className="truncate text-muted-foreground" title={c.feature}>
              {c.feature}
              {formatFeatureValue(c.value) && (
                <span className="text-xs opacity-70"> ({formatFeatureValue(c.value)})</span>
              )}
            </span>
            <div className="flex h-3 items-center bg-muted rounded-sm overflow-hidden">
              <div
                className={
                  isZero
                    ? "h-full w-0"
                    : isPositive
                      ? "h-full bg-positive rounded-sm"
                      : "h-full bg-negative rounded-sm"
                }
                style={{ width: `${widthPct}%` }}
              />
            </div>
            <span
              className={
                "tabular-nums text-right font-medium " +
                (isZero ? "text-muted-foreground" : isPositive ? "text-positive" : "text-negative")
              }
            >
              {formatSigned(c.effect_pct)}
            </span>
          </li>
        )
      })}
    </ul>
  )
}
