import type { Snapshot } from "@/lib/types"

interface SparklineProps {
  history: Snapshot[]
  width?: number
  height?: number
}

/**
 * Minimal inline-SVG sparkline of mid-price history, ascending by date.
 * No charting library -- just a polyline scaled to fit the viewbox.
 */
export function Sparkline({ history, width = 80, height = 24 }: SparklineProps) {
  if (history.length < 2) {
    return <span className="text-xs text-muted-foreground">not enough data</span>
  }

  const values = history.map((h) => h.mid)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const padding = 2

  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * (width - padding * 2) + padding
      const y = height - padding - ((v - min) / range) * (height - padding * 2)
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(" ")

  const trendUp = values[values.length - 1] >= values[0]

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Price trend sparkline, ${trendUp ? "up" : "down"} overall`}
    >
      <polyline
        points={points}
        fill="none"
        stroke={trendUp ? "var(--positive)" : "var(--negative)"}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}
