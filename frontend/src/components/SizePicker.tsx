import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

interface SizePickerProps {
  sizes: number[]
  value: number | null
  onChange: (size: number) => void
  disabled?: boolean
}

function formatSize(size: number): string {
  return Number.isInteger(size) ? String(size) : size.toFixed(1)
}

export function SizePicker({ sizes, value, onChange, disabled }: SizePickerProps) {
  const sorted = Array.from(new Set(sizes)).sort((a, b) => a - b)

  return (
    <Select
      value={value !== null ? String(value) : undefined}
      onValueChange={(v) => {
        if (v) onChange(Number(v))
      }}
      disabled={disabled || sorted.length === 0}
    >
      <SelectTrigger aria-label="Select size" className="w-28">
        <SelectValue placeholder="Size" />
      </SelectTrigger>
      <SelectContent>
        {sorted.map((size) => (
          <SelectItem key={size} value={String(size)}>
            {formatSize(size)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
