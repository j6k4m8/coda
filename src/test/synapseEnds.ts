/**
 * Checks on synapses-between clouds for the live suites: that `location: 'both'` puts each row's
 * partner end a short hop from its drawn end, and that two clouds hold the same points.
 */

import { expect } from 'vitest'

import type { PointsValue } from '../core/values'
import { quantileSorted } from '../core/stats'
import { PARTNER_POSITION_COLUMNS } from '../data/source'

/** Nanometres from each presynaptic point to its postsynaptic partner, one per row. */
function hopLengths(points: PointsValue): number[] {
  const [xs, ys, zs] = PARTNER_POSITION_COLUMNS.map((c) => points.attributes.data[c]!)
  const p = points.positions
  return xs!.map((x, i) =>
    Math.hypot(
      Number(x) - p[i * 3]!,
      Number(ys![i]) - p[i * 3 + 1]!,
      Number(zs![i]) - p[i * 3 + 2]!,
    ),
  )
}

/** The median gap, `quantileSorted`'s interpolation so "median" means one thing here. */
function medianHop(gaps: readonly number[]): number {
  return quantileSorted(
    [...gaps].sort((a, b) => a - b),
    0.5,
  )
}

/**
 * Every row's partner end a short hop from its drawn end: none past a micrometre, and most not on
 * top of it (a positive median), which a source copying one coordinate into both would fail.
 */
export function expectShortHops(points: PointsValue): void {
  expect(points.attributes.length).toBeGreaterThan(0)
  const gaps = hopLengths(points)
  expect(Math.max(...gaps)).toBeLessThan(1000)
  expect(medianHop(gaps)).toBeGreaterThan(0)
}

/** Each point as text, sorted: two clouds with the same points in any order compare equal. */
export function sortedPoints(points: PointsValue): string[] {
  const p = points.positions
  const out: string[] = []
  for (let i = 0; i < p.length; i += 3) out.push(`${p[i]},${p[i + 1]},${p[i + 2]}`)
  return out.sort()
}
