/**
 * Where a skeleton's soma is: which node stands for it, and how a published soma *position* becomes
 * one. Source-agnostic — a source supplies the position and its own tolerance.
 */

import type { SkeletonGeometry } from '../core/values'
import { SWC_SOMA } from '../core/values'

/**
 * The node of `positions` (xyz interleaved) nearest `at`, within `tolerance`; -1 where none is.
 * A tie goes to the later node. Split corrections snap their anchors with it, and `withSomaAt` a
 * soma position.
 */
export function nearestNode(
  positions: Float32Array,
  at: readonly number[],
  tolerance: number,
): number {
  let best = -1
  let bestSq = tolerance * tolerance
  for (let i = 0; i * 3 < positions.length; i++) {
    const dx = positions[i * 3]! - at[0]!
    const dy = positions[i * 3 + 1]! - at[1]!
    const dz = positions[i * 3 + 2]! - at[2]!
    const sq = dx * dx + dy * dy + dz * dz
    if (sq <= bestSq) {
      best = i
      bestSq = sq
    }
  }
  return best
}

/**
 * The soma-labelled node with the largest radius — the centre of a soma that several nodes
 * outline — or -1 where none is labelled. What `resolveRoot('soma')` roots at, and the node the
 * split and the heal reroot at.
 */
export function widestSomaNode(skeleton: SkeletonGeometry): number {
  const { compartments, radii } = skeleton
  let best = -1
  if (!compartments) return best
  for (let i = 0; i < skeleton.parents.length; i++) {
    if (compartments[i] !== SWC_SOMA) continue
    if (best < 0 || radii[i]! > radii[best]!) best = i
  }
  return best
}

/** Per cached skeleton, the soma it was labelled at and the labelled copy. */
const labelledAt = new WeakMap<SkeletonGeometry, { key: string; out: SkeletonGeometry }>()

/**
 * `skeleton` with its node nearest `soma` labelled `SWC_SOMA`, or `skeleton` itself where there is
 * nothing to do: it already labels a soma, or no node lies within `tolerance` — a property of
 * the source's somata, so the caller's number (neuPrint's `SOMA_SNAP_NM`).
 *
 * For a source that publishes a soma *position* but no soma *node*, without which
 * `resolveRoot('soma')` falls back to the largest fragment — on a fragmented skeleton often a piece
 * of arbour (male-CNS 48122: the soma is a 38-node fragment 1.2 µm from `somaLocation`, the largest
 * fragment's root 200 µm away). A new object and new arrays, never written in place, because the
 * geometry is shared through `cachedGeometry` — and **memoised on that shared object**, so the
 * labelled copy keeps one identity across partial publishes and re-runs. Every cache keyed on a
 * skeleton (the heal, the split, the 3D scene's segments) would otherwise miss on every Run.
 */
export function withSomaAt(
  skeleton: SkeletonGeometry,
  soma: readonly [number, number, number],
  tolerance: number,
): SkeletonGeometry {
  if (skeleton.compartments?.includes(SWC_SOMA)) return skeleton
  const key = `${soma.join(',')}:${tolerance}`
  const held = labelledAt.get(skeleton)
  if (held?.key === key) return held.out
  const node = nearestNode(skeleton.positions, soma, tolerance)
  let out = skeleton
  if (node >= 0) {
    const compartments = skeleton.compartments
      ? Uint8Array.from(skeleton.compartments)
      : new Uint8Array(skeleton.parents.length)
    compartments[node] = SWC_SOMA
    out = { ...skeleton, compartments }
  }
  labelledAt.set(skeleton, { key, out })
  return out
}
