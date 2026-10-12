/**
 * A set of skeletons flattened for one crossing of the Python bridge.
 *
 * Every Pyodide call that takes skeletons sends the same arrays — a `parents` soup and an
 * `offsets` index saying where each neuron starts — and the rules about them are exactly the
 * kind that look like details until one is missed. They were written twice, in `cleanOps.ts` and
 * in the Topology node, and only one copy carried the comment.
 *
 * - **Parent indices stay neuron-local.** The Python side slices each neuron out and hands
 *   fastcore row numbers as node ids, so adding a global offset here produces a forest of
 *   dangling references rather than a tree — and fastcore answers, plausibly, about the wrong
 *   graph.
 * - **The buffers are built rather than borrowed.** `callPython` transfers, so anything passed in
 *   is detached the moment the call is posted; a `subarray` view onto a cached skeleton would
 *   take the caller's own geometry with it.
 * - **A soma is one node per neuron, neuron-local, `-1` for none** — `widestSomaNode`, packed here
 *   so the heal and the split, which both reroot there, cannot pick different nodes.
 *
 * Callers layer their own per-call buffers on top — positions and radii for a clean, synapse
 * counts for a split — over the same `offsets`, which is what lets the result be scattered
 * straight back onto the geometry they already hold.
 */

import type { SkeletonsValue } from '../../core/values'
import { skeletonPointCount } from '../../core/values'
import { widestSomaNode } from '../../data/soma'

export interface PackedSkeletons {
  /** Parent index per node across the whole set, `-1` for a root. Neuron-local. */
  readonly parents: Int32Array
  /** Where each neuron starts, counted in nodes. Length is `items.length + 1`. */
  readonly offsets: Int32Array
  /** Per neuron, its soma node (`widestSomaNode`), neuron-local; `-1` where none is labelled. */
  readonly somas: Int32Array
  /** Total node count — `offsets[items.length]`, named so callers need not reach for it. */
  readonly total: number
}

export function packSkeletons(skeletons: Pick<SkeletonsValue, 'items'>): PackedSkeletons {
  const total = skeletonPointCount(skeletons)
  const parents = new Int32Array(total)
  const offsets = new Int32Array(skeletons.items.length + 1)
  const somas = new Int32Array(skeletons.items.length)

  let at = 0
  for (let n = 0; n < skeletons.items.length; n++) {
    const item = skeletons.items[n]!
    const count = item.parents.length
    parents.set(item.parents.subarray(0, count), at)
    at += count
    offsets[n + 1] = at
    somas[n] = widestSomaNode(item)
  }
  return { parents, offsets, somas, total }
}

/**
 * Every node's xyz over `packSkeletons`' `offsets` — the per-call buffer both a clean and a
 * healing split add, and built rather than borrowed for the transfer rule above.
 */
export function packPositions(
  skeletons: Pick<SkeletonsValue, 'items'>,
  offsets: Int32Array,
): Float32Array {
  const points = new Float32Array(offsets[skeletons.items.length]! * 3)
  for (let n = 0; n < skeletons.items.length; n++) {
    const item = skeletons.items[n]!
    points.set(item.positions.subarray(0, item.parents.length * 3), offsets[n]! * 3)
  }
  return points
}
