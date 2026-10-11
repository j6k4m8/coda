/**
 * Simplify Skeletons, headless: the card's params, the flattening, and the gather back.
 *
 * `cleanOps.ts`' arrangement and its two rules — the item count never changes, and distances
 * are micrometres on the card and nanometres on the wire — with one difference that decides
 * everything else here: **both operations keep a subset of the nodes and move none of them.**
 * Pruning removes whole twigs; every one of the thinning methods keeps the nodes it keeps where
 * they were. So the Python hands back *which rows survived* rather than new geometry, and
 * everything that rides on a node is gathered through that list exactly:
 *
 * - positions and radii, unchanged rather than interpolated;
 * - the source's `compartments`;
 * - a computed `split` and `nodeValues`, which Clean Skeletons has to drop. Those label the
 *   nodes of the tree they were computed on, and a resample mints new nodes; here every
 *   surviving node *is* one of those nodes, in the same place, so its label is still true of it.
 *   What changes is what a reader may conclude from the set — a flow computed with the twigs
 *   still on counted their synapses — which is a statement about order the card's guide makes;
 * - tags, through `snappedTags`: a tag on a survivor stays on it and one on a removed node is
 *   snapped to the nearest survivor — never dropped, never left naming another node.
 */

import type { ParamValues } from '../../core/node'
import type { SkeletonGeometry, SkeletonsValue } from '../../core/values'
import { boundsOf } from '../../core/values'
import { NM_PER_UM } from '../../data/units'
import type {
  SimplifyMethod,
  SimplifySkeletonsRequest,
  SimplifySkeletonsResult,
} from '../../pyodide/skeletons'
import { checkItemCount, labelled, snappedTags } from './cleanOps'
import { packPositions, packSkeletons } from './skeletonPacking'

/** Every control on the Simplify Skeletons card, in the card's units (µm, µm²). */
export interface SkeletonSimplifyParams {
  prune: number
  repeat: boolean
  method: SimplifyMethod
  factor: number
  tolerance: number
  area: number
}

/**
 * A context's params in, `SkeletonSimplifyParams` out — the one place that says what the card
 * means, and what the notebook emitter reads too (`skeletonCleanParamsFrom`'s reason).
 */
export function skeletonSimplifyParamsFrom(params: ParamValues): SkeletonSimplifyParams {
  return {
    prune: Number(params.prune),
    repeat: params.repeat === true,
    method: String(params.method) as SimplifyMethod,
    factor: Number(params.factor),
    tolerance: Number(params.tolerance),
    area: Number(params.area),
  }
}

/**
 * Whether a thinning method is set to something that removes nodes — the one statement of it. The
 * request sends `none` where this is false, so the Python has no guard of its own, and the
 * notebook emitter asks this rather than restating the thresholds.
 */
export function thins(params: SkeletonSimplifyParams): boolean {
  switch (params.method) {
    case 'every':
      return params.factor > 1
    // A tolerance of 0 still drops nodes that are exactly collinear, fastcore's own reading.
    case 'rdp':
      return params.tolerance >= 0
    case 'vw':
      return params.area > 0
    default:
      return false
  }
}

/** Whether these params ask for anything at all. */
export function isSimplifyNoOp(params: SkeletonSimplifyParams): boolean {
  return params.prune <= 0 && !thins(params)
}

/**
 * Whether these params ask a question a voxel coordinate cannot answer — `checkCleanUnits`'
 * second argument. Keeping every Nth node counts hops; a zero tolerance asks only "collinear?",
 * which is the same question in any unit.
 */
export function simplifyUsesDistance(params: SkeletonSimplifyParams): boolean {
  if (params.prune > 0) return true
  if (params.method === 'rdp') return params.tolerance > 0
  return params.method === 'vw' && params.area > 0
}

/**
 * Lay a skeleton set out flat, in nanometres, with the params folded in.
 *
 * Coordinates and parents only — see `SimplifySkeletonsRequest`. Built rather than borrowed,
 * `cleanRequestFrom`'s reason: `callPython` transfers.
 */
export function simplifyRequestFrom(
  skeletons: Pick<SkeletonsValue, 'items'>,
  params: SkeletonSimplifyParams,
): SimplifySkeletonsRequest {
  const { parents, offsets, somas } = packSkeletons(skeletons)
  return {
    points: packPositions(skeletons, offsets),
    parents,
    offsets,
    somas,
    prune: params.prune * NM_PER_UM,
    repeat: params.repeat,
    method: thins(params) ? params.method : 'none',
    factor: params.factor,
    tolerance: params.tolerance * NM_PER_UM,
    // An area, so the scale is squared: 1 µm² is a million nm².
    area: params.area * NM_PER_UM * NM_PER_UM,
  }
}

/** `values` at the rows `keep` names, `stride` values a row, as a new array of the same kind. */
function gather<T extends Float32Array | Uint8Array>(
  values: T,
  keep: Int32Array,
  stride = 1,
): T {
  const out = new (values.constructor as new (length: number) => T)(keep.length * stride)
  for (let i = 0; i < keep.length; i++) {
    for (let k = 0; k < stride; k++) out[i * stride + k] = values[keep[i]! * stride + k]!
  }
  return out
}

/**
 * The same set with fewer nodes, everything per node gathered through `keep`.
 *
 * Tags go through `snappedTags`, Clean Skeletons' rule, handed `keep`: a tag on a survivor stays
 * on it, and one on a removed node lands on the nearest survivor.
 */
export function skeletonsFromKeep(
  original: SkeletonsValue,
  result: SimplifySkeletonsResult,
): SkeletonsValue {
  checkItemCount('Simplify Skeletons', result.offsets, original.items.length, 'neurons')

  const items: SkeletonGeometry[] = original.items.map((item, n) => {
    const from = result.offsets[n]!
    const to = result.offsets[n + 1]!
    const keep = result.keep.subarray(from, to)
    // Nothing removed: the item itself, which keeps every downstream memo keyed on its arrays.
    if (keep.length === item.parents.length) return item
    const simplified: SkeletonGeometry = {
      id: item.id,
      positions: gather(item.positions, keep, 3),
      radii: gather(item.radii, keep),
      parents: result.parents.slice(from, to),
    }
    const nodeValues =
      item.nodeValues &&
      Object.fromEntries(
        Object.entries(item.nodeValues).map(([name, values]) => [name, gather(values, keep)]),
      )
    return {
      ...simplified,
      ...labelled(item.compartments && gather(item.compartments, keep)),
      ...(item.split ? { split: gather(item.split, keep) } : {}),
      ...(nodeValues ? { nodeValues } : {}),
      ...snappedTags(item, simplified, keep),
    }
  })

  return { ...original, items, bounds: boundsOf(items.map((item) => item.positions)) }
}
