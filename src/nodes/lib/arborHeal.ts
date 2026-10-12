/**
 * A skeleton that arrived in pieces, joined into one tree — for the surfaces that can only lay out
 * one: the Neuron Dendrogram, and Split Axon/Dendrite's editor and corrections.
 *
 * Apart from `arborOps.ts` because this calls Python: `arborOps` is pure and reachable from the
 * daylight worker, and a value import of `src/pyodide` there would put the Pyodide engine in that
 * worker's graph (`topologyOps.ts`' rule; `src/test/importGraph.test.ts` holds it).
 *
 * The join is navis-fastcore's `heal_skeleton` through the Clean Skeletons capability, every other
 * step off and no distance cap. Memoised per geometry (`perGeometry`: `parents` and `positions`,
 * since the bridges are chosen by distance) rather than per item, because Split Axon/Dendrite hands
 * back a new item on every correction over the same arrays — so every surface, and a Run after a
 * card has drawn the neuron, shares one join. The heal also roots it at a labelled soma; every
 * reader builds an undirected arbour from it, so which end it was rooted at draws nothing different.
 */

import type { SkeletonGeometry } from '../../core/values'
import { perGeometry } from '../../core/values'
import { untilAborted } from '../../data/memoPromise'
import { runCleanSkeletons } from '../../pyodide/skeletons'
import { healedBridges, skeletonRoots } from './arborOps'
import { HEAL_ONLY, cleanRequestFrom } from './cleanOps'

/** A skeleton joined into one tree, and what the join added. */
export interface Healed {
  readonly skeleton: SkeletonGeometry
  /** Per node, whether its edge to its parent was added by the heal. Absent when nothing was. */
  readonly bridges?: Uint8Array
  /** How many pieces were joined; 1 for a skeleton that was whole. */
  readonly pieces: number
}

/** What a heal found for one geometry: the joined tree in place of the skeleton. */
type Join = Omit<Healed, 'skeleton'> & { readonly parents: Int32Array }

/** Per geometry, the join once known, and the heal under way — shared by everybody asking. */
const slots = perGeometry((): { join?: Join; pending?: Promise<Join> } => ({}))

/** `skeleton` with a join's tree — itself, by identity, where the join added nothing. */
function healedWith(skeleton: SkeletonGeometry, join: Join): Healed {
  return {
    skeleton:
      join.parents === skeleton.parents ? skeleton : { ...skeleton, parents: join.parents },
    bridges: join.bridges,
    pieces: join.pieces,
  }
}

/**
 * The answer where it needs no Python: a whole skeleton as it is, or one healed before. The
 * runtime is ~10 MB, so a card asking this first never downloads it for a neuron with nothing to
 * heal. `pieces` is the count of fragments otherwise, for a caller saying what it is waiting on.
 */
export function healedNow(skeleton: SkeletonGeometry): Healed | { pieces: number } {
  const slot = slots(skeleton)
  if (slot.join) return healedWith(skeleton, slot.join)
  const pieces = skeletonRoots(skeleton.parents).length
  if (pieces > 1) return { pieces }
  slot.join = { parents: skeleton.parents, pieces }
  return { skeleton, pieces }
}

/**
 * `skeleton` joined into one tree where it arrived in pieces. A heal already under way for it — the
 * card's, when Run arrives first — is joined rather than started again, and `signal` stops only
 * this caller waiting, never a heal somebody else is still waiting on.
 */
export function healSkeleton(
  skeleton: SkeletonGeometry,
  signal?: AbortSignal,
): Promise<Healed> {
  const now = healedNow(skeleton)
  if ('skeleton' in now) return Promise.resolve(now)
  const slot = slots(skeleton)
  // Kept only while running, so a failed heal is tried again by the next caller.
  slot.pending ??= join(skeleton, now.pieces)
    .then((found) => (slot.join = found))
    .finally(() => {
      slot.pending = undefined
    })
  return untilAborted(slot.pending, signal).then((found) => healedWith(skeleton, found))
}

/** One heal through Python, checked. */
async function join(skeleton: SkeletonGeometry, pieces: number): Promise<Join> {
  const result = await runCleanSkeletons(cleanRequestFrom({ items: [skeleton] }, HEAL_ONLY))
  // Healing renumbers nothing, so the coordinates and radii are the ones already held — which is
  // checked rather than trusted, a renumbered tree being a wrong arbour with no error.
  if (result.parents.length !== skeleton.parents.length) {
    throw new Error('Joining the skeleton’s fragments changed its node count.')
  }
  return {
    parents: result.parents,
    bridges: healedBridges(skeleton.parents, result.parents),
    pieces,
  }
}
