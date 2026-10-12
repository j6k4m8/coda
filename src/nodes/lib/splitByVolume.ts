/**
 * Split Axon/Dendrite by ROI: a skeleton node inside any of the named volumes takes one
 * compartment, and every other node the other one.
 *
 * The inside test is Points in Volumes' (`meshInside.ts`): a skeleton node is a point, so the box
 * prefilter, the cached trees and the oblique ray all carry over, and so does the cost model — a
 * ray per node per candidate volume, warned about through `warnRayWork`. Nothing here walks the
 * tree, so it needs no root, no heal and no synapses: a fragmented skeleton splits piece by piece.
 * See `docs/nodes-morphology.md`.
 */

import type { Slicer } from '../../core/slice'
import { sliced } from '../../core/slice'
import type { MeshGeometry, MeshesValue, SkeletonGeometry } from '../../core/values'
import { CODE_AXON, CODE_DENDRITE } from '../../core/values'
import { buildInsideTests, volumeBoxes } from './meshInside'

/** Which compartment the named volumes hold; every node outside them is the other one. */
export type VolumeCompartment = 'axon' | 'dendrite'

/**
 * The wired volumes a split by ROI reads: those named, or every one where none is named. `unknown`
 * is the names no wired volume carries — a typo, or a region ROI Meshes was not asked for — which
 * would otherwise read as a neuron with nothing inside.
 */
export function pickVolumes(
  volumes: Pick<MeshesValue, 'items'>,
  names: readonly string[],
): { items: MeshGeometry[]; unknown: string[] } {
  if (names.length === 0) return { items: volumes.items, unknown: [] }
  const wanted = new Set(names)
  const items = volumes.items.filter((item) => wanted.has(item.id))
  const found = new Set(items.map((item) => item.id))
  return { items, unknown: names.filter((name) => !found.has(name)) }
}

/**
 * Which skeleton nodes are inside any of `volumes`, 1 or 0 per node, and how many neurons have no
 * node inside at all — usually the wrong ROI or a frame mismatch rather than a neuron that is all
 * one compartment, so the caller says it.
 *
 * A mask rather than labels, because it is what the rays answer: which compartment the volumes
 * hold is a relabelling of it (`volumeLabels`), so flipping that control casts nothing again.
 *
 * With a `margin`, a node outside every volume but within that distance of one's surface is in
 * too — `buildInsideTests`' margin.
 */
export async function nodesInVolumes(
  skeletons: readonly SkeletonGeometry[],
  volumes: readonly MeshGeometry[],
  hooks: Slicer = {},
  /** The boxes the caller already counted rays with, so they are built once. */
  prefilter = volumeBoxes(volumes),
  /** In the skeletons' own units; 0 is none. */
  margin = 0,
): Promise<{ inside: Uint8Array[]; untouched: number }> {
  const tests = await buildInsideTests(volumes, prefilter, hooks, margin)
  const inside = skeletons.map((item) => new Uint8Array(item.parents.length))

  // One walk over every node of every neuron, sliced so a Cancel reaches it.
  const starts = [0]
  for (const item of skeletons) starts.push(starts.at(-1)! + item.parents.length)
  let item = 0
  await sliced(starts.at(-1)!, hooks, (node) => {
    while (node >= starts[item + 1]!) item++
    const at = node - starts[item]!
    const p = skeletons[item]!.positions
    if (tests.containsAny(p[at * 3]!, p[at * 3 + 1]!, p[at * 3 + 2]!)) inside[item]![at] = 1
  })
  return { inside, untouched: inside.filter((mask) => !mask.includes(1)).length }
}

/** Each neuron's labels: a node the mask marks takes `compartment`, every other the other one. */
export function volumeLabels(
  inside: readonly Uint8Array[],
  compartment: VolumeCompartment,
): Uint8Array[] {
  const [hit, miss] =
    compartment === 'axon' ? [CODE_AXON, CODE_DENDRITE] : [CODE_DENDRITE, CODE_AXON]
  return inside.map((mask) => {
    const labels = new Uint8Array(mask.length).fill(miss)
    for (let i = 0; i < mask.length; i++) if (mask[i]) labels[i] = hit
    return labels
  })
}
