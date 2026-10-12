/**
 * Split Axon/Dendrite by tag: a node carrying one of the named axon or dendrite tags takes that
 * compartment, **and so does everything distal of it** — away from the soma. CATMAID tracers place
 * a tag where a compartment starts, so the tag marks the base of a subtree.
 *
 * What the rest of the neuron is depends on which kinds of tag are named: with axon tags alone it
 * is dendrite, with dendrite tags alone axon, and with both it is linker — the stretch distal of
 * neither, which is the primary neurite between the soma and the first tag.
 *
 * "Distal" is the hand corrections' (`splitCorrections.ts`): the same soma-rooted arbour
 * (`splitArbor`) and the same fill (`fillDistal`). Not tags *as* corrections, which are anchored
 * by position and snapped: a tag already names its node. Tags are applied in pre-order, so a tag
 * inside another's subtree is applied after it and wins — a dendrite tag on a branch distal of an
 * axon tag makes that branch dendrite. A node named in both lists ends up dendrite, the dendrite
 * tags being applied after the axon ones at the same node.
 */

import type { SkeletonGeometry } from '../../core/values'
import { CODE_AXON, CODE_DENDRITE, CODE_LINKER } from '../../core/values'
import { skeletonRoots } from './arborOps'
import { fillDistal, splitArbor } from './splitCorrections'
import type { SplitStatus } from './splitStatus'

/** One neuron's split by tag: its labels, and `ok` or why there are none. */
export interface TagSplit {
  readonly labels: Uint8Array
  readonly status: SplitStatus
}

/**
 * Split one neuron by its tags. `tree` is the skeleton whose parents are walked — the healed one,
 * where a heal was asked for — and must keep the delivered node numbering, which every heal does.
 * A forest is refused (`multiple roots`), as the flow split refuses it: a tag on a fragment the
 * soma's walk never reaches has no "distal", and the fragment's labels would be the default.
 */
export function splitByTags(
  tree: SkeletonGeometry,
  axonTags: readonly string[],
  dendriteTags: readonly string[],
): TagSplit {
  const n = tree.parents.length
  const axon = taggedNodes(tree, axonTags)
  const dendrite = taggedNodes(tree, dendriteTags)
  const hasAxon = axon.length > 0
  const hasDendrite = dendrite.length > 0
  if (!hasAxon && !hasDendrite) return { labels: new Uint8Array(n), status: 'no tags' }
  if (skeletonRoots(tree.parents).length > 1) {
    return { labels: new Uint8Array(n), status: 'multiple roots' }
  }

  let rest = CODE_AXON
  if (hasAxon && hasDendrite) rest = CODE_LINKER
  else if (hasAxon) rest = CODE_DENDRITE
  const labels = new Uint8Array(n).fill(rest)
  const arbor = splitArbor(tree)
  const starts = [
    ...axon.map((node) => ({ node, code: CODE_AXON })),
    ...dendrite.map((node) => ({ node, code: CODE_DENDRITE })),
  ].sort((a, b) => arbor.tin[a.node]! - arbor.tin[b.node]!)
  for (const { node, code } of starts) fillDistal(arbor, node, code, labels)
  return { labels, status: 'ok' }
}

/** The nodes carrying any of `names`, each once. */
function taggedNodes(tree: SkeletonGeometry, names: readonly string[]): number[] {
  const nodes = new Set<number>()
  for (const name of names) for (const node of tree.tags?.[name] ?? []) nodes.add(node)
  return [...nodes]
}

/**
 * Every tag name on any of the skeletons, sorted — what a warning offers when the names asked for
 * match nothing, CATMAID's tags being free text whose spelling differs between projects.
 */
export function tagNames(skeletons: readonly SkeletonGeometry[]): string[] {
  const names = new Set<string>()
  for (const item of skeletons) for (const name of Object.keys(item.tags ?? {})) names.add(name)
  return [...names].sort()
}
