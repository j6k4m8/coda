import { describe, expect, it } from 'vitest'

import type { SkeletonGeometry } from '../../core/values'
import { CODE_AXON, CODE_DENDRITE, CODE_LINKER, SWC_SOMA } from '../../core/values'
import { ySkeleton } from './__fixtures__/ySkeleton'
import { splitByTags, tagNames } from './splitByTag'

const A = CODE_AXON
const D = CODE_DENDRITE
const L = CODE_LINKER

/** The Y (0 ── 1 ── 2 ── 3, with 2 ── 4 ── 5) carrying `tags`. */
function tagged(tags: Record<string, number[]>, base = ySkeleton()): SkeletonGeometry {
  return {
    ...base,
    tags: Object.fromEntries(
      Object.entries(tags).map(([name, nodes]) => [name, Int32Array.from(nodes)]),
    ),
  }
}

describe('splitByTags', () => {
  it('makes a tagged node and everything distal of it axon, and the rest dendrite', () => {
    const split = splitByTags(tagged({ axon: [4] }), ['axon'], [])
    expect(split.status).toBe('ok')
    expect([...split.labels]).toEqual([D, D, D, D, A, A])
  })

  it('makes the rest axon where only dendrite tags are named', () => {
    const split = splitByTags(tagged({ dendrite: [3] }), [], ['dendrite'])
    expect([...split.labels]).toEqual([A, A, A, D, A, A])
  })

  it('makes the cable distal of neither tag linker where both kinds are named', () => {
    const split = splitByTags(tagged({ axon: [4], dendrite: [3] }), ['axon'], ['dendrite'])
    expect([...split.labels]).toEqual([L, L, L, D, A, A])
  })

  it('lets a tag inside another tag’s subtree win', () => {
    // Axon from the branch point, then a dendrite tag further out on one arm.
    const split = splitByTags(tagged({ axon: [2], dendrite: [4] }), ['axon'], ['dendrite'])
    expect([...split.labels]).toEqual([L, L, A, A, D, D])
  })

  it('reads several names for one compartment, CATMAID’s tags being free text', () => {
    const split = splitByTags(
      tagged({ axon: [3], 'axon start': [4] }),
      ['axon', 'axon start'],
      [],
    )
    expect([...split.labels]).toEqual([D, D, D, A, A, A])
  })

  it('measures distal from the soma, not from where the file starts', () => {
    // Soma at the end of the long arm: distal of the branch point is now the root side and the
    // short arm, and the long arm back to the soma is not.
    const base = { ...ySkeleton(), compartments: Uint8Array.from([0, 0, 0, 0, 0, SWC_SOMA]) }
    const split = splitByTags(tagged({ axon: [2] }, base), ['axon'], [])
    expect([...split.labels]).toEqual([A, A, A, A, D, D])
  })

  it('says `no tags` for a neuron carrying none of the named tags, and labels nothing', () => {
    const split = splitByTags(tagged({ ends: [3, 5] }), ['axon'], [])
    expect(split.status).toBe('no tags')
    expect([...split.labels].every((code) => code === 0)).toBe(true)
  })

  it('refuses a neuron in pieces, whose far fragments have no "distal"', () => {
    const pieces = ySkeleton(undefined, [-1, 0, 1, 2, -1, 4])
    const split = splitByTags(tagged({ axon: [4] }, pieces), ['axon'], [])
    expect(split.status).toBe('multiple roots')
  })
})

describe('tagNames', () => {
  it('lists every tag on any skeleton, once and sorted', () => {
    expect(
      tagNames([tagged({ soma: [0], ends: [3] }), tagged({ ends: [5] }), ySkeleton()]),
    ).toEqual(['ends', 'soma'])
  })
})
