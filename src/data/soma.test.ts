import { describe, expect, it } from 'vitest'
import type { SkeletonGeometry } from '../core/values'
import { SWC_AXON, SWC_SOMA } from '../core/values'
import { withSomaAt } from './soma'

/** neuPrint's, the one caller's: half of a 4 µm fly soma. */
const SOMA_SNAP_NM = 2000

/** Three nodes along x, in nanometres. */
function line(): SkeletonGeometry {
  return {
    id: '1',
    positions: new Float32Array([0, 0, 0, 1000, 0, 0, 5000, 0, 0]),
    radii: new Float32Array(3),
    parents: new Int32Array([-1, 0, 1]),
  }
}

describe('withSomaAt', () => {
  it('labels the node nearest the soma position, leaving the input untouched', () => {
    const skeleton = line()
    const labelled = withSomaAt(skeleton, [4200, 0, 0], SOMA_SNAP_NM)
    expect([...labelled.compartments!]).toEqual([0, 0, SWC_SOMA])
    // Shared through the geometry cache, so never written in place.
    expect(skeleton).not.toHaveProperty('compartments')
  })

  it('accepts up to half a 4 µm soma and nothing further', () => {
    // A position is the soma's centre, so a node further than its radius is a skeleton that
    // does not reach the cell body — labelling one would root the arbour somewhere arbitrary.
    const skeleton = line()
    expect(
      withSomaAt(skeleton, [5000 + SOMA_SNAP_NM, 0, 0], SOMA_SNAP_NM).compartments?.[2],
    ).toBe(SWC_SOMA)
    expect(
      withSomaAt(line(), [5001 + SOMA_SNAP_NM, 0, 0], SOMA_SNAP_NM).compartments,
    ).toBeUndefined()
  })

  it('hands back one labelled copy per cached skeleton, so caches keyed on it still hit', () => {
    // Assembly runs on every partial publish and every Run over the same cached object.
    const skeleton = line()
    expect(withSomaAt(skeleton, [0, 0, 0], SOMA_SNAP_NM)).toBe(
      withSomaAt(skeleton, [0, 0, 0], SOMA_SNAP_NM),
    )
    expect(withSomaAt(skeleton, [5000, 0, 0], SOMA_SNAP_NM)).not.toBe(
      withSomaAt(skeleton, [0, 0, 0], SOMA_SNAP_NM),
    )
  })

  it('keeps the source’s own labels, and its own soma where it has one', () => {
    const coded = { ...line(), compartments: new Uint8Array([0, SWC_AXON, 0]) }
    expect([...withSomaAt(coded, [0, 0, 0], SOMA_SNAP_NM).compartments!]).toEqual([
      SWC_SOMA,
      SWC_AXON,
      0,
    ])
    const withSoma = { ...line(), compartments: new Uint8Array([0, SWC_SOMA, 0]) }
    expect(withSomaAt(withSoma, [0, 0, 0], SOMA_SNAP_NM)).toBe(withSoma)
  })
})
