import { describe, expect, it } from 'vitest'

import { CODE_AXON, CODE_DENDRITE } from '../../core/values'
import { chain } from './__fixtures__/chain'
import { cube } from './__fixtures__/cube'
import { nodesInVolumes, pickVolumes, volumeLabels } from './splitByVolume'

// A 1.6 µm cube around x = 500 nm: the chain's first two nodes are inside, the last two out.
const ME = cube('ME(R)', [500, 0, 0], 800)
const LO = cube('LO(R)', [50_000, 0, 0], 800)

describe('nodesInVolumes', () => {
  it('marks the nodes inside any ROI', async () => {
    const { inside } = await nodesInVolumes([chain('1', 4, 1000)], [ME, LO])
    expect([...inside[0]!]).toEqual([1, 1, 0, 0])
  })

  it('also marks a node outside an ROI but within the margin of its surface', async () => {
    // The cube's face is at x = 1,300: the third node is 700 nm past it, the fourth 1,700.
    const { inside } = await nodesInVolumes(
      [chain('1', 4, 1000)],
      [ME, LO],
      {},
      undefined,
      1000,
    )
    expect([...inside[0]!]).toEqual([1, 1, 1, 0])
  })

  it('counts the neurons with no node inside any ROI', async () => {
    const { inside, untouched } = await nodesInVolumes(
      [chain('1', 4, 1000), chain('2', 4, 1000, 20_000)],
      [ME, LO],
    )
    expect(untouched).toBe(1)
    expect([...inside[1]!]).toEqual([0, 0, 0, 0])
  })
})

describe('volumeLabels', () => {
  it('gives a marked node the compartment the ROIs hold, and every other node the other', () => {
    const mask = [Uint8Array.from([1, 1, 0, 0])]
    expect([...volumeLabels(mask, 'axon')[0]!]).toEqual([
      CODE_AXON,
      CODE_AXON,
      CODE_DENDRITE,
      CODE_DENDRITE,
    ])
    expect([...volumeLabels(mask, 'dendrite')[0]!]).toEqual([
      CODE_DENDRITE,
      CODE_DENDRITE,
      CODE_AXON,
      CODE_AXON,
    ])
  })
})

describe('pickVolumes', () => {
  it('takes every wired volume where none is named', () => {
    expect(pickVolumes({ items: [ME, LO] }, []).items.map((v) => v.id)).toEqual([
      'ME(R)',
      'LO(R)',
    ])
  })

  it('takes the named ones, and says which names matched nothing', () => {
    const picked = pickVolumes({ items: [ME, LO] }, ['LO(R)', 'LOP(R)'])
    expect(picked.items.map((v) => v.id)).toEqual(['LO(R)'])
    expect(picked.unknown).toEqual(['LOP(R)'])
  })
})
