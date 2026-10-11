/**
 * Classifying an edge list's synapses by the compartment at each end.
 *
 * Two hand-built neurons, drawn so that each end's answer can be read off the picture (nm):
 *
 *     1 (presynaptic, all axon)   ●────●────●      y = -500
 *     2's axon branch             ●────●           y = -400
 *     2's dendrite                ●────●────●      y =    0
 *                                 x=0  1000 2000
 *
 * Neuron 2's axon runs 100 nm from neuron 1. So a synapse whose T-bar is on 1 at (1000, -450) and
 * whose PSD is on 2's dendrite at (1000, -50) is axo-dendritic, and **read from the T-bar alone it
 * is axo-axonic**, 2's nearest node to that point being on its axon. The ids are digits because an
 * edge only counts when both ends are neuron ids (`readEdgePairs`). That pair of answers is the
 * whole reason the seam carries both ends, and the test that pins it.
 */

import { describe, expect, it } from 'vitest'

import { column, tableSchema } from '../../core/types'
import type {
  CellValue,
  PointsValue,
  SkeletonGeometry,
  SkeletonsValue,
} from '../../core/values'
import {
  CODE_AXON,
  CODE_DENDRITE,
  CODE_LINKER,
  EMPTY_BOUNDS,
  makeTable,
} from '../../core/values'
import { synapsesBetweenTable } from '../../data/source'
import {
  CLASSIFIED_SYNAPSES_SCHEMA,
  CONNECTION_CLASSES,
  classifiedSynapsesTable,
  classifySynapses,
  connectionClass,
  connectionEdgesSchema,
  connectionEdgesTable,
  readEdgePairs,
  rowTallies,
  rowsOnPairs,
  weightMismatch,
} from './connectionCompartments'

const A = CODE_AXON
const D = CODE_DENDRITE
const L = CODE_LINKER

function skeleton(
  id: string,
  points: number[][],
  parents: number[],
  split?: number[],
): SkeletonGeometry {
  return {
    id,
    positions: Float32Array.from(points.flat()),
    radii: new Float32Array(points.length).fill(100),
    parents: Int32Array.from(parents),
    ...(split ? { split: Uint8Array.from(split) } : {}),
  }
}

const neuronA = () =>
  skeleton(
    '1',
    [
      [0, -500, 0],
      [1000, -500, 0],
      [2000, -500, 0],
    ],
    [-1, 0, 1],
    [A, A, A],
  )
const neuronB = () =>
  skeleton(
    '2',
    [
      [0, 0, 0],
      [1000, 0, 0],
      [2000, 0, 0],
      [0, -400, 0],
      [1000, -400, 0],
    ],
    [-1, 0, 1, 0, 3],
    [D, D, D, A, A],
  )

function set(...items: SkeletonGeometry[]): SkeletonsValue {
  return { kind: 'skeletons', items, bounds: EMPTY_BOUNDS, units: 'nm' } as SkeletonsValue
}

interface Row {
  pre: string
  post: string
  at: [number, number, number]
  partner?: [number, number, number]
}

/** A synapses-between cloud; with every row carrying `partner`, a both-ends one. */
function cloud(rows: Row[]): PointsValue {
  const both = rows.every((r) => r.partner)
  const partner = both
    ? ([0, 1, 2].map((axis) => rows.map((r) => r.partner![axis]!)) as [
        CellValue[],
        CellValue[],
        CellValue[],
      ])
    : undefined
  return {
    kind: 'points',
    positions: Float32Array.from(rows.flatMap((r) => r.at)),
    attributes: synapsesBetweenTable(
      {
        neuronId: rows.map((r) => r.pre),
        type: rows.map(() => null),
        partnerId: rows.map((r) => r.post),
        partnerType: rows.map(() => null),
        polarity: rows.map(() => 'pre'),
        confidence: rows.map(() => null),
      },
      partner,
    ),
    bounds: EMPTY_BOUNDS,
    units: 'nm',
  }
}

const names = (indices: readonly number[]) => indices.map((k) => CONNECTION_CLASSES[k]!.name)

/** The pairs to keep, as `classifySynapses` takes them: `'1>2'` is neuron 1 onto neuron 2. */
function wanted(...pairs: string[]): Map<string, Map<string, true>> {
  const out = new Map<string, Map<string, true>>()
  for (const pair of pairs) {
    const [pre, post] = pair.split('>') as [string, string]
    if (!out.has(pre)) out.set(pre, new Map())
    out.get(pre)!.set(post, true)
  }
  return out
}

describe('connectionClass', () => {
  it('names the four classes by the code at each end, presynaptic first', () => {
    expect(names([connectionClass(A, A), connectionClass(A, D)])).toEqual([
      'axo-axonic',
      'axo-dendritic',
    ])
    expect(names([connectionClass(D, A), connectionClass(D, D)])).toEqual([
      'dendro-axonic',
      'dendro-dendritic',
    ])
  })

  it('puts either end on the linker in linker, and an unread end in unsplit, which wins', () => {
    expect(names([connectionClass(L, D), connectionClass(A, L)])).toEqual(['linker', 'linker'])
    expect(names([connectionClass(undefined, D), connectionClass(L, 0)])).toEqual([
      'unsplit',
      'unsplit',
    ])
  })
})

describe('classifySynapses', () => {
  const tbar: [number, number, number] = [1000, -450, 0]
  const psd: [number, number, number] = [1000, -50, 0]

  it('reads each end on its own neuron at its own site', async () => {
    const result = await classifySynapses(
      cloud([{ pre: '1', post: '2', at: tbar, partner: psd }]),
      set(neuronA(), neuronB()),
      10_000,
    )
    expect(result.columns.connection).toEqual(['axo-dendritic'])
    expect(result.columns.preCompartment).toEqual(['axon'])
    expect(result.columns.postCompartment).toEqual(['dendrite'])
    expect(result.oneCoordinate).toBe(0)
  })

  it('reads both ends at the one position a single-coordinate cloud has, and counts them', async () => {
    // The same synapse without its PSD: b's nearest node to the T-bar is on b's axon.
    const result = await classifySynapses(
      cloud([{ pre: '1', post: '2', at: tbar }]),
      set(neuronA(), neuronB()),
      10_000,
    )
    expect(result.columns.connection).toEqual(['axo-axonic'])
    expect(result.oneCoordinate).toBe(1)
  })

  it('keeps only the listed pairs, and tallies per pair', async () => {
    const rows: Row[] = [
      { pre: '1', post: '2', at: tbar, partner: psd },
      { pre: '2', post: '1', at: [1000, -400, 0], partner: [1000, -500, 0] },
      { pre: '1', post: '2', at: tbar, partner: [0, -400, 0] },
    ]
    // The pair filter is `rowsOnPairs`', run once before classifying, as the node does.
    const result = await classifySynapses(
      rowsOnPairs(cloud(rows), wanted('1>2')),
      set(neuronA(), neuronB()),
      10_000,
    )
    expect(result.rows).toBe(2)
    expect([...result.counts.get('1')!.get('2')!]).toEqual([1, 1, 0, 0, 0, 0])
    expect(result.counts.has('2')).toBe(false)
  })

  it('counts an end it cannot read as unsplit, by reason, rather than dropping the synapse', async () => {
    const unsplitB = { ...neuronB(), split: undefined }
    const rows: Row[] = [
      { pre: '1', post: '3', at: tbar, partner: psd }, // no skeleton for c
      { pre: '1', post: '2', at: tbar, partner: psd }, // b carries no split
      { pre: '1', post: '1', at: tbar, partner: [1000, 50_000, 0] }, // 50 µm off a
    ]
    const result = await classifySynapses(cloud(rows), set(neuronA(), unsplitB), 10_000)
    expect(result.columns.connection).toEqual(['unsplit', 'unsplit', 'unsplit'])
    expect(result.reasons).toEqual({ noSkeleton: 1, noSplit: 1, tooFar: 1 })
    expect(result.columns.preCompartment).toEqual(['axon', 'axon', 'axon'])
    expect(result.columns.postCompartment).toEqual([null, null, null])
  })

  it('writes the classes onto the rows on listed pairs, in the schema it advertises', async () => {
    const synapses = rowsOnPairs(
      cloud([
        { pre: '2', post: '1', at: [0, 0, 0], partner: [0, -500, 0] },
        { pre: '1', post: '2', at: tbar, partner: psd },
      ]),
      wanted('1>2'),
    )
    const out = classifiedSynapsesTable(
      synapses,
      await classifySynapses(synapses, set(neuronA(), neuronB()), 10_000),
    )
    expect(out.attributes.schema).toEqual(CLASSIFIED_SYNAPSES_SCHEMA)
    expect([...out.positions]).toEqual(tbar)
    expect(out.attributes.data.connection).toEqual(['axo-dendritic'])
    expect(out.attributes.data.partnerY).toEqual([-50])
  })

  it('advertises the both-ends schema whatever arrives, with a one-ended row reading null', async () => {
    const synapses = cloud([{ pre: '1', post: '2', at: tbar }])
    const out = classifiedSynapsesTable(
      synapses,
      await classifySynapses(synapses, set(neuronA(), neuronB()), 10_000),
    )
    expect(out.attributes.schema).toEqual(CLASSIFIED_SYNAPSES_SCHEMA)
    expect(out.attributes.data.partnerX).toEqual([null])
  })
})

describe('rowsOnPairs', () => {
  it('keeps the rows on a listed pair, in order, and the frame', () => {
    const synapses = cloud([
      { pre: '1', post: '2', at: [0, 0, 0] },
      { pre: '2', post: '1', at: [1, 1, 1] },
      { pre: '1', post: '2', at: [2, 2, 2] },
    ])
    const kept = rowsOnPairs(synapses, wanted('1>2'))
    expect([...kept.positions]).toEqual([0, 0, 0, 2, 2, 2])
    expect(kept.attributes.data.neuronId).toEqual(['1', '1'])
    expect(kept.units).toBe('nm')
  })
})

describe('readEdgePairs', () => {
  it('names a pair only when both ends are neuron ids, and counts the rows that hold labels', () => {
    // Adjacency's old per-neuron labels: text, and no id a source could match.
    const table = makeTable(tableSchema(column('pre', 'str'), column('post', 'str')), {
      pre: ['10013', 'KCg-d 19102', null, '10013'],
      post: ['19102', 'MBON01 10013', '10013', '19102'],
    })
    const read = readEdgePairs(table, 'pre', 'post')
    expect(read.ends).toEqual([['10013', '19102'], null, null, ['10013', '19102']])
    expect([read.pairs, read.repeated, read.notIds]).toEqual([1, 1, 1])
    expect(read.example).toBe('"KCg-d 19102" → "MBON01 10013"')
    expect([read.pres, read.posts]).toEqual([['10013'], ['19102']])
  })
})

describe('the edge table', () => {
  const edges = makeTable(
    tableSchema(
      column('preId', 'str'),
      column('postId', 'str'),
      column('weight', 'i64'),
      column('roi', 'str'),
    ),
    {
      preId: ['1', '2', '1', null],
      postId: ['2', '1', '2', '1'],
      weight: [3, 1, 3, 2],
      roi: ['LO', 'ME', 'LO', 'ME'],
    },
  )
  const counts = new Map([['1', new Map([['2', Uint32Array.from([1, 2, 0, 0, 0, 0])]])]])
  const { ends } = readEdgePairs(edges, 'preId', 'postId')

  it('wide: every row in order, a column per class and the total, the input weight untouched', () => {
    const table = connectionEdgesTable(edges, rowTallies(ends, counts), 'wide')
    expect(table.schema).toEqual(connectionEdgesSchema(edges.schema, 'wide'))
    expect(table.data.weightAxonAxon).toEqual([1, 0, 1, null])
    expect(table.data.weightAxonDendrite).toEqual([2, 0, 2, null])
    expect(table.data.synapses).toEqual([3, 0, 3, null])
    expect(table.data.weight).toEqual([3, 1, 3, 2])
  })

  it('long: six rows per edge, weight replaced in its slot and connection after it', () => {
    const table = connectionEdgesTable(edges, rowTallies(ends, counts), 'long')
    expect(table.schema).toEqual(connectionEdgesSchema(edges.schema, 'long'))
    expect(table.schema.columns.map((c) => c.name)).toEqual([
      'preId',
      'postId',
      'weight',
      'connection',
      'roi',
    ])
    expect(table.length).toBe(24)
    expect(table.data.connection!.slice(0, 6)).toEqual(CONNECTION_CLASSES.map((c) => c.name))
    expect(table.data.roi!.slice(6, 12)).toEqual(new Array(6).fill('ME'))
  })

  it('long sums to wide, edge by edge', () => {
    const wide = connectionEdgesTable(edges, rowTallies(ends, counts), 'wide')
    const long = connectionEdgesTable(edges, rowTallies(ends, counts), 'long')
    for (let row = 0; row < edges.length; row++) {
      const part = long.data.weight!.slice(row * 6, row * 6 + 6)
      const total = part.every((v) => v === null)
        ? null
        : part.reduce<number>((a, v) => a + Number(v), 0)
      expect(total).toBe(wide.data.synapses![row])
    }
  })

  it('compares each edge’s weight with what was counted, both ways', () => {
    // Row 1 counted 3 of 3; row 2 found nothing of its 1; row 3 is the repeat; row 4 names no pair.
    expect(weightMismatch(edges, rowTallies(ends, counts))).toEqual({ fewer: 1, more: 0 })
  })

  it('appends weight and connection to a table without a weight', () => {
    const bare = tableSchema(column('preId', 'str'), column('postId', 'str'))
    expect(connectionEdgesSchema(bare, 'long').columns.map((c) => c.name)).toEqual([
      'preId',
      'postId',
      'weight',
      'connection',
    ])
  })
})
