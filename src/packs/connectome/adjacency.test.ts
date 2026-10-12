/**
 * The Adjacency node's two outputs, and the chain the second one exists for.
 *
 * The reshape itself, zero-dropping rule included, is `connectivityOps.test.ts`'s. What is pinned
 * here is that the two ports describe one fetch and cannot disagree, and that `Links` really does
 * land on `Build Network` without
 * three column pickers being re-chosen by hand. That last one is the whole reason the port was
 * added: before it, a connection matrix could be drawn by the Heatmap and nothing else, because
 * no node in the registry turned a matrix back into an edge list.
 */

import { describe, expect, it } from 'vitest'

import { addEdge, addNode, deserializeGraph, emptyGraph } from '../../core/graph'
import type { CodaGraph } from '../../core/graph'
import { inferGraph } from '../../core/inference'
import { defaultParams } from '../../core/node'
import { defaultOutputPorts } from '../../core/ports'
import { requireNodeDef } from '../../core/registry'
import { Scheduler } from '../../core/scheduler'
import { columnNames, schemaOf } from '../../core/types'
import { isMatrixValue, isNetworkValue, isTableValue } from '../../core/values'
import { MockSource } from '../../data/mock/MockSource'
import type { DataSource } from '../../data/source'
import { MATRIX_LINKS_SCHEMA } from '../../nodes/lib/connectivityOps'
import '../../nodes'
import { searchFor } from '../../test/findNeurons'
import { node } from '../../test/graph'

const source: DataSource = new MockSource({ latencyMs: 0 })

/**
 * dataset → find(LC.\*) onto find(everything) → adjacency → net.build
 *
 * Two different populations rather than one onto itself, and the mock is why: `LC.*` onto `LC.*`
 * is a real 4 × 4 answer with no connections in it at all, so a self-onto-self pipeline would
 * assert nothing about the reshape. `LC.*` onto every Traced neuron is 4 × 34 with 11 links —
 * sparse enough that "the zeros were dropped" is a claim the row count can actually carry.
 */
function pipeline(params: Record<string, unknown> = {}): CodaGraph {
  let g = emptyGraph('adjacency-test')
  g = addNode(g, node('ds', 'neuron.dataset', { dataset: 'optic-lobe-mini' }))
  g = addNode(
    g,
    node('find', 'neuron.findNeurons', searchFor({ type: 'LC.*', status: 'Traced' })),
  )
  g = addNode(g, node('all', 'neuron.findNeurons', searchFor({ type: '', status: 'Traced' })))
  g = addNode(g, node('adj', 'neuron.adjacency', params))
  // Per neuron Build Network's named defaults resolve on Links; grouped by type the ids are
  // empty and the ends are the types, which is the one case somebody picks by hand.
  const byType = params.groupByType === true
  g = addNode(
    g,
    node('net', 'net.build', byType ? { source: 'preType', target: 'postType' } : {}),
  )
  for (const id of ['find', 'all', 'adj']) {
    g = addEdge(g, {
      source: 'ds',
      sourceHandle: 'dataset',
      target: id,
      targetHandle: 'dataset',
    })
  }
  g = addEdge(g, {
    source: 'find',
    sourceHandle: 'neurons',
    target: 'adj',
    targetHandle: 'sources',
  })
  g = addEdge(g, {
    source: 'all',
    sourceHandle: 'neurons',
    target: 'adj',
    targetHandle: 'targets',
  })
  g = addEdge(g, { source: 'adj', sourceHandle: 'links', target: 'net', targetHandle: 'edges' })
  return g
}

describe('neuron.adjacency — ports', () => {
  it('keeps Matrix first, so a dragged link starts at the picture', () => {
    // `neuron.roiConnectivity`'s order and its reason: the footer's summary then reads `N × M`.
    expect(defaultOutputPorts(requireNodeDef('neuron.adjacency')).map((p) => p.id)).toEqual([
      'matrix',
      'links',
    ])
  })

  it('types the Links table exactly with nothing wired', () => {
    // Each end's id and type, and a value, whatever the data — so unlike the wide half this shape
    // is a constant, and a picker downstream fills the moment the wire is drawn.
    const bare = addNode(emptyGraph('bare'), node('adj', 'neuron.adjacency'))
    expect(schemaOf(inferGraph(bare).nodes['adj']?.outputs['links'])).toEqual(
      MATRIX_LINKS_SCHEMA,
    )
  })

  it('names its columns as Connectivity does, so Build Network needs nothing picked', () => {
    // Connectivity's names, so an id-reading node takes either edge list with the same pickers —
    // and Build Network's named defaults resolve on it. The sweep over every producer is in
    // `buildNetwork.test.ts`.
    expect(columnNames(MATRIX_LINKS_SCHEMA)).toEqual([
      'preId',
      'preType',
      'postId',
      'postType',
      'weight',
    ])
  })
})

describe('neuron.adjacency — Group by type', () => {
  it('starts off on a new card, and stays grouped on a saved one written before that', () => {
    expect(defaultParams(requireNodeDef('neuron.adjacency')).groupByType).toBe(false)
    // A stored node without the key was made when grouping was the default.
    const stored = JSON.stringify({
      version: 1,
      nodes: [{ id: 'adj', type: 'neuron.adjacency', position: { x: 0, y: 0 }, params: {} }],
      edges: [],
    })
    expect(
      deserializeGraph(stored).graph?.nodes.find((n) => n.id === 'adj')?.params.groupByType,
    ).toBe(true)
  })
})

describe('neuron.adjacency — values', () => {
  it('emits the same connections twice over, as a matrix and as its non-zero cells', async () => {
    const scheduler = new Scheduler({ resolveSource: () => source })
    await scheduler.run(pipeline({ groupByType: true }), { mode: 'full' })
    expect(scheduler.info('adj').state).toBe('ok')

    const matrix = scheduler.output('adj', 'matrix')
    const links = scheduler.output('adj', 'links')
    if (!isMatrixValue(matrix) || !isTableValue(links)) {
      throw new Error('adjacency did not produce a matrix and a table')
    }

    // Derived from the matrix rather than fetched again, so the two cannot disagree about
    // lines, grouping or weights — `neuron.roiConnectivity`'s rule in the other direction.
    let nonZero = 0
    for (const value of matrix.values) if (value !== 0) nonZero++
    expect(links.length).toBe(nonZero)
    expect(links.length).toBeGreaterThan(0)
    // And strictly fewer rows than cells, or the zeros were not dropped and Build Network is
    // about to be handed a complete graph.
    expect(links.length).toBeLessThan(matrix.values.length)

    // Grouped by type, a line is a type: no id, and the label is the type.
    const labels = new Set([...matrix.rowLabels, ...matrix.colLabels])
    for (const cell of links.data.preType!) expect(labels.has(String(cell))).toBe(true)
    for (const cell of links.data.postType!) expect(labels.has(String(cell))).toBe(true)
    expect(new Set([...links.data.preId!, ...links.data.postId!])).toEqual(new Set([null]))
    for (const cell of links.data.weight!) expect(Number(cell)).not.toBe(0)
  })

  it('puts the neurons’ ids in the id columns per neuron, never their display labels', async () => {
    /*
     * The bug this replaced: per neuron, the links carried the matrix's labels, which on neuPrint
     * read `KCg-d 19102`, and Connection Compartments fed them looked up a neuron by that name
     * and counted zero synapses on every edge. Ids here are the ones the axes were asked for.
     */
    const scheduler = new Scheduler({ resolveSource: () => source })
    await scheduler.run(pipeline(), { mode: 'full' })
    const links = scheduler.output('adj', 'links')
    const sources = scheduler.output('find', 'neurons')
    const targets = scheduler.output('all', 'neurons')
    if (!isTableValue(links) || !isTableValue(sources) || !isTableValue(targets)) {
      throw new Error('adjacency did not run')
    }
    expect(links.length).toBeGreaterThan(0)
    const typeOf = new Map<unknown, unknown>()
    for (const table of [sources, targets]) {
      table.data.neuronId!.forEach((id, i) => typeOf.set(id, table.data.type![i]))
    }
    links.data.preId!.forEach((id, i) => {
      expect(sources.data.neuronId).toContain(id)
      expect(links.data.preType![i]).toBe(typeOf.get(id))
    })
    links.data.postId!.forEach((id, i) => {
      expect(targets.data.neuronId).toContain(id)
      expect(links.data.postType![i]).toBe(typeOf.get(id))
    })
  })

  it('reaches Build Network with no columns picked by hand, per neuron', async () => {
    const scheduler = new Scheduler({ resolveSource: () => source })
    await scheduler.run(pipeline(), { mode: 'full' })
    expect(scheduler.info('net').state).toBe('ok')

    const network = scheduler.output('net', 'network')
    const links = scheduler.output('adj', 'links')
    if (!isNetworkValue(network) || !isTableValue(links)) throw new Error('no network')
    expect(network.nodes.length).toBeGreaterThan(0)
    // One link per row: the reshape already merged whatever the matrix merged, so there are no
    // parallel rows left for `net.build` to aggregate.
    expect(network.edges.length).toBe(links.length)
  })
})

describe('neuron.adjacency — Reduce Matrix onto skeletons', () => {
  it('reaches Attach Attributes on neuronId, with nothing picked', async () => {
    /*
     * The curated "Reduce Matrix and Attach Attributes" chain. Reduce Matrix writes the ids the
     * matrix records as `neuronId`, so Attach Attributes' default matches without being told to
     * read `label` — which per neuron on neuPrint is `T4a 12345` and matches nothing.
     */
    let g = pipeline()
    g = addNode(g, node('reduce', 'core.reduceMatrix', { stats: ['sum'] }))
    g = addNode(g, node('skel', 'neuron.skeletons'))
    g = addNode(g, node('attach', 'neuron.attachAttributes'))
    for (const [source, sourceHandle, target, targetHandle] of [
      ['adj', 'matrix', 'reduce', 'in'],
      ['ds', 'dataset', 'skel', 'dataset'],
      ['find', 'neurons', 'skel', 'neurons'],
      ['skel', 'skeletons', 'attach', 'in'],
      ['reduce', 'out', 'attach', 'table'],
    ] as const) {
      g = addEdge(g, { source, sourceHandle, target, targetHandle })
    }
    const scheduler = new Scheduler({ resolveSource: () => source })
    await scheduler.run(g, { mode: 'full' })
    const reduced = scheduler.output('reduce', 'out')
    const attached = scheduler.output('attach', 'out')
    if (!isTableValue(reduced) || !attached || !('attributes' in attached)) {
      throw new Error('the chain did not run')
    }
    const find = scheduler.output('find', 'neurons')
    if (!isTableValue(find)) throw new Error('no seeds')
    expect(reduced.data.neuronId).toEqual(find.data.neuronId)
    const sums = attached.attributes.data.sum ?? []
    expect(sums.length).toBeGreaterThan(0)
    expect(sums.every((v) => typeof v === 'number')).toBe(true)
  })
})
