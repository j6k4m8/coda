/**
 * The Connection Compartments node, driven through the real scheduler against the mock connectome.
 *
 * `connectionCompartments.test.ts` in `nodes/lib` pins the classification on hand-built geometry.
 * This covers what that file cannot: that the schema the card advertises is the one it builds
 * (invariant 3), that the fetch reaches every listed pair at both ends, and that every synapse of an
 * edge is counted somewhere — here, on skeletons labelled axon throughout, all of them axo-axonic,
 * summing to Connectivity's own weight.
 *
 * The split comes from a `test.*` node rather than Split Axon/Dendrite, whose split runs in Pyodide.
 * What it hands on is the mock's own skeletons, so each synapse end lands on the arbour the mock
 * generated it on.
 */

import { beforeAll, describe, expect, it, vi } from 'vitest'

import { addEdge, addNode, emptyGraph } from '../../core/graph'
import type { CodaGraph } from '../../core/graph'
import { inferGraph } from '../../core/inference'
import { registerNode } from '../../core/registry'
import { Scheduler } from '../../core/scheduler'
import { T, column, columnNames, tableSchema } from '../../core/types'
import { CODE_AXON, isPointsValue, isTableValue, makeTable } from '../../core/values'
import { idText } from '../../core/ids'
import { MockSource } from '../../data/mock/MockSource'
import { registerSource, requireSource } from '../../data/source'
import { requireDataset } from '../../nodes/lib/datasetParam'
import '../../nodes'
import { searchFor } from '../../test/findNeurons'
import { node } from '../../test/graph'
import { CONNECTION_CLASSES } from '../../nodes/lib/connectionCompartments'

const mock = new MockSource({ latencyMs: 0 })

beforeAll(() => {
  registerSource(mock)
  registerNode({
    type: 'test.edgeTable',
    label: 'Edge table',
    category: 'utility',
    cost: 'cheap',
    inputs: [],
    outputs: [{ id: 'out', label: 'Edges', type: T.table() }],
    params: [{ id: 'rows', kind: 'string', label: 'Rows', default: '{}' }],
    inferOutputs: (ctx) => {
      const rows = JSON.parse(String(ctx.params.rows)) as Record<string, string[]>
      return {
        out: T.table(tableSchema(...Object.keys(rows).map((name) => column(name, 'str')))),
      }
    },
    evaluate: (ctx) => {
      const rows = JSON.parse(String(ctx.params.rows)) as Record<string, string[]>
      const schema = tableSchema(...Object.keys(rows).map((name) => column(name, 'str')))
      return { out: makeTable(schema, rows) }
    },
  })
  registerNode({
    type: 'test.axonSkeletons',
    label: 'Axon skeletons',
    category: 'utility',
    cost: 'expensive',
    inputs: [
      { id: 'dataset', label: 'Dataset', type: T.dataset() },
      { id: 'edges', label: 'Edges', type: T.table() },
    ],
    outputs: [{ id: 'out', label: 'Skeletons', type: T.skeletons() }],
    inferOutputs: () => ({ out: T.skeletons() }),
    evaluate: async (ctx) => {
      const dataset = requireDataset(ctx.input('dataset'))
      const edges = ctx.input('edges')
      if (!isTableValue(edges)) throw new Error('no edges')
      const ids = new Set<string>()
      for (const name of ['preId', 'postId']) {
        for (const cell of edges.data[name] ?? []) {
          const id = idText(cell)
          if (id !== null) ids.add(id)
        }
      }
      const skeletons = await ctx.resolveSource(dataset.sourceId).fetchSkeletons!({
        datasetId: dataset.datasetId,
        neuronIds: [...ids],
      })
      return {
        out: {
          ...skeletons,
          items: skeletons.items.map((item) => ({
            ...item,
            split: new Uint8Array(item.parents.length).fill(CODE_AXON),
          })),
        },
      }
    },
  })
})

/** dataset → find → connectivity → connection compartments, with skeletons split all-axon. */
function pipeline(params: Record<string, unknown> = {}): CodaGraph {
  let g = emptyGraph('connection-compartments-test')
  g = addNode(g, node('ds', 'neuron.dataset', { dataset: 'optic-lobe-mini' }))
  g = addNode(
    g,
    node('find', 'neuron.findNeurons', searchFor({ type: 'LC4', status: 'Traced' })),
  )
  g = addNode(g, node('conn', 'neuron.connectivity', {}))
  g = addNode(g, node('skel', 'test.axonSkeletons'))
  g = addNode(g, node('cc', 'connectome:connectionCompartments', params))
  const wire = (source: string, handle: string, target: string, into: string) => {
    g = addEdge(g, { source, sourceHandle: handle, target, targetHandle: into })
  }
  wire('ds', 'dataset', 'find', 'dataset')
  wire('ds', 'dataset', 'conn', 'dataset')
  wire('find', 'neurons', 'conn', 'neurons')
  wire('ds', 'dataset', 'skel', 'dataset')
  wire('conn', 'connections', 'skel', 'edges')
  wire('ds', 'dataset', 'cc', 'dataset')
  wire('conn', 'connections', 'cc', 'edges')
  wire('skel', 'out', 'cc', 'skeletons')
  return g
}

async function run(params: Record<string, unknown> = {}) {
  const graph = pipeline(params)
  const sched = new Scheduler({ resolveSource: (id) => requireSource(id) })
  const summary = await sched.run(graph, { mode: 'full' })
  const edges = sched.output('cc', 'edges')
  const synapses = sched.output('cc', 'synapses')
  const input = sched.output('conn', 'connections')
  if (!isTableValue(edges) || !isTableValue(input) || !isPointsValue(synapses)) {
    throw new Error(`the node did not run: ${JSON.stringify(summary)}`)
  }
  const advertised = inferGraph(graph).nodes.cc?.outputs
  return { edges, synapses, input, advertised, warning: sched.warning('cc') ?? '' }
}

const schemaOf = (type: unknown) =>
  type && typeof type === 'object' && 'schema' in type
    ? columnNames((type as { schema: never }).schema)
    : undefined

/** The node fed a two-row table straight from a test node, so its columns are whatever is given. */
async function runOn(rows: Record<string, string[]>, params: Record<string, unknown> = {}) {
  let g = emptyGraph('connection-compartments-labels')
  g = addNode(g, node('ds', 'neuron.dataset', { dataset: 'optic-lobe-mini' }))
  g = addNode(g, node('edges', 'test.edgeTable', { rows: JSON.stringify(rows) }))
  g = addNode(g, node('skel', 'test.axonSkeletons'))
  g = addNode(g, node('cc', 'connectome:connectionCompartments', params))
  const wire = (source: string, handle: string, target: string, into: string) => {
    g = addEdge(g, { source, sourceHandle: handle, target, targetHandle: into })
  }
  wire('ds', 'dataset', 'skel', 'dataset')
  wire('edges', 'out', 'skel', 'edges')
  wire('ds', 'dataset', 'cc', 'dataset')
  wire('edges', 'out', 'cc', 'edges')
  wire('skel', 'out', 'cc', 'skeletons')
  const sched = new Scheduler({ resolveSource: (id) => requireSource(id) })
  await sched.run(g, { mode: 'full' })
  return { info: sched.info('cc'), issues: inferGraph(g).nodes.cc?.issues ?? [] }
}

describe('Connection Compartments', () => {
  it('refuses an edge list of labels rather than fetching nothing and blaming the weights', async () => {
    // Adjacency's per-neuron links: `type id` in one cell, which no source can look up.
    const { info } = await runOn(
      { source: ['LC4 1001', 'LC4 1002'], target: ['LPLC2 2001', 'LPLC2 2002'] },
      { pre: 'source', post: 'target' },
    )
    expect(info.state).toBe('error')
    expect(info.error).toMatch(/should hold neuron ids.*"LC4 1001" → "LPLC2 2001"/)
  })

  it('says before a Run when both pickers have fallen onto one column', async () => {
    const { issues } = await runOn({ source: ['1'], target: ['2'] })
    expect(issues.map((i) => (typeof i === 'string' ? i : i.message)).join(' ')).toMatch(
      /both read "source"/,
    )
  })

  it('builds the wide table it advertises, every synapse of an edge counted', async () => {
    const { edges, synapses, input, advertised } = await run()
    expect(columnNames(edges.schema)).toEqual(schemaOf(advertised?.edges))
    expect(columnNames(synapses.attributes.schema)).toEqual(schemaOf(advertised?.synapses))
    expect(edges.length).toBe(input.length)
    expect(edges.length).toBeGreaterThan(0)
    // All-axon skeletons: every synapse is axo-axonic, and an edge's total is its weight.
    expect(edges.data.weightAxonAxon).toEqual(input.data.weight)
    expect(edges.data.synapses).toEqual(input.data.weight)
    expect(new Set(edges.data.weightUnsplit)).toEqual(new Set([0]))
    const total = (input.data.weight ?? []).reduce<number>((a, w) => a + Number(w), 0)
    expect(synapses.attributes.length).toBe(total)
    expect(new Set(synapses.attributes.data.connection)).toEqual(new Set(['axo-axonic']))
  })

  it('builds the long table it advertises, six rows an edge summing to its weight', async () => {
    const { edges, input, advertised } = await run({ layout: 'long' })
    expect(columnNames(edges.schema)).toEqual(schemaOf(advertised?.edges))
    expect(edges.length).toBe(input.length * CONNECTION_CLASSES.length)
    for (let row = 0; row < input.length; row++) {
      const part = edges.data.weight!.slice(row * 6, row * 6 + 6).map(Number)
      expect(part.reduce((a, b) => a + b, 0)).toBe(Number(input.data.weight![row]))
    }
  })

  it('counts a synapse beyond the snap distance as unsplit and says why', async () => {
    const { edges, input, warning } = await run({ maxSnap: 0 })
    expect(edges.data.weightUnsplit).toEqual(input.data.weight)
    expect(warning).toMatch(/further than `Max snap distance`/)
  })

  it('re-classifies on a layout change without fetching again, and Clear Cache fetches', async () => {
    // `Layout` and `Max snap distance` shape the answer, not the question.
    const fetches = vi.spyOn(mock, 'fetchSynapsesBetween')
    const sched = new Scheduler({ resolveSource: (id) => requireSource(id) })
    let graph = pipeline()
    await sched.run(graph, { mode: 'full' })
    graph = {
      ...graph,
      nodes: graph.nodes.map((n) =>
        n.id === 'cc' ? { ...n, params: { ...n.params, layout: 'long', maxSnap: 5 } } : n,
      ),
    }
    await sched.run(graph, { mode: 'full' })
    expect(isTableValue(sched.output('cc', 'edges'))).toBe(true)
    expect(fetches).toHaveBeenCalledTimes(1)
    sched.clearNodeCache(graph, 'cc')
    await sched.run(graph, { mode: 'full' })
    expect(fetches).toHaveBeenCalledTimes(2)
    fetches.mockRestore()
  })

  it('replays what the source warned on a cached answer, and says how old it is', async () => {
    // The mock ignores `Min confidence` and says so: a re-classification from the held rows must
    // say it again, or the card reads clean on an answer the source qualified.
    const sched = new Scheduler({ resolveSource: (id) => requireSource(id) })
    let graph = pipeline({ minConfidence: 0.5 })
    await sched.run(graph, { mode: 'full' })
    expect(sched.warning('cc')).toMatch(/confidence/i)
    graph = {
      ...graph,
      nodes: graph.nodes.map((n) =>
        n.id === 'cc' ? { ...n, params: { ...n.params, layout: 'long' } } : n,
      ),
    }
    await sched.run(graph, { mode: 'full' })
    expect(sched.warning('cc')).toMatch(/confidence/i)
    expect(sched.fetchedAt('cc')).toBeTypeOf('number')
  })
})
