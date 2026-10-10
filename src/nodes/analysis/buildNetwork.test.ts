/**
 * Build Network's pickers against every node that emits Connectivity's edge list.
 *
 * `Source`, `Target` and `Weight` default to `preId`, `postId` and `weight`, which Connectivity,
 * Adjacency's Links and Synapses to Edges all emit, so wiring any of the three into Build Network
 * needs nothing picked. They defaulted to `''` once — "first column" for both ends — and every
 * link came out a self-loop until somebody set `Target`. Resolution only, against the inferred
 * schemas: what a picker lands on is decided before anything runs.
 */

import { describe, expect, it } from 'vitest'

import { addEdge, addNode, emptyGraph } from '../../core/graph'
import type { CodaGraph } from '../../core/graph'
import { inferGraph } from '../../core/inference'
import type { ParamValues } from '../../core/node'
import { defaultParams, makeInferContext } from '../../core/node'
import { requireNodeDef } from '../../core/registry'
import type { CodaType } from '../../core/types'
import { T, column, tableSchema } from '../../core/types'
import '../index'
import { node } from '../../test/graph'

const def = requireNodeDef('net.build')

/** The nodes given, chained by `wires` (`[from, port, to, port]`), with `from`'s `port` into Build Network. */
function into(
  types: Record<string, string>,
  [from, port]: [string, string],
  wires: [string, string, string, string][] = [],
): CodaGraph {
  let g = emptyGraph('build-network-defaults')
  for (const [id, type] of Object.entries(types)) g = addNode(g, node(id, type))
  g = addNode(g, node('net', 'net.build'))
  for (const [source, sourceHandle, target, targetHandle] of [
    ...wires,
    [from, port, 'net', 'edges'] as [string, string, string, string],
  ]) {
    g = addEdge(g, { source, sourceHandle, target, targetHandle })
  }
  return g
}

const PRODUCERS: Record<string, CodaGraph> = {
  Connectivity: into({ p: 'neuron.connectivity' }, ['p', 'connections']),
  'Adjacency Links': into({ p: 'neuron.adjacency' }, ['p', 'links']),
  'Synapses to Edges': into(
    { between: 'neuron.synapsesBetween', p: 'neuron.synapseEdges' },
    ['p', 'out'],
    [['between', 'points', 'p', 'in']],
  ),
}

function ctxFor(
  inputs: Record<string, CodaType | undefined>,
  params: ParamValues = defaultParams(def),
) {
  return makeInferContext(def, params, inputs)
}

const resolvedOn = (graph: CodaGraph, params?: ParamValues) => {
  const ctx = ctxFor(inferGraph(graph).nodes['net']!.inputs, params)
  return [ctx.column('source'), ctx.column('target'), ctx.column('weight')]
}

describe('Build Network on an edge list', () => {
  for (const [name, graph] of Object.entries(PRODUCERS)) {
    it(`resolves preId → postId, weighted, from ${name} with nothing picked`, () => {
      expect(resolvedOn(graph)).toEqual(['preId', 'postId', 'weight'])
    })
  }

  it('reads a stored empty Source/Target as the named defaults, and a stored empty Weight as off', () => {
    // What a card saved before the defaults were named holds.
    const stored = { source: '', target: '', weight: '' }
    expect(resolvedOn(PRODUCERS.Connectivity!, stored)).toEqual(['preId', 'postId', undefined])
  })

  it('falls back to the first column for both on a table without the names, which validate flags', () => {
    const ctx = ctxFor({
      edges: T.table(tableSchema(column('source', 'str'), column('target', 'str'))),
    })
    expect([ctx.column('source'), ctx.column('target')]).toEqual(['source', 'source'])
    expect(def.validate!(ctx)).toEqual([
      expect.stringMatching(/same column, so every link would be a self-loop/),
    ])
  })
})
