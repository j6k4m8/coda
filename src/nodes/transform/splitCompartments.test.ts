/**
 * Split Axon/Dendrite, as a node: what it promises at edit time against what a run hands over,
 * and the sentences it owes when the join between skeletons and synapses goes quietly wrong.
 *
 * The split itself is navis's and is checked by `pnpm probe:split`; the engine is mocked here, as
 * in `compartmentOps.test.ts`, and the inputs are the demo dataset's own skeletons and synapses so
 * the wiring the node is built for is the wiring tested.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { EvalContext, ParamValues } from '../../core/node'
import { defaultParams, makeInferContext } from '../../core/node'
import { requireNodeDef } from '../../core/registry'
import { T, column, tableSchema } from '../../core/types'
import type { PointsValue, SkeletonsValue, Value } from '../../core/values'
import { isPointsValue, isSkeletonsValue, isTableValue } from '../../core/values'
import { MockSource } from '../../data/mock/MockSource'
import { getConnectome } from '../../data/mock/generate'
import type * as TopologyBridge from '../../pyodide/topology'
import type { CleanSkeletonsResult } from '../../pyodide/skeletons'
import { writeCorrection } from '../lib/splitCorrections'
import { fakeSplit } from '../../test/splitStub'
import '../index'

vi.mock('../../pyodide/topology', async (importOriginal) => ({
  ...(await importOriginal<typeof TopologyBridge>()),
  runSplitCompartments: vi.fn(),
}))
const { runSplitCompartments } = await import('../../pyodide/topology')
const mockedRun = vi.mocked(runSplitCompartments)

// The heal behind corrections on a fragmented neuron; each test that reaches it says what it joins.
vi.mock('../../pyodide/skeletons', () => ({ runCleanSkeletons: vi.fn() }))
const { runCleanSkeletons } = await import('../../pyodide/skeletons')
const mockedHeal = vi.mocked(runCleanSkeletons)

const DEF = requireNodeDef('neuron.splitCompartments')
const source = new MockSource({ latencyMs: 0 })
const DATASET = 'optic-lobe-mini'

function neuronIds(count: number): string[] {
  return getConnectome(DATASET)!
    .neurons.filter((n) => n.type === 'LC4')
    .slice(0, count)
    .map((n) => String(n.neuronId))
}

async function inputs(
  count = 2,
): Promise<{ skeletons: SkeletonsValue; synapses: PointsValue }> {
  const ids = neuronIds(count)
  return {
    skeletons: await source.fetchSkeletons({ datasetId: DATASET, neuronIds: ids }),
    synapses: await source.fetchSynapses({ datasetId: DATASET, neuronIds: ids, unit: 'sites' }),
  }
}

async function run(
  values: Record<string, Value | undefined>,
  params: Partial<ParamValues> = {},
) {
  const warnings: string[] = []
  const ctx: EvalContext = {
    params: { ...defaultParams(DEF), ...params } as ParamValues,
    refresh: false,
    reportFetched: () => undefined,
    warn: (message: string) => warnings.push(message),
    publish: () => undefined,
    input: (port) => values[port],
    inputKey: (port) => (values[port] ? `${port}-key` : undefined),
    column: () => undefined,
    columns: () => [],
    inputPorts: () => [],
    outputPorts: () => [],
    resolveSource: () => source,
    signal: new AbortController().signal,
    progress: () => {},
  }
  return { outputs: await DEF.evaluate(ctx), warnings }
}

beforeEach(() => {
  mockedRun.mockReset()
  mockedRun.mockImplementation(fakeSplit)
})

describe('neuron.splitCompartments', () => {
  it('is cheap, since everything it reads arrives on a wire', () => {
    expect(DEF.cost).toBe('cheap')
  })

  it('runs what it promised at edit time: every output’s schema, before anything ran', async () => {
    const { skeletons, synapses } = await inputs()
    const inferred = DEF.inferOutputs!(
      makeInferContext(
        DEF,
        {},
        {
          skeletons: T.skeletons(skeletons.attributes.schema),
          synapses: T.points(synapses.attributes.schema),
        },
      ),
    )
    const { outputs } = await run({ skeletons, synapses })
    const { out, labelled, summary } = outputs
    if (!isSkeletonsValue(out) || !isPointsValue(labelled) || !isTableValue(summary)) {
      throw new Error('expected skeletons, points and a table')
    }

    expect(inferred['out']).toEqual(T.skeletons(out.attributes.schema))
    expect(inferred['labelled']).toEqual(T.points(labelled.attributes.schema))
    expect(inferred['summary']).toEqual(T.neurons(summary.schema))
  })

  it('labels every arbour it split and every synapse of those neurons', async () => {
    const { skeletons, synapses } = await inputs()
    const { outputs, warnings } = await run({ skeletons, synapses })
    const out = outputs['out'] as SkeletonsValue
    const labelled = outputs['labelled'] as PointsValue

    for (const item of out.items) expect(item.split?.length).toBe(item.parents.length)
    expect(labelled.attributes.data['compartment']!.every((c) => c !== null)).toBe(true)
    expect(warnings).toEqual([])
  })

  it('writes synapse flow onto the arbours only when asked, as a fraction of each peak', async () => {
    mockedRun.mockImplementation((request) =>
      fakeSplit(request).then((r) =>
        request.flow
          ? {
              ...r,
              flow: Float32Array.from({ length: request.parents.length }, (_, i) => i % 5),
            }
          : r,
      ),
    )
    const { skeletons, synapses } = await inputs()
    const plain = (await run({ skeletons, synapses })).outputs['out'] as SkeletonsValue
    expect(plain.items.every((item) => item.nodeValues === undefined)).toBe(true)

    const out = (await run({ skeletons, synapses }, { flow: true })).outputs[
      'out'
    ] as SkeletonsValue
    for (const item of out.items) {
      const flow = item.nodeValues!['flow']!
      expect(flow.length).toBe(item.parents.length)
      expect(Math.max(...flow)).toBe(1)
    }
  })

  it('applies hand corrections without asking Python again, and counts one that lands nowhere', async () => {
    const { skeletons, synapses } = await inputs()
    const first = skeletons.items[0]!
    const p = first.positions
    const correction = (x: number, y: number, z: number) =>
      writeCorrection({ neuron: first.id, at: [x, y, z], scope: 'distal', to: 'linker' })

    await run({ skeletons, synapses })
    const calls = mockedRun.mock.calls.length
    const { outputs, warnings } = await run(
      { skeletons, synapses },
      // The root, outward: the whole first neuron becomes linker. And one far from everything.
      { corrections: [correction(p[0]!, p[1]!, p[2]!), correction(9e9, 9e9, 9e9)] },
    )
    // The automatic split was remembered; only the corrections pass ran.
    expect(mockedRun.mock.calls.length).toBe(calls)

    const out = outputs['out'] as SkeletonsValue
    expect(out.items[0]!.split!.every((code) => code === 3)).toBe(true)
    expect(out.items[1]!.split!.some((code) => code !== 3)).toBe(true)
    const summary = outputs['summary']
    if (!isTableValue(summary)) throw new Error('expected a table')
    expect(summary.data['corrections']).toEqual([1, 0])
    expect(warnings.join(' ')).toMatch(/1 of 2 hand corrections no longer land/)
  })

  it('applies a correction across the pieces of a neuron healed to be split', async () => {
    // The first neuron cut in two halfway along its node list. A split healed it, so a correction
    // from its root outward must reach the far piece too — measured on the delivered parents it
    // reached only the piece the root is on.
    const { skeletons, synapses } = await inputs()
    const whole = skeletons.items[0]!
    const cut = Int32Array.from(whole.parents)
    const at = Math.floor(cut.length / 2)
    cut[at] = -1
    const pieces = {
      ...skeletons,
      items: [{ ...whole, parents: cut }, ...skeletons.items.slice(1)],
    }
    mockedHeal.mockResolvedValue({
      parents: Int32Array.from(whole.parents),
    } as CleanSkeletonsResult)
    const p = whole.positions
    const params = {
      heal: true,
      corrections: [
        writeCorrection({
          neuron: whole.id,
          at: [p[0]!, p[1]!, p[2]!],
          scope: 'distal',
          to: 'linker',
        }),
      ],
    }
    // Twice: the second Run is answered from the join the first one made — the editor's, in use —
    // and must still correct the joined tree rather than the pieces.
    for (const pass of [1, 2]) {
      const { outputs, warnings } = await run({ skeletons: pieces, synapses }, params)
      expect(mockedHeal, `pass ${pass}`).toHaveBeenCalledTimes(1)
      const split = (outputs['out'] as SkeletonsValue).items[0]!.split!
      expect(split[at], `pass ${pass}`).toBe(3)
      expect(split.every((code) => code === 3)).toBe(true)
      expect(warnings.join(' ')).not.toMatch(/no longer land/)
      // The geometry downstream is the one delivered: healing is for the arbour, not the output.
      expect((outputs['out'] as SkeletonsValue).items[0]!.parents).toBe(cut)
    }
  })

  it('says how many neurons could not be split, and points a fragmented one at Heal', async () => {
    mockedRun.mockImplementation((request) =>
      fakeSplit(request).then((r) => ({ ...r, status: Int32Array.from([1, 0]) })),
    )
    const { skeletons, synapses } = await inputs()
    const { outputs, warnings } = await run({ skeletons, synapses })
    expect((outputs['out'] as SkeletonsValue).items[0]!.split).toBeUndefined()
    expect(warnings.join(' ')).toMatch(/1 of 2 neurons could not be split.*Heal fragmented/)
  })

  it('counts synapses of neurons that have no skeleton here, which a mis-wired cloud is', async () => {
    const all = await inputs(2)
    const one = await inputs(1)
    const { warnings } = await run({ skeletons: one.skeletons, synapses: all.synapses })
    expect(warnings.join(' ')).toMatch(/synapses belong to neurons with no skeleton here/)
  })

  it('refuses synapses in another unit, which would snap to whichever node was nearest', async () => {
    const { skeletons, synapses } = await inputs(1)
    await expect(
      run({ skeletons, synapses: { ...synapses, units: 'voxels' } }),
    ).rejects.toThrow(/synapses are in voxels/)
  })

  it('asks for a polarity column before anything runs', () => {
    const ctx = makeInferContext(
      DEF,
      {},
      {
        synapses: T.points(tableSchema(column('neuronId', 'str'), column('x', 'f64'))),
      },
    )
    expect(DEF.validate!(ctx).join(' ')).toMatch(/`polarity`/)
  })
})
