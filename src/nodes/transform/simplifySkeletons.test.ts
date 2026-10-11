/**
 * Simplify Skeletons: the flattening, the gather back through `keep`, and the node.
 *
 * The Python is mocked, `cleanSkeletons.test.ts`' arrangement; `scripts/probe-skeletons.mjs` runs
 * the real `coda_simplify_skeletons` against the real wheel, and owns which nodes fastcore keeps.
 * This file owns what the JavaScript does with the answer — above all that everything riding on a
 * node is gathered through the *same* `keep`, which is the one way a split, a label or a radius
 * could end up on somebody else's node without anything failing.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { defaultParams, makeInferContext, offeredOptions } from '../../core/node'
import type { ParamValues } from '../../core/node'
import { requireNodeDef } from '../../core/registry'
import { column, tableSchema } from '../../core/types'
import type { SkeletonsValue } from '../../core/values'
import { makeTable } from '../../core/values'
import { MockSource } from '../../data/mock/MockSource'
import { NM_PER_UM } from '../../data/units'
import type { SimplifySkeletonsRequest, SimplifySkeletonsResult } from '../../pyodide/skeletons'
import type { SkeletonSimplifyParams } from '../lib/simplifyOps'
import {
  isSimplifyNoOp,
  simplifyRequestFrom,
  simplifyUsesDistance,
  skeletonsFromKeep,
} from '../lib/simplifyOps'
import '../index'
import { gateNote } from '../../assistant/catalogue'

vi.mock('../../pyodide/skeletons', () => ({ runSimplifySkeletons: vi.fn() }))
const { runSimplifySkeletons } = await import('../../pyodide/skeletons')
const mockedRun = vi.mocked(runSimplifySkeletons)

const OFF: SkeletonSimplifyParams = {
  prune: 0,
  repeat: false,
  method: 'none',
  factor: 2,
  tolerance: 0.1,
  area: 0.01,
}

/**
 * A Y carrying everything that rides on a node, a two-node neuron, and an empty one.
 *
 * Node 0 root, 1 the branch point, 2 a short twig, 3–4 the long arm. Every per-node array is
 * different at every node, so a gather through the wrong index cannot come out right by accident.
 */
function fixture(): SkeletonsValue {
  return {
    kind: 'skeletons',
    items: [
      {
        id: '11',
        positions: new Float32Array([
          0, 0, 0, 1000, 0, 0, 1000, 500, 0, 2000, 0, 0, 3000, 0, 0,
        ]),
        radii: new Float32Array([10, 20, 30, 40, 50]),
        parents: new Int32Array([-1, 0, 1, 1, 3]),
        compartments: new Uint8Array([1, 3, 3, 2, 2]),
        split: new Uint8Array([0, 1, 1, 2, 2]),
        nodeValues: { flow: new Float32Array([0, 0.5, 0.1, 0.9, 0.2]) },
        tags: { ends: Int32Array.from([2, 4]), soma: Int32Array.from([0]) },
      },
      {
        id: '22',
        positions: new Float32Array([5000, 0, 0, 6000, 0, 0]),
        radii: new Float32Array([60, 70]),
        parents: new Int32Array([-1, 0]),
      },
      {
        id: '33',
        positions: new Float32Array([]),
        radii: new Float32Array([]),
        parents: new Int32Array([]),
      },
    ],
    attributes: makeTable(tableSchema(column('neuronId', 'str'), column('type', 'str')), {
      neuronId: ['11', '22', '33'],
      type: ['LC4', 'LC6', 'LC10'],
    }),
    bounds: { min: [0, 0, 0], max: [6000, 500, 0] },
    units: 'nm',
  }
}

/** The Y with its short twig (node 2) pruned; the other two untouched. */
const PRUNED: SimplifySkeletonsResult = {
  keep: Int32Array.from([0, 1, 3, 4, 0, 1]),
  parents: Int32Array.from([-1, 0, 1, 2, -1, 0]),
  offsets: Int32Array.from([0, 4, 6, 6]),
}

/** Every node kept — what the engine returns when nothing qualified. */
function keepAll(request: SimplifySkeletonsRequest): SimplifySkeletonsResult {
  const keep = new Int32Array(request.parents.length)
  for (let n = 0; n + 1 < request.offsets.length; n++) {
    for (let i = request.offsets[n]!; i < request.offsets[n + 1]!; i++) {
      keep[i] = i - request.offsets[n]!
    }
  }
  return { keep, parents: request.parents.slice(), offsets: request.offsets.slice() }
}

describe('simplifyOps — is anything switched on', () => {
  it('reads the untouched card as a pass-through', () => {
    expect(isSimplifyNoOp(OFF)).toBe(true)
    expect(isSimplifyNoOp({ ...OFF, method: 'every', factor: 1 })).toBe(true)
    expect(isSimplifyNoOp({ ...OFF, method: 'vw', area: 0 })).toBe(true)
  })

  it('reads each control that removes something', () => {
    expect(isSimplifyNoOp({ ...OFF, prune: 1 })).toBe(false)
    expect(isSimplifyNoOp({ ...OFF, method: 'every', factor: 3 })).toBe(false)
    expect(isSimplifyNoOp({ ...OFF, method: 'vw', area: 0.5 })).toBe(false)
    // A zero tolerance still drops exactly collinear nodes, fastcore's own reading.
    expect(isSimplifyNoOp({ ...OFF, method: 'rdp', tolerance: 0 })).toBe(false)
  })
})

describe('simplifyOps — units', () => {
  it('counts a length, a tolerance and an area as distances', () => {
    expect(simplifyUsesDistance({ ...OFF, prune: 1 })).toBe(true)
    expect(simplifyUsesDistance({ ...OFF, method: 'rdp', tolerance: 0.5 })).toBe(true)
    expect(simplifyUsesDistance({ ...OFF, method: 'vw', area: 0.5 })).toBe(true)
  })

  it('does not count hops, or "exactly collinear", as one', () => {
    expect(simplifyUsesDistance({ ...OFF, method: 'every', factor: 4 })).toBe(false)
    expect(simplifyUsesDistance({ ...OFF, method: 'rdp', tolerance: 0 })).toBe(false)
  })
})

describe('simplifyOps — the flattening', () => {
  it('keeps parents neuron-local and sends the soma', () => {
    const request = simplifyRequestFrom(fixture(), OFF)
    expect(Array.from(request.offsets)).toEqual([0, 5, 7, 7])
    expect(Array.from(request.parents)).toEqual([-1, 0, 1, 1, 3, -1, 0])
    // The `soma` tag on node 0 is what `widestSomaNode` reads off a CATMAID skeleton.
    expect(request.somas[0]).toBe(0)
    expect(request.somas[1]).toBe(-1)
  })

  it('converts the controls to nanometres, the area squared', () => {
    const request = simplifyRequestFrom(fixture(), {
      ...OFF,
      prune: 3,
      tolerance: 0.5,
      area: 0.02,
    })
    expect(request.prune).toBe(3 * NM_PER_UM)
    expect(request.tolerance).toBe(0.5 * NM_PER_UM)
    // The one that would be wrong by a factor of a thousand and still draw.
    expect(request.area).toBeCloseTo(0.02 * NM_PER_UM * NM_PER_UM)
    expect(request.points[3]).toBe(1000)
  })

  it('builds its own buffers rather than handing over the upstream ones', () => {
    const value = fixture()
    const request = simplifyRequestFrom(value, OFF)
    expect(request.points).not.toBe(value.items[0]!.positions)
    expect(request.parents).not.toBe(value.items[0]!.parents)
  })
})

describe('simplifyOps — the gather', () => {
  it('gathers every per-node array through the same rows', () => {
    const out = skeletonsFromKeep(fixture(), PRUNED).items[0]!
    expect(Array.from(out.parents)).toEqual([-1, 0, 1, 2])
    expect(Array.from(out.positions)).toEqual([0, 0, 0, 1000, 0, 0, 2000, 0, 0, 3000, 0, 0])
    expect(Array.from(out.radii)).toEqual([10, 20, 40, 50])
    expect(Array.from(out.compartments!)).toEqual([1, 3, 2, 2])
    expect(Array.from(out.split!)).toEqual([0, 1, 2, 2])
    expect(Array.from(out.nodeValues!['flow']!)).toEqual(
      Array.from(Float32Array.from([0, 0.5, 0.9, 0.2])),
    )
  })

  it('keeps a tag on its node, and snaps one on a removed node to the nearest survivor', () => {
    const out = skeletonsFromKeep(fixture(), PRUNED).items[0]!
    // `ends` was on the pruned twig's tip (2, at 1000/500) and on node 4 (now row 3). The tip
    // snaps to the branch point, row 1, the survivor nearest to where it was.
    expect(Array.from(out.tags!['ends']!).sort()).toEqual([1, 3])
    expect(Array.from(out.tags!['soma']!)).toEqual([0])
  })

  it('hands back an untouched neuron as itself', () => {
    const value = fixture()
    const out = skeletonsFromKeep(value, PRUNED)
    expect(out.items[1]).toBe(value.items[1])
    expect(out.items[2]).toBe(value.items[2])
    expect(out.attributes).toBe(value.attributes)
    expect(out.units).toBe('nm')
  })

  it('recomputes the bounds over what is left', () => {
    const out = skeletonsFromKeep(fixture(), PRUNED)
    // The twig was the only node off the x axis.
    expect(out.bounds.max[1]).toBe(0)
  })

  it('refuses a result with a different number of neurons in it', () => {
    expect(() =>
      skeletonsFromKeep(fixture(), { ...PRUNED, offsets: PRUNED.offsets.slice(0, 3) }),
    ).toThrow(/no longer line up/)
  })
})

// ---------------------------------------------------------------------------
// The node
// ---------------------------------------------------------------------------

/** The node's own `evaluate`, with a warn spy. */
async function simplify(value: SkeletonsValue, params: Partial<SkeletonSimplifyParams>) {
  const def = requireNodeDef('neuron.simplifySkeletons')
  const warnings: string[] = []
  const outputs = await def.evaluate({
    params: { ...defaultParams(def), ...params } as ParamValues,
    refresh: false,
    reportFetched: () => undefined,
    warn: (message: string) => warnings.push(message),
    publish: () => undefined,
    input: () => value,
    inputKey: () => 'in-key',
    column: () => undefined,
    columns: () => [],
    inputPorts: () => [],
    outputPorts: () => [],
    resolveSource: () => new MockSource(),
    signal: new AbortController().signal,
    progress: () => {},
  })
  return { out: outputs['out'] as SkeletonsValue, warnings }
}

const validate = (params: ParamValues) => {
  const def = requireNodeDef('neuron.simplifySkeletons')
  return (
    def.validate?.(makeInferContext(def, { ...defaultParams(def), ...params }, {})) ?? []
  ).join(' ')
}

beforeEach(() => {
  mockedRun.mockReset()
})

describe('neuron.simplifySkeletons', () => {
  it('is expensive, so a length field does not fire a run per keystroke', () => {
    expect(requireNodeDef('neuron.simplifySkeletons').cost).toBe('expensive')
  })

  it('makes the thinning methods one control, each with only its own setting showing', () => {
    const params = requireNodeDef('neuron.simplifySkeletons').params ?? []
    const shown = (id: string, method: string) =>
      params.find((p) => p.id === id)?.visibleIf?.({ method })
    expect(shown('factor', 'every')).toBe(true)
    expect(shown('factor', 'rdp')).toBe(false)
    expect(shown('tolerance', 'rdp')).toBe(true)
    expect(shown('area', 'vw')).toBe(true)
    expect(shown('area', 'every')).toBe(false)
  })

  it('says so when nothing is switched on, and catches a length typed in nanometres', () => {
    expect(validate({})).toMatch(/pass through unchanged/)
    expect(validate({ prune: 5 })).not.toMatch(/nanometres/)
    expect(validate({ prune: 5000 })).toMatch(/check you did not enter nanometres/)
  })

  it('passes the skeletons through without calling Python when nothing is on', async () => {
    const value = fixture()
    const { out } = await simplify(value, {})
    expect(out).toBe(value)
    expect(mockedRun).not.toHaveBeenCalled()
  })

  it('forwards every control, in nanometres', async () => {
    mockedRun.mockImplementation((request) => Promise.resolve(keepAll(request)))
    await simplify(fixture(), { prune: 4, repeat: true, method: 'vw', area: 0.5 })
    expect(mockedRun.mock.calls[0]![0]).toMatchObject({
      prune: 4000,
      repeat: true,
      method: 'vw',
      area: 500_000,
    })
  })

  it('carries a computed split through, and says nothing about it', async () => {
    // The difference from Clean Skeletons, which has to drop it and warn.
    mockedRun.mockResolvedValue(PRUNED)
    const { out, warnings } = await simplify(fixture(), { prune: 1 })
    expect(out.items[0]!.split).toBeDefined()
    expect(warnings).toEqual([])
  })

  it('refuses a length on voxel coordinates, and lets every Nth through', async () => {
    mockedRun.mockImplementation((request) => Promise.resolve(keepAll(request)))
    const voxels = { ...fixture(), units: 'voxels' as const }
    await expect(simplify(voxels, { prune: 2 })).rejects.toThrow(/are in voxels/)
    await expect(simplify(voxels, { method: 'every', factor: 3 })).resolves.toBeDefined()
  })
})

describe('neuron.cleanSkeletons — keep every Nth, moved', () => {
  const def = () => requireNodeDef('neuron.cleanSkeletons')
  const options = (method: string) => {
    const param = (def().params ?? []).find((p) => p.id === 'method')
    if (param?.kind !== 'enum' || typeof param.options === 'function') throw new Error('enum')
    return offeredOptions(param.options, method)
  }

  it('is not offered to a new card', () => {
    expect(options('none').map((o) => o.value)).toEqual(['none', 'resample'])
  })

  it('is never advertised to the assistant, not even as the gate on `Keep every`', () => {
    // A gate note probes the other params' values; probing a retired one would tell a model to
    // set it, one line below the list that leaves it out.
    const factor = (def().params ?? []).find((p) => p.id === 'factor')!
    expect(gateNote(def(), factor)).toBe('')
  })

  it('is still offered, with where it went, on a card that already holds it', () => {
    expect(options('downsample').find((o) => o.value === 'downsample')?.note).toMatch(
      /Simplify Skeletons/,
    )
    const d = def()
    const issues = d.validate?.(
      makeInferContext(d, { ...defaultParams(d), method: 'downsample', factor: 4 }, {}),
    )
    expect((issues ?? []).join(' ')).toMatch(/moved to Simplify Skeletons/)
  })
})
