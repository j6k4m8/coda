/**
 * Simplify Skeletons: prune short twigs, then thin what is left.
 *
 * Clean Skeletons' sibling, and a separate node because it asks a different question. Clean
 * *repairs* a reconstruction — joins it, smooths it, lays its nodes out evenly — and every one
 * of its steps can move a node or make a new one. This *removes* nodes and moves none: a twig
 * goes whole, and every thinning method keeps the nodes it keeps where they were. That is what
 * lets it carry a computed split, per-node values and tags straight through (`simplifyOps.ts`),
 * where Clean has to drop the first two.
 *
 * Two steps, in this order:
 *
 * 1. **Prune twigs** — `navis.prune_twigs`' operation, run as `fastcore.prune_twigs` with two
 *    protections added (below). A twig is a leaf back to its branch point, measured along the
 *    cable as traced; first, because thinning first would cut the corners off every twig and
 *    make it shorter than it is. `Repeat` is navis' `recursive=True`.
 * 2. **Downsample** — one of three, never two, for the reason Clean's `method` is one enum.
 *    Keep every Nth node (`navis.downsample_neuron`'s operation, fastcore's
 *    `downsample_skeleton`), or one of fastcore's two geometry-aware methods, which spend the
 *    same node budget where the neurite actually bends.
 *
 * ## What it never removes
 *
 * Roots, branch points and leaves survive every thinning method, on fastcore's own rule; the
 * **soma** survives everything, by being handed to both calls. A **fragment with no branch point**
 * is never pruned, however short — fastcore would delete it whole, being all twig, and a neuron
 * that arrived unbranched would come back empty. And like every geometry transform here, it never
 * changes how many neurons there are.
 */

import { registerNode } from '../../core/registry'
import { T, attributeSchema } from '../../core/types'
import { isSkeletonsValue } from '../../core/values'
import { runSimplifySkeletons } from '../../pyodide/skeletons'
import { checkCleanUnits, micrometreMixup } from '../lib/cleanOps'
import {
  isSimplifyNoOp,
  simplifyRequestFrom,
  simplifyUsesDistance,
  skeletonSimplifyParamsFrom,
  skeletonsFromKeep,
} from '../lib/simplifyOps'

registerNode({
  type: 'neuron.simplifySkeletons',
  label: 'Simplify Skeletons',
  category: 'transform',
  description: 'Prune short twigs and downsample skeletons, keeping the neuron’s shape.',
  guide:
    'Remove nodes from skeletons: prune twigs shorter than a cable length, then keep every Nth ' +
    'node or only the nodes where the neurite bends. Roots, branch points, leaves and the soma ' +
    'always survive, and no node moves, so an axon/dendrite split computed upstream comes ' +
    'through intact. Distances are in micrometres.',
  /*
   * `expensive`, Clean Skeletons' reason: a fastcore call in a Python runtime that is a ten
   * megabyte download on first use, and `cheap` would run it on each keystroke of a length.
   */
  cost: 'expensive',
  inputs: [{ id: 'in', label: 'Skeletons', type: T.skeletons() }],
  outputs: [{ id: 'out', label: 'Skeletons', type: T.skeletons() }],
  params: [
    {
      id: 'prune',
      kind: 'number',
      label: 'Prune twigs (µm)',
      default: 0,
      min: 0,
      step: 1,
      help: 'Remove twigs — a leaf back to its branch point — shorter than this, measured along the cable. 0 prunes nothing.',
    },
    {
      id: 'repeat',
      kind: 'boolean',
      label: 'Repeat until none are left',
      default: false,
      visibleIf: (params) => Number(params.prune) > 0,
      help: 'Removing a twig can leave the branch it hung from as a new short twig. On, pruning runs again until there is none; off, it runs once.',
    },
    {
      id: 'method',
      kind: 'enum',
      label: 'Downsample',
      default: 'none',
      options: [
        { value: 'none', label: 'keep every node' },
        { value: 'every', label: 'keep every Nth node' },
        { value: 'rdp', label: 'drop nodes that do not bend', note: 'Ramer–Douglas–Peucker' },
        { value: 'vw', label: 'drop nodes adding least area', note: 'Visvalingam–Whyatt' },
      ],
      help: '"keep every Nth node" ignores geometry. The other two keep the nodes where the neurite curves and drop them along straight stretches, so the same node count looks more like the neuron.',
    },
    {
      id: 'factor',
      kind: 'int',
      label: 'Keep every',
      default: 2,
      min: 2,
      max: 100,
      step: 1,
      visibleIf: (params) => params.method === 'every',
      help: 'Keep one node in every N along each unbranched stretch.',
    },
    {
      id: 'tolerance',
      kind: 'number',
      label: 'Tolerance (µm)',
      default: 0.1,
      min: 0,
      step: 0.05,
      visibleIf: (params) => params.method === 'rdp',
      help: 'How far the simplified neurite may move from the traced one. 0 drops only nodes that lie exactly on a straight line.',
    },
    {
      id: 'area',
      kind: 'number',
      label: 'Smallest area (µm²)',
      default: 0.01,
      min: 0,
      step: 0.01,
      visibleIf: (params) => params.method === 'vw',
      help: 'Drop a node while the triangle it makes with its two neighbours is smaller than this. Squared units: 0.01 µm² is a 0.1 µm bump over 0.2 µm.',
    },
  ],

  // Kind and schema straight through, Clean Skeletons' answer: no column is touched.
  inferOutputs: (ctx) => ({ out: T.skeletons(attributeSchema(ctx.inputs.in, 'nodes')) }),

  validate: (ctx) => {
    const params = skeletonSimplifyParamsFrom(ctx.params)
    const issues: string[] = []
    if (isSimplifyNoOp(params))
      issues.push('Nothing is switched on, so the skeletons pass through unchanged.')
    // The nanometre mix-up, caught at edit time rather than after the wait. Thresholds are each
    // past anything a neuron could mean: a twig longer than most neurons, a 100 µm wobble, a
    // bump of 100 µm over 200 µm.
    if (params.prune >= 1000) issues.push(micrometreMixup('Prune twigs (µm)', params.prune))
    if (params.method === 'rdp' && params.tolerance >= 100)
      issues.push(micrometreMixup('Tolerance (µm)', params.tolerance))
    if (params.method === 'vw' && params.area >= 10_000)
      issues.push(micrometreMixup('Smallest area (µm²)', params.area, 'µm²'))
    return issues
  },

  evaluate: async (ctx) => {
    const value = ctx.input('in')
    if (!isSkeletonsValue(value))
      throw new Error('Simplify Skeletons takes a set of skeletons.')
    if (value.items.length === 0) throw new Error('No skeletons on the input')

    const params = skeletonSimplifyParamsFrom(ctx.params)
    // A pass-through for a card still being filled in, Clean Skeletons' reason.
    if (isSimplifyNoOp(params)) return { out: value }

    checkCleanUnits(value, simplifyUsesDistance(params))

    ctx.progress(0.01, `${value.items.length} neurons`)
    const result = await runSimplifySkeletons(simplifyRequestFrom(value, params), {
      onProgress: ctx.progress,
      signal: ctx.signal,
    })
    return { out: skeletonsFromKeep(value, result) }
  },
})
