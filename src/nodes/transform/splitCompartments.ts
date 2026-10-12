/**
 * Split Axon/Dendrite: every skeleton node labelled axon, dendrite or linker, as data on the wire.
 *
 * Four methods, one table of what each needs (`SPLIT_METHODS`). The two from synapses, synapse
 * flow centrality and label propagation, are navis's through `compartmentOps.splitSkeletons` —
 * the call Neuron Topology makes, so this card and that one cannot disagree. By ROI
 * (`splitByVolume.ts`) and by tag (`splitByTag.ts`) run here and need no synapses; see
 * `docs/nodes-morphology.md`. What this node adds is where the answer goes: onto each skeleton as
 * `SkeletonGeometry.split`, which the 3D View shades by, and onto each synapse as a `compartment`
 * column, which anything that reads a table can group by. Topology keeps only per-neuron totals; those are this node's `Summary`.
 *
 * **`cheap`, because nothing here fetches.** The skeletons and the synapses arrive on wires, so
 * invariant 6 does not apply, and a method's knobs are the controls whose honest setting is
 * found by moving them and looking at the arbour. The Heatmap's clustering is the precedent for a
 * Pyodide call in a `cheap` node. The cost is that the first run in a session pays the runtime
 * download (~10 MB), on the first edit after the node is wired rather than on a Run.
 *
 * **The synapses are a wire, never a fetch.** A Synapses node of the same neurons, both
 * polarities, is the input the synapse methods are defined over. A `Synapses Between` cloud is not: its
 * `neuronId` is the presynaptic neuron on every row, so a postsynaptic neuron's inputs would be
 * filed under its partner.
 */

import type { EvalContext, ParamDef, ParamValues } from '../../core/node'
import { registerNode } from '../../core/registry'
import { T, columnNames } from '../../core/types'
import type { MeshesValue, PointsValue, SkeletonsValue } from '../../core/values'
import {
  emptyPoints,
  isMeshesValue,
  isPointsValue,
  isSkeletonsValue,
  skeletonPointCount,
} from '../../core/values'
import { listed } from '../../core/prose'
import { memoPromise, untilAborted } from '../../data/memoPromise'
import { NM_PER_UM, checkGeometryUnits } from '../../data/units'
import type {
  NeuronSynapses,
  PythonSplitSettings,
  SetSplit,
  SplitSettings,
} from '../lib/compartmentOps'
import type { SynapseAssignment } from '../lib/topologyOps'
import { assignSynapses, parentDistances } from '../lib/topologyOps'
import { hasSplit } from '../lib/splitStatus'
import { nodesInVolumes, pickVolumes, volumeLabels } from '../lib/splitByVolume'
import { splitByTags, tagNames } from '../lib/splitByTag'
import { grownBoxes, volumeBoxes } from '../lib/meshInside'
import type { RayCount } from '../lib/pointsInMeshes'
import { countRays, warnRayCount, warnRayWork } from '../lib/pointsInMeshes'
import {
  SPLIT_METHODS,
  SPLIT_METHOD_OPTIONS,
  VOLUME_COMPARTMENT_OPTIONS,
  WEIGHTING_OPTIONS,
  correctSplit,
  labelSynapses,
  labelledSynapsesSchema,
  readSplitSettings,
  requireSynapseFrame,
  requireVolumeFrame,
  sitesByNeuron,
  splitParams,
  splitSkeletons,
  splitSummarySchema,
  splitSummaryTable,
  splitWarning,
  splitMethodSpec,
  whenMethod,
  withSplit,
} from '../lib/compartmentOps'
import type { SplitCorrection } from '../lib/splitCorrections'
import { CORRECTIONS_PARAM, readCorrections } from '../lib/splitCorrections'
import { healSkeleton } from '../lib/arborHeal'

/**
 * What a split reads off its inputs, remembered per input so that a hand correction — an edit to
 * `corrections` alone, re-running this `cheap` node — is a TypeScript pass over labels already
 * computed rather than another trip through Python or another ray per node.
 *
 * Everything is keyed on the skeletons value, which the scheduler hands back by identity until
 * something upstream re-runs, and in levels, because half of the work depends on neither the
 * method nor its knobs: every node's parent distance (per skeletons), and the synapses grouped by
 * neuron and placed on their nodes (per synapse cloud). Those are done once, where a dragged slider
 * would otherwise redo them, and hold a copy of them, per step. Only the latest cloud and the
 * latest volumes per skeletons value are kept: a new one is a new input, and the old answers say
 * nothing about it. Below them, the splits are per setting and bounded.
 *
 * Held without a cancel signal and raced against each run's own (`untilAborted`): a cheap node's
 * next edit aborts the previous run, and a shared promise carrying that signal would fail every
 * later run that reused it. Hand-capped `Map`s rather than `LruMap`, `memoPromise` taking a `Map`
 * — the exception `core/lruMap.ts` records.
 */
const MEMO_ENTRIES = 8

/** Oldest out first, a `Map` iterating in insertion order. */
function cap(held: Map<string, unknown>): void {
  while (held.size > MEMO_ENTRIES) held.delete(held.keys().next().value!)
}

const distanceMemo = new WeakMap<SkeletonsValue, Float32Array[]>()

/** Every node's distance to its parent, per neuron. */
function distancesOf(skeletons: SkeletonsValue): Float32Array[] {
  let distances = distanceMemo.get(skeletons)
  if (!distances) {
    distances = skeletons.items.map((item) => parentDistances(item))
    distanceMemo.set(skeletons, distances)
  }
  return distances
}

/** A synapse cloud placed on the skeletons, and the Python splits made from it. */
interface Placed {
  readonly points: PointsValue
  readonly synapses: Map<string, NeuronSynapses>
  readonly assignments: SynapseAssignment[]
  readonly splits: Map<string, Promise<SetSplit>>
}
const placedMemo = new WeakMap<SkeletonsValue, Placed>()

function place(skeletons: SkeletonsValue, points: PointsValue): Placed {
  let placed = placedMemo.get(skeletons)
  if (placed?.points !== points) {
    const synapses = sitesByNeuron(points)
    const distances = distancesOf(skeletons)
    placed = {
      points,
      synapses,
      assignments: skeletons.items.map((item, i) =>
        assignSynapses(item, synapses.get(item.id)?.sites ?? [], distances[i]),
      ),
      splits: new Map(),
    }
    placedMemo.set(skeletons, placed)
  }
  return placed
}

/** A Python method's split. */
async function automaticSplit(
  skeletons: SkeletonsValue,
  placed: Placed,
  settings: PythonSplitSettings,
  /** Write the method's per-node value: flow, or the propagation's probability. */
  values: boolean,
  run: Pick<EvalContext, 'progress' | 'signal'>,
): Promise<SetSplit> {
  /*
   * Keyed on the settings alone — they carry the method, and a method's knobs only when it is
   * chosen. The per-node value is always asked for: Python computes it either way and it is four
   * bytes a node to send back, where keying on the switch made ticking it re-run the whole split.
   * Whether it reaches the skeletons is decided here, after the memo.
   */
  const split = memoPromise(
    placed.splits,
    JSON.stringify(settings),
    () =>
      splitSkeletons(skeletons, placed.synapses, settings, {
        distances: distancesOf(skeletons),
        assignments: placed.assignments,
        values: true,
        onProgress: run.progress,
      }),
    { keep: 'resolved' },
  )
  cap(placed.splits)
  const found = await untilAborted(split, run.signal)
  return values ? found : { ...found, values: undefined }
}

/**
 * The skeletons corrections are applied on: each split neuron that carries a correction and arrived
 * in pieces, joined — it was healed to be split at all, and an arbour built from the delivered
 * parents reaches only the fragment its root is on, so a correction anywhere else lands nowhere and
 * is counted stale. The join is `arborHeal.ts`', the one the editor draws, so a click and the Run
 * agree on what is distal. Nobody else pays: an uncorrected or whole neuron is passed as it is.
 */
async function correctionTrees(
  skeletons: SkeletonsValue,
  split: SetSplit,
  corrections: readonly SplitCorrection[],
  signal: AbortSignal | undefined,
): Promise<SkeletonsValue> {
  if (corrections.length === 0) return skeletons
  const corrected = new Set(corrections.map((c) => c.neuron))
  // `healSkeleton` answers a whole neuron, or one joined before, without Python.
  const items = await Promise.all(
    skeletons.items.map(async (item, i) =>
      hasSplit(split.status[i]!) && corrected.has(item.id)
        ? (await healSkeleton(item, signal)).skeleton
        : item,
    ),
  )
  return { ...skeletons, items }
}

/**
 * A split by tag as a `SetSplit`, each neuron split on its own tree — healed first where that was
 * asked for, which keeps the node numbering the tags name. Nothing is memoised here: the heal is
 * (`arborHeal.ts`, per geometry) and so is the arbour (`splitArbor`), and what is left is one pass
 * over each neuron's nodes.
 */
async function tagSplit(
  skeletons: SkeletonsValue,
  placed: Placed | undefined,
  settings: Extract<SplitSettings, { method: 'tag' }>,
  ctx: Pick<EvalContext, 'signal' | 'warn'>,
): Promise<SetSplit> {
  const named = [...settings.axonTags, ...settings.dendriteTags]
  const carried = tagNames(skeletons.items)
  const unknown = named.filter((name) => !carried.includes(name))
  if (unknown.length > 0) {
    ctx.warn(
      carried.length === 0
        ? 'These skeletons carry no node tags.'
        : `No skeleton here carries ${quoted(unknown)}. The tags they carry: ${quoted(carried)}.`,
    )
  }
  const splits = await Promise.all(
    skeletons.items.map(async (item) => {
      const tree = settings.heal ? (await healSkeleton(item, ctx.signal)).skeleton : item
      return splitByTags(tree, settings.axonTags, settings.dendriteTags)
    }),
  )
  return {
    labels: splits.map((split) => split.labels),
    status: splits.map((split) => split.status),
    assignments: placed?.assignments ?? skeletons.items.map(() => undefined),
    distances: distancesOf(skeletons),
  }
}

/** Which nodes are inside the ROIs, and what finding that out cost, for the warning's replay. */
interface VolumeMask {
  readonly inside: Uint8Array[]
  readonly untouched: number
  /** The work `warnRayWork` priced, for the warning's replay. */
  readonly work: RayCount
}

/**
 * The ROI split's masks, per volumes value and set of ROI names — not per `inside`, which only
 * relabels a mask, so flipping it casts nothing.
 */
const volumeMemo = new WeakMap<
  SkeletonsValue,
  { volumes: MeshesValue; masks: Map<string, Promise<VolumeMask>> }
>()

/**
 * A split by ROI as a `SetSplit`: every neuron `ok`, its labels from the mask, and its synapses
 * placed on it where a cloud is wired so Summary can still count each compartment's.
 */
async function volumeSplit(
  skeletons: SkeletonsValue,
  placed: Placed | undefined,
  volumes: MeshesValue,
  settings: Extract<SplitSettings, { method: 'roi' }>,
  ctx: Pick<EvalContext, 'progress' | 'signal' | 'warn'>,
): Promise<SetSplit> {
  requireVolumeFrame(skeletons, volumes)
  // The skeletons' units, which `requireVolumeFrame` has just made the volumes' too.
  if (settings.margin > 0) {
    checkGeometryUnits('The', skeletons, 'skeletons', 'Skeletons', MARGIN_CONSEQUENCE)
  }
  const margin = settings.margin * NM_PER_UM
  const { items, unknown } = pickVolumes(volumes, settings.volumes)
  if (unknown.length > 0) {
    if (items.length === 0) {
      throw new Error(`None of the ROIs named (${quoted(unknown)}) is among the wired volumes.`)
    }
    const one = unknown.length === 1
    ctx.warn(
      `${one ? 'An ROI' : 'ROIs'} named here ${one ? 'is' : 'are'} not among the wired volumes: ` +
        `${quoted(unknown)}.`,
    )
  }

  let memo = volumeMemo.get(skeletons)
  if (memo?.volumes !== volumes) {
    volumeMemo.set(skeletons, (memo = { volumes, masks: new Map() }))
  }
  const key = JSON.stringify([margin, items.map((item) => item.id)])
  const cached = memo.masks.has(key)
  const pending = memoPromise(
    memo.masks,
    key,
    async () => {
      // The cost said before anything is cast: a ray per node per box holding it, and with a
      // margin a search per node per grown box — an upper bound, a node a ray puts inside asking
      // none. Every node against every volume bounds both.
      const prefilter = volumeBoxes(items)
      const bound = skeletonPointCount(skeletons) * items.length
      const work = warnRayWork(
        ctx,
        { rays: bound, searches: margin > 0 ? bound : 0 },
        () => {
          const grown = margin > 0 ? grownBoxes(prefilter.boxes, margin) : undefined
          let rays = 0
          let searches = 0
          for (const { positions } of skeletons.items) {
            rays += countRays(positions, prefilter.candidateCount)
            if (grown) searches += countRays(positions, grown.candidateCount)
          }
          return { rays, searches }
        },
        items.length,
      )
      // Progress only: the promise is shared, and the run's own signal races it below.
      const hooks = { progress: ctx.progress }
      const found = await nodesInVolumes(skeletons.items, items, hooks, prefilter, margin)
      return { ...found, work }
    },
    { keep: 'resolved' },
  )
  cap(memo.masks)
  const { inside, untouched, work } = await untilAborted(pending, ctx.signal)
  // A remembered mask replays its warning, or the card's warning comes and goes with the cache.
  if (cached) warnRayCount(ctx, work, items.length)
  if (untouched > 0) {
    ctx.warn(
      `${untouched} of ${skeletons.items.length} neurons have no node inside any of the ROIs, so ` +
        `they are all ${settings.inside === 'axon' ? 'dendrite' : 'axon'}. Check the ROIs, ` +
        `that the skeletons and the volumes are in the same space, or try a \`Margin\`.`,
    )
  }

  return {
    labels: volumeLabels(inside, settings.inside),
    status: skeletons.items.map(() => 'ok'),
    assignments: placed?.assignments ?? skeletons.items.map(() => undefined),
    distances: distancesOf(skeletons),
  }
}

/** Names as the card's warnings write them: in backticks, listed. */
const quoted = (names: readonly string[]): string => listed(names.map((name) => `\`${name}\``))

/** Why a margin needs nanometres: it is a distance in µm. */
const MARGIN_CONSEQUENCE = 'a `Margin` in µm cannot be measured against them.'

/** What `validate` and `evaluate` say when a method's input is not wired. */
const NO_SYNAPSES =
  'Wire a Synapses node of the same neurons, with both polarities, into `Synapses`. ' +
  'The ROI and tag methods run without one.'
const NO_VOLUMES = 'Wire the regions to split by into `Volumes`, e.g. from ROI Meshes.'

/** `Synapses` when nothing was wired there: an empty cloud with the column the port promises. */
const NO_LABELLED_SYNAPSES = emptyPoints(labelledSynapsesSchema(undefined))

/** The two columns the split reads off a synapse cloud. */
const NEEDED_COLUMNS = ['neuronId', 'polarity'] as const

const isFlow = whenMethod('flow')
const isPropagation = whenMethod('propagation')
const isRoi = whenMethod('roi')
const isTag = whenMethod('tag')
const walksTree = (params: ParamValues): boolean => splitMethodSpec(params).walksTree

/**
 * A split by tag's two controls. Names typed rather than picked, for the ROI names' reason: the
 * tags are in the wired skeletons, which a param's options cannot see before a Run. A run whose
 * names match nothing says which tags the skeletons do carry.
 */
function tagParams(): ParamDef[] {
  const tags = (compartment: 'axon' | 'dendrite'): ParamDef => ({
    id: `${compartment}Tags`,
    kind: 'string',
    label: compartment === 'axon' ? 'Axon tags' : 'Dendrite tags',
    default: '',
    chips: true,
    placeholder: `add a tag, e.g. ${compartment}`,
    group: 'split',
    visibleIf: isTag,
    help:
      `Node tags that mark where the ${compartment} starts: a tagged node and everything ` +
      `distal of it, away from the soma, are ${compartment}.`,
  })
  return [tags('axon'), tags('dendrite')]
}

/** A split by ROI's two controls: which wired volumes count, and what they hold. */
function roiParams(): ParamDef[] {
  return [
    {
      id: 'inside',
      kind: 'enum',
      label: 'The ROIs hold',
      default: 'axon',
      group: 'split',
      visibleIf: isRoi,
      options: [...VOLUME_COMPARTMENT_OPTIONS],
      help: 'Which compartment the ROIs contain. A node inside any of them is that; every other node is the other one.',
    },
    /*
     * Names typed rather than picked: a volume's name is in the wired value, which a param's
     * options cannot see before a Run. Empty is every wired volume, which is the simplest set-up —
     * ask ROI Meshes for just those regions. A name that matches no wired volume is said on a Run.
     */
    {
      id: 'volumes',
      kind: 'string',
      label: 'ROIs',
      default: '',
      chips: true,
      placeholder: 'add an ROI, e.g. ME(R)',
      group: 'split',
      visibleIf: isRoi,
      help: 'Which of the wired volumes hold the compartment, by name, e.g. ME(R). Leave empty to use every wired volume.',
    },
    // Grow only: a region's edge drawn a little tight is the case asked for, not a generous one.
    {
      id: 'margin',
      kind: 'number',
      label: 'Margin (µm)',
      default: 0,
      min: 0,
      step: 0.5,
      group: 'split',
      visibleIf: isRoi,
      help: 'Also count a node as inside an ROI when it is within this distance of the ROI’s surface, for neurites that run along a region’s edge. 0 counts only nodes inside.',
    },
  ]
}

/**
 * Label propagation's knobs, navis's `split_axon_dendrite_prop` arguments under the names on the
 * card. The defaults are navis's but for `Clamping`, which says why. The two iteration controls are advanced: the cap is what a
 * "did not converge" warning asks somebody to raise, and the stop rule is navis's own.
 */
function propagationParams(): ParamDef[] {
  return [
    {
      id: 'weighting',
      kind: 'enum',
      label: 'Weights',
      default: 'balance',
      group: 'split',
      visibleIf: isPropagation,
      options: [...WEIGHTING_OPTIONS],
      help: 'How strongly each node starts out as axon or dendrite. By synapse count: by how many more outputs than inputs it has, or the reverse. Balanced (navis’s default) also evens out the two totals, so the commoner synapse type does not win by numbers alone. None: every node with synapses counts the same.',
    },
    {
      id: 'alpha',
      kind: 'number',
      label: 'Clamping',
      /*
       * 0.5, not navis's 1. Unclamped, each step replaces a node's scores by its neighbours'
       * average, and a tree is bipartite, so the scores can flip between two states for ever: a
       * nine-node Y with outputs on one arm and inputs on the other never converges at 1 and splits
       * cleanly at 0.5, as navis's five example neurons all do. See docs/nodes-morphology.md.
       */
      default: 0.5,
      min: 0,
      max: 1,
      step: 0.1,
      group: 'split',
      visibleIf: isPropagation,
      help: 'How free a node with synapses is to change its starting label: 1, free; 0, held fixed; between, pulled back towards it. navis’s default is 1, which can flip back and forth on a tree and never settle; 0.5 here.',
    },
    {
      id: 'maxIter',
      kind: 'number',
      label: 'Max iterations',
      default: 10_000,
      min: 1,
      step: 1000,
      advanced: true,
      group: 'split',
      visibleIf: isPropagation,
      help: 'Stop after this many steps. A neuron that gets there is reported as not converged; its labels are kept, but may be noise.',
    },
    {
      id: 'tol',
      kind: 'number',
      label: 'Tolerance',
      default: 1,
      min: 0,
      step: 0.1,
      advanced: true,
      group: 'split',
      visibleIf: isPropagation,
      help: 'When to stop. 1 or more: once no node has changed label for that many steps (navis’s default, 1). Below 1: once the largest change in any node’s score falls below it.',
    },
  ]
}

registerNode({
  type: 'neuron.splitCompartments',
  label: 'Split Axon/Dendrite',
  category: 'transform',
  description:
    'Label every skeleton node axon, dendrite or linker: from where its synapses are (navis’s synapse flow centrality or label propagation), from the ROIs it lies in, or from node tags.',
  guide:
    'Labels each neuron’s skeleton axon, dendrite or linker: from where its synapses are, by ' +
    'navis’s synapse flow centrality or by label propagation; from which wired ROIs it lies in; ' +
    'or from node tags a tracer placed (CATMAID). Wire in skeletons, and a Synapses cloud of the ' +
    'same neurons for the synapse methods. Open the card full size to inspect a split on a ' +
    'dendrogram and correct it by hand.',
  cost: 'cheap',
  inputs: [
    { id: 'skeletons', label: 'Skeletons', type: T.skeletons() },
    /*
     * Optional because a split by ROI or by tag needs none: wired, it still counts each
     * compartment's synapses and labels the cloud; unwired, those counts read empty. The two
     * Python methods cannot run without it, and say so on a Run.
     */
    { id: 'synapses', label: 'Synapses', type: T.points(), required: false },
    // The regions a split by ROI tests each node against — ROI Meshes, typically.
    { id: 'volumes', label: 'Volumes', type: T.meshes(), required: false },
  ],
  /*
   * Skeletons first, so a wire dragged off the card starts at the labelled arbour — the output the
   * node is named for.
   */
  outputs: [
    { id: 'out', label: 'Skeletons', type: T.skeletons() },
    { id: 'labelled', label: 'Synapses', type: T.points() },
    { id: 'summary', label: 'Summary', type: T.neurons() },
  ],
  /*
   * One group, and it changes the data — which is the point. The thresholds are the knobs whose
   * honest setting is found by moving them while looking at the arbour, so they are declared a
   * data group: the full-size view puts them in a tab with a note, and a dashboard cell (which
   * draws only presentational params and data groups) puts them beside a 3D View showing the
   * result. The node is `cheap`, so each move re-splits on its own.
   */
  paramGroups: [
    { id: 'split', label: 'Split', affectsData: true },
    { id: 'display', label: 'Display' },
  ],
  params: [
    /*
     * How to split: navis's flow centrality (`split_axon_dendrite`) or label propagation
     * (`split_axon_dendrite_prop`), by ROI or by tag — `SPLIT_METHODS`. A saved card has no key
     * and reads as flow, its default, which is what it was. A method's own knobs are `visibleIf` on it, which also keeps them out of the
     * provenance key while hidden (invariant 4), so switching away and back finds the cached split.
     */
    {
      id: 'method',
      kind: 'enum',
      label: 'Method',
      default: 'flow',
      group: 'split',
      options: [...SPLIT_METHOD_OPTIONS],
      help: 'Synapse flow cuts the neuron where the most input-to-output paths run and labels axon, dendrite and linker. Label propagation spreads labels out from where outputs and inputs sit, and labels axon and dendrite only. By ROI labels nodes by whether they lie inside the wired volumes. By tag reads node tags a tracer placed where the axon or the dendrite starts (CATMAID).',
    },
    // The heal is for the methods that walk the tree; a split by ROI tests each node on its own.
    ...splitParams({ group: 'split' }).map((param) => ({
      ...param,
      visibleIf: param.id === 'heal' ? walksTree : isFlow,
    })),
    ...propagationParams(),
    ...roiParams(),
    ...tagParams(),
    /*
     * Hand corrections, written by the expanded view's dendrogram — see `splitCorrections.ts`.
     * Data, in the provenance key, so a saved file and a share link carry them. Not in the Split
     * group, which would put a "clear all" for somebody's afternoon of proofreading on a dashboard
     * cell's rail; the card shows the count, and the inspector can clear them.
     */
    {
      id: CORRECTIONS_PARAM,
      kind: 'ids',
      label: 'Corrections',
      noun: 'corrections',
      default: [],
      help: 'Branches reassigned by hand in the expanded view. They are applied on top of the automatic split, so they still hold when a threshold changes.',
    },
    /*
     * Opt-in, because it is the largest thing the split sends back — a float per node — and only
     * a scene that draws it wants it. Data rather than presentation: it changes what `Skeletons`
     * carries, so it is in the provenance key and in the same group as the thresholds it shows.
     */
    {
      id: 'flow',
      kind: 'boolean',
      label: 'Write synapse flow',
      help: 'Also store each node’s synapse flow on the skeletons, as a fraction of that neuron’s peak, for the 3D View’s “by node value” colours. The linker is every node at or above Linker threshold.',
      default: false,
      group: 'split',
      visibleIf: isFlow,
    },
    // The propagation's counterpart of `flow`: how sure each node's label is.
    {
      id: 'probability',
      kind: 'boolean',
      label: 'Write probability',
      help: 'Also store each node’s probability on the skeletons, for the 3D View’s “by node value” colours: 1 where the propagation is sure of the label, 0.5 where axon and dendrite are a toss-up.',
      default: false,
      group: 'split',
      visibleIf: isPropagation,
    },
    /*
     * The editor's synapse ticks. Presentational — what the dendrogram draws, never what the node
     * returns — so it stales nothing, and remembered, so a dashboard reopens as it was left. Off the
     * card, which has no dendrogram to draw them on.
     */
    {
      id: 'showSynapses',
      kind: 'boolean',
      label: 'Show synapses',
      help: 'Draw the synapses on the expanded view’s dendrogram: outputs on one side of a branch, inputs on the other.',
      default: true,
      presentational: true,
      advanced: true,
      group: 'display',
    },
  ],

  inferOutputs: (ctx) => ({
    out: T.skeletons(ctx.attributes('skeletons')),
    labelled: T.points(labelledSynapsesSchema(ctx.attributes('synapses'))),
    summary: T.neurons(splitSummarySchema(ctx.attributes('skeletons'))),
  }),

  /*
   * The frame check waits for `evaluate`: units and a template space live on the value, and a type
   * carries a kind and a schema only — `neuron.pointsInVolumes` says the same at the same place.
   */
  validate: (ctx) => {
    /*
     * Each method's own inputs, said before a Run: the ports are optional because the methods
     * differ, so the framework's "required" mark cannot say it for them.
     */
    const spec = splitMethodSpec(ctx.params)
    if (spec.needsVolumes && !ctx.inputs.volumes) return [NO_VOLUMES]
    if (spec.needsSynapses && !ctx.inputs.synapses) return [NO_SYNAPSES]
    const settings = readSplitSettings(ctx.params)
    if (
      settings.method === 'tag' &&
      settings.axonTags.length + settings.dendriteTags.length === 0
    ) {
      return ['Name the node tags that mark the axon, the dendrite, or both.']
    }
    const schema = ctx.attributes('synapses')
    if (!schema) return []
    const names = columnNames(schema)
    const missing = NEEDED_COLUMNS.filter((name) => !names.includes(name))
    return missing.length > 0
      ? [
          `The synapses need ${quoted(missing)} to be split by. ` +
            `Wire in a Synapses node of the same neurons, with both polarities.`,
        ]
      : []
  },

  evaluate: async (ctx) => {
    const skeletons = ctx.input('skeletons')
    if (!isSkeletonsValue(skeletons)) {
      throw new Error(
        'Nothing is wired into `Skeletons`. Wire in skeletons, e.g. from the Skeletons node.',
      )
    }
    const settings = readSplitSettings(ctx.params)
    const spec = SPLIT_METHODS[settings.method]
    const wired = ctx.input('synapses')
    const points = isPointsValue(wired) ? wired : undefined
    if (points) requireSynapseFrame(skeletons, points)
    const placed = points && place(skeletons, points)
    const wiredVolumes = ctx.input('volumes')
    const volumes = isMeshesValue(wiredVolumes) ? wiredVolumes : undefined
    // What each method needs, from the table `validate` reads too, before any method runs.
    if (spec.needsSynapses && !placed) throw new Error(NO_SYNAPSES)
    if (spec.needsVolumes && !volumes) throw new Error(NO_VOLUMES)

    let automatic: SetSplit
    if (settings.method === 'tag') {
      automatic = await tagSplit(skeletons, placed, settings, ctx)
    } else if (settings.method === 'roi') {
      automatic = await volumeSplit(skeletons, placed, volumes!, settings, ctx)
    } else {
      const values = spec.value !== undefined && ctx.params[spec.value.param] === true
      automatic = await automaticSplit(skeletons, placed!, settings, values, ctx)
    }
    const corrections = readCorrections(ctx.params[CORRECTIONS_PARAM])
    const trees = await correctionTrees(skeletons, automatic, corrections, ctx.signal)
    const { split, applied, stale } = correctSplit(trees, automatic, corrections)
    if (stale > 0) {
      ctx.warn(
        `${stale} of ${corrections.length} hand corrections no longer land on a split neuron ` +
          `here: the neuron is not on the input, was not split, or its skeleton has changed. ` +
          `They are kept; remove them in the expanded view.`,
      )
    }

    const unsplit = splitWarning(split.status, {
      column: 'the `splitStatus` column of Summary',
    })
    if (unsplit) ctx.warn(unsplit)
    /*
     * The other half of the join, which fails just as quietly: synapses of neurons that have no
     * skeleton here are not used, and a cloud wired from a different query would leave every
     * neuron `no synapses` while looking like a fine input.
     */
    if (placed) {
      const ids = new Set(skeletons.items.map((item) => item.id))
      let stray = 0
      for (const [id, group] of placed.synapses) if (!ids.has(id)) stray += group.rows.length
      if (stray > 0) {
        ctx.warn(
          `${stray} of ${placed.points.attributes.length} synapses belong to neurons with no ` +
            `skeleton here, so they were not used and their \`compartment\` is empty.`,
        )
      }
    }

    return {
      out: withSplit(skeletons, split),
      // No synapses in, none out: a split by ROI or tag with nothing on `Synapses` labels no rows.
      labelled: placed
        ? labelSynapses(placed.points, skeletons, placed.synapses, split)
        : NO_LABELLED_SYNAPSES,
      summary: splitSummaryTable(skeletons, split, applied),
    }
  },
})
