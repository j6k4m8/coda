/**
 * Connection Compartments: an edge list with each edge's synapses counted by the compartment at
 * either end — axo-dendritic, axo-axonic, dendro-axonic, dendro-dendritic, linker and unsplit.
 *
 * The counting is `nodes/lib/connectionCompartments.ts`; what is here is the fetch and the
 * warnings. **`expensive`, because it fetches**: every synapse between the edge list's presynaptic
 * and postsynaptic neurons, in one `fetchSynapsesBetween` bound at both ends with `location:
 * 'both'`, kept where the pair is one the table lists. The query answers the cross product, which
 * is why the node says how much of what came back it threw away.
 *
 * The compartments come from a `Skeletons` wire carrying a split (Split Axon/Dendrite), never from a
 * split made here: the thresholds and the hand corrections are that card's, and a second copy of
 * them here would be two answers to one question.
 */

import { packNode } from '../../core/registry'
import { cannotFetchBetween } from './synapses'
import { T } from '../../core/types'
import type { DatasetValue, PointsValue, SkeletonsValue, TableValue } from '../../core/values'
import { isSkeletonsValue, isTableValue } from '../../core/values'
import type { SynapsesBetweenRequest } from '../../data/source'
import { requireSkeletonUnits, requireSynapseFrame } from '../../nodes/lib/compartmentOps'
import {
  datasetRequest,
  requireDataset,
  sourceLabel,
  sourceSupports,
} from '../../nodes/lib/datasetParam'
import { MAX_NEURONS, warnAboveParam, warnSideCount } from '../../nodes/lib/limitParams'
import { minConfidenceParam, minSynapseConfidence } from '../../nodes/lib/synapseParams'
import type {
  ByPair,
  Classified,
  ConnectionLayout,
} from '../../nodes/lib/connectionCompartments'
import {
  CLASSIFIED_SYNAPSES_SCHEMA,
  classifiedSynapsesTable,
  classifySynapses,
  connectionEdgesSchema,
  connectionEdgesTable,
  readEdgePairs,
  rowTallies,
  rowsOnPairs,
  weightMismatch,
} from '../../nodes/lib/connectionCompartments'

/** Past this many synapses thrown away, the cross-product cost is worth a sentence. */
const DISCARD_WARN = 10_000

const layoutOf = (value: unknown): ConnectionLayout => (value === 'long' ? 'long' : 'wide')

export const connectionCompartmentsNode = packNode({
  type: 'connectome:connectionCompartments',
  label: 'Connection Compartments',
  category: 'query',
  description:
    'Count each edge’s synapses by the compartment at either end: axo-dendritic, ' +
    'axo-axonic, dendro-axonic, dendro-dendritic, linker or unsplit.',
  guide:
    'Takes an edge list and skeletons split by Split Axon/Dendrite, fetches every synapse ' +
    'between the listed pairs, and counts each edge’s synapses by the compartment at the ' +
    'presynaptic and the postsynaptic end. Each end is placed on its own neuron’s skeleton at ' +
    'its own site. A synapse that cannot be placed is counted as unsplit, so the counts of an ' +
    'edge always add up to its synapses.',
  cost: 'expensive',
  // The last cloud fetched for each edge list is held while that list is: see `held`.
  dataCache: true,
  inputs: [
    { id: 'dataset', label: 'Dataset', type: T.dataset() },
    { id: 'edges', label: 'Edges', type: T.table() },
    {
      id: 'skeletons',
      label: 'Skeletons',
      type: T.skeletons(),
      /*
       * Any skeletons fit the type, and a plan wiring a bare Skeletons node here applies cleanly
       * and counts every synapse as unsplit: 0/5 live runs split first without this line, 8/10
       * with it (`gemma4:31b-cloud`, `lean`). See `docs/assistant.md`.
       */
      producedBy: { type: 'neuron.splitCompartments', port: 'out' },
    },
  ],
  outputs: [
    { id: 'edges', label: 'Edges', type: T.table() },
    { id: 'synapses', label: 'Synapses', type: T.points() },
  ],
  params: [
    {
      id: 'pre',
      kind: 'column',
      label: 'Presynaptic',
      from: 'edges',
      default: 'preId',
      help: 'The id column of each edge’s presynaptic neuron.',
    },
    {
      id: 'post',
      kind: 'column',
      label: 'Postsynaptic',
      from: 'edges',
      default: 'postId',
      help: 'The id column of each edge’s postsynaptic neuron.',
    },
    {
      id: 'layout',
      kind: 'enum',
      label: 'Layout',
      default: 'wide',
      options: [
        { value: 'wide', label: 'wide (a column per class)' },
        { value: 'long', label: 'long (a row per class)' },
      ],
      help: 'Wide keeps one row per edge and adds weightAxonAxon, weightAxonDendrite, weightDendriteAxon, weightDendriteDendrite, weightLinker, weightUnsplit and their total, synapses. Long has six rows per edge, a connection column naming the class and weight replaced by its count.',
    },
    {
      id: 'maxSnap',
      kind: 'number',
      label: 'Max snap distance (µm)',
      default: 10,
      min: 0,
      step: 1,
      advanced: true,
      help: 'A synapse further than this from its neuron’s skeleton is counted as unsplit rather than given the nearest node’s compartment.',
    },
    minConfidenceParam(
      'Drop a synapse when either of its ends scores below this; 0 keeps every one.',
    ),
    warnAboveParam({
      threshold: MAX_NEURONS,
      min: 1,
      counting: 'fetching synapses for more than this many neurons on either side',
    }),
  ],

  inferOutputs: (ctx) => ({
    edges: T.table(connectionEdgesSchema(ctx.attributes('edges'), layoutOf(ctx.params.layout))),
    synapses: T.points(CLASSIFIED_SYNAPSES_SCHEMA),
  }),

  validate: (ctx) => {
    const issues: string[] = []
    // Rule 3 hands both pickers the first column of a table without `preId`/`postId`, and two
    // pickers on one column describe no edge at all.
    const pre = ctx.column('pre')
    if (pre !== undefined && pre === ctx.column('post')) {
      issues.push(
        `\`Presynaptic\` and \`Postsynaptic\` both read "${pre}". Set them to the two id ` +
          `columns of the edge list.`,
      )
    }
    const dataset = ctx.inputs.dataset
    if (dataset && !sourceSupports(dataset, 'synapsesBetween')) {
      issues.push(cannotFetchBetween(sourceLabel(dataset)))
    }
    return issues
  },

  evaluate: async (ctx) => {
    const dataset = requireDataset(ctx.input('dataset'))
    const source = ctx.resolveSource(dataset.sourceId)
    if (!source.fetchSynapsesBetween) {
      throw new Error(cannotFetchBetween(source.label))
    }
    const table = ctx.input('edges')
    if (!isTableValue(table)) {
      throw new Error('Wire an edge list into `Edges`, for example from Connectivity.')
    }
    const skeletons = ctx.input('skeletons')
    if (!isSkeletonsValue(skeletons)) {
      throw new Error(
        'Wire split skeletons into `Skeletons`, from Split Axon/Dendrite’s Skeletons output.',
      )
    }
    // Before the fetch: skeletons in no known unit would be refused after it anyway.
    requireSkeletonUnits(skeletons)
    const preColumn = ctx.column('pre')
    const postColumn = ctx.column('post')
    if (!preColumn || !postColumn) {
      throw new Error('Set `Presynaptic` and `Postsynaptic` to the edge list’s two id columns.')
    }

    const edgePairs = readEdgePairs(table, preColumn, postColumn)
    const { pres, posts, pairs, notIds } = edgePairs
    if (notIds > 0) {
      const what =
        `\`${preColumn}\` and \`${postColumn}\` should hold neuron ids, and ` +
        `${pairs === 0 ? 'they' : `${notIds} of ${table.length} rows`} hold something ` +
        `else (the first reads ${edgePairs.example}).`
      const fix =
        'Set `Presynaptic` and `Postsynaptic` to the id columns. Adjacency’s links name ' +
        'neurons by label rather than id, so use Connectivity for the edge list.'
      if (pairs === 0) throw new Error(`${what} ${fix}`)
      ctx.warn(`${what} Those rows are not counted. ${fix}`)
    }

    const limit = Number(ctx.params.limit)
    const cost =
      'Every synapse between the two sides arrives, then only the listed pairs are kept.'
    warnSideCount(ctx, 'the presynaptic side', pres.length, limit, cost)
    warnSideCount(ctx, 'the postsynaptic side', posts.length, limit, cost)
    ctx.progress(0.1, `${pairs} edges, ${pres.length} → ${posts.length} neurons`)

    // An empty side asks nothing, and every source answers that with an empty cloud unasked.
    const entry = await fetchCloud(
      ctx,
      source.fetchSynapsesBetween.bind(source),
      {
        ...datasetRequest(dataset),
        sourceIds: pres,
        targetIds: posts,
        location: 'both',
        minConfidence: minSynapseConfidence(ctx.params),
        onProgress: (fraction, note) => ctx.progress(0.1 + fraction * 0.8, note),
        signal: ctx.signal,
      },
      dataset,
      table,
      edgePairs.wanted,
    )
    const { cloud, fetched } = entry
    requireSynapseFrame(skeletons, cloud)

    // `Layout` shapes the table and nothing before it, so an edit to it re-uses the classification.
    const maxSnapNm = Math.max(0, Number(ctx.params.maxSnap)) * 1000
    const known = entry.classified.get(skeletons)
    let classified = known?.maxSnapNm === maxSnapNm ? known.result : undefined
    if (!classified) {
      ctx.progress(0.9, 'classifying')
      classified = await classifySynapses(cloud, skeletons, maxSnapNm, {
        signal: ctx.signal,
        progress: (fraction) => ctx.progress(0.9 + fraction * 0.1, 'classifying'),
      })
      entry.classified.set(skeletons, { maxSnapNm, result: classified })
    }
    warnClassified(ctx.warn, classified, {
      fetched,
      pairs,
      pres: pres.length,
      posts: posts.length,
    })
    if (edgePairs.repeated > 0) {
      ctx.warn(
        `${edgePairs.repeated} edge rows repeat a pair already listed, and each carries the ` +
          `whole pair's counts. Summing the table counts those synapses again.`,
      )
    }
    const tallies = rowTallies(edgePairs.ends, classified.counts)
    const { fewer, more } = weightMismatch(table, tallies)
    if (fewer + more > 0) {
      const parts = [
        fewer > 0 && `${fewer} with fewer synapses than their \`weight\``,
        more > 0 && `${more} with more`,
      ].filter(Boolean)
      ctx.warn(
        `${fewer + more} edges count differently from their \`weight\` column: ` +
          `${parts.join(' and ')}. \`Min confidence\` drops synapses, and a \`weight\` that is ` +
          `not a synapse count (a normalised one, or a sum over a group) will not match.`,
      )
    }

    return {
      edges: connectionEdgesTable(table, tallies, layoutOf(ctx.params.layout)),
      synapses: classifiedSynapsesTable(cloud, classified),
    }
  },
})

/**
 * The last cloud fetched for each edge list, so moving `Layout` or `Max snap distance` — which
 * shape the answer but not the question — does not re-fetch a cross product. Held per edge-list
 * value, weakly, so it lives exactly as long as the table it answers, and cards on different edge
 * lists do not evict each other. Checked against the dataset value (handed back by identity until
 * its node re-runs), `Min confidence` and the listed pairs, which also carry the column choice.
 * **Only the rows on a listed pair are kept** (`rowsOnPairs`): the raw cross product is mostly
 * rows no edge lists. Beside it, the classification per skeletons value and snap distance, so a
 * `Layout` edit re-shapes the table without classifying again.
 *
 * The **resolved** cloud, never the promise: an expensive node runs only on a Run, so sharing a
 * fetch in flight buys nothing and would hand one run's signal to another. A hit says how old it
 * is (`reportFetched`) and replays what the source warned, or the card would read clean on a
 * cloud the source had truncated. Clear Cache (`ctx.refresh`) drops it and fetches again.
 */
const held = new WeakMap<
  TableValue,
  {
    dataset: DatasetValue
    key: string
    cloud: PointsValue
    /** Rows the source returned, of which `cloud` is the ones on listed pairs. */
    fetched: number
    warnings: readonly string[]
    at: number
    classified: WeakMap<SkeletonsValue, { maxSnapNm: number; result: Classified }>
  }
>()

async function fetchCloud(
  ctx: { refresh: boolean; warn: (message: string) => void; reportFetched(at: number): void },
  fetch: (req: SynapsesBetweenRequest) => Promise<PointsValue>,
  req: SynapsesBetweenRequest,
  dataset: DatasetValue,
  edges: TableValue,
  wanted: ByPair<true>,
) {
  // The two sides are derived from the pairs, so the pairs and the threshold are the question.
  const pairs = [...wanted].map(([pre, posts]) => [pre, [...posts.keys()]])
  const key = JSON.stringify([req.minConfidence, pairs])
  if (ctx.refresh) held.delete(edges)
  const hit = held.get(edges)
  if (hit?.dataset === dataset && hit.key === key) {
    for (const message of hit.warnings) ctx.warn(message)
    ctx.reportFetched(hit.at)
    return hit
  }
  const warnings: string[] = []
  const raw = await fetch({
    ...req,
    onWarn: (message) => {
      warnings.push(message)
      ctx.warn(message)
    },
  })
  const at = Date.now()
  const entry = {
    dataset,
    key,
    cloud: rowsOnPairs(raw, wanted),
    fetched: raw.attributes.length,
    warnings,
    at,
    classified: new WeakMap(),
  }
  held.set(edges, entry)
  ctx.reportFetched(at)
  return entry
}

/** What could not be classified, and how much of the fetch was not on a listed edge. */
function warnClassified(
  warn: (message: string) => void,
  classified: Classified,
  fetch: { fetched: number; pairs: number; pres: number; posts: number },
): void {
  const { reasons, oneCoordinate, rows, unsplit } = classified
  if (unsplit > 0) {
    const parts = [
      reasons.noSkeleton > 0 && `${reasons.noSkeleton} ends on a neuron with no skeleton wired`,
      reasons.noSplit > 0 && `${reasons.noSplit} on a skeleton the split did not reach`,
      reasons.tooFar > 0 && `${reasons.tooFar} further than \`Max snap distance\` from it`,
    ].filter(Boolean)
    warn(`${unsplit} of ${rows} synapses are counted as unsplit: ${parts.join(', ')}.`)
  }
  if (oneCoordinate > 0) {
    warn(
      `${oneCoordinate} of ${rows} synapses have one position for both ends, so both ` +
        `compartments were read there. Where an axon and a dendrite interdigitate, the ` +
        `postsynaptic end may be on the wrong one.`,
    )
  }
  const discarded = fetch.fetched - rows
  if (discarded >= DISCARD_WARN && discarded > rows) {
    warn(
      `The query returned every synapse between ${fetch.pres} presynaptic and ${fetch.posts} ` +
        `postsynaptic neurons, ${fetch.fetched}, and ${rows} of them lie on the ` +
        `${fetch.pairs} listed edges.`,
    )
  }
}
