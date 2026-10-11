/**
 * Connection compartments: every synapse of an edge list classified by the compartment at each
 * end — axo-dendritic, axo-axonic, dendro-axonic, dendro-dendritic, linker or unsplit — and
 * counted per edge.
 *
 * **Each end is read on its own neuron's skeleton, at its own position.** The presynaptic site
 * snaps to the presynaptic skeleton and the postsynaptic site to the postsynaptic one; why one
 * coordinate for both is not enough is `SynapseDraw`'s. A source with one position per synapse is
 * classified from it at both ends, and the node says how many.
 *
 * **Unsplit is a class, not a dropped row.** A synapse whose neuron has no skeleton here, whose
 * skeleton carries no split, or whose site is further than the snap distance from its skeleton is
 * counted as `unsplit`, so the six counts of an edge always add up to the synapses fetched for it.
 * A synapse that is dropped would make the four real classes look like the whole edge.
 *
 * Headless and pure: the node fetches, this counts. See `docs/compartments-plan.md`.
 */

import type { NeuronId } from '../../core/ids'
import { ID_COLUMN_NAME, idText, isNeuronId } from '../../core/ids'
import { column, findColumn, tableSchema } from '../../core/types'
import type { TableSchema } from '../../core/types'
import type {
  CellValue,
  PointsValue,
  SkeletonGeometry,
  SkeletonsValue,
  TableValue,
} from '../../core/values'
import { CODE_AXON, CODE_DENDRITE, CODE_LINKER, makeTable, splitName } from '../../core/values'
import type { Slicer } from '../../core/slice'
import { sliced } from '../../core/slice'
import { selectPointsAt } from './iterables'
import { PARTNER_POSITION_COLUMNS, synapsesBetweenSchema } from '../../data/source'
import { foldColumns } from './tableOps'
import { nodeLocator } from './topologyOps'

/**
 * The six classes, in the order the wide layout's columns and the long layout's rows take. The
 * first four columns are fish2's spelling of the published per-compartment weights, so a computed
 * table and a published one read the same.
 */
export const CONNECTION_CLASSES = [
  { name: 'axo-axonic', column: 'weightAxonAxon' },
  { name: 'axo-dendritic', column: 'weightAxonDendrite' },
  { name: 'dendro-axonic', column: 'weightDendriteAxon' },
  { name: 'dendro-dendritic', column: 'weightDendriteDendrite' },
  { name: 'linker', column: 'weightLinker' },
  { name: 'unsplit', column: 'weightUnsplit' },
] as const

/** Each class's index, by name — so reordering `CONNECTION_CLASSES` reorders and mislabels nothing. */
const classAt = (name: (typeof CONNECTION_CLASSES)[number]['name']) =>
  CONNECTION_CLASSES.findIndex((c) => c.name === name)
const LINKER = classAt('linker')
const UNSPLIT = classAt('unsplit')
/** Presynaptic end (axon, dendrite) by postsynaptic end (axon, dendrite). */
const BY_ENDS = [
  [classAt('axo-axonic'), classAt('axo-dendritic')],
  [classAt('dendro-axonic'), classAt('dendro-dendritic')],
] as const

const known = (code: number | undefined) =>
  code === CODE_AXON || code === CODE_DENDRITE || code === CODE_LINKER

export type ConnectionLayout = 'wide' | 'long'

/** The wide layout's total, beside the six classes: what was counted, not the input's `weight`. */
const SYNAPSES_COLUMN = 'synapses'
/** The long layout's class column, after `weight` — where `Split by region` puts `roi`. */
const CONNECTION_COLUMN = 'connection'
const WEIGHT_COLUMN = 'weight'

/**
 * One synapse's class from the code at each end; `undefined` is an end that could not be read.
 * Unsplit wins over linker: an end nobody could place says nothing about the other end's class.
 */
export function connectionClass(pre: number | undefined, post: number | undefined): number {
  if (!known(pre) || !known(post)) return UNSPLIT
  if (pre === CODE_LINKER || post === CODE_LINKER) return LINKER
  return BY_ENDS[pre === CODE_AXON ? 0 : 1][post === CODE_AXON ? 0 : 1]
}

/** Why an end could not be read, counted for the node's warning. */
export interface UnsplitReasons {
  /** The end's neuron has no skeleton on the `Skeletons` wire. */
  noSkeleton: number
  /** It has one, and the split did not reach it (`no synapses`, `multiple roots`). */
  noSplit: number
  /** The site is further than the snap distance from every node of its skeleton. */
  tooFar: number
}

/** Every split skeleton by its neuron id; the first item wins where an id repeats. */
function skeletonsById(
  skeletons: Pick<SkeletonsValue, 'items'>,
): Map<string, SkeletonGeometry> {
  const out = new Map<string, SkeletonGeometry>()
  for (const item of skeletons.items) if (!out.has(item.id)) out.set(item.id, item)
  return out
}

/* ------------------------------------------------------------------------------------------
 * The edge list, read once.
 * ---------------------------------------------------------------------------------------- */

/** Presynaptic id → postsynaptic id → something: a pair looked up without building a key. */
export type ByPair<V> = ReadonlyMap<string, ReadonlyMap<string, V>>

export interface EdgePairs {
  /** Per input row, its two neuron ids, or null for a row naming no pair. */
  readonly ends: readonly (readonly [string, string] | null)[]
  /** Each listed pair once. */
  readonly wanted: ByPair<true>
  /** The two sides, first-seen: what the fetch is bound to. */
  readonly pres: NeuronId[]
  readonly posts: NeuronId[]
  /** Distinct pairs, and rows repeating one already listed. */
  readonly pairs: number
  readonly repeated: number
  /** Rows whose two ends hold something that is not a neuron id, and the first of them. */
  readonly notIds: number
  readonly example?: string
}

/**
 * The pairs an edge table names, **both ends neuron ids**. Text is not enough: Adjacency's links
 * once named a neuron `KCg-d 19102`, which matches nothing when handed to a source, and every edge
 * then counted zero with nothing to say why. Such a row names no pair, and is counted so the node
 * can say so. A row with an empty end is a gap and is not.
 */
export function readEdgePairs(
  table: TableValue,
  preColumn: string,
  postColumn: string,
): EdgePairs {
  const preCells = table.data[preColumn] ?? []
  const postCells = table.data[postColumn] ?? []
  const ends: (readonly [string, string] | null)[] = []
  const wanted = new Map<string, Map<string, true>>()
  const pres = new Set<string>()
  const posts = new Set<string>()
  let pairs = 0
  let repeated = 0
  let notIds = 0
  let example: string | undefined
  for (let row = 0; row < table.length; row++) {
    const pre = idText(preCells[row] ?? null)
    const post = idText(postCells[row] ?? null)
    if (pre === null || post === null) {
      ends.push(null)
      continue
    }
    if (!isNeuronId(pre) || !isNeuronId(post)) {
      notIds++
      example ??= `"${pre}" → "${post}"`
      ends.push(null)
      continue
    }
    ends.push([pre, post])
    let targets = wanted.get(pre)
    if (!targets) wanted.set(pre, (targets = new Map()))
    if (targets.has(post)) repeated++
    else {
      targets.set(post, true)
      pairs++
    }
    pres.add(pre)
    posts.add(post)
  }
  return {
    ends,
    wanted,
    pres: [...pres],
    posts: [...posts],
    pairs,
    repeated,
    notIds,
    ...(example ? { example } : {}),
  }
}

/* ------------------------------------------------------------------------------------------
 * Classification.
 * ---------------------------------------------------------------------------------------- */

export interface Classified {
  /** Per pair, the six counts. A pair with no synapses is absent. */
  readonly counts: ByPair<Uint32Array>
  /** Synapses classified: every row of the cloud. */
  readonly rows: number
  /** The synapses output's three class columns, one cell per row of the cloud. */
  readonly columns: Readonly<
    Record<'preCompartment' | 'postCompartment' | 'connection', CellValue[]>
  >
  readonly reasons: UnsplitReasons
  /** Rows counted as unsplit. */
  readonly unsplit: number
  /** Rows classified from one coordinate, because the cloud carried no partner end. */
  readonly oneCoordinate: number
}

/**
 * Classify every row of a synapses-between cloud (`neuronId` presynaptic, `partnerId`
 * postsynaptic) against the split skeletons. Which rows are wanted is `rowsOnPairs`' question,
 * answered once before this: the node hands over only rows on a listed pair.
 *
 * The presynaptic end is the point; the postsynaptic end is `partnerX/Y/Z` where the cloud carries
 * them, and the point again where it does not. One grid per skeleton touched, for this call only:
 * held across runs it is about five times the skeleton's own size, invisible to the memory meter,
 * to save a few milliseconds per skeleton — the node holds the result instead (`held`).
 */
export async function classifySynapses(
  cloud: PointsValue,
  skeletons: Pick<SkeletonsValue, 'items'>,
  maxSnapNm: number,
  run: Slicer = {},
): Promise<Classified> {
  const byId = skeletonsById(skeletons)
  const reasons: UnsplitReasons = { noSkeleton: 0, noSplit: 0, tooFar: 0 }
  const maxSq = maxSnapNm * maxSnapNm
  const p = cloud.positions

  /** Per neuron id: its split and search, or why an end on it cannot be read. One lookup per end. */
  const ends = new Map<
    string,
    { split: Uint8Array; locate: ReturnType<typeof nodeLocator> } | 'noSkeleton' | 'noSplit'
  >()
  const endOf = (id: string) => {
    let end = ends.get(id)
    if (end === undefined) {
      const skeleton = byId.get(id)
      if (!skeleton) end = 'noSkeleton'
      else if (!skeleton.split) end = 'noSplit'
      else end = { split: skeleton.split, locate: nodeLocator(skeleton) }
      ends.set(id, end)
    }
    return end
  }

  /** The code at one end, or undefined with the reason counted. */
  const codeAt = (id: string, x: number, y: number, z: number): number | undefined => {
    const end = endOf(id)
    if (typeof end === 'string') {
      reasons[end]++
      return undefined
    }
    const node = end.locate(x, y, z, maxSq)
    if (node < 0) {
      reasons.tooFar++
      return undefined
    }
    return end.split[node]
  }

  const data = cloud.attributes.data
  const preCells = data[ID_COLUMN_NAME] ?? []
  const postCells = data.partnerId ?? []
  const [pxs, pys, pzs] = PARTNER_POSITION_COLUMNS.every((c) =>
    findColumn(cloud.attributes.schema, c),
  )
    ? PARTNER_POSITION_COLUMNS.map((c) => data[c]!)
    : []

  const counts = new Map<string, Map<string, Uint32Array>>()
  const rows = cloud.attributes.length
  const preCompartment = new Array<CellValue>(rows)
  const postCompartment = new Array<CellValue>(rows)
  const connection = new Array<CellValue>(rows)
  let unsplit = 0
  let oneCoordinate = 0
  // Sliced: a snap per end on the main thread, so a Cancel has to be able to reach it.
  await sliced(rows, run, (row) => {
    // An end with no id has no skeleton either, so it is read as one and counted unsplit.
    const pre = idText(preCells[row] ?? null) ?? ''
    const post = idText(postCells[row] ?? null) ?? ''
    const x = p[row * 3]!
    const y = p[row * 3 + 1]!
    const z = p[row * 3 + 2]!
    const preCode = codeAt(pre, x, y, z)
    // A missing partner coordinate on a both-ends cloud is that row's one end, not a zero.
    const px = pxs?.[row]
    let postCode: number | undefined
    if (typeof px === 'number') {
      postCode = codeAt(post, px, Number(pys![row]), Number(pzs![row]))
    } else {
      oneCoordinate++
      postCode = codeAt(post, x, y, z)
    }
    const cls = connectionClass(preCode, postCode)
    if (cls === UNSPLIT) unsplit++
    let byPost = counts.get(pre)
    if (!byPost) counts.set(pre, (byPost = new Map()))
    let tally = byPost.get(post)
    if (!tally) byPost.set(post, (tally = new Uint32Array(CONNECTION_CLASSES.length)))
    tally[cls]!++
    preCompartment[row] = preCode === undefined ? null : splitName(preCode)
    postCompartment[row] = postCode === undefined ? null : splitName(postCode)
    connection[row] = CONNECTION_CLASSES[cls]!.name
  })
  return {
    counts,
    rows,
    columns: { preCompartment, postCompartment, connection },
    reasons,
    unsplit,
    oneCoordinate,
  }
}

/* ------------------------------------------------------------------------------------------
 * The classified cloud — schema and value halves.
 * ---------------------------------------------------------------------------------------- */

/**
 * The synapses output: a synapses-between cloud with both ends, then the three class columns.
 * Always the both-ends schema — a one-coordinate source's rows carry null partner positions — so
 * the type is known before a Run, whichever source answers.
 */
export const CLASSIFIED_SYNAPSES_SCHEMA: TableSchema = foldColumns(
  synapsesBetweenSchema('both'),
  [
    column('preCompartment', 'str'),
    column('postCompartment', 'str'),
    column(CONNECTION_COLUMN, 'str'),
  ],
)

/** `CLASSIFIED_SYNAPSES_SCHEMA`'s value half: the cloud classified, with its classes. */
export function classifiedSynapsesTable(
  cloud: PointsValue,
  classified: Classified,
): PointsValue {
  const data: Record<string, CellValue[]> = { ...cloud.attributes.data, ...classified.columns }
  // A one-coordinate source's cloud has no partner columns: its rows read null there.
  for (const name of PARTNER_POSITION_COLUMNS) {
    data[name] ??= new Array<CellValue>(classified.rows).fill(null)
  }
  return { ...cloud, attributes: makeTable(CLASSIFIED_SYNAPSES_SCHEMA, data) }
}

/* ------------------------------------------------------------------------------------------
 * The edge table — schema and value halves, both layouts.
 * ---------------------------------------------------------------------------------------- */

/** The edges output for either layout. */
export function connectionEdgesSchema(
  input: TableSchema | undefined,
  layout: ConnectionLayout,
): TableSchema {
  const base = input ?? tableSchema()
  if (layout === 'wide') {
    return foldColumns(base, [
      ...CONNECTION_CLASSES.map((c) => column(c.column, 'i64', 'synapses')),
      column(SYNAPSES_COLUMN, 'i64', 'synapses'),
    ])
  }
  // Long: `weight` replaced in its slot by the class count, `connection` straight after it.
  const weight = column(WEIGHT_COLUMN, 'i64', 'synapses')
  const kept = base.columns.filter((c) => c.name !== CONNECTION_COLUMN)
  const at = kept.findIndex((c) => c.name === WEIGHT_COLUMN)
  const connection = column(CONNECTION_COLUMN, 'str')
  return at < 0
    ? tableSchema(...kept, weight, connection)
    : tableSchema(...kept.slice(0, at), weight, connection, ...kept.slice(at + 1))
}

const NO_SYNAPSES = new Uint32Array(CONNECTION_CLASSES.length)

/**
 * Each input row's six counts: its pair's (zeros where none were found), or null for a row naming
 * no pair. Read once per run, by both layouts and by `weightMismatch`.
 */
export function rowTallies(
  ends: EdgePairs['ends'],
  counts: ByPair<Uint32Array>,
): (Uint32Array | null)[] {
  return ends.map((pair) => (pair ? (counts.get(pair[0])?.get(pair[1]) ?? NO_SYNAPSES) : null))
}

/**
 * The edges with their classes counted, every input row carried in order.
 *
 * Counts are per pair, so an edge row repeating a pair carries that pair's counts again — the
 * node warns. A row naming no pair counts nothing and reads null.
 */
export function connectionEdgesTable(
  table: TableValue,
  tallies: readonly (Uint32Array | null)[],
  layout: ConnectionLayout,
): TableValue {
  const schema = connectionEdgesSchema(table.schema, layout)

  if (layout === 'wide') {
    const data: Record<string, CellValue[]> = { ...table.data }
    CONNECTION_CLASSES.forEach((c, k) => {
      data[c.column] = tallies.map((t) => (t ? t[k]! : null))
    })
    data[SYNAPSES_COLUMN] = tallies.map((t) => (t ? total(t) : null))
    return makeTable(schema, data)
  }

  // Long: each input cell once per class, then the class count and its name — one array a column.
  const per = CONNECTION_CLASSES.length
  const rows = tallies.length
  const data: Record<string, CellValue[]> = {}
  for (const c of schema.columns) {
    const from = table.data[c.name]
    if (!from || c.name === WEIGHT_COLUMN || c.name === CONNECTION_COLUMN) continue
    const out = new Array<CellValue>(rows * per)
    for (let row = 0; row < rows; row++) {
      const cell = from[row] ?? null
      for (let k = 0; k < per; k++) out[row * per + k] = cell
    }
    data[c.name] = out
  }
  const weight = new Array<CellValue>(rows * per)
  const connection = new Array<CellValue>(rows * per)
  for (let row = 0; row < rows; row++) {
    const t = tallies[row]
    for (let k = 0; k < per; k++) {
      weight[row * per + k] = t ? t[k]! : null
      connection[row * per + k] = CONNECTION_CLASSES[k]!.name
    }
  }
  data[WEIGHT_COLUMN] = weight
  data[CONNECTION_COLUMN] = connection
  return makeTable(schema, data)
}

const total = (tally: Uint32Array) => tally.reduce((a, b) => a + b, 0)

/**
 * Edge rows whose own `weight` counts fewer or more synapses than were classified here. Not an
 * error: a confidence floor drops some, and a `weight` that is not a synapse count will not match.
 */
export function weightMismatch(
  table: TableValue,
  tallies: readonly (Uint32Array | null)[],
): { fewer: number; more: number } {
  const weights = table.data[WEIGHT_COLUMN]
  let fewer = 0
  let more = 0
  if (!weights) return { fewer, more }
  tallies.forEach((tally, row) => {
    const weight = weights[row]
    if (typeof weight !== 'number' || !tally) return
    const counted = total(tally)
    if (counted < Math.round(weight)) fewer++
    else if (counted > Math.round(weight)) more++
  })
  return { fewer, more }
}

/**
 * The cloud's rows on a `wanted` pair — what the node keeps of a cross-product fetch for the
 * session, where the raw cloud is mostly rows no edge lists.
 */
export function rowsOnPairs(cloud: PointsValue, wanted: ByPair<true>): PointsValue {
  const pres = cloud.attributes.data[ID_COLUMN_NAME] ?? []
  const posts = cloud.attributes.data.partnerId ?? []
  const rows: number[] = []
  for (let row = 0; row < cloud.attributes.length; row++) {
    const pre = idText(pres[row] ?? null)
    const post = idText(posts[row] ?? null)
    if (pre !== null && post !== null && wanted.get(pre)?.has(post)) rows.push(row)
  }
  return selectPointsAt(cloud, rows)
}
