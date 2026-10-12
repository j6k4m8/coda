/**
 * Shapes and rules shared by everything that builds a connectivity answer *locally*.
 *
 * `neuronFilter.ts`'s reason, one question over. That module exists because CAVE became the
 * second backend to filter neurons in the browser, and a second copy of "what does this pattern
 * mean" would have had one graph answer differently on two backends. This is the same seam for
 * connectivity: every source's `fetchAdjacency`, and a user-supplied edge set, fold an edge list
 * into its matrix through `matrixFromEdges` here. The fold is not obvious enough to write twice —
 * an untyped neuron's bucket and a pair outside the requested rows both produce a plausible wrong
 * matrix rather than an error.
 */

import { ID_COLUMN_NAME, type NeuronId } from '../core/ids'
import type { AdjacencyMatrix, MatrixLine, TableValue } from '../core/values'
import { makeMatrix } from '../core/values'

/** One connection, oriented presynaptic → postsynaptic whatever answered it. */
export interface Edge {
  pre: NeuronId
  post: NeuronId
  weight: number
}

/**
 * Neuron id to cell type, from a dataset's own neuron index.
 *
 * Memoised on the *table's identity*, which is safe rather than merely likely to hit:
 * `cacheGet` promotes a cached table into `cache.ts`'s module-level map and hands back the same
 * object, so repeat `neuronIndex` calls for one dataset return an identical `TableValue`. Worth
 * the memo because this is called once per hop per direction — `Hops: 3, Direction: both` built
 * the same 108,000-entry map six times in one Run before it existed. Same idiom as
 * `searchIndexFor`, `statsFor` and `cornersByBucket`.
 *
 * A neuron with no type is **absent** rather than mapped to null or to the empty string, so a
 * caller can tell "not typed" from "typed as nothing" with a single `Map.get`.
 */
const typeCache = new WeakMap<TableValue, Map<NeuronId, string>>()

export function typesOf(index: TableValue): Map<NeuronId, string> {
  const cached = typeCache.get(index)
  if (cached) return cached
  const lookup = new Map<NeuronId, string>()
  const ids = index.data[ID_COLUMN_NAME]
  const types = index.data.type
  if (ids && types) {
    for (let i = 0; i < index.length; i++) {
      const type = types[i]
      if (typeof type === 'string' && type) lookup.set(String(ids[i]), type)
    }
  }
  typeCache.set(index, lookup)
  return lookup
}

/**
 * One axis of an adjacency over the neurons asked about: a first-seen dedup of their keys, each
 * with its line, and `at` to find an edge end's position. **The one statement of the key rule**:
 * grouped by type a typed neuron joins its type's line (no id); an untyped one keeps a line of its
 * own under its id, which is `matrixFromEdges`' first rule. Per neuron the line is the neuron,
 * typed where the source knows it.
 */
function axisLines(
  ids: readonly NeuronId[],
  types: ReadonlyMap<NeuronId, string>,
  groupByType: boolean,
  labelOf: (key: string, line: MatrixLine) => string,
): { labels: string[]; lines: MatrixLine[]; at: (id: NeuronId) => number | undefined } {
  const keyOf = groupByType ? (id: NeuronId) => types.get(id) || id : (id: NeuronId) => id
  const position = new Map<string, number>()
  const labels: string[] = []
  const lines: MatrixLine[] = []
  for (const id of ids) {
    const key = keyOf(id)
    if (position.has(key)) continue
    const type = types.get(id) ?? null
    const line = groupByType && type ? { id: null, type } : { id, type }
    position.set(key, lines.length)
    lines.push(line)
    labels.push(labelOf(key, line))
  }
  return { labels, lines, at: (id) => position.get(keyOf(id)) }
}

/**
 * Fold an edge list into the matrix an Adjacency node asked for.
 *
 * Two rules in here, and each produces a plausible wrong picture rather than an error:
 *
 *  - **An untyped neuron keeps its own id as its key** rather than joining a bucket called
 *    "null" — `profileStats`' rule, and for its reason: merging them puts a fictitious type at
 *    the top of the list. On male-CNS that bucket would be one of the largest rows in the matrix.
 *  - **A pair outside the requested rows or columns is dropped, not appended.** The axes are the
 *    ids that were *asked about*; a source free to add a row would answer a different question,
 *    and the caller has already sized its picture around the list it sent.
 */
export function matrixFromEdges(
  edges: readonly Edge[],
  sourceIds: readonly NeuronId[],
  targetIds: readonly NeuronId[],
  types: ReadonlyMap<NeuronId, string>,
  /** Absent is off: a request that does not ask to group is per neuron. */
  groupByType: boolean | undefined,
  /** A display label other than the key, e.g. neuPrint's `type id`; the lines are unaffected. */
  labelOf: (key: string, line: MatrixLine) => string = (key) => key,
): AdjacencyMatrix {
  const rows = axisLines(sourceIds, types, groupByType === true, labelOf)
  const cols = axisLines(targetIds, types, groupByType === true, labelOf)
  const width = cols.lines.length
  const values = new Float64Array(rows.lines.length * width)
  for (const edge of edges) {
    const r = rows.at(edge.pre)
    const c = cols.at(edge.post)
    if (r === undefined || c === undefined) continue
    values[r * width + c] = (values[r * width + c] ?? 0) + edge.weight
  }
  return {
    ...makeMatrix(rows.labels, cols.labels, values, 'synapses', 'count'),
    lines: { rows: rows.lines, cols: cols.lines },
  }
}
