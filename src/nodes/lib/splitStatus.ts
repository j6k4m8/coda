/**
 * A neuron's split status, in words, and which of them carry labels.
 *
 * Here rather than beside the Python bridge's numeric codes because the methods that produce a
 * status are not all Python: a split by ROI runs in TypeScript, and a split by tag adds a status
 * (`no tags`) no Python code returns. Headless and import-free, so the daylight worker can read it
 * without pulling in `src/pyodide` (`topologyOps.ts` is reachable from there).
 */

/** `ok`, why a neuron has no split, or — `not converged` — that it has one to treat with care. */
export type SplitStatus =
  'ok' | 'multiple roots' | 'no synapses' | 'no tags' | 'not converged' | 'not split'

/**
 * Whether a status carries labels: a split, or a label propagation that hit its cap and still
 * labelled, as navis does.
 */
export function hasSplit(status: SplitStatus): boolean {
  return status === 'ok' || status === 'not converged'
}
