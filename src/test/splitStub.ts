/**
 * A Split Axon/Dendrite answer with no Python: every node dendrite, each neuron's first node axon,
 * every neuron "ok". For suites that need a split of the right shape to run what sits around it —
 * vitest has no Pyodide. The real split is `pnpm probe:split`'s.
 */

import type { SplitCompartmentsRequest, SplitCompartmentsResult } from '../pyodide/topology'

export function fakeSplit(
  request: Pick<SplitCompartmentsRequest, 'parents' | 'offsets'>,
): Promise<SplitCompartmentsResult> {
  const compartment = new Int32Array(request.parents.length).fill(1)
  for (let i = 0; i + 1 < request.offsets.length; i++) compartment[request.offsets[i]!] = 2
  return Promise.resolve({ compartment, status: new Int32Array(request.offsets.length - 1) })
}
