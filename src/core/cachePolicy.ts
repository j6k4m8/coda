/**
 * How long one node may answer from the data it downloaded — the per-node half of the data cache.
 *
 * A node declaring `dataCache` reads through a store that keeps what a server returned: a month in
 * IndexedDB for tables, the session for geometry. That default is right for a released connectome
 * and wrong for a spreadsheet somebody edits daily, and the node cannot tell which it is reading.
 * So the person who can is given the choice, per card.
 *
 * **On the node, never in its params.** A param takes part in the provenance key (invariant 4), so
 * shortening a lifetime would re-run the node — and re-running is the one thing the setting is not
 * about. It changes *when* the data is read again, never what `evaluate` makes of it. The field is
 * `GraphNode.cache`, written only when somebody chose something, like `hints`.
 *
 * **What a lifetime does is the scheduler's.** A result whose data is older than its node's
 * lifetime lapses: the node and everything downstream turn stale with their outputs still on
 * screen, and the next run of the node reads the server rather than the store. See `Scheduler`'s
 * `lapse`. Nothing fetches on a timer; a lapse changes a badge.
 *
 * **Data cache only.** `Don't cache` means every execution reads the server; it does not make the
 * node re-run on an unchanged graph, whose result cache still answers for its key.
 */

/** A node's own lifetime for its data cache. */
export interface CachePolicy {
  /**
   * The longest the data behind a result may be held, in ms. `0` is **Don't cache** — every
   * execution reads the server — which is a lifetime with nothing in it rather than a second field.
   */
  maxAgeMs: number
}

const HOUR = 60 * 60 * 1000
const DAY = 24 * HOUR

const AGED = 'Read the server again once the data is older than this'

/**
 * The choices a menu offers, in order. A `policy` of `undefined` is the node's own default — no
 * field at all — so a row's policy goes straight to `setNodeCachePolicy`.
 *
 * Presets rather than free entry: the cases somebody has are "edited all day", "edited daily" and
 * "released", and a typed `6h` is a parser and a validation message for a distinction nobody acts
 * on. Nothing past thirty days, which is already the table store's own expiry — a longer lifetime
 * would claim more than the store keeps.
 */
export const CACHE_LIFETIMES: ReadonlyArray<{
  label: string
  policy: CachePolicy | undefined
  /** What picking it means, for the row's tooltip. */
  title: string
}> = [
  { label: 'Default', policy: undefined, title: 'What this node keeps by default' },
  {
    label: 'Don’t cache',
    policy: { maxAgeMs: 0 },
    title: 'Read the server every time it runs',
  },
  { label: '1 hour', policy: { maxAgeMs: HOUR }, title: AGED },
  { label: '1 day', policy: { maxAgeMs: DAY }, title: AGED },
  { label: '1 week', policy: { maxAgeMs: 7 * DAY }, title: AGED },
  { label: '30 days', policy: { maxAgeMs: 30 * DAY }, title: AGED },
]

/** Which preset a policy is — the ticked row — or -1 for one written by hand. */
export function cachePreset(policy: CachePolicy | undefined): number {
  return CACHE_LIFETIMES.findIndex((p) => p.policy?.maxAgeMs === policy?.maxAgeMs)
}

/**
 * A stored policy, or undefined when there is none worth keeping.
 *
 * Lenient-but-checked, like the loader's other passes: anything not a finite, non-negative number
 * of ms is dropped rather than refused, and a file that says nothing loads as the default. A value
 * off the preset list is kept — a hand-edited `6h` means what it says — and simply leaves every
 * row of the menu unticked.
 */
export function validCachePolicy(raw: unknown): CachePolicy | undefined {
  if (!raw || typeof raw !== 'object') return undefined
  const ms = (raw as { maxAgeMs?: unknown }).maxAgeMs
  if (typeof ms !== 'number' || !Number.isFinite(ms) || ms < 0) return undefined
  return { maxAgeMs: ms }
}

/**
 * When a result read at `fetchedAt` lapses, or undefined when it never does.
 *
 * Never for **Don't cache**: that policy is spent at execution (each one reads the server), and a
 * result lapsing the instant it lands would make an unchanged graph re-run on every Run — the
 * result cache is deliberately out of this setting's reach.
 */
export function lapsesAt(
  fetchedAt: number | undefined,
  policy: CachePolicy | undefined,
): number | undefined {
  if (fetchedAt === undefined || !policy || policy.maxAgeMs <= 0) return undefined
  return fetchedAt + policy.maxAgeMs
}

/** Whether the data read at `fetchedAt` is older than the policy allows, at `now`. */
export function cacheExpired(
  fetchedAt: number | undefined,
  policy: CachePolicy | undefined,
  now: number = Date.now(),
): boolean {
  const due = lapsesAt(fetchedAt, policy)
  return due !== undefined && now > due
}

/** The preset's name for a policy, or a plain duration for one written by hand. */
export function cacheLifetimeLabel(policy: CachePolicy | undefined): string {
  const preset = CACHE_LIFETIMES[cachePreset(policy)]
  if (preset) return preset.label
  const ms = policy!.maxAgeMs
  if (ms % DAY === 0) return count(ms / DAY, 'day')
  if (ms >= HOUR) return count(Math.round(ms / HOUR), 'hour')
  return count(Math.max(1, Math.round(ms / 60_000)), 'minute')
}

/**
 * `n noun(s)`. Not `style/format`'s `plural`: this module is under `core/graph.ts`, which
 * `nodes/lib` imports, and that layer must not reach `src/style` at any depth (invariant 1).
 */
function count(n: number, noun: string): string {
  return `${n} ${noun}${n === 1 ? '' : 's'}`
}
