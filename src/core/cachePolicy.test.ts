import { describe, expect, it } from 'vitest'

import { cacheExpired, cacheLifetimeLabel, lapsesAt, validCachePolicy } from './cachePolicy'

const HOUR = 3_600_000

describe('cache policy', () => {
  it('lapses only a positive lifetime, and only data that said when it was read', () => {
    expect(lapsesAt(1000, { maxAgeMs: HOUR })).toBe(1000 + HOUR)
    // Don't cache is spent at execution; a result lapsing as it lands would re-run every Run.
    expect(lapsesAt(1000, { maxAgeMs: 0 })).toBeUndefined()
    expect(lapsesAt(undefined, { maxAgeMs: HOUR })).toBeUndefined()
    expect(lapsesAt(1000, undefined)).toBeUndefined()
  })

  it('is expired strictly past the lifetime', () => {
    expect(cacheExpired(0, { maxAgeMs: HOUR }, HOUR)).toBe(false)
    expect(cacheExpired(0, { maxAgeMs: HOUR }, HOUR + 1)).toBe(true)
  })

  it('keeps a hand-written lifetime and drops anything that is not one', () => {
    expect(validCachePolicy({ maxAgeMs: 6 * HOUR })).toEqual({ maxAgeMs: 6 * HOUR })
    expect(validCachePolicy({ maxAgeMs: 0 })).toEqual({ maxAgeMs: 0 })
    for (const bad of [undefined, null, 'a week', { maxAgeMs: -1 }, { maxAgeMs: Infinity }]) {
      expect(validCachePolicy(bad)).toBeUndefined()
    }
  })

  it('names a preset by its label and anything else by its length', () => {
    expect(cacheLifetimeLabel(undefined)).toBe('Default')
    expect(cacheLifetimeLabel({ maxAgeMs: 0 })).toBe('Don’t cache')
    expect(cacheLifetimeLabel({ maxAgeMs: HOUR })).toBe('1 hour')
    expect(cacheLifetimeLabel({ maxAgeMs: 6 * HOUR })).toBe('6 hours')
    expect(cacheLifetimeLabel({ maxAgeMs: 2 * 24 * HOUR })).toBe('2 days')
    expect(cacheLifetimeLabel({ maxAgeMs: 10 * 60_000 })).toBe('10 minutes')
  })
})
