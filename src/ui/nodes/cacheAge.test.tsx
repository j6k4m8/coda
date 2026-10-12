// @vitest-environment jsdom

/**
 * `cached 3d ago ⟳` — the card's one statement about data it did not just fetch.
 *
 * The gap it closes: a node reading through `loadCachedTable` sits green and answers in
 * milliseconds whether the copy is four seconds or four weeks old, because a hit and a fresh read
 * are indistinguishable from the rows. So the age is *reported* by `evaluate` and kept in the
 * scheduler's cache entry, which is what makes it survive a result being restored rather than
 * recomputed — the case where nothing runs and anything in `NodeRunInfo` would be gone.
 *
 * Driven through the real store and a real Scheduler, because the interesting half is the wiring
 * rather than the drawing: the report reaching the card, and the click reaching both caches.
 */

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { registerNode } from '../../core/registry'
import { T, column, tableSchema } from '../../core/types'
import { tableFromRows } from '../../core/values'
import { MockSource } from '../../data/mock/MockSource'
import { registerSource } from '../../data/source'
import '../../nodes'
import { useGraphStore } from '../../store/graphStore'
import { clearStorage } from '../../test/jsdomStubs'
import { CacheAge } from './CacheAge'
import { CacheLifetimeMenu } from './CacheLifetimeMenu'

const DAY = 86_400_000

/** A node that reports a fetch time, standing in for an annotation source. */
let reportAt = 0
let runs = 0

beforeAll(() => {
  registerSource(new MockSource({ latencyMs: 0 }))
  registerNode({
    type: 'test.cacheAge',
    label: 'cache age',
    category: 'utility',
    cost: 'expensive',
    dataCache: true,
    inputs: [],
    outputs: [{ id: 'out', label: 'Out', type: T.table() }],
    inferOutputs: () => ({ out: T.table() }),
    evaluate: (ctx) => {
      runs += 1
      // A cache hit reports the *stored* time; a fresh read reports now. The node cannot tell
      // which it got, which is the whole reason this is reported rather than inferred.
      ctx.reportFetched(ctx.refresh ? Date.now() : reportAt)
      return { out: tableFromRows(tableSchema(column('x', 'i64')), [{ x: 1 }]) }
    },
  })
})

beforeEach(() => {
  clearStorage()
  useGraphStore.getState().newGraph()
  runs = 0
  reportAt = Date.now() - 3 * DAY
})

afterEach(cleanup)

describe('the age itself', () => {
  it('draws nothing when the node reported no fetch', () => {
    // The honest state for a Filter: it has no data cache, so there is no age and no control.
    const { container } = render(<CacheAge fetchedAt={undefined} onRefresh={() => undefined} />)
    expect(container.textContent).toBe('')
  })

  it('says how old, to the largest whole unit', () => {
    render(<CacheAge fetchedAt={Date.now() - 3 * DAY} onRefresh={() => undefined} />)
    expect(screen.getByRole('button').textContent).toContain('cached 3d ago')
  })

  it('says so even when the data is seconds old', () => {
    /*
     * Not gated on an age threshold. A line that appears only when something is wrong is a line
     * nobody learns to look at — the rule that keeps geometry units printed when they are the
     * expected ones — and `0s` today is what makes `28d` believable in a month.
     */
    render(<CacheAge fetchedAt={Date.now()} onRefresh={() => undefined} />)
    expect(screen.getByRole('button').textContent).toContain('cached 0s ago')
  })

  it('calls back on a click, without letting it reach the card behind', () => {
    const onRefresh = vi.fn()
    const onCard = vi.fn()
    render(
      <div onClick={onCard}>
        <CacheAge fetchedAt={Date.now()} onRefresh={onRefresh} />
      </div>,
    )
    fireEvent.click(screen.getByRole('button'))
    expect(onRefresh).toHaveBeenCalledOnce()
    // The foot sits on a draggable, selectable card; a refresh must not also select it.
    expect(onCard).not.toHaveBeenCalled()
  })
})

describe('the report reaching the card', () => {
  it('survives a result being restored rather than recomputed', async () => {
    /*
     * The reason this lives in the scheduler's *cache entry* and not in `NodeRunInfo`. A second
     * run over an unchanged graph re-executes nothing, so a run-time report would be gone while
     * the stale table it described stayed on screen.
     */
    const store = useGraphStore.getState()
    const id = store.addNode('test.cacheAge', { x: 0, y: 0 })
    await store.runAll()
    expect(runs).toBe(1)
    expect(useGraphStore.getState().nodeFetchedAt(id)).toBe(reportAt)

    await useGraphStore.getState().runAll()
    expect(runs).toBe(1)
    expect(useGraphStore.getState().nodeFetchedAt(id)).toBe(reportAt)
  })

  it('is replaced by a refresh, which reaches both caches', async () => {
    const store = useGraphStore.getState()
    const id = store.addNode('test.cacheAge', { x: 0, y: 0 })
    await store.runAll()
    const before = useGraphStore.getState().nodeFetchedAt(id)!
    expect(Date.now() - before).toBeGreaterThan(DAY)

    // What the ⟳ does. Invalidating alone would re-run the node and get the same stale age back
    // out of IndexedDB, which is the failure the whole pair exists for.
    useGraphStore.getState().clearNodeCache(id)
    await useGraphStore.getState().runNode(id)

    await waitFor(() => expect(runs).toBe(2))
    expect(Date.now() - useGraphStore.getState().nodeFetchedAt(id)!).toBeLessThan(DAY)
  })
})

/*
 * The node's cache lifetime, chosen on this label. The scheduler's half — what a lapse does to the
 * badge, the key and the next run — is `scheduler.test.ts`'; this is the label's words and the
 * gesture that reaches the setting.
 */
describe('the cache lifetime', () => {
  it('says fetched rather than cached when nothing is kept', () => {
    render(
      <CacheAge fetchedAt={Date.now()} onRefresh={() => undefined} policy={{ maxAgeMs: 0 }} />,
    )
    expect(screen.getByRole('button').textContent).toContain('fetched 0s ago')
  })

  it('says expired once the data is older than the lifetime, and not before', () => {
    const { rerender } = render(
      <CacheAge
        fetchedAt={Date.now() - 3 * DAY}
        onRefresh={() => undefined}
        policy={{ maxAgeMs: 7 * DAY }}
      />,
    )
    expect(screen.getByRole('button').textContent).not.toContain('expired')
    rerender(
      <CacheAge
        fetchedAt={Date.now() - 3 * DAY}
        onRefresh={() => undefined}
        policy={{ maxAgeMs: DAY }}
      />,
    )
    const button = screen.getByRole('button')
    expect(button.textContent).toContain('cached 3d ago · expired')
    expect(button.className).toContain('coda-node__cache--expired')
  })

  it('opens its own menu on a right-click, and not the card’s', () => {
    // React Flow's node menu is an `onContextMenu` on the wrapper this label sits inside.
    const onMenu = vi.fn()
    const onCard = vi.fn()
    render(
      <div onContextMenu={onCard}>
        <CacheAge fetchedAt={Date.now()} onRefresh={() => undefined} onMenu={onMenu} />
      </div>,
    )
    fireEvent.contextMenu(screen.getByRole('button'), { clientX: 40, clientY: 60 })
    expect(onMenu).toHaveBeenCalledWith({ x: 40, y: 60 })
    expect(onCard).not.toHaveBeenCalled()
  })

  it('leaves the browser’s menu alone where there is nothing to choose', () => {
    const onCard = vi.fn()
    render(
      <div onContextMenu={onCard}>
        <CacheAge fetchedAt={Date.now()} onRefresh={() => undefined} />
      </div>,
    )
    fireEvent.contextMenu(screen.getByRole('button'))
    expect(onCard).toHaveBeenCalledOnce()
  })

  it('is chosen from the menu as one undoable edit, ticked where it stands', () => {
    const store = useGraphStore.getState()
    const id = store.addNode('test.cacheAge', { x: 0, y: 0 })
    const onClose = vi.fn()
    const menu = () => (
      <CacheLifetimeMenu
        at={{ x: 0, y: 0 }}
        nodeId={id}
        policy={useGraphStore.getState().graph.nodes.find((n) => n.id === id)?.cache}
        onRefresh={() => undefined}
        onClose={onClose}
      />
    )
    const { rerender } = render(menu())
    expect(
      screen.getByRole('menuitemradio', { name: /Default/ }).getAttribute('aria-checked'),
    ).toBe('true')

    fireEvent.click(screen.getByRole('menuitemradio', { name: /1 day/ }))
    const cacheOf = () => useGraphStore.getState().graph.nodes.find((n) => n.id === id)?.cache
    expect(cacheOf()).toEqual({ maxAgeMs: DAY })
    expect(onClose).toHaveBeenCalled()
    rerender(menu())
    expect(
      screen.getByRole('menuitemradio', { name: /1 day/ }).getAttribute('aria-checked'),
    ).toBe('true')

    // No run: the lifetime is not in the provenance key, so nothing about the result moved.
    expect(runs).toBe(0)
    useGraphStore.getState().undo()
    expect(cacheOf()).toBeUndefined()
  })
})
