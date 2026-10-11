/**
 * How long a node may answer from the data it downloaded — the choice behind a right-click on
 * the card's `cached 3d ago ⟳`, and the same rows in the node menu.
 *
 * **On the label, because the label is where the question arises.** Somebody reading "3d" on a
 * base edited daily is deciding two things: whether to re-read it now (the click) and whether it
 * should ever have been three days (this). A right-click answers the second without moving the
 * pointer. It is not the only way in — a touch screen has no right-click — so `NodeContextMenu`
 * carries the same rows, through `CacheLifetimeRows`, rather than a second list of presets.
 *
 * The presets and what they mean are `core/cachePolicy.ts`'s; nothing here re-states them.
 */

import type { CachePolicy } from '../../core/cachePolicy'
import { CACHE_LIFETIMES, cachePreset } from '../../core/cachePolicy'
import { useGraphStore } from '../../store/graphStore'
import { ContextMenu } from '../menu/ContextMenu'

/** The preset rows, ticked where the node's policy matches. Picking one is one undo step. */
export function CacheLifetimeRows({
  nodeId,
  policy,
  onPicked,
}: {
  nodeId: string
  policy: CachePolicy | undefined
  onPicked: () => void
}) {
  const setNodeCachePolicy = useGraphStore((s) => s.setNodeCachePolicy)
  const ticked = cachePreset(policy)
  return (
    <>
      {CACHE_LIFETIMES.map((preset, i) => {
        const checked = i === ticked
        return (
          <button
            key={preset.label}
            type="button"
            role="menuitemradio"
            aria-checked={checked}
            className="context-menu__item"
            title={preset.title}
            onClick={() => {
              setNodeCachePolicy(nodeId, preset.policy)
              onPicked()
            }}
          >
            {preset.label}
            <span aria-hidden="true">{checked ? '✓' : ''}</span>
          </button>
        )
      })}
    </>
  )
}

interface CacheLifetimeMenuProps {
  /** Where the right-click landed, in client coordinates. */
  at: { x: number; y: number }
  nodeId: string
  policy: CachePolicy | undefined
  /** What a left click on the label does: drop the node's data cache and run it. */
  onRefresh: () => void
  onClose: () => void
}

export function CacheLifetimeMenu({
  at,
  nodeId,
  policy,
  onRefresh,
  onClose,
}: CacheLifetimeMenuProps) {
  return (
    // Portalled: the card is inside React Flow's transformed pane, which would scale the menu
    // with the zoom and cap its stacking order.
    <ContextMenu at={at} onClose={onClose} portal label="Cache lifetime">
      <div className="context-menu__caption">Cache lifetime</div>
      <CacheLifetimeRows nodeId={nodeId} policy={policy} onPicked={onClose} />
      <div className="context-menu__sep" />
      <button
        type="button"
        className="context-menu__item"
        title="Forget the data this node downloaded and run it"
        onClick={() => {
          onClose()
          onRefresh()
        }}
      >
        Fetch Again Now
      </button>
    </ContextMenu>
  )
}
