/**
 * The Connectome pack's card drawings for its own `connectome:` types, merged into `NODE_GLYPHS`
 * by `ui/glyphs.ts`. The pack's nodes that keep built-in ids are drawn there, with the rest.
 *
 * Data only, held to `ui/glyphs.ts`' own lint rule: `nodes.html` draws these with no React and no
 * node definitions in it.
 */

import type { GlyphShape } from '../../ui/glyphs'

const glyphs: Readonly<Record<string, readonly GlyphShape[]>> = {
  /*
   * The dataset's stack of discs — the category's own silhouette, so the family reads — with a
   * plus beside it: a dataset you add parts to. The stack's right side stops short to leave the
   * plus room, the way the Custom backend nodes' drawing leaves room for its connectors.
   */
  'connectome:customDataset': [
    ['ellipse', { cx: '9.5', cy: '6.5', rx: '6', ry: '2.4' }],
    ['path', { d: 'M3.5 6.5v10c0 1.3 2.7 2.4 6 2.4' }],
    ['path', { d: 'M3.5 11.5c0 1.3 2.7 2.4 6 2.4M15.5 6.5V11' }],
    ['path', { d: 'M18 13.5v7M14.5 17h7' }],
  ],
  /*
   * Synapses Between's pair of points between two bars — the synapses of an edge list — with Split
   * Axon/Dendrite's one-side-solid-one-faint for the compartment at each end. Opacity, not colour:
   * the glyph has one ink, and the dashed outline is taken by "a user's selection".
   */
  'connectome:connectionCompartments': [
    ['path', { d: 'M3.6 6.4v11.2M20.4 6.4v11.2' }],
    ['circle', { cx: '7.8', cy: '12', r: '2.2', fill: 'currentColor', stroke: 'none' }],
    [
      'circle',
      {
        cx: '16.2',
        cy: '12',
        r: '2.2',
        fill: 'currentColor',
        fillOpacity: '.35',
        stroke: 'none',
      },
    ],
    ['path', { d: 'M10.6 12h2.6' }],
    ['path', { d: 'M12.4 10.6 13.8 12l-1.4 1.4' }],
  ],
}

export default glyphs
