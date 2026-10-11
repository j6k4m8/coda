# Plan: more than one way to split an axon from a dendrite

Split Axon/Dendrite had one method, synapse flow centrality (navis's `split_axon_dendrite`).
This adds three more behind a `Method` control on the same node: **label propagation** (navis's
`split_axon_dendrite_prop`), **by ROI** and **by tag**. Status: *all four phases built
(2026-10-11)*. Decisions taken with the user on 2026-10-11 are marked as such; the rest are
proposals.

## What stays the same, whatever the method

The node's contract is the labels, not the method, and everything downstream reads only the labels:

- **One output shape.** Every method writes `SkeletonGeometry.split` (one `CODE_*` per node: axon,
  dendrite, linker, or 0 for unassigned), the `Synapses` output's `compartment` column, and the
  `Summary` table. Connection Compartments, the 3D View's compartment colours and Topology's
  per-compartment columns need no change.
- **Hand corrections apply on top of every method.** They are positions and a scope, applied to
  whatever labels the method produced (`applyCorrections`), so they survive a change of method the
  same way they survive a threshold change. The editor, the dendrogram and the segregation index
  are method-blind.
- **Heal and the soma root.** Healing (`Heal fragmented skeletons`) and rooting at the soma apply to
  every method that walks the tree. The editor and the corrections already draw and correct on the
  healed tree (`arborHeal.ts`).
- **The status vocabulary grows by what is new**: today `ok`, `multiple roots`, `no synapses`. Each
  method adds only the refusals it can actually reach (below), and `splitWarning` keeps saying them
  in one sentence.

**Topology keeps flow only.** Its split is a checkbox and its thresholds are shared through
`splitParams`; the methods are this node's.

## The control

`Method`, an enum, default **"synapse flow"** so every saved card keeps its split (a plain
default: a card saved before the control was flow, so `absentMeans` is not needed). Options: "synapse flow", "label propagation", "by ROI", "by tag". Each
method's own params sit behind `visibleIf` on it, which is also what keeps a hidden method's params
out of the provenance key (invariant 4), so switching method and back returns the cached split.

**Inputs.** `Synapses` stays, and becomes *optional* for ROI and tag (*decided*): they do not need
synapses to split. When wired, `Summary` still counts synapses per compartment and the `Synapses`
output still gets its `compartment` column; when not, those columns are empty and the card says
why. Flow and propagation keep it required (`validate` says so per method). ROI adds a `Volumes`
input.

## 1. Label propagation

navis's `split_axon_dendrite_prop`, ported. *Decided:* in Python, in `topology.py` beside the flow
split, so `pnpm probe:split` can check it against navis node for node, as it does flow.

**How navis does it** (`navis/morpho/manipulation.py`, `navis/graph/graph_utils.py`):

1. Count pre- and postsynapses per node (nearest node, a KD-tree query). A node with more pre than
   post is labelled `pre`, more post `post`, a tie unlabelled. Its weight is `|post - pre|`, and
   with `use_weights="balance"` (the default) `pre` weights are scaled by `n_post / n_pre` so the
   commoner synapse type does not win by numbers alone.
2. Propagate: `F ← S · F` repeatedly, where `S` is the row-normalised adjacency **weighted by edge
   length** (`parent_dist`), undirected. `alpha` clamps the seeds (1 = no clamping, navis's
   default; 0 = fully clamped; between = soft).
3. Stop after `max_iter` (10,000) or when the hard labels have not changed for `tol` (1)
   iterations.
4. A node scoring higher for `pre` is axon, for `post` dendrite; a node no signal reached (another
   fragment) is unassigned.

**What it does not do**, and the card should say so in its help: no linker (only axon and
dendrite), and no cell-body fibre. navis warns that it "might return garbage" where axon and
dendrite are not well segregated, or where one synapse type dominates — the tell being slow
convergence or probabilities near 0.5 over large parts of the neuron.

**Port.** No scipy in the Pyodide lock (numpy and fastcore only), and none needed: on a tree, `S ·
F` is a sum over parent and child edges. *Built* as a gather over the adjacency in scipy's CSR
order and one `np.add.reduceat` per step, which is what matches navis to the last bit (below).
Synapses are already assigned to nodes on the TypeScript side (`assignSynapses`), so the request
carries per-node pre/post counts, as the flow split's does.

**Params** (in the Split group): `Weights` ("balance", "by count", "none"), `Clamping` (alpha,
 0–1; *decided* default 0.5, not navis's 1), and `Max iterations` / `Tolerance` as advanced.

**Extra output.** The probability per node (navis's `compartment_prob`, the larger normalised
score) as a node value, `axonProbability`, beside `flow` in `NODE_VALUES`, so the 3D View can shade
how sure the split is. Opt-in like `Write synapse flow`. *Not built:* one `Write node values`
switch for both — each method keeps its own, which reads better on the card.

**Refusals.** `no synapses` (no pre or no post at all: nothing to propagate from), and a new
**`not converged`** status when `max_iter` is reached, which is navis's own warning sign, said
rather than printed.

### Built (phase 1)

- `SplitSettings` is one shape per method (`{ heal } & ({ method: 'flow', … } | { method:
  'propagation', … })`), so each method keys its cached split on its own knobs only, and nothing
  reads "not propagation" as flow — the ROI and tag methods add a member each. What each method
  needs, writes and reads is one table, `SPLIT_METHODS`, described in `docs/nodes-morphology.md`.
- `Method` ("synapse flow" / "label propagation") on the node, flow's two knobs and `Write synapse
  flow` behind it; `Weights`, `Clamping`, and advanced `Max iterations` / `Tolerance` for the
  propagation, plus `Write probability` (node value `axonProbability`). Topology has no `Method`
  and reads flow's settings (now `SPLIT_METHODS.flow.read`).
- `_propagate_one` in `topology.py`. **`pnpm probe:split` holds it to navis** on navis's five
  example neurons at three settings: every node agrees, probabilities within 5e-6, with one
  recorded exception below.
- **Not converged keeps its labels**, as navis does: status `not converged`, which `hasSplit`
  counts as split (so the labels are stored, corrected and counted) and `splitWarning` says in a
  sentence of its own. The plan's first idea, refusing, was stricter than navis: on two of its five
  example neurons navis itself runs to 10,000 iterations without converging and returns labels.
- **The port sums as scipy does, and that turned out to matter.** navis stops once the hard labels
  have held for two steps; a dendrite front reaching unreached nodes changes no hard label (both
  scores 0, so `argmax` already says `post`), so *when* it stops can turn on the last bit of a
  score. Summed in float64, then element by element in float32, the port stopped two iterations
  early on one neuron and left two nodes unlabelled. Laying the adjacency out as scipy's CSR and
  taking row totals by numpy's own reduction (`np.add.reduceat`, which is scipy's `A.sum`) fixed
  the defaults exactly. One case remains: `alpha=0` with no weights, where scipy's compiled product
  and numpy still differ by one ulp at a tied score (measured 2.4e-7 after everything else
  matched), so navis stops two iterations later and labels two more front nodes. The probe allows
  that: every node labelled by both agrees, and at most 0.1% are labelled by one side only.
- **Finding: navis's default `alpha=1` oscillates on a tree.** With no clamping each step replaces
  a node's scores by its neighbours' average, and a tree is bipartite, so the scores can flip
  between two states forever. A nine-node Y with outputs on one arm and inputs on the other never
  converges at the default (alternating labels, `not converged`) and splits cleanly at `alpha` 0.5
  or 0. See the open questions.

## 2. By ROI

The user names the ROIs that hold one compartment; a skeleton node inside any of them gets that
compartment, and every other node the other one. *Decided:* a `Volumes` input (e.g. ROI Meshes),
a picker of which volumes count, and a dropdown saying whether they are the **axon** or the
**dendrite**; everything outside them is the other compartment.

- **The inside test is Points in Volumes'** (`meshInside.ts`: `volumeBoxes` to reject on the box,
  `buildInsideTests` for the ray test). A skeleton node is a point, so nothing new is needed below
  the node.
- **The picker.** Volume names are only known once ROI Meshes has run, so the picker lists the
  names on the wired value after a Run; **empty means every wired volume**, which is also the
  simplest setup (wire ROI Meshes with just the axon's regions selected). *Proposal* — the
  alternative is a free-text list.
- **No root, no heal, no synapses needed.** A fragmented skeleton splits fine, piece by piece.
- **Cost** is a ray test per node per candidate volume, the same order Points in Volumes warns
  about (`RAY_WARN`), so it warns through the same function with nodes in place of points.
- **Refusals.** None of its own: a neuron entirely outside the ROIs is all the other compartment,
  which is an answer, not a failure. The warning counts neurons with **no node inside** any picked
  ROI, since that is usually a wrong ROI or a frame mismatch (`requireVolumeFrame` refuses the
  latter outright).

### Built (phase 2)

- `Method` gains "by ROI", with `The ROIs hold` (the axon / the dendrite) and `ROIs`, a chips
  field of volume names where empty is every wired volume. Names are typed: a volume's name is in
  the wired value, which a param's options cannot see before a Run, so the picker the plan proposed
  became free entry with "all" as the empty state. A name no wired volume carries is warned about;
  all of them missing is a refusal.
- `nodes/lib/splitByVolume.ts`, headless and sliced, on `meshInside.ts` (`containsAny`, stopping
  at the first enclosing volume). The rays answer a **mask**, which nodes are inside, and `The ROIs
  hold` only relabels it (`volumeLabels`), so the mask is memoised per skeletons value, volumes and
  ROI names alone: neither a hand correction nor flipping the compartment casts a ray again. The
  ray cost is counted first (`countRays`, shared with Points in Volumes) and the count rides on the
  memo entry, so a remembered mask replays the warning.
- `Margin (µm)`, grow only (decided): a node outside every volume but within that distance of a
  surface counts as inside. It lives in `meshInside.buildInsideTests` (`margin`), so Points in
  Volumes could take it too: Distance between's `hasWithin` (`geometryIndex.meshIndexes`), the
  early-exit "any triangle within d" with the strict-comparison slack, on the trees the rays use,
  asked only inside each volume's box grown by the margin — a tree rejects a far point at its root
  in a few hundred nanoseconds, a box in a few. Rays stay on the ungrown boxes. The cost is counted
  up front, a ray per node per box and a search per node per grown box, the searches priced at
  Distance between's measured `hasWithin` rate inside `warnRayWork`, so the one threshold stays a
  wait. A margin needs nanometres. The mask is keyed on the margin too, so a new margin recasts
  the rays: they are the cheap half, a search costing about eleven.
- Synapse placement is shared by every method: per skeletons value, every parent distance; per
  synapse cloud, the sites grouped by neuron and placed on nodes. `splitSkeletons` takes the
  placement (`assignments`) rather than redoing it per setting.
- **The method dispatch is in TypeScript.** `SplitSettings` gains a `roi` member;
  `splitSkeletons` takes only `PythonSplitSettings`, and Topology reads flow's alone
  (`SPLIT_METHODS.flow.read`). `SplitMethod` is the card's list and `PythonSplitMethod` the
  bridge's, so the bridge never appears to accept a method Python does not run.
- `Synapses` is optional, `Volumes` new and optional, and `validate` says per method which one is
  missing, the framework's required mark no longer being able to. Without synapses the per-
  compartment counts are **null**, not zero (`CompartmentStats` counts are `number | null`), and
  the `Synapses` output is an empty cloud with the `compartment` column the port promises.
- No heal under ROI: nothing walks the tree. Measured in the app on the demo data, with ROI Meshes
  asked for `LOP(R)`: the run labels, and 28 of 30 neurons reach no node into it, which the card
  says.

## 3. By tag

For CATMAID skeletons, whose tracers place node tags such as `axon` or `dendrite` where a
compartment starts. *Decided:* a tag marks where its compartment begins, and **it and everything
distal of it** (away from the soma) take that compartment. With only one kind of tag set, every
other node is the other compartment.

**Params:** `Axon tags` and `Dendrite tags`, each a list of tag names (CATMAID's are free text, and
projects differ: `axon`, `axon start`, `dendrite`, …). Tag names, like volume names, are only
known after a fetch, so each is a text list with the tags found on the run offered as suggestions
(*deferred*: a run whose names match nothing names the tags it found instead).

**The root** is the soma (`resolveRoot('soma')`), and a fragmented neuron is walked on its healed
tree, so "distal" means away from the cell body.

**Value-model change: skeletons have no tags today.** CATMAID's `compact-detail` request already
asks for them (`with_tags: true`), and `decodeCompactSkeleton` drops them. This needs:

- `SkeletonGeometry.tags?: Readonly<Record<string, Int32Array>>`: tag name to node indices.
- **Carried through every transform**, the `split`/`compartments` rule: kept by one that only moves
  nodes (Mirror, Transform), and by one that renumbers them (Clean Skeletons' resample and
  downsample, slicing) re-snapped by **position** to the nearest new node, the rule corrections
  already follow, because a tag is a point on the cable rather than a node id. *Proposal.*
- `skeletonBuffers` / `valueBytes`, the geometry cache key, and the SWC writer (which has no tag
  column; tags are dropped there, said in its note).
- Other sources publish no tags, so the method's `validate` says "this dataset's skeletons carry no
  node tags" when the wired skeletons' source has none (a `SourceCapabilities` flag, `nodeTags`;
  *not built*, see below).

**Refusals.** A new `no tags` status for a neuron carrying none of the named tags.

### Built (phase 3)

- **Value model.** `SkeletonGeometry.tags` (name → node indices), counted by `skeletonBuffers`.
  CATMAID's `decodeCompactSkeleton` keeps them, dropping an id the skeleton does not hold. A node
  tagged `soma` also becomes the source's soma label (`compartments`, `SWC_SOMA`), as neuPrint's
  `somaLocation` does through `withSomaAt` — so every soma reader (the split's root, the dendrogram,
  Topology) finds CATMAID's soma, not only this method. Transforms that move nodes (Mirror,
  Transform, the heal, `withSplit`) spread the item and keep tags untouched; Clean Skeletons, the
  one renumbering transform, re-snaps each tagged node to the nearest cleaned node (decided:
  re-snap, not drop). SWC export does not write them, said in its note.
- **The method.** `nodes/lib/splitByTag.ts`, headless. A tag is applied the way a `distal` hand
  correction is: the same soma-rooted arbour (`splitArbor`), the subtree as a run of the pre-order,
  tags in pre-order so a nested tag wins. The rest is dendrite (axon tags only), axon (dendrite tags
  only) or **linker** (both, decided). A forest is `multiple roots`, as flow refuses it, and
  `Heal fragmented skeletons` applies (`walksTree`); the heal keeps node numbering, so the tags still
  name the right nodes.
- **Controls.** `Axon tags` and `Dendrite tags`, chips of free text. `validate` asks for at least
  one name. A run whose names match no tag on any skeleton warns with the tags the skeletons do
  carry (`tagNames`) — or that they carry none, CATMAID being the source that publishes them. That
  warning stands in for the `nodeTags` capability proposed above: `validate` cannot see which source
  the wired skeletons came from, and the run can.
- Synapses optional, as for ROI: wired, Summary counts each compartment's synapses.

## Phases

1. **The switch and label propagation.** `Method` with "synapse flow" and "label propagation";
   Python port; `probe:split` extended to compare the propagation against navis; the probability
   node value. Smallest change that proves the multi-method contract.
2. **By ROI.** `Volumes` input, picker, axon/dendrite dropdown, the optional `Synapses` input.
3. **Tags in the value model, then by tag.** CATMAID decode, transforms, then the method.
4. **Help, changelog, assistant.** The help page gets a section per method; the assistant catalogue
   picks up the enum by itself, but `Method` probably wants a `feeds`-style note so a request like
   "split by the ROIs the axon is in" reaches "by ROI" (measured live, as `producedBy` was).

### Built (phase 4)

- The help page is rewritten around `Method`: a table of the four, then a section each, with
  `Inputs`, the statuses ("no tags", "not converged") and the fixing advice widened to every method.
  The clamping default's departure from navis is said in one sentence, being behaviour a navis user
  would not predict.
- Changelog: a feature card on the 2026-10-11 entry, and an item for CATMAID's node tags (and the
  `soma` tag becoming the soma).
- Assistant: **no note needed**, measured — 10/10 "by ROI" and 5/5 "by tag" on region- and
  tag-phrased requests (`docs/assistant.md`).

## Open questions

0. ~~The propagation's default `Clamping`.~~ *Decided 2026-10-11:* **0.5**, not navis's 1, which
   can oscillate (above). The help says so.
0. **The node's cost under propagation.** The node is `cheap` (every edit re-runs it). A neuron
   that does not converge runs all of `Max iterations`: measured natively at ~2.8 ms per step on a
   100k-node tree, so ~28 s per neuron at the default 10,000, likely a minute or more in Pyodide —
   re-run on every `Clamping` edit. Progress is now reported every 200 steps, so the bar moves.
   Options: keep `cheap`; make the node `expensive` while the method is label propagation; or a
   lower default `Max iterations`.

1. ~~**Both kinds of tag set.**~~ *Decided 2026-10-11:* the cable distal of neither is **linker**.
2. **The ROI and tag pickers before a Run:** suggestions from the last run, or the full list from
   the source where one is cheap (neuPrint's ROI list is)? *Proposal:* last run only, to start.
3. **Exports.** The flow split has no Python/R emitter yet; label propagation is a one-line navis
   call and could be the first. Out of scope here unless wanted.
