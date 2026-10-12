Split Axon/Dendrite labels every node of a skeleton as axon, dendrite or linker. Pick how with `Method`:

| Method | Splits by | Needs |
| ------ | --------- | ----- |
| "synapse flow" | where input-to-output paths run through the neuron | synapses |
| "label propagation" | where outputs and inputs sit, spread along the cable | synapses |
| "by ROI" | which brain regions the cable lies in | volumes |
| "by tag" | node tags a tracer placed where the axon or dendrite starts | CATMAID skeletons |

If in doubt, start with synapse flow; use "by ROI" or "by tag" when you already know where the axon is.

Both synapse methods run in the Python runtime, so the first split in a session waits for a ~10 MB download. After that it is fast, and moving a setting re-splits straight away.

## Synapse flow

This is navis's `split_axon_dendrite` with synapse flow centrality, the method from Schneider-Mizell et al. (2016):

1. For every piece of cable, count how many paths from an input (postsynaptic site) to an output (presynaptic site) run through it. That count is the synapse flow.
2. Everything at or above `Linker threshold` × the neuron's peak flow is the linker. With the default 0.9, that is the cable carrying the top 10% of flow, usually the stretch between the dendrite and the axon.
3. Cutting out the linker leaves a few pieces. A piece whose share of the neuron's outputs is at least `Axon threshold` times its share of the inputs is axon; the rest is dendrite.

```coda-params
neuron.splitCompartments: flowThresh, splitVal, heal, flow
```

With `Write synapse flow` on, the skeletons also carry each node's flow as a fraction of that neuron's peak. Set the 3D View's skeleton `Colour` to "by node value" to see it; everything above `Linker threshold` is linker.

## Label propagation

This is navis's `split_axon_dendrite_prop`. Nodes with more outputs than inputs start out as axon, nodes with more inputs than outputs as dendrite, and those labels are then spread along the cable, weighted by its length, until they stop changing. It labels axon and dendrite only; there is no linker.

```coda-params
neuron.splitCompartments: weighting, alpha, probability
```

With `Write probability` on, colour the 3D View by node value to see where the boundary between axon and dendrite is soft.

`Clamping` defaults to 0.5 rather than navis's 1; set it to 1 to match navis exactly.

A neuron that has not settled after `Max iterations` keeps its labels with the status "not converged", and the node warns. Raise `Max iterations` (an advanced setting), or check the neuron in the full-size view.

## By ROI

Wire the regions into `Volumes`, typically from ROI Meshes, and say what they hold. There is no linker.

```coda-params
neuron.splitCompartments: inside, volumes, margin
```

Say the axons of a set of visual projection neurons end in the lobula: wire ROI Meshes with "LO(R)" into `Volumes`, and leave `The ROIs hold` on "the axon".

```coda-graph
caption: Split neurons by the region their axons end in.
neuron.findNeurons as find
neuron.skeletons as skel
neuron.roiMeshes as rois
neuron.splitCompartments as split { method: roi }
out.viewer3d as view
find -> skel:neurons
skel -> split:skeletons
rois -> split:volumes
split:out -> view:skeletons
```

The node warns about neurons with no node inside any of the regions. Usually that is the wrong region, or skeletons and volumes from two different template spaces.

## By tag

CATMAID tracers often tag the node where the axon or the dendrite starts. Name those tags in `Axon tags` and `Dendrite tags`: a tagged node and everything distal of it, away from the soma, take that compartment.

```coda-params
neuron.splitCompartments: axonTags, dendriteTags
```

What the rest of the neuron becomes depends on which tags you name:

- axon tags only: the rest is dendrite;
- dendrite tags only: the rest is axon;
- both: the rest is linker, which is usually the primary neurite between the soma and the first tag.

A tag inside the subtree of another wins, so a dendrite tag on a branch beyond an axon tag makes that branch dendrite. Tags are free text and projects spell them differently (`axon`, `axon start`, …), so each field takes several names. If none of them matches, the node warns with the tags the skeletons do carry.

"Distal" is measured from the node tagged `soma`, or from the skeleton's root where there is none.

## Inputs

The synapse methods need a synapse cloud for the same neurons, with both inputs and outputs. That is the `Synapses` node with `Polarity` left on "both". "By ROI" and "by tag" do not need synapses, but with a cloud wired in, `Summary` still counts each compartment's synapses.

A neuron can come back unsplit, or split with a caveat. The `splitStatus` column in `Summary` says which:

- "no synapses": it has only inputs or only outputs, which is common for sensory and motor neurons cut off at the edge of the volume.
- "multiple roots": the skeleton arrived in several pieces. Tick `Heal fragmented skeletons` to join them before splitting. The joins are only used for the split; the skeleton you get back is unchanged.
- "no tags": it carries none of the tags named in `Axon tags` or `Dendrite tags`.
- "not converged": a label propagation stopped at `Max iterations`; see Label propagation.

> [!WARNING] Not Synapses Between
> A `Synapses Between` cloud puts the presynaptic neuron in `neuronId` on every row, so a
> postsynaptic neuron's inputs would be filed under its partner. Use a plain `Synapses` node.

## A typical chain

```coda-graph
caption: Split a set of neurons and look at them in 3D.
neuron.findNeurons as find
neuron.skeletons as skel
neuron.synapses as syn
neuron.splitCompartments as split
out.viewer3d as view
find -> skel:neurons
find -> syn:neurons
skel -> split:skeletons
syn -> split:synapses
split:out -> view:skeletons
```

In the 3D View, set the skeleton `Colour` to "by compartment (axon/dendrite)".

## Inspecting a split

Run the node, then open the card full size with ⤢. You can also put it on a dashboard, next to a 3D View of the same skeletons.

The full-size view shows one neuron at a time as an unrooted dendrogram, coloured by compartment. Page through the neurons with ‹ ›. The dot marks the root, and the bar shows the neuron's segregation index (`SI`): 1 when inputs and outputs are completely apart, 0 when they are mixed evenly. If the index is low for a neuron you know is well polarised, the split probably needs fixing.

Turn on `Synapses` in the bar to draw the synapses on the dendrogram, outputs on one side of each branch and inputs on the other. A misplaced split usually shows up as a run of outputs on cable labelled dendrite, or the reverse.

The `Summary` output has the same numbers for every neuron at once: cable and synapse counts per compartment, `segregationIndex`, and `corrections`, the number of hand corrections applied to that neuron. Sorting by `segregationIndex` is a quick way to find the neurons worth a second look.

## Fixing a split

Try the method's settings first, since they apply to every neuron. For synapse flow:

- Axon and dendrite swapped, or twigs near the linker landing on the wrong side: change `Axon threshold`. Lower it to call more of the neuron axon, raise it to call more of it dendrite.
- Linker too short or too long: change `Linker threshold`. Lowering it makes the linker longer.

If one method gets a neuron wrong, try another.

For a single neuron, correct it by hand. Click a branch on the dendrogram and choose from the menu:

- "Away from the root → axon / dendrite / linker" relabels the branch and everything beyond it, as seen from the root.
- "Everything else → …" relabels the rest of the neuron instead.
- "Make this the root" moves the root to that end of the branch. Do this first when the part you want to relabel lies between the click and the current root. "Reset root" goes back to the root the neuron opened on.

Say the split put the axon's first branch point on the dendrite. Click the branch just past it and choose "Away from the root → axon". The colours, the `SI` and Summary update straight away; there is no need to re-run.

Each correction is listed beside the dendrogram for the neuron on screen. ✕ removes one, and "Reset this neuron" removes them all. Corrections are applied in order, so a later one wins where two overlap.

Corrections are applied on top of the automatic split, so they still hold when you change a threshold or the method. They are saved with the node, in the workflow file and in share links.

> [!NOTE] Corrections that no longer land
> A correction remembers where you clicked and which root it was made from. If the skeleton
> changes, for example because you fetched it from another source, a click may no longer land
> on the neuron. The node then warns "N of M hand corrections no longer land", keeps them, and
> leaves the labels as they were. Remove them in the full-size view.

> [!NOTE] Healed skeletons
> With `Heal fragmented skeletons` on, the full-size view draws a fragmented neuron with its
> pieces joined, as it was split, and corrections are applied to the joined neuron. The first
> time, it shows "Joining the skeleton’s N pieces…" while that runs.

## Outputs

- `Skeletons`: the input skeletons with the split stored on them, for the 3D View's compartment colours. A neuron that could not be split has no labels.
- `Synapses`: the input cloud with a `compartment` column (`axon`, `dendrite`, `linker`, or empty). Empty when no cloud is wired in.
- `Summary`: one row per neuron, with the skeletons' own columns first.

To count each connection's synapses by the compartment at either end (axo-dendritic, axo-axonic and so on), wire `Skeletons` into [Connection Compartments](#connectome:connectionCompartments).

> [!TIP] Clean first, then split
> [Clean Skeletons](#neuron.cleanSkeletons) renumbers the nodes, so it drops a split and says so.
> Put it before this node, not after. Node tags survive it: each moves to the nearest new node.
> [Simplify Skeletons](#neuron.simplifySkeletons) only removes nodes, so a split survives it.
