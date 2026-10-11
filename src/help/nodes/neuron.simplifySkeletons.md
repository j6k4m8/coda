## What Simplify Skeletons does

Simplify Skeletons makes skeletons smaller by removing nodes. It can do two things, each optional and always in this order:

1. **Prune twigs**: remove short side branches, as navis's `prune_twigs` does.
2. **Downsample**: thin out what is left, as navis's `downsample_neuron` does, or with one of two methods that look at the shape of the neurite.

No node is ever moved, only removed. The nodes that remain keep their positions, radii and labels exactly.

```coda-graph
caption: Prune and downsample before drawing a large set of neurons.
dataset.hemibrain as ds
neuron.findNeurons as find
neuron.skeletons as skel
neuron.simplifySkeletons as simp { prune: 2, method: every }
out.viewer3d as v3d
ds -> find:dataset
ds -> skel:dataset
find -> skel:neurons
skel -> simp
simp -> v3d:skeletons
```

Everything is off by default, so a freshly added node passes its skeletons through unchanged. The work runs in the in-browser Python runtime (navis-fastcore), so the first run of a session waits for a download of about 10 MB.

## Pruning twigs

A twig is the cable from a leaf back to the nearest branch point. `Prune twigs` removes every twig shorter than the given length, measured along the cable. Use it to get rid of the short spurs that tracing or skeletonisation leaves behind:

```coda-params
neuron.simplifySkeletons: prune, repeat
```

Removing a twig can turn the branch it hung from into a new, shorter twig. By default pruning runs once, so those are left alone. With `Repeat until none are left` on, it runs again until no twig is shorter than the threshold. This is navis's `recursive=True`, and with a large threshold it can strip a neuron down to its main paths.

A few things are never pruned:

- **The soma.** A twig that ends in the soma is kept, however short.
- **Roots.**
- **Fragments without a branch point.** A short, unbranched piece of a neuron is all "twig", so pruning would delete it outright, and an unbranched neuron would come back empty. These are left as they are. To join fragments into one tree first, use `Heal fragments` on [Clean Skeletons](#neuron.cleanSkeletons).

## Downsampling

`Downsample` picks how to thin the skeleton:

```coda-params
neuron.simplifySkeletons: method, factor, tolerance, area
```

| Method | Keeps | Setting |
| ------ | ----- | ------- |
| "keep every Nth node" | one node in every N along each unbranched stretch | `Keep every` |
| "drop nodes that do not bend" | nodes where removing them would move the neurite by more than the tolerance (Ramer–Douglas–Peucker) | `Tolerance (µm)` |
| "drop nodes adding least area" | nodes whose triangle with their two neighbours is larger than the threshold (Visvalingam–Whyatt) | `Smallest area (µm²)` |

"keep every Nth node" ignores the geometry, which is fine for skeletons that are already evenly sampled. The other two keep nodes where the neurite curves and drop them along straight stretches, so the same number of nodes looks more like the neuron. Of the two, Ramer–Douglas–Peucker can keep a single sharp spike and flatten everything around it, while Visvalingam–Whyatt removes detail more evenly.

`Smallest area` is in square micrometres: 0.01 µm² is a bump of 0.1 µm over 0.2 µm of cable.

All three methods always keep roots, branch points, leaves and the soma, so the result is still the same neuron.

## What happens to labels

Because surviving nodes are the original nodes, everything attached to them comes through unchanged:

- compartment labels from the source
- an axon/dendrite split from [Split Axon/Dendrite](#neuron.splitCompartments), and per-node values such as synapse flow
- node tags (e.g. CATMAID's `ends`). A tag on a removed node moves to the nearest node that is left.

> [!TIP] Split before or after?
> A split computed before simplifying is kept, but it was computed with the pruned twigs (and
> their synapses) still on the neuron. If those twigs shouldn't count, split after simplifying.

> [!NOTE] Distances need nanometre coordinates
> If the skeletons arrive in voxels, `Prune twigs`, `Tolerance` and `Smallest area` are refused.
> "keep every Nth node" counts nodes, so it works in any units.

## Simplify or Clean?

[Clean Skeletons](#neuron.cleanSkeletons) re-samples to an even spacing, which creates new nodes and can increase the count. Use it before measurements that average per node, or before NBLAST. Use Simplify Skeletons when you want fewer nodes, e.g. to draw hundreds of neurons in a [3D View](#out.viewer3d).

## Exporting

The notebook export calls the same navis-fastcore functions Coda runs, plus a small helper for the pruning rules above, so it gives the same skeletons. The R Markdown export leaves a note pointing to the notebook.
