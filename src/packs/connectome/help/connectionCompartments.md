## What it counts

Connection Compartments takes an edge list and asks, for every synapse behind each edge, which compartment it sits on at either end. The presynaptic end is looked up on the presynaptic neuron's split, the postsynaptic end on the postsynaptic neuron's:

| Class              | Presynaptic end | Postsynaptic end |
| ------------------ | --------------- | ---------------- |
| `axo-dendritic`    | axon            | dendrite         |
| `axo-axonic`       | axon            | axon             |
| `dendro-dendritic` | dendrite        | dendrite         |
| `dendro-axonic`    | dendrite        | axon             |
| `linker`           | either end on the linker | |
| `unsplit`          | either end could not be placed | |

So an edge with a `weight` of 40 might come back as 31 axo-dendritic, 6 axo-axonic and 3 linker. The six counts of an edge always add up to the synapses fetched for it.

## How it works

1. The node fetches every synapse between the neurons on the presynaptic side and those on the postsynaptic side, and keeps the ones on a pair your table lists.
2. Each end is snapped to the nearest node of its own neuron's skeleton, at its own position: the presynaptic site onto the presynaptic skeleton, the postsynaptic site onto the postsynaptic one.
3. That node's compartment, from [Split Axon/Dendrite](#neuron.splitCompartments), is the end's compartment.

The compartments come from the skeletons you wire in. This node does no splitting of its own, so the thresholds and any hand corrections you made on the split card are what you get here.

## A typical chain

```coda-graph
caption: Count a neuron's outgoing connections by compartment.
neuron.findNeurons as find
neuron.connectivity as conn { direction: outputs }
neuron.skeletons as skel
neuron.synapses as syn
neuron.splitCompartments as split
connectome:connectionCompartments as cc
find -> conn:neurons
conn:connections -> cc:edges
conn:neuronSet -> skel:neurons
conn:neuronSet -> syn:neurons
skel -> split:skeletons
syn -> split:synapses
split:out -> cc:skeletons
```

The skeletons have to cover the neurons at **both** ends of the edges, which is why the chain fetches them from Connectivity's `Neuron Set` rather than from Find Neurons: `Neuron Set` holds every neuron that appears in the connections, seeds and partners alike.

Wire the same dataset into this node as into Connectivity. Any edge list works as long as it has a presynaptic and a postsynaptic id column, e.g. Synapses to Edges or a table of your own.

## Settings

```coda-params
connectome:connectionCompartments: pre, post, layout, maxSnap, minConfidence
```

`Presynaptic` and `Postsynaptic` default to `preId` and `postId`, Connectivity's own columns.

## Outputs

- `Edges`: your edge list with the counts added. With `Layout` on "wide", each row gains `weightAxonAxon`, `weightAxonDendrite`, `weightDendriteAxon`, `weightDendriteDendrite`, `weightLinker`, `weightUnsplit` and their total, `synapses`. With "long", each edge becomes six rows, a `connection` column names the class and `weight` holds its count, which is the shape a Bar Chart or Group By wants.
- `Synapses`: every synapse that was counted, with `preCompartment`, `postCompartment` and `connection`. Colour a 3D View by `connection` to see where on the arbour each class sits.

## Unsplit synapses

A synapse is counted as `unsplit` when either end could not be placed. The node's warning says how many and why:

- the neuron at that end has no skeleton wired in;
- its skeleton carries no split (Split Axon/Dendrite's `splitStatus` says why, often "no synapses" or "multiple roots");
- the site is further than `Max snap distance` from the skeleton, usually because the skeleton is a coarse one or from a different source than the synapses.

A large `weightUnsplit` is a sign to look at the skeletons rather than at the biology.

> [!NOTE] Counts that differ from `weight`
> The `synapses` total is what was fetched and counted, and it can differ from the edge list's own
> `weight`. `Min confidence` drops low-scoring synapses, and a `weight` that is normalised or summed
> over a cell type will never match. The node warns with the number of edges that differ.

> [!WARNING] One position for both ends
> Some data sources record a single position per synapse. There, both ends are read at that one
> point, and where an axon and a dendrite of the same neuron interdigitate the postsynaptic end can
> land on the wrong one. The node warns with how many synapses were read this way.

## What it costs

Like Synapses Between, the node fetches every synapse between the two sides, not just the listed pairs, and throws the rest away. For a dozen seeds and their partners that is quick; for a few hundred neurons on each side it can be a large download. It warns above `Warn above` neurons on either side, and when most of what came back (and at least 10,000 synapses) was not on a listed edge, it says so.

The fetched synapses are kept for the session, so changing `Layout` or `Max snap distance` re-counts without fetching again. Changing the edge list or `Min confidence` fetches again.
