"""Axon/dendrite split, as navis performs it.

The seventh capability on the bridge and, like the six before it, nearly free: it declares the
same two packages NBLAST does, so on a runtime that has scored or cleaned anything it costs a
`runPython` of these definitions and nothing else.

## This is navis's algorithm, not one like it

`navis.split_axon_dendrite(metric='synapse_flow_centrality')` is transcribed here rather than
approximated, because the split is the one number on this card somebody will compare against a
paper. Checked against navis 2.0.0-rc.1's own source, and worth writing down because two of the
defaults are not what a reading of the docstring suggests:

- **`mode='sum'`, not `'centrifugal'`.** `split_axon_dendrite` calls `synapse_flow_centrality(x)`
  with no mode at all, and that function's default is `'sum'`. Centrifugal is the mode the
  literature discusses and the one an implementer reaches for; it gives a different linker.
- **`flow_thresh=0.9`**, so the linker is every node at or above 90% of peak flow — not the peak
  node, and not a single cut point. The compartments are the connected components of what is
  *left* once the linker is removed, which is why a neuron can come back with two dendritic
  fragments rather than exactly one of each.

navis itself delegates the metric to `fastcore.synapse_flow_centrality`, so the number computed
here is the same function on the same wheel rather than a second implementation of it.

## Where Coda's version has to differ, and why

**navis rejects a multi-rooted neuron outright**; this reports it. A fragmented reconstruction is
common enough that failing the whole batch on one would make the node useless on exactly the
datasets whose skeletons arrive in pieces — so `status` says which neurons could not be split and
the card names them, beside the `heal` flag below that fixes it. Same call `cleanOps.ts` makes about an item that cannot be processed: it stays
in the collection, and the count is said out loud.

**Healing is opt-in, and happens here rather than in `skeletons.py`.** A skeleton derived from a
segmentation routinely arrives as a forest — every fish2 body sampled was 13 to 627 pieces — and
navis's answer is to call `heal_skeleton` first. With `heal` on, `_split_one` does exactly that
before asking whether it has one root. It is off by default because a bridge is an edge nobody
traced and it carries synapse flow, so it can move the linker. The coordinates cross the bridge
only when it is on; the split itself never reads them.

**Synapses arrive already assigned to nodes.** navis reads a `node_id` off its connector table;
Coda's synapses are loose coordinates, so `assignSynapses` in `nodes/lib/topologyOps.ts` does the
nearest-node lookup on the JavaScript side — where the geometry still has its units and a test can
see it. This function is handed the per-node counts navis would have computed itself.
"""

import numpy as np
import navis_fastcore as fc

# Mirrored in `topology.ts` as `COMPARTMENT_*`. Two languages, so the agreement is asserted by
# `topology.test.ts` rather than shared; the numbers are ours, since navis labels with strings and
# a string per node would be the largest thing crossing the bridge.
UNASSIGNED = 0
DENDRITE = 1
AXON = 2
LINKER = 3

# What `status` reports per neuron.
OK = 0
MULTIPLE_ROOTS = 1
NO_SYNAPSES = 2
# Label propagation hit its iteration cap: navis's own warning sign that the answer may be noise.
# The labels are still returned, as navis returns them.
NOT_CONVERGED = 3

# navis's own defaults, named rather than inlined so the two that surprise people are greppable.
FLOW_MODE = "sum"
# navis's `split='prepost'`, whose ratio threshold is 1 unless you spell it `split='prepost:0.5'`.
# The default here, and the value the caller overrides when the split comes out the wrong way up.
SPLIT_VAL = 1.0
# A component holding under 1% of either synapse population is left unlabelled here and picked up
# by the nearest-labelled-node pass below. navis's comment: a small side branch with few synapses
# is noise, and assigning it on its own ratio introduces that noise into the split.
MIN_FRACTION = 0.01


def _components(parents, is_linker):
    """Connected components of the arbour with the linker cut out.

    Both ends of every edge touching the linker are detached, so a component holds either only
    linker nodes or only non-linker ones and the two can never be pooled by accident.
    """
    n = len(parents)
    node_ids = np.arange(n, dtype=np.int32)
    masked = parents.astype(np.int32).copy()

    has_parent = masked >= 0
    parent_is_linker = np.zeros(n, dtype=bool)
    parent_is_linker[has_parent] = is_linker[masked[has_parent]]
    masked[is_linker | parent_is_linker] = -1

    return node_ids, fc.connected_components(node_ids, masked)


def _connecting_nodes(parents, allowed, seeds):
    """`seeds` plus whatever has to come along to connect them, within `allowed`.

    navis's `graph.connecting_nodes` on the subgraph induced by `allowed`. On a forest the
    minimal connecting subgraph is found by pruning: repeatedly drop any node with at most one
    remaining neighbour that is not itself a seed. What survives is exactly the union of the paths
    between the seeds, per component.

    This is the step that reunites an axon which the linker cut into pieces - and it is why
    navis's linker is *smaller* than the flow threshold alone would make it. Leaving it out looked
    almost right: 99.78% agreement on navis's own example neuron, with the missing 1% being a
    nine-node run of primary neurite that navis calls axon and a threshold alone calls linker.
    """
    n = len(parents)
    neighbours = [[] for _ in range(n)]
    for i in range(n):
        p = int(parents[i])
        if p < 0 or not allowed[i] or not allowed[p]:
            continue
        neighbours[i].append(p)
        neighbours[p].append(i)

    keep = allowed.copy()
    degree = np.array([len(neighbours[i]) for i in range(n)], dtype=np.int32)
    stack = [i for i in range(n) if keep[i] and not seeds[i] and degree[i] <= 1]
    while stack:
        i = stack.pop()
        if not keep[i] or seeds[i] or degree[i] > 1:
            continue
        keep[i] = False
        for j in neighbours[i]:
            if not keep[j]:
                continue
            degree[j] -= 1
            if not seeds[j] and degree[j] <= 1:
                stack.append(j)
    return keep


def _heal(parents, coords):
    """Every fragment joined into one tree, as `skeletons.py`'s heal does with no distance cap.

    `heal_skeleton` returns parents in the numbering it was given, so the labels computed on the
    healed tree land on the caller's own nodes — which is what lets the card colour the skeleton
    it drew, bridges and all, without drawing the bridges.
    """
    ids = np.arange(len(parents), dtype=np.int64)
    # float64 here, per neuron that heals, rather than for the whole batch up front: most
    # neurons in a mixed set have one root and never read their coordinates at all.
    healed = fc.heal_skeleton(
        ids,
        parents.astype(np.int64),
        np.ascontiguousarray(coords, dtype=np.float64),
        method="ALL",
        max_dist=None,
    )
    return np.asarray(healed, dtype=np.int32)


def _reroot_at(ids, parents, soma):
    """`parents` rerooted at `soma`, or as they are where it is -1 or already the root.

    `skeletons.py` holds the same three lines for the heal: each module runs in a namespace of its
    own (`runtime.ts`), so neither can call the other's.
    """
    if soma < 0 or parents[soma] < 0:
        return parents
    return np.asarray(fc.reroot(ids, parents, [soma]), dtype=parents.dtype)


def _split_one(
    parents, presynapses, postsynapses, flow_thresh, split_val, soma, coords=None
):
    """One neuron. Returns `(compartment, flow, status)`.

    `flow_thresh` and `split_val` are navis's two tuning knobs, in navis's own units: the linker
    is `max(flow) * flow_thresh`, and a component is axon when its pre/post fraction ratio is at
    or above `split_val`. navis spells the second one inside the `split` argument
    (`split='prepost:0.5'`), which is why it is easy to miss that it exists at all.

    `coords` is the opt-in heal: given, a fragmented neuron is joined into one tree first.
    `soma` is the node to root at, or -1 — navis's `reroot_soma=True`. After the heal, because
    `heal_skeleton` keeps the largest fragment's root and the soma is often in another fragment.
    """
    n = len(parents)
    compartment = np.full(n, UNASSIGNED, dtype=np.int32)
    flow = np.zeros(n, dtype=np.float32)

    if n == 0:
        return compartment, flow, OK

    node_ids = np.arange(n, dtype=np.int32)

    # Healed before the root check, which is the whole point: that check is what refuses a forest.
    roots = int((parents < 0).sum())
    if coords is not None and roots > 1:
        parents = _heal(parents, coords)
        roots = int((parents < 0).sum())

    # navis raises here. We report instead - see the module docstring.
    if roots != 1:
        return compartment, flow, MULTIPLE_ROOTS

    parents = _reroot_at(node_ids, parents, soma)

    if int(presynapses.sum()) == 0 or int(postsynapses.sum()) == 0:
        # A neuron with only one polarity has no flow to speak of: every node's centrality is 0,
        # the linker is the whole cell, and the "split" would label everything linker. That is a
        # meaningless answer rather than a failure, so it is named. Sensory and motor neurons in
        # a partially reconstructed volume land here routinely.
        return compartment, flow, NO_SYNAPSES

    raw = fc.synapse_flow_centrality(
        node_ids=node_ids,
        parent_ids=parents.astype(np.int32),
        presynapses=presynapses.astype(np.uint32),
        postsynapses=postsynapses.astype(np.uint32),
        mode=FLOW_MODE,
    ).astype(np.float64)

    # navis does not use fastcore's answer as it stands, and this is the correction - found by
    # running both and diffing, not by reading either. A branch point is given the *maximum flow
    # of its children*, because at the fork into the cell body fibre the flow does not pass
    # through the branch point child -> parent, it crosses it from one child to the other, and
    # fastcore's walk therefore records a dip exactly where the linker should be widest. It
    # affected 549 of 4,465 nodes on navis's own example neuron - every branch point - and it
    # moved the linker from 216 nodes to 391, which is most of the compartment boundary.
    #
    # Note it *overwrites* rather than taking `max(own, children)`: that is navis's line, and on
    # a branch point whose own flow is higher the two differ.
    flow = raw.copy()
    kinds = fc.classify_nodes(node_ids, parents.astype(np.int32))
    is_branch = kinds == 2
    if is_branch.any():
        child_max = np.zeros(n, dtype=np.float64)
        has_parent = parents >= 0
        np.maximum.at(child_max, parents[has_parent].astype(np.int64), raw[has_parent])
        flow[is_branch] = child_max[is_branch]
    # navis casts the column to int before thresholding; kept, because the threshold is a `>=`
    # and a fractional flow either side of it is a different linker.
    flow = np.floor(flow).astype(np.float32)

    peak = float(flow.max())
    if peak <= 0:
        return compartment, flow, NO_SYNAPSES

    is_linker = flow >= peak * flow_thresh
    compartment[is_linker] = LINKER

    _, cc = _components(parents, is_linker)

    # One entry per non-linker component, in no particular order.
    keep = ~is_linker
    labels = np.unique(cc[keep])

    # One pass each, not one per component. The comment below records that an axon is "routinely
    # cut into several components", so `labels` is tens rather than two — and the comprehension
    # this replaces built two full-length boolean masks and a fancy-index copy *per label*, which
    # on fifty components over seventeen thousand nodes is millions of elementwise operations and
    # a few hundred temporary arrays. `bincount` answers both vectors in one sweep.
    index = np.searchsorted(labels, cc[keep])
    n_pre = np.bincount(
        index, weights=presynapses[keep].astype(np.float64), minlength=len(labels)
    )
    n_post = np.bincount(
        index, weights=postsynapses[keep].astype(np.float64), minlength=len(labels)
    )

    total_pre = n_pre.sum()
    total_post = n_post.sum()
    # navis fills a division by zero with 0 rather than letting it become NaN, so that a neuron
    # with no presynapses outside the linker still gets a split rather than an exception.
    frac_pre = n_pre / total_pre if total_pre > 0 else np.zeros_like(n_pre)
    frac_post = n_post / total_post if total_post > 0 else np.zeros_like(n_post)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = frac_pre / frac_post

    # Too few synapses either way to judge: left unlabelled and attached below.
    ratio[np.maximum(frac_pre, frac_post) < MIN_FRACTION] = np.nan

    # A lookup indexed by component rather than a mask per component, for the reason above. NaN
    # stays unassigned, which is what the nearest-labelled-node pass below picks up.
    verdict = np.where(np.isnan(ratio), UNASSIGNED, np.where(ratio >= split_val, AXON, DENDRITE))
    compartment[keep] = verdict[index].astype(np.int32)

    # An axon or a dendrite is routinely cut into several components by the linker running through
    # it, so navis stitches each back together *using the linker* and takes those nodes out of it.
    # Axon first, then dendrite against what is left, which is navis's order and decides the few
    # nodes both could claim.
    is_axon = compartment == AXON
    if is_axon.any():
        reunited = _connecting_nodes(parents, is_axon | (compartment == LINKER), is_axon)
        compartment[reunited & (compartment == LINKER)] = AXON

    is_dend = compartment == DENDRITE
    if is_dend.any():
        reunited = _connecting_nodes(parents, is_dend | (compartment == LINKER), is_dend)
        compartment[reunited & (compartment == LINKER)] = DENDRITE

    # Everything still unassigned takes the compartment of its geodesically nearest labelled node
    # - navis's final pass, and the reason a small twig ends up on the arbour it grows out of
    # rather than in a compartment of its own. Undirected and unweighted, as navis asks for it.
    missing = np.flatnonzero(compartment == UNASSIGNED)
    if len(missing):
        labelled = np.flatnonzero(compartment != UNASSIGNED)
        if len(labelled):
            # `(distances, nearest)` - fastcore's order, and deliberately *not* navis's
            # `graph._geodesic_nearest`, which returns `(closest_id, distance)`. Unpacked the
            # other way round this assigns compartments by indexing with a distance: it throws
            # nothing, labels most nodes plausibly, and left 185 of them unlabelled here.
            _, nearest = fc.geodesic_nearest(
                node_ids=node_ids,
                parent_ids=parents.astype(np.int32),
                sources=missing.astype(np.int32),
                targets=labelled.astype(np.int32),
                directed=False,
            )
            # An orphan with no reachable labelled node comes back as -1 and stays unassigned.
            reachable = nearest >= 0
            compartment[missing[reachable]] = compartment[nearest[reachable].astype(np.int64)]

    return compartment, flow, OK


def _propagate_one(
    parents, presynapses, postsynapses, coords, weighting, alpha, max_iter, tol, heal, progress=None
):
    """One neuron by label propagation. Returns `(compartment, probability, status)`.

    navis's `split_axon_dendrite_prop` with `label_only=True`, ported line for line where it can
    be, and `pnpm probe:split` holds the two to the same answer. What differs is only how:

    - navis builds a scipy sparse adjacency and multiplies by it. There is no scipy in the Pyodide
      lock, and none is needed on a tree: `S @ F` is each node's neighbours' scores, weighted by
      edge length and divided by its own total edge length — a gather over the edges in scipy's CSR
      order and one `reduceat` over both labels, in float32 as navis's matrix is.
    - navis snaps each synapse to its nearest node itself; here the counts arrive per node,
      assigned on the TypeScript side as the flow split's are.

    Only axon and dendrite: label propagation has no linker. A node no signal reached (a fragment
    without synapses, when not healed) stays unassigned, with probability 0.
    """
    n = len(parents)
    compartment = np.full(n, UNASSIGNED, dtype=np.int32)
    probability = np.zeros(n, dtype=np.float32)
    if n == 0:
        return compartment, probability, OK

    if heal and int((parents < 0).sum()) > 1:
        parents = _heal(parents, coords)

    pre = presynapses.astype(np.float32)
    post = postsynapses.astype(np.float32)
    if pre.sum() == 0 or post.sum() == 0:
        return compartment, probability, NO_SYNAPSES

    # navis's `_calc_pre_post`: a node with more outputs than inputs is a `pre` seed, more inputs a
    # `post` seed, a tie no seed; its weight is the difference.
    is_pre = pre > post
    is_post = post > pre
    if weighting == "none":
        weight = np.ones(n, dtype=np.float32)
    else:
        weight = np.abs(post - pre)
        if weighting == "balance":
            # Scaled by the synapse totals, so the commoner type does not win by numbers alone.
            weight[is_pre] *= np.float32(post.sum() / pre.sum())
    # Columns in navis's sorted label order: `post`, then `pre`.
    Y = np.zeros((n, 2), dtype=np.float32)
    Y[is_post, 0] = weight[is_post]
    Y[is_pre, 1] = weight[is_pre]

    # The undirected adjacency, each edge weighted by its length, rows normalised — laid out as
    # scipy's CSR would be (by row, then column) and summed in float32 in that order. Not for
    # speed: navis stops when the hard labels have not changed for two iterations, and a dendrite
    # front reaching unreached nodes changes none (both score 0, so `argmax` already says `post`).
    # When it stops therefore turns on float rounding elsewhere; summed in float64 instead, the
    # port stopped two iterations early on one example neuron and left two nodes unassigned. What
    # remains is scipy's compiled product against numpy's, one ulp apart (2.4e-7) at a tied score:
    # with `alpha=0` and no weights navis then stops two iterations later. `probe:split` allows it.
    child = np.flatnonzero(parents >= 0)
    parent = parents[child].astype(np.int64)
    length = np.sqrt(((coords[child] - coords[parent]) ** 2).sum(axis=1), dtype=np.float32)
    rows = np.concatenate([child, parent])
    cols = np.concatenate([parent, child])
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    weight_edge = np.concatenate([length, length])[order]
    has_edges = np.zeros(n, dtype=bool)
    has_edges[rows] = True
    starts = np.flatnonzero(np.diff(rows, prepend=-1))
    # A row's total by numpy's own reduction over it, which is what scipy's `A.sum(axis=1)` is: an
    # element-by-element float32 sum differs in the last bit, and that alone moved the stop.
    row_sum = np.ones(n, dtype=np.float32)
    row_sum[has_edges] = np.add.reduceat(weight_edge, starts)
    row_sum[row_sum == 0] = 1
    s_col = ((np.float32(1.0) / row_sum)[rows] * weight_edge)[:, None]
    # One gather buffer for every step, and no scatter where every node has an edge (any tree of
    # more than one node, healed or whole). Both columns in one `reduceat`, which sums each column
    # in the same order the per-column version did — checked bitwise against it.
    gathered = np.empty((len(cols), 2), dtype=np.float32)
    every_node = bool(has_edges.all())

    def step(F):
        np.take(F, cols, axis=0, out=gathered)
        np.multiply(gathered, s_col, out=gathered)
        summed = np.add.reduceat(gathered, starts, axis=0)
        if every_node:
            return summed
        out = np.zeros_like(F)
        out[has_edges] = summed
        return out

    # navis's clamping: 0 holds the seeds fixed, 1 lets them change freely, between is soft.
    seed = np.flatnonzero(is_pre | is_post)
    held = Y[seed] if alpha == 0 else (1 - alpha) * Y[seed]
    F = Y.copy()
    F_max = F.max() + 1e-16 if tol < 1 else None
    # `argmax` over the two columns, with its tie rule (to `post`), at a fraction of the cost.
    prev_hard = F[:, 1] > F[:, 0]
    n_hard = 0
    converged = False
    for it in range(max_iter):
        F_new = step(F)
        if alpha == 0:
            F_new[seed] = held
        elif alpha < 1:
            F_new[seed] = alpha * F_new[seed] + held
        if tol >= 1:
            # navis counts iterations with unchanged hard labels without resetting the count.
            hard = F_new[:, 1] > F_new[:, 0]
            if np.array_equal(hard, prev_hard):
                n_hard += 1
            converged = n_hard > tol
            prev_hard = hard
        else:
            converged = np.abs(F_new - F).max() / F_max < tol
        F = F_new
        if converged:
            break
        # A neuron that does not converge runs every iteration, so say where it is, not just which.
        if progress is not None and it % 200 == 199:
            progress(it + 1)

    total = F.sum(axis=1)
    reached = F.max(axis=1) > 0
    norm = np.zeros_like(F)
    norm[reached] = F[reached] / total[reached, None]
    diff = norm[:, 1] - norm[:, 0]
    compartment[reached & (diff > 0)] = AXON
    compartment[reached & (diff < 0)] = DENDRITE
    probability[reached] = norm[reached].max(axis=1)
    # Labelled either way, as navis does: hitting the cap is a warning sign rather than an empty
    # answer, and the status is how the card says it.
    return compartment, probability, OK if converged else NOT_CONVERGED


def coda_split_compartments(request, report=None):
    """A whole set of skeletons split in one crossing.

    Flat in and flat out, `coda_clean_skeletons`' arrangement and for its reason: a hundred
    skeletons is a hundred thousand nodes, and an array of objects would be that many clones at
    the `postMessage` boundary. The node count never changes here, so `offsets` is reused rather
    than rebuilt - which is what lets the caller scatter the result straight back onto the
    geometry it already holds.
    """
    req = request.to_py()

    parents = np.frombuffer(req["parents"], dtype=np.int32)
    presynapses = np.frombuffer(req["presynapses"], dtype=np.uint32)
    postsynapses = np.frombuffer(req["postsynapses"], dtype=np.uint32)
    offsets = np.frombuffer(req["offsets"], dtype=np.int32)
    flow_thresh = float(req.get("flowThresh", 0.9))
    split_val = float(req.get("splitVal", SPLIT_VAL))
    heal = bool(req.get("heal", False))
    somas = np.frombuffer(req["somas"], dtype=np.int32)
    propagate = req.get("method") == "propagation"
    # Healing and propagation (its edges are weighted by length) are what need to know where a
    # node is, so the coordinates are sent only then — and kept a float32 view here.
    coords = (
        np.frombuffer(req["points"], dtype=np.float32).reshape(-1, 3) if heal or propagate else None
    )
    if coords is not None and len(coords) != len(parents):
        raise ValueError(f"positions need one point per node: {len(coords)} for {len(parents)}")
    if propagate:
        # Always sent with the method: the card's defaults are the ones that apply, not navis's.
        weighting = str(req["weighting"])
        alpha = float(req["alpha"])
        max_iter = int(req["maxIter"])
        tol = float(req["tol"])
    count = len(offsets) - 1

    compartment = np.full(len(parents), UNASSIGNED, dtype=np.int32)
    status = np.zeros(max(count, 0), dtype=np.int32)
    # Opt-in, for the reason the return below records: a float per node is the largest thing this
    # call can send back, and only a caller that draws the flow wants it.
    flow = np.zeros(len(parents), dtype=np.float32) if req.get("flow") else None
    probability = np.zeros(len(parents), dtype=np.float32) if req.get("probability") else None

    for i in range(count):
        a, b = int(offsets[i]), int(offsets[i + 1])
        if b <= a:
            continue
        if propagate:
            # Called only within this iteration, so `i` is this neuron's.
            progress = report and (
                lambda done: report(
                    (i + done / max_iter) / count, f"neuron {i + 1} of {count}, step {done:,}"
                )
            )
            comp_i, prob_i, status_i = _propagate_one(
                parents[a:b],
                presynapses[a:b],
                postsynapses[a:b],
                coords[a:b],
                weighting,
                alpha,
                max_iter,
                tol,
                heal,
                progress,
            )
            if probability is not None:
                probability[a:b] = prob_i
        else:
            comp_i, flow_i, status_i = _split_one(
                parents[a:b],
                presynapses[a:b],
                postsynapses[a:b],
                flow_thresh,
                split_val,
                int(somas[i]),
                coords=None if coords is None else coords[a:b],
            )
            if flow is not None:
                flow[a:b] = flow_i
        compartment[a:b] = comp_i
        status[i] = status_i
        if report is not None:
            report((i + 1) / count, f"neuron {i + 1} of {count}")

    # `flow` crosses only when asked for. It is the number the linker threshold was applied to —
    # navis's sum-mode flow with the branch-point correction, floored — and shipping it is four
    # bytes times every node of every neuron in a Run, so a caller that does not draw it (Topology)
    # does not pay for it. Split Axon/Dendrite always asks, to keep its memo independent of its
    # `Write synapse flow` switch. Zero for a neuron that was not split.
    out = {
        "compartment": np.ascontiguousarray(compartment, dtype=np.int32),
        "status": np.ascontiguousarray(status, dtype=np.int32),
    }
    if flow is not None:
        out["flow"] = np.ascontiguousarray(flow, dtype=np.float32)
    if probability is not None:
        out["probability"] = np.ascontiguousarray(probability, dtype=np.float32)
    return out
