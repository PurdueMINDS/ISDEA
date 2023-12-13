#
import pytest
import numpy as onp
import networkx
import etexood
import math
import more_itertools as xitertools
import torch
import torch_geometric as thgeo
from typing import Sequence, Tuple, cast
from etexood.dtypes import NPINTS, NPFLOATS


def test_ini() -> None:
    R"""
    Test initialization.

    Args
    ----

    Returns
    -------
    """


def benchmarks() -> Sequence[Tuple[int, NPINTS, NPINTS]]:
    R"""
    Generate benchmarks.

    Args
    ----

    Returns
    -------
    - graphs
        Graphs with all-pair shortest distances.
    """
    #
    n = 100
    graphs = []
    for (i, graph) in enumerate(
        (networkx.erdos_renyi_graph(n, 0.055, seed=42), networkx.dual_barabasi_albert_graph(n, 9, 2, 0.5, seed=42)),
    ):
        #
        (srcs, dsts) = onp.array(graph.edges).T

        #
        rng = onp.random.RandomState(42)
        mask_forward = rng.uniform(0.0, 1.0, (len(graph.edges),)) < 0.85
        mask_inverse = rng.uniform(0.0, 1.0, (len(graph.edges),)) < 0.85

        #
        (srcs_forward, dsts_forward) = (srcs[mask_forward], dsts[mask_forward])
        (srcs_inverse, dsts_inverse) = (dsts[mask_inverse], srcs[mask_inverse])
        srcs = onp.concatenate((srcs_forward, srcs_inverse))
        dsts = onp.concatenate((dsts_forward, dsts_inverse))
        adjs = onp.stack((srcs, dsts))

        #
        eids_forward = srcs * n + dsts
        eids_inverse = dsts * n + srcs
        assert len(eids_forward) == len(onp.unique(eids_forward))
        assert len(eids_inverse) == len(onp.unique(eids_inverse))
        if onp.any(onp.isin(eids_forward, eids_inverse)).item():
            #
            print("Benchmark graph {:d} has node pairs which are bidirectly connected.".format(i))

        # A trivial check for shortest distance computation.
        xadjs = onp.zeros((2, 0), dtype=adjs.dtype)
        dists = etexood.batches.hop.shortest(n, adjs, xadjs, num_hops=n - 1)
        (rows, cols) = onp.nonzero(dists == 1)
        assert onp.all(onp.sort(srcs * n + dsts) == onp.sort(rows * n + cols)).item()
        graphs.append((n, srcs, dsts))
    return graphs


#
BENCHMARKS0 = list(
    xitertools.flatten(
        [
            xitertools.flatten(
                [[(n, srcs, dsts, num_hops, bidirect) for bidirect in (False, True)] for num_hops in [1, 2, 3]],
            )
            for (n, srcs, dsts) in benchmarks()
        ],
    ),
)
BENCHMARKS1 = [(n, srcs, dsts, 2, i % 2 == 1) for (i, (n, srcs, dsts)) in enumerate(benchmarks())]


@pytest.mark.parametrize(("n", "srcs", "dsts", "num_hops", "bidirect"), BENCHMARKS0)
def test_node(*, n: int, srcs: NPINTS, dsts: NPINTS, num_hops: int, bidirect: bool) -> None:
    R"""
    Test hop sampling centered at nodes.

    Args
    ----
    - n
        Number of nodes.
    - srcs
        Source nodes.
    - dsts
        Destination nodes.
    - num_hops
        Numnber of hops.
    - bidirect
        Treat given graph as bidirected graph.
        Inversed relations will be automatically added to the graph.

    Returns
    -------
    """
    # Generate full graph with potential inversed relations.
    num_nodes = n
    num_edges = max(len(srcs), len(dsts))
    num_relations = num_edges
    triplets = onp.stack((srcs, dsts, onp.arange(num_edges, dtype=onp.int64))).T
    if bidirect:
        #
        adjs = onp.concatenate((triplets[:, [0, 1]], triplets[:, [1, 0]])).T
        rels = onp.concatenate((triplets[:, 2], triplets[:, 2] + num_relations))
    else:
        #
        adjs = triplets[:, :2].T.copy()
        rels = triplets[:, 2].copy()
    adjs.setflags(write=False)
    rels.setflags(write=False)

    # Randomly truncate 5% edges.
    num_edges_reject = int(num_edges * (1 + int(bidirect) * 0.05))
    adjs_reject = adjs[:, onp.random.RandomState(42).permutation(num_edges * (1 + int(bidirect)))[:num_edges_reject]]

    #
    batch_size = int(math.floor(math.sqrt(float(num_nodes))))
    num_batches = int(math.ceil(float(num_nodes) / float(batch_size)))
    indices = onp.random.RandomState(42).permutation(num_nodes)
    tree = etexood.batches.hop.ComputationSubsetNode(num_nodes, adjs, rels, num_hops=num_hops)
    for i in range(num_batches):
        # Get centering nodes of current batch.
        nodes = indices[i * batch_size : min((i + 1) * batch_size, num_nodes)]

        #
        for xadjs in (onp.zeros((2, 0), dtype=adjs.dtype), adjs_reject):
            #
            (_, vids0, adjs0, rels0) = tree.sample("tree", nodes, tree.masks_edge_accept(xadjs))
            (_, vids1, adjs1, rels1) = etexood.batches.hop.computation_tree(
                nodes,
                adjs,
                rels,
                xadjs,
                bidirect=False,
                num_relations=num_relations,
                num_nodes=num_nodes,
                num_hops=num_hops,
            )

            # Get sampled and reduced graph edge indices in full graph.
            # Pay attention that edge ID order matters in comparison.
            eids0 = vids0[adjs0[0]] * n + vids0[adjs0[1]]
            eids1 = vids1[adjs1[0]] * n + vids1[adjs1[1]]
            ords0 = onp.argsort(eids0)
            ords1 = onp.argsort(eids1)
            assert onp.all(eids0[ords0] == eids1[ords1]).item()
            assert onp.all(rels0[ords0] == rels1[ords1]).item()


@pytest.mark.parametrize(("n", "srcs", "dsts", "num_hops", "bidirect"), BENCHMARKS0)
def test_edge(*, n: int, srcs: NPINTS, dsts: NPINTS, num_hops: int, bidirect: bool) -> None:
    R"""
    Test 2-hop sampling centered at edges.

    Args
    ----
    - n
        Number of nodes.
    - srcs
        Source nodes.
    - dsts
        Destination nodes.
    - num_hops
        Numnber of hops.
    - bidirect
        Treat given graph as bidirected graph.
        Inversed relations will be automatically added to the graph.

    Returns
    -------
    """
    # Generate full graph with potential inversed relations.
    num_nodes = n
    num_edges = max(len(srcs), len(dsts))
    num_relations = num_edges
    triplets = onp.stack((srcs, dsts, onp.arange(num_edges, dtype=onp.int64))).T
    if bidirect:
        #
        adjs = onp.concatenate((triplets[:, [0, 1]], triplets[:, [1, 0]])).T
        rels = onp.concatenate((triplets[:, 2], triplets[:, 2] + num_relations))
    else:
        #
        adjs = triplets[:, :2].T.copy()
        rels = triplets[:, 2].copy()
    adjs.setflags(write=False)
    rels.setflags(write=False)

    # Randomly truncate 5% edges.
    num_edges_reject = int(num_edges * (1 + int(bidirect) * 0.05))
    adjs_reject = adjs[:, onp.random.RandomState(42).permutation(num_edges * (1 + int(bidirect)))[:num_edges_reject]]

    #
    batch_size = int(math.floor(math.sqrt(float(num_edges))))
    num_batches = int(math.ceil(float(num_edges) / float(batch_size)))
    indices = onp.random.RandomState(42).permutation(num_edges)
    tree = etexood.batches.hop.ComputationSubsetEdge(num_nodes, adjs, rels, num_hops=num_hops)
    for i in range(num_batches):
        # Get centering nodes of current batch.
        # Pay attention that we only care given direction in centering even if full graph is bidirected.
        # Pay attention that sampled subgraph should be bidirected if full graph is bidirected.
        edges = indices[i * batch_size : min((i + 1) * batch_size, num_nodes)]
        nodes = onp.unique(onp.concatenate((triplets[:, 0][edges], triplets[:, 1][edges])))

        #
        for xadjs in (onp.zeros((2, 0), dtype=adjs.dtype), adjs_reject):
            #
            (_, vids0, adjs0, rels0) = tree.sample("tree", triplets[edges, :2], tree.masks_edge_accept(xadjs))
            (_, vids1, adjs1, rels1) = etexood.batches.hop.computation_tree(
                nodes,
                adjs,
                rels,
                xadjs,
                bidirect=bidirect,
                num_relations=num_relations,
                num_nodes=num_nodes,
                num_hops=num_hops,
            )

            # Get sampled and reduced graph edge indices in full graph.
            # Pay attention that edge ID order matters in comparison.
            eids0 = vids0[adjs0[0]] * n + vids0[adjs0[1]]
            eids1 = vids1[adjs1[0]] * n + vids1[adjs1[1]]
            ords0 = onp.argsort(eids0)
            ords1 = onp.argsort(eids1)
            assert onp.all(eids0[ords0] == eids1[ords1]).item()
            assert onp.all(rels0[ords0] == rels1[ords1]).item()


@torch.no_grad()
def encode(gnns: Sequence[thgeo.nn.RGCNConv], vfts_: NPFLOATS, adjs_: NPINTS, rels_: NPINTS, /) -> NPINTS:
    R"""
    Encode given subgraph.

    Args
    ----
    - gnns
        GNNs.
    - vfts_
        Node features.
    - adjs_
        Adjacency list.
    - rels_
        Relations.

    Returns
    -------
    - vrps
        Node representations.
    """
    #
    vfts = torch.from_numpy(vfts_).to(torch.float32)
    adjs = torch.from_numpy(adjs_).to(torch.int64)
    rels = torch.from_numpy(rels_).to(torch.int64)
    vrps = vfts
    for gnn in gnns:
        #
        vrps = torch.sigmoid(gnn.forward(vrps, adjs, rels))
    return cast(NPINTS, vrps.data.numpy())


@pytest.mark.parametrize(("n", "srcs", "dsts", "num_hops", "bidirect"), BENCHMARKS1)
def test_gnn(*, n: int, srcs: NPINTS, dsts: NPINTS, num_hops: int, bidirect: bool) -> None:
    R"""
    A special check to ensure computation is properly contained in its subgraph.
    Case with isolation node is considered here.

    Args
    ----
    - n
        Number of nodes.
    - srcs
        Source nodes.
    - dsts
        Destination nodes.
    - num_hops
        Numnber of hops.
    - bidirect
        Treat given graph as bidirected graph.
        Inversed relations will be automatically added to the graph.

    Returns
    -------
    """
    # Generate full graph with potential inversed relations.
    # Add some isolation nodes for this test.
    num_nodes = n + int(math.floor(math.sqrt(float(n))))
    num_edges = max(len(srcs), len(dsts))
    num_relations = num_edges
    triplets = onp.stack((srcs, dsts, onp.arange(num_edges, dtype=onp.int64))).T
    if bidirect:
        #
        adjs = onp.concatenate((triplets[:, [0, 1]], triplets[:, [1, 0]])).T
        rels = onp.concatenate((triplets[:, 2], triplets[:, 2] + num_relations))
    else:
        #
        adjs = triplets[:, :2].T.copy()
        rels = triplets[:, 2].copy()
    adjs.setflags(write=False)
    rels.setflags(write=False)

    #
    rng = torch.Generator("cpu").manual_seed(42)
    gnns = [thgeo.nn.RGCNConv(3, 3, num_relations) for _ in range(num_hops)]
    for gnn in gnns:
        # Initialize.
        gnn.weight.data.uniform_(-1.0, 1.0, generator=rng)
        gnn.root.data.uniform_(-1.0, 1.0, generator=rng)
        gnn.bias.data.uniform_(-1.0, 1.0, generator=rng)

        # Freeze.
        gnn.weight.requires_grad = False
        gnn.root.requires_grad = False
        gnn.bias.requires_grad = False
    feats = onp.random.RandomState(42).normal(0.0, 1.0, (num_nodes, 3))

    # Use larger batches for GNN computation.
    # We do not need to test edge rejection here since it is unrelated.
    batch_size = int(math.ceil(float(num_nodes) / 3.0))
    num_batches = 3
    indices = onp.random.RandomState(42).permutation(num_nodes)
    tree = etexood.batches.hop.ComputationSubsetNode(num_nodes, adjs, rels, num_hops=num_hops)
    masks_edge_accept = onp.ones((num_edges * (1 + int(bidirect)),), dtype=onp.bool_)
    for i in range(num_batches):
        # Get centering nodes of current batch.
        nodes = indices[i * batch_size : min((i + 1) * batch_size, num_nodes)]

        # Computation tree should contain even isolated nodes.
        (uids0, vids0, adjs0, rels0) = tree.sample("tree", nodes, masks_edge_accept)
        assert not onp.any(uids0[0][nodes] < 0).item()

        # Computation graph should contain even isolated nodes.
        (uids1, vids1, adjs1, rels1) = tree.sample("graph", nodes, masks_edge_accept)
        assert not onp.any(uids1[nodes] < 0).item()

        # We expect target nodes get the same embeddings on both computation tree and computation graph.
        # We check if two embeddings are close rather than same, since we may find nuance for the same input under
        # some randomly initialized weights.
        vrps0 = encode(gnns, feats[vids0], adjs0, rels0)[uids0[0][nodes]]
        vrps1 = encode(gnns, feats[vids1], adjs1, rels1)[uids1[nodes]]
        assert onp.all(onp.isclose(vrps0, vrps1)).item()


def test_fin() -> None:
    R"""
    Test finalization.

    Args
    ----

    Returns
    -------
    """


def main():
    R"""
    Main execution.

    Args
    ----

    Returns
    -------
    """
    #
    test_ini()
    for test_hop in [test_node, test_edge]:
        #
        for (n, srcs, dsts, num_hops, bidirect) in BENCHMARKS0:
            #
            test_hop(n=n, srcs=srcs, dsts=dsts, num_hops=num_hops, bidirect=bidirect)
    for (n, srcs, dsts, num_hops, bidirect) in BENCHMARKS1:
        #
        test_gnn(n=n, srcs=srcs, dsts=dsts, num_hops=num_hops, bidirect=bidirect)
    test_fin()


#
if __name__ == "__main__":
    #
    main()
