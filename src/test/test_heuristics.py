#
import pytest
import numpy as onp
import networkx
import etexood
import os
import shutil
import more_itertools as xitertools
from typing import Sequence, Tuple
from etexood.dtypes import NPINTS


#
ROOT = os.path.join("debug", "cache")


def test_ini() -> None:
    R"""
    Test initialization.

    Args
    ----

    Returns
    -------
    """
    #
    if os.path.isdir(ROOT):
        #
        shutil.rmtree(ROOT)
    while os.path.isdir(ROOT):
        #
        pass


def benchmarks() -> Sequence[Tuple[str, int, NPINTS, NPINTS]]:
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
    for (i, (title, graph)) in enumerate(
        (
            ("er", networkx.erdos_renyi_graph(n, 0.055, seed=42)),
            ("ba", networkx.dual_barabasi_albert_graph(n, 9, 2, 0.5, seed=42)),
        ),
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
        graphs.append((title, n, srcs, dsts))
    return graphs


#
BENCHMARKS = list(
    xitertools.flatten(
        [
            xitertools.flatten(
                [[(title, n, srcs, dsts, num_hops, bidirect) for bidirect in (False, True)] for num_hops in [2, 3]],
            )
            for (title, n, srcs, dsts) in benchmarks()
        ],
    ),
)


@pytest.mark.parametrize(("title", "n", "srcs", "dsts", "num_hops", "bidirect"), BENCHMARKS)
def test_heuristics0(*, title: str, n: int, srcs: NPINTS, dsts: NPINTS, num_hops: int, bidirect: bool) -> None:
    R"""
    Test hop sampling centered at nodes.

    Args
    ----
    - title
        Data title.
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
    pairs = onp.stack((srcs, dsts))
    adjs.setflags(write=False)
    rels.setflags(write=False)
    pairs.setflags(write=False)

    # Prepare cache.
    cache = os.path.join(ROOT, "~".join((title, "dx2" if bidirect else "dx1")))

    # An heuristics collector without multiprocessing.
    if os.path.isdir(cache):
        #
        shutil.rmtree(cache)
    os.makedirs(cache, exist_ok=False)
    estimator = etexood.batches.heuristics.HeuristicsForest0(
        cache,
        num_nodes,
        adjs,
        rels,
        num_hops=num_hops,
        num_processes=1,
        unit=1.0,
    )
    estimator.forest(onp.arange(num_nodes))
    estimator.forest(onp.arange(num_nodes))
    estimator.collect(pairs)
    estimator.collect(pairs)
    heuristics0 = estimator.load(pairs)

    # An heuristics collector with multiprocessing.
    if os.path.isdir(cache):
        #
        shutil.rmtree(cache)
    os.makedirs(cache, exist_ok=False)
    estimator = etexood.batches.heuristics.HeuristicsForest0(
        cache,
        num_nodes,
        adjs,
        rels,
        num_hops=num_hops,
        num_processes=4,
        unit=1.0,
    )
    estimator.forest(onp.arange(num_nodes))
    estimator.forest(onp.arange(num_nodes))
    estimator.collect(pairs)
    estimator.collect(pairs)
    heuristics1 = estimator.load(pairs)

    #
    assert onp.all(heuristics0 == heuristics1).item()

    # Regular heuristics collector should load from preprocessed cache.
    estimator = etexood.batches.heuristics.HeuristicsForest0(
        cache,
        num_nodes,
        adjs,
        rels,
        num_hops=num_hops,
        num_processes=4,
        unit=1.0,
    )
    estimator.collect(pairs)
    estimator.collect(pairs)
    heuristics0 = estimator.load(pairs)

    # Pay attention that heuristics is the distance from arbitrary node to target node.
    xadjs = onp.zeros((2, 0), dtype=adjs.dtype)
    shortests = etexood.batches.hop.shortest(num_nodes, adjs, xadjs, num_hops=num_hops)
    heuristics1 = onp.stack((shortests[pairs[1], pairs[0]], shortests[pairs[0], pairs[1]]), axis=1)

    #
    assert onp.all(heuristics0[:, :2] == heuristics1).item()


@pytest.mark.parametrize(("title", "n", "srcs", "dsts", "num_hops", "bidirect"), BENCHMARKS)
def test_heuristics1(*, title: str, n: int, srcs: NPINTS, dsts: NPINTS, num_hops: int, bidirect: bool) -> None:
    R"""
    Test hop sampling centered at nodes.

    Args
    ----
    - title
        Data title.
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
    pairs = onp.stack((srcs, dsts))
    adjs.setflags(write=False)
    rels.setflags(write=False)
    pairs.setflags(write=False)

    # Prepare cache.
    cache = os.path.join(ROOT, "~".join((title, "dx2" if bidirect else "dx1")))

    # An heuristics collector without multiprocessing.
    if os.path.isdir(cache):
        #
        shutil.rmtree(cache)
    os.makedirs(cache, exist_ok=False)
    estimator = etexood.batches.heuristics.HeuristicsForest1(
        cache,
        num_nodes,
        adjs,
        rels,
        num_hops=num_hops,
        num_processes=1,
        unit=1.0,
    )
    estimator.forest(pairs)
    estimator.forest(pairs)
    estimator.collect(pairs)
    estimator.collect(pairs)
    heuristics0 = estimator.load(pairs)

    # An heuristics collector with multiprocessing.
    if os.path.isdir(cache):
        #
        shutil.rmtree(cache)
    os.makedirs(cache, exist_ok=False)
    estimator = etexood.batches.heuristics.HeuristicsForest1(
        cache,
        num_nodes,
        adjs,
        rels,
        num_hops=num_hops,
        num_processes=4,
        unit=1.0,
    )
    estimator.forest(pairs)
    estimator.forest(pairs)
    estimator.collect(pairs)
    estimator.collect(pairs)
    heuristics1 = estimator.load(pairs)

    #
    assert onp.all(heuristics0 == heuristics1).item()

    # Regular heuristics collector should load from preprocessed cache.
    estimator = etexood.batches.heuristics.HeuristicsForest1(
        cache,
        num_nodes,
        adjs,
        rels,
        num_hops=num_hops,
        num_processes=4,
        unit=1.0,
    )
    heuristics0 = estimator.load(pairs)

    # Pay attention that heuristics is the distance from arbitrary node to target node.
    buf = []
    for (src, dst) in pairs.T.tolist():
        #
        xadjs = onp.array([(src, dst), (dst, src)]).T
        shortests = etexood.batches.hop.shortest(num_nodes, adjs, xadjs, num_hops=num_hops)
        buf.append((shortests[dst, src], shortests[src, dst]))
    heuristics1 = onp.array(buf)

    #
    assert onp.all(heuristics0[:, :2] == heuristics1).item()


def test_fin() -> None:
    R"""
    Test finalization.

    Args
    ----

    Returns
    -------
    """
    #
    if os.path.isdir(ROOT):
        #
        shutil.rmtree(ROOT)
    while os.path.isdir(ROOT):
        #
        pass


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
    for (title, n, srcs, dsts, num_hops, bidirect) in BENCHMARKS:
        #
        test_heuristics0(title=title, n=n, srcs=srcs, dsts=dsts, num_hops=num_hops, bidirect=bidirect)
    for (title, n, srcs, dsts, num_hops, bidirect) in BENCHMARKS:
        #
        test_heuristics1(title=title, n=n, srcs=srcs, dsts=dsts, num_hops=num_hops, bidirect=bidirect)
    test_fin()


#
if __name__ == "__main__":
    #
    main()
