#
import pytest
import numpy as onp
import networkx
import etexood
import os
import shutil
import more_itertools as xitertools
import itertools
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
def test_enclose(*, title: str, n: int, srcs: NPINTS, dsts: NPINTS, num_hops: int, bidirect: bool) -> None:
    R"""
    Test enclose subgraph translation.

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

    # An enclose subgraph translator without multiprocessing.
    # Pay attention that it relies on result from heuristics collector type 1 (with pair removal).
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
    estimator.collect(pairs)
    translator0 = etexood.batches.enclose.Enclose(
        cache,
        num_nodes,
        adjs,
        rels,
        num_hops=num_hops,
        num_processes=1,
        unit=1.0,
    )
    translator0.translate()
    (vpts0, epts0, vids0, vfts0, adjs0, rels0) = translator0.load(pairs)

    # An enclose subgraph translator with multiprocessing.
    # Pay attention that it relies on result from heuristics collector type 1 (with pair removal).
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
    estimator.collect(pairs)
    translator1 = etexood.batches.enclose.Enclose(
        cache,
        num_nodes,
        adjs,
        rels,
        num_hops=num_hops,
        num_processes=4,
        unit=1.0,
    )
    translator1.translate()
    (vpts1, epts1, vids1, vfts1, adjs1, rels1) = translator1.load(pairs)

    #
    assert onp.all(vpts0 == vpts1).item()
    assert onp.all(epts0 == epts1).item()
    assert onp.all(vids0 == vids1).item()
    assert onp.all(vfts0 == vfts1).item()
    assert onp.all(adjs0 == adjs1).item()
    assert onp.all(rels0 == rels1).item()

    # Regular enclose subgraph translator should load from preprocessed cache.
    translator = etexood.batches.enclose.Enclose(
        cache,
        num_nodes,
        adjs,
        rels,
        num_hops=num_hops,
        num_processes=4,
        unit=1.0,
    )
    (vpts0, epts0, vids0, vfts0, adjs0, rels0) = translator.load(pairs)

    #
    buf_vid = []
    buf_vft = []
    buf_adj = []
    buf_rel = []
    for (src, dst) in pairs.T.tolist():
        #
        (vids1, vfts1, adjs1, rels1) = etexood.batches.enclose.enclose(
            num_nodes,
            adjs,
            rels,
            src,
            dst,
            num_hops=num_hops,
        )
        buf_vid.append(vids1)
        buf_vft.append(vfts1)
        buf_adj.append(adjs1)
        buf_rel.append(rels1)
    vpts1 = onp.array([0] + list(itertools.accumulate(len(vids1) for vids1 in buf_vid)))
    epts1 = onp.array([0] + list(itertools.accumulate(len(rels1) for rels1 in buf_rel)))
    vids1 = onp.concatenate(buf_vid)
    vfts1 = onp.concatenate(buf_vft)
    adjs1 = onp.concatenate(buf_adj, axis=1)
    rels1 = onp.concatenate(buf_rel)

    #
    assert len(vpts0) == len(vpts1) and len(vpts0) == len(pairs.T) + 1
    assert len(epts0) == len(epts1) and len(epts0) == len(pairs.T) + 1
    for i in range(len(pairs.T)):
        #
        bgn0 = vpts0[i].item()
        end0 = vpts0[i + 1].item()
        vord0 = onp.argsort(vids0[bgn0:end0])

        #
        bgn1 = vpts1[i].item()
        end1 = vpts1[i + 1].item()
        vord1 = onp.argsort(vids1[bgn1:end1])

        #
        assert onp.all(vids0[bgn0:end0][vord0] == vids1[bgn1:end1][vord1]).item()
        assert onp.all(vfts0[bgn0:end0][vord0] == vfts1[bgn1:end1][vord1]).item()

        #
        bgn0 = epts0[i].item()
        end0 = epts0[i + 1].item()
        eord0 = onp.argsort(adjs0[0, bgn0:end0] * num_nodes + adjs0[1, bgn0:end0])

        #
        bgn1 = epts1[i].item()
        end1 = epts1[i + 1].item()
        eord1 = onp.argsort(adjs1[0, bgn1:end1] * num_nodes + adjs1[1, bgn1:end1])

        #
        assert onp.all(adjs0[0, bgn0:end0][eord0] == adjs1[0, bgn1:end1][eord1]).item()
        assert onp.all(adjs0[1, bgn0:end0][eord0] == adjs1[1, bgn1:end1][eord1]).item()
        assert onp.all(rels0[bgn0:end0][eord0] == rels1[bgn1:end1][eord1]).item()


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
        test_enclose(title=title, n=n, srcs=srcs, dsts=dsts, num_hops=num_hops, bidirect=bidirect)
    test_fin()


#
if __name__ == "__main__":
    #
    main()
