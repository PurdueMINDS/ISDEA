#
import os
import pytest
import etexood
import shutil
import math
import numpy as onp
import more_itertools as xitertools


#
ROOT = os.path.join("debug", "log")
DATA = "data"
CACHE = os.path.join("debug", "cache")


#
TEST_FROM_FILE = [os.path.join(DATA, "FD1-trans"), os.path.join(DATA, "FD1-ind")]
TEST_FROM_FILE_REUSE = list(xitertools.flatten([[(path, flag) for flag in (False, True)] for path in TEST_FROM_FILE]))


def test_ini() -> None:
    R"""
    Test initialization.

    Args
    ----

    Returns
    -------
    """
    #
    for directory in [ROOT, CACHE]:
        #
        if os.path.isdir(directory):
            #
            shutil.rmtree(directory)
        while os.path.isdir(directory):
            #
            pass


@pytest.mark.parametrize("path", TEST_FROM_FILE)
def test_batch_node(*, path: str) -> None:
    R"""
    Test node batching.

    Args
    ----
    - path
        Path.

    Returns
    -------
    """
    #
    path_dataset = path
    path_cache = os.path.join(CACHE, os.path.basename(path_dataset))

    #
    prefix = "~".join(["batch", "node"])
    suffix = os.path.basename(path)
    unique = etexood.loggings.create_framework_directory(ROOT, prefix, "", suffix, sleep=1.1, max_waits=11)
    logger = etexood.loggings.create_logger(unique, os.path.basename(unique), level_file=None, level_console=None)
    os.makedirs(path_cache, exist_ok=True)

    #
    dataset = etexood.datasets.DatasetTriplet.from_file(logger, path_dataset)

    #
    num_nodes = len(dataset._entity2id)
    tripelts = dataset.triplets_train
    adjs = tripelts[:, :2].T
    rels = tripelts[:, 2]

    # Get 2 batch sizes to be tested.
    batch_size_min = 2 ** int(math.floor(math.log2(math.sqrt(float(num_nodes)))))
    batch_size_max = 2 ** int(math.ceil(math.log2(float(num_nodes))))

    #
    minibatch = etexood.batches.batch.MinibatchNode(logger, path_cache).register(
        etexood.batches.hop.ComputationSubsetNode(num_nodes, adjs, rels, num_hops=2),
    )

    #
    num_epochs = 2
    for batch_size in (batch_size_min, batch_size_max):
        #
        logger.info("-- Generate node minibatch schedule of batch size {:d}.".format(batch_size))
        minibatch.generate("nodes{:d}".format(batch_size), onp.arange(num_nodes), batch_size)

    #
    for batch_size in (batch_size_min, batch_size_max):
        #
        logger.info("-- Load and traverse node minibatch schedule of batch size {:d}.".format(batch_size))
        minibatch.load("nodes{:d}".format(batch_size))

        #
        for eid in range(1, num_epochs + 1):
            #
            num_updates = onp.zeros((num_nodes,), dtype=onp.int64)

            #
            minibatch.epoch(eid, num_epochs)
            for bid in range(minibatch.num_batches()):
                #
                (ucenters, uids, vids, _, _) = minibatch.batch(bid)
                onp.add.at(num_updates, vids[ucenters], 1)
                assert onp.all(uids[vids] == onp.arange(len(vids))).item()
            assert onp.all(num_updates == 1).item()


@pytest.mark.parametrize(("path", "reusable"), TEST_FROM_FILE_REUSE)
def test_batch_edge_heuristics(*, path: str, reusable: bool) -> None:
    R"""
    Test edge batching with heuristics.

    Args
    ----
    - path
        Path.
    - reusable
        Reusablility.

    Returns
    -------
    """
    #
    path_dataset = path
    path_cache = os.path.join(CACHE, os.path.basename(path_dataset))

    #
    prefix = "~".join(["batch", "edge"])
    suffix = os.path.basename(path)
    unique = etexood.loggings.create_framework_directory(ROOT, prefix, "", suffix, sleep=1.1, max_waits=11)
    logger = etexood.loggings.create_logger(unique, os.path.basename(unique), level_file=None, level_console=None)
    os.makedirs(path_cache, exist_ok=True)

    #
    dataset = etexood.datasets.DatasetTriplet.from_file(logger, path_dataset)

    #
    num_nodes = len(dataset._entity2id)
    num_relations = len(dataset._relation2id)
    tripelts_observe = dataset.triplets_observe
    adjs_observe = tripelts_observe[:, :2].T
    rels_observe = tripelts_observe[:, 2]
    tripelts_train = dataset.triplets_train
    adjs_train = tripelts_train[:, :2].T
    rels_train = tripelts_train[:, 2]

    # Get 2 batch sizes to be tested.
    num_edges = len(rels_train)
    negative_rate = 2
    batch_size_min = 2 ** int(math.floor(math.log2(float(num_edges)))) * (1 + negative_rate)
    batch_size_max = 2 ** int(math.ceil(math.log2(float(num_edges)))) * (1 + negative_rate)

    #
    minibatch = etexood.batches.batch.MinibatchEdgeHeuristics(logger, path_cache).register(
        etexood.batches.heuristics.HeuristicsForest1(
            path_cache,
            num_nodes,
            adjs_observe,
            rels_observe,
            num_hops=2,
            num_processes=4,
            unit=1.0,
        ),
        etexood.batches.hop.ComputationSubsetEdge(num_nodes, adjs_observe, rels_observe, num_hops=2),
    )

    #
    num_epochs = 2
    for batch_size in (batch_size_min, batch_size_max):
        #
        logger.info("-- Generate edge minibatch schedule of batch size {:d}.".format(batch_size))
        minibatch.generate(
            "edges{:d}".format(batch_size),
            adjs_train,
            rels_train,
            batch_size,
            negative_rate=negative_rate,
            rng=onp.random.RandomState(42),
            num_epochs=1 if reusable else num_epochs,
            reusable=reusable,
        )

    #
    for batch_size in (batch_size_min, batch_size_max):
        #
        logger.info("-- Load and traverse edge minibatch schedule of batch size {:d}.".format(batch_size))
        minibatch.load("edges{:d}".format(batch_size))

        #
        for eid in range(1, num_epochs + 1):
            #
            buf = []

            #
            minibatch.epoch(eid, num_epochs)
            for bid in range(minibatch.num_batches()):
                #
                (
                    adjs_target,
                    rels_target,
                    heus_target,
                    lbls_target,
                    uids_observe,
                    vids_observe,
                    adjs_observe,
                    rels_observe,
                ) = minibatch.batch(bid)
                num_pairs = len(lbls_target) // (1 + negative_rate)
                buf.extend(
                    (
                        (
                            vids_observe[adjs_target[0, :num_pairs]] * num_nodes
                            + vids_observe[adjs_target[1, :num_pairs]]
                        )
                        * num_relations
                        + rels_target[:num_pairs]
                    ).tolist(),
                )
                assert len(lbls_target) % (1 + negative_rate) == 0
                assert onp.all(uids_observe[vids_observe] == onp.arange(len(vids_observe))).item()
                assert onp.all(uids_observe[vids_observe[adjs_target]] == adjs_target).item()
                assert onp.all(
                    onp.logical_or(
                        onp.reshape(adjs_target[0, :num_pairs], (num_pairs, 1))
                        == onp.reshape(adjs_target[0, num_pairs:], (num_pairs, negative_rate)),
                        onp.reshape(adjs_target[1, :num_pairs], (num_pairs, 1))
                        == onp.reshape(adjs_target[1, num_pairs:], (num_pairs, negative_rate)),
                    )
                ).item()
                # \\:assert onp.all(
                # \\:    onp.reshape(adjs_target[0, :num_pairs], (num_pairs, 1))
                # \\:    == onp.reshape(adjs_target[0, num_pairs:], (num_pairs, negative_rate)),
                # \\:).item()
                assert onp.all(
                    onp.reshape(rels_target[:num_pairs], (num_pairs, 1))
                    == onp.reshape(rels_target[num_pairs:], (num_pairs, negative_rate))
                ).item()
                assert heus_target.shape == (len(rels_target), minibatch._heuristics.NUM_HEURISTICS)
                assert onp.all(lbls_target[:num_pairs] == 1).item()
                assert onp.all(lbls_target[num_pairs:] == 0).item()

            #
            tids0 = onp.array(buf)
            tids1 = (adjs_train[0] * num_nodes + adjs_train[1]) * num_relations + rels_train
            assert onp.all(onp.sort(tids0) == onp.sort(tids1)).item()


@pytest.mark.parametrize(("path", "reusable"), TEST_FROM_FILE_REUSE)
def test_batch_edge_enclose(*, path: str, reusable: bool) -> None:
    R"""
    Test edge batching with enclosed subgraph.

    Args
    ----
    - path
        Path.
    - reusable
        Reusablility.

    Returns
    -------
    """
    #
    path_dataset = path
    path_cache = os.path.join(CACHE, os.path.basename(path_dataset))

    #
    prefix = "~".join(["batch", "edge"])
    suffix = os.path.basename(path)
    unique = etexood.loggings.create_framework_directory(ROOT, prefix, "", suffix, sleep=1.1, max_waits=11)
    logger = etexood.loggings.create_logger(unique, os.path.basename(unique), level_file=None, level_console=None)
    os.makedirs(path_cache, exist_ok=True)

    #
    dataset = etexood.datasets.DatasetTriplet.from_file(logger, path_dataset)

    #
    num_nodes = len(dataset._entity2id)
    num_relations = len(dataset._relation2id)
    tripelts_observe = dataset.triplets_observe
    adjs_observe = tripelts_observe[:, :2].T
    rels_observe = tripelts_observe[:, 2]
    tripelts_train = dataset.triplets_train
    adjs_train = tripelts_train[:, :2].T
    rels_train = tripelts_train[:, 2]

    # Get 2 batch sizes to be tested.
    num_edges = len(rels_train)
    negative_rate = 2
    batch_size_min = 2 ** int(math.floor(math.log2(float(num_edges)))) * (1 + negative_rate)
    batch_size_max = 2 ** int(math.ceil(math.log2(float(num_edges)))) * (1 + negative_rate)

    # Heuristics version will serve as a preprocessing step for enclosed subgraph version.
    minibatch_prep = etexood.batches.batch.MinibatchEdgeHeuristics(logger, path_cache).register(
        etexood.batches.heuristics.HeuristicsForest1(
            path_cache,
            num_nodes,
            adjs_observe,
            rels_observe,
            num_hops=2,
            num_processes=4,
            unit=1.0,
        ),
        etexood.batches.hop.ComputationSubsetEdge(num_nodes, adjs_observe, rels_observe, num_hops=2),
    )

    # We will not generate since it will load cached generation from heuristics version.
    num_epochs = 2
    for batch_size in (batch_size_min, batch_size_max):
        #
        logger.info("-- Generate edge minibatch schedule of batch size {:d}.".format(batch_size))
        minibatch_prep.generate(
            "edges{:d}".format(batch_size),
            adjs_train,
            rels_train,
            batch_size,
            negative_rate=negative_rate,
            rng=onp.random.RandomState(42),
            num_epochs=1 if reusable else num_epochs,
            reusable=reusable,
        )

    # Enclosed subgraph schedule is created after preprocessing.
    minibatch = etexood.batches.batch.MinibatchEdgeEnclose(logger, path_cache).register(
        etexood.batches.enclose.Enclose(
            path_cache,
            num_nodes,
            adjs_observe,
            rels_observe,
            num_hops=2,
            num_processes=4,
            unit=1.0,
        ),
    )
    minibatch.__annotate__()
    minibatch.reusable = reusable

    #
    for batch_size in (batch_size_min, batch_size_max):
        #
        logger.info("-- Load and traverse edge minibatch schedule of batch size {:d}.".format(batch_size))
        minibatch.load("edges{:d}".format(batch_size))

        #
        for eid in range(1, num_epochs + 1):
            #
            buf_eid = []

            #
            minibatch.epoch(eid, num_epochs)
            for bid in range(minibatch.num_batches()):
                # Pay attention that each pair will generate independent subgraph, thus there is no strict relations
                # between corresponding positive and negative adjacency list as heuristics version.
                (
                    adjs_target,
                    rels_target,
                    lbls_target,
                    vpts_observe,
                    epts_observe,
                    vids_observe,
                    vfts_observe,
                    adjs_observe,
                    rels_observe,
                ) = minibatch.batch(bid)
                num_pairs = len(lbls_target) // (1 + negative_rate)
                assert onp.all(lbls_target[:num_pairs] == 1).item()
                assert onp.all(lbls_target[num_pairs:] == 0).item()
                assert onp.all(
                    onp.reshape(rels_target[:num_pairs], (num_pairs, 1))
                    == onp.reshape(rels_target[num_pairs:], (num_pairs, negative_rate))
                ).item()

                #
                bias = 0
                for i in range(len(lbls_target)):
                    #
                    vbgn_subgraph = vpts_observe[i].item()
                    vend_subgraph = vpts_observe[i + 1].item()
                    ebgn_subgraph = epts_observe[i].item()
                    eend_subgraph = epts_observe[i + 1].item()
                    vids_subgraph = vids_observe[vbgn_subgraph:vend_subgraph]
                    adjs_subgraph = adjs_observe[:, ebgn_subgraph:eend_subgraph] - bias

                    #
                    pair_subgraph = vids_subgraph[adjs_target[:, i] - bias]
                    src_subgraph = pair_subgraph[0].item()
                    dst_subgraph = pair_subgraph[1].item()
                    rel_subgraph = rels_target[i].item()
                    if (lbls_target[i] == 1).item():
                        #
                        eid_subgraph = (src_subgraph * num_nodes + dst_subgraph) * num_relations + rel_subgraph
                        buf_eid.append(eid_subgraph)
                    assert onp.all(adjs_subgraph >= 0).item() and onp.all(adjs_subgraph < len(vids_subgraph)).item()
                    bias += len(vids_subgraph)
                assert tuple(vfts_observe.shape) == (len(vids_observe), 2)

            #
            assert onp.all(
                onp.sort(onp.array(buf_eid))
                == onp.sort((adjs_train[0] * num_nodes + adjs_train[1]) * num_relations + rels_train)
            ).item()


def test_fin() -> None:
    R"""
    Test finalization.

    Args
    ----

    Returns
    -------
    """
    #
    for directory in [ROOT, CACHE]:
        #
        if os.path.isdir(directory):
            #
            shutil.rmtree(directory)
        while os.path.isdir(directory):
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
    for path in TEST_FROM_FILE:
        #
        test_batch_node(path=path)
    for test_batch_edge in [test_batch_edge_heuristics, test_batch_edge_enclose]:
        #
        for (path, reusable) in TEST_FROM_FILE_REUSE:
            #
            test_batch_edge(path=path, reusable=reusable)
    test_fin()


#
if __name__ == "__main__":
    #
    main()
