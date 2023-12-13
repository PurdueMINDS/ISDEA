#
import os
import pytest
import etexood
import shutil
import torch
import numpy as onp
import math
from typing import Sequence
from etexood.dtypes import NPFLOATS


#
ROOT = os.path.join("debug", "log")
DATA = "data"
CACHE = os.path.join("debug", "cache")


#
TEST_FROM_FILE_EVAL = [
    # \\:(os.path.join(DATA, "FD1-ind"), True, "enclose", "grail", "binary"),
    # \\:(os.path.join(DATA, "FD1-ind"), False, "enclose", "grail", "distance"),
    (os.path.join(DATA, "FD1-ind"), True, "heuristics", "dssgnn", "binary"),
    (os.path.join(DATA, "FD1-ind"), False, "heuristics", "dssgnn", "distance"),
    # \\:(os.path.join(DATA, "FD1-ind"), True, "bellman", "nbfnet", "binary"),
    # \\:(os.path.join(DATA, "FD1-ind"), False, "bellman", "nbfnet", "distance"),
]
TEST_FROM_FILE_TRAIN = [
    # \\:(os.path.join(DATA, "FD1-trans"), True, "enclose", "grail", "binary"),
    # \\:(os.path.join(DATA, "FD1-trans"), False, "enclose", "grail", "distance"),
    (os.path.join(DATA, "FD1-trans"), True, "heuristics", "dssgnn", "binary"),
    (os.path.join(DATA, "FD1-trans"), False, "heuristics", "dssgnn", "distance"),
    # \\:(os.path.join(DATA, "FD1-trans"), True, "bellman", "nbfnet", "binary"),
    # \\:(os.path.join(DATA, "FD1-trans"), False, "bellman", "nbfnet", "distance"),
]


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


def validate(vrps: NPFLOATS, nodes: Sequence[int], /, *, num_hops: int) -> None:
    R"""
    Validate node presentations.

    Args
    ----
    - vrps
        Node representations.
    - nodes
        Validating nodes.
    - num_hops
        Number of hops.

    Returns
    -------
    """
    # Ensure representation shape
    assert vrps.ndim == 3 and vrps.shape[1] > 2

    #
    num_layers = int(math.ceil(math.log2(float(len(vrps)))))
    assert len(vrps) == 2**num_layers - 1

    def get_pas_fd(u: int, /) -> Sequence[int]:
        R"""
        Get related nodes in upper number of hops.

        Args
        ----
        - u
            Node.

        Returns
        -------
        - pas
            Related nodes in upper number of hops.
        """
        #
        buf = []
        for _ in range(num_hops + 1):
            #
            if u >= 0:
                #
                buf.append(u)
                u = (u + 1) // 2 - 1
        return buf[1:]

    #
    for u in nodes:
        #
        layer = int(math.floor(math.log2(float(u + 1))))
        lower = 2**layer - 1
        upper = 2 ** (layer + 1) - 1
        assert lower <= u and u < upper

        # If two nodes of the same layer has distance 2 to the power of number of hops, they are members in
        # different family subtree, but of the same role, thus their embedding should be the same.
        v = u + 2**num_hops
        assert v >= upper or onp.all(onp.isclose(vrps[u], vrps[v])).item(), str((u, v))

        # Nodes of the same layer should have the same embedding when father and mother relations are flipped.
        r = 1 - u % 2
        v = (2**layer - 1) * 3 - u
        assert onp.all(onp.isclose(vrps[u, r], vrps[v, 1 - r])).item(), str(((u, r), (v, 1 - r)))
        assert onp.all(onp.isclose(vrps[u, 1 - r], vrps[v, r])).item(), str(((u, 1 - r), (v, r)))

        # Consecutive fathers should havce the same embedding if their full hops are within the full graph.
        # Same thing applies for mothers.
        r = 1 - u % 2
        v = u * 2 + 1 + r
        assert (
            int(math.floor(math.log2(float(v + 1)))) + num_hops >= num_layers
            or tuple(get_pas_fd(u)) != tuple(get_pas_fd(v))
            or torch.all(torch.isclose(vrps[u], vrps[v])).item()
        ), str((u, v))


@pytest.mark.parametrize(("path", "bidirect", "sample", "name_model", "name_loss"), TEST_FROM_FILE_EVAL)
def test_evaluate(*, path: str, bidirect: bool, sample: str, name_model: str, name_loss: str) -> None:
    R"""
    Test evaluation.

    Args
    ----
    - path
        Path.
    - bidirect
        Treat given observed graph as bidirected graph.
        Inversed relations will be automatically added to the graph.
    - sample
        Sampling data format.
    - name_model
        Model name.
    - name_loss
        Loss function name.

    Returns
    -------
    """
    # Cache must start from empty.
    path_dataset = path
    path_cache = os.path.join(CACHE, os.path.basename(path_dataset))
    if os.path.isdir(path_cache):
        #
        shutil.rmtree(path_cache)
    os.makedirs(path_cache, exist_ok=False)

    #
    prefix = "~".join(["transform", "evaluate"])
    suffix = os.path.basename(path)
    unique = etexood.loggings.create_framework_directory(ROOT, prefix, "", suffix, sleep=1.1, max_waits=11)
    logger = etexood.loggings.create_logger(unique, os.path.basename(unique), level_file=None, level_console=None)

    #
    dataset = etexood.datasets.DatasetTriplet.from_file(logger, path_dataset)

    # Prepare observed graph.
    num_nodes = len(dataset._entity2id)
    num_relations = len(dataset._relation2id)
    tripelts_observe = dataset.triplets_observe
    adjs_observe = tripelts_observe[:, :2].T
    rels_observe = tripelts_observe[:, 2]
    assert onp.all(onp.isin(rels_observe, [0, 1])).item()

    #
    if bidirect:
        #
        logger.info("-- Augment observation by inversion.")
        adjs_observe = onp.concatenate((adjs_observe[[0, 1]], adjs_observe[[1, 0]]), axis=1)
        rels_observe = onp.concatenate((rels_observe, rels_observe + num_relations))

    # Test on all triplets.
    tripelts_target = dataset._triplets
    adjs_target = tripelts_target[:, :2].T
    rels_target = tripelts_target[:, 2]

    # Lock triplets in memory.
    adjs_observe.setflags(write=False)
    rels_observe.setflags(write=False)
    adjs_target.setflags(write=False)
    rels_target.setflags(write=False)

    #
    logger.info("-- Create a randomly initialized model.")
    num_hops = 3
    num_hiddens = 4
    kwargs = {
        "activate": "relu",
        "dropout": 0.0,
        "num_bases": 4,
        "kernel": "gin",
        "train_eps": True,
        "dss_aggr": "mean",
        "ablate": "both",
    }
    model = etexood.models.create_model(
        num_nodes,
        num_relations * (1 + int(bidirect)),
        num_hops,
        num_hiddens,
        name_model,
        kwargs,
    )

    #
    seed = 42
    device = torch.device("cpu")
    model.reset_parameters(torch.Generator("cpu").manual_seed(seed))
    model = model.to(device)

    #
    num_epochs = 2
    seed = 42
    negative_rate = 5
    batch_size_node = 64
    batch_size_edge = 128 * (1 + negative_rate)
    transformer = etexood.frameworks.transform.Evaluator(
        logger,
        num_nodes,
        adjs_observe,
        rels_observe,
        path_cache,
        sample,
        bidirect=bidirect,
        num_relations=num_relations,
        num_hops=num_hops,
        num_processes=4,
        unit_process=1.0,
        device=device,
    )
    transformer.generate(
        adjs_target,
        rels_target,
        1,
        batch_size_node=batch_size_node,
        batch_size_edge=batch_size_edge,
        negative_rate=negative_rate,
        seed=seed,
        reusable_edge=True,
    )

    # Force to generate heuristics once before loading.
    etexood.frameworks.transform.Evaluator(
        logger,
        num_nodes,
        adjs_observe,
        rels_observe,
        path_cache,
        "heuristics",
        bidirect=bidirect,
        num_relations=num_relations,
        num_hops=num_hops,
        num_processes=4,
        unit_process=1.0,
        device=device,
    ).generate(
        adjs_target,
        rels_target,
        1,
        batch_size_node=batch_size_node,
        batch_size_edge=batch_size_edge,
        negative_rate=negative_rate,
        seed=seed,
        reusable_edge=True,
    )

    #
    transformer.load(
        1,
        batch_size_node=batch_size_node,
        batch_size_edge=batch_size_edge,
        negative_rate=negative_rate,
        seed=seed,
        reusable_edge=True,
    )

    # Collect all node embeddings and validate.
    if name_model == "dssgnn":
        #
        logger.info("-- Collect full node representations:")
        validate(transformer.embed(model).data.numpy().astype(onp.float64), list(range(num_nodes)), num_hops=num_hops)

    #
    logger.info("-- Evaluate:")
    for eind in range(1, num_epochs + 1):
        #
        transformer.test(
            model,
            name_loss,
            ks=[2, 10],
            negative_rate=negative_rate,
            margin=1.0,
            eind=eind,
            emax=num_epochs,
        )


@pytest.mark.parametrize(("path", "bidirect", "sample", "name_model", "name_loss"), TEST_FROM_FILE_TRAIN)
def test_train(*, path: str, bidirect: bool, sample: str, name_model: str, name_loss: str) -> None:
    R"""
    Test training.

    Args
    ----
    - path
        Path.
    - bidirect
        Treat given observed graph as bidirected graph.
        Inversed relations will be automatically added to the graph.
    - sample
        Sampling data format.
    - name_model
        Model name.
    - name_loss
        Loss function name.

    Returns
    -------
    """
    # Cache must start from empty.
    path_dataset = path
    path_cache = os.path.join(CACHE, os.path.basename(path_dataset))
    if os.path.isdir(path_cache):
        #
        shutil.rmtree(path_cache)
    os.makedirs(path_cache, exist_ok=False)

    #
    prefix = "~".join(["transform", "evaluate"])
    suffix = os.path.basename(path)
    unique = etexood.loggings.create_framework_directory(ROOT, prefix, "", suffix, sleep=1.1, max_waits=11)
    logger = etexood.loggings.create_logger(unique, os.path.basename(unique), level_file=None, level_console=None)

    #
    dataset = etexood.datasets.DatasetTriplet.from_file(logger, path_dataset)

    # Prepare observed graph.
    num_nodes = len(dataset._entity2id)
    num_relations = len(dataset._relation2id)
    tripelts_observe = dataset.triplets_observe
    adjs_observe = tripelts_observe[:, :2].T
    rels_observe = tripelts_observe[:, 2]
    assert onp.all(onp.isin(rels_observe, [0, 1])).item()

    #
    if bidirect:
        #
        logger.info("-- Augment observation by inversion.")
        adjs_observe = onp.concatenate((adjs_observe[[0, 1]], adjs_observe[[1, 0]]), axis=1)
        rels_observe = onp.concatenate((rels_observe, rels_observe + num_relations))

    # Test on all triplets.
    tripelts_target = dataset.triplets_train
    adjs_target = tripelts_target[:, :2].T
    rels_target = tripelts_target[:, 2]

    # Lock triplets in memory.
    adjs_observe.setflags(write=False)
    rels_observe.setflags(write=False)
    adjs_target.setflags(write=False)
    rels_target.setflags(write=False)

    #
    logger.info("-- Create a randomly initialized model.")
    num_hops = 2
    num_hiddens = 4
    kwargs = {
        "activate": "relu",
        "dropout": 0.0,
        "num_bases": 4,
        "kernel": "gin",
        "train_eps": True,
        "dss_aggr": "mean",
        "ablate": "both",
    }
    model = etexood.models.create_model(
        num_nodes,
        num_relations * (1 + int(bidirect)),
        num_hops,
        num_hiddens,
        name_model,
        kwargs,
    )

    #
    seed = 42
    device = torch.device("cpu")
    clip_grad_norm = 1.0
    model.reset_parameters(torch.Generator("cpu").manual_seed(seed))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    #
    num_epochs = 2
    seed = 42
    negative_rate = 2
    batch_size_node = 64
    batch_size_edge = 128 * (1 + negative_rate)
    transformer = etexood.frameworks.transform.Trainer(
        logger,
        num_nodes,
        adjs_observe,
        rels_observe,
        path_cache,
        sample,
        bidirect=bidirect,
        num_relations=num_relations,
        num_hops=num_hops,
        num_processes=4,
        unit_process=1.0,
        device=device,
    )
    transformer.generate(
        adjs_target,
        rels_target,
        num_epochs,
        batch_size_node=batch_size_node,
        batch_size_edge=batch_size_edge,
        negative_rate=negative_rate,
        seed=seed,
        reusable_edge=False,
    )

    # Force to generate heuristics once before loading.
    etexood.frameworks.transform.Trainer(
        logger,
        num_nodes,
        adjs_observe,
        rels_observe,
        path_cache,
        "heuristics",
        bidirect=bidirect,
        num_relations=num_relations,
        num_hops=num_hops,
        num_processes=4,
        unit_process=1.0,
        device=device,
    ).generate(
        adjs_target,
        rels_target,
        num_epochs,
        batch_size_node=batch_size_node,
        batch_size_edge=batch_size_edge,
        negative_rate=negative_rate,
        seed=seed,
        reusable_edge=False,
    )

    #
    transformer.load(
        num_epochs,
        batch_size_node=batch_size_node,
        batch_size_edge=batch_size_edge,
        negative_rate=negative_rate,
        seed=seed,
        reusable_edge=False,
    )

    # Train.
    # Training loss (bce) should has converging tendency.
    buf = []
    for eind in range(1, num_epochs + 1):
        #
        loss = transformer.tune(
            model,
            name_loss,
            optimizer,
            negative_rate=negative_rate,
            margin=1.0,
            clip_grad_norm=clip_grad_norm,
            eind=eind,
            emax=num_epochs,
        )
        buf.append(loss)
    losses = onp.array(buf)
    # \\:assert onp.all(losses[:-1] > losses[1:]).item(), (name_model, name_loss, losses)


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
    for path, bidirect, sample, name_model, name_loss in TEST_FROM_FILE_EVAL:
        #
        test_evaluate(path=path, bidirect=bidirect, sample=sample, name_model=name_model, name_loss=name_loss)
    for path, bidirect, sample, name_model, name_loss in TEST_FROM_FILE_TRAIN:
        #
        test_train(path=path, bidirect=bidirect, sample=sample, name_model=name_model, name_loss=name_loss)
    test_fin()


#
if __name__ == "__main__":
    #
    main()
