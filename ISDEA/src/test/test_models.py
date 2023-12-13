#
import os
import pytest
import etexood
import shutil
import networkx
import numpy as onp
import torch
import math
import more_itertools as xitertools
from typing import Sequence, Tuple, TypeVar, Mapping, Any, cast
from etexood.dtypes import NPINTS


#
# \\:SelfSpecialGraIL = TypeVar("SelfSpecialGraIL", bound="SpecialGraIL")
SelfSpecialDSSGNNExcl = TypeVar("SelfSpecialDSSGNNExcl", bound="SpecialDSSGNNExcl")


#
ROOT = os.path.join("debug", "log")


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
# \\:MODELS = ["distmult", "transe", "rgcn", "grail", "nbfnet", "dssgnn"]
MODELS = ["dssgnn"]
BENCHMARKS = list(
    xitertools.flatten(
        [
            [(n, srcs, dsts, 2, i % 2 == 1, name) for (i, (n, srcs, dsts)) in enumerate(benchmarks())]
            for name in MODELS
        ],
    ),
)


@pytest.mark.xfail(raises=RuntimeError)
def test_model_unknown() -> None:
    R"""
    Test unknown model creation.

    Args
    ----

    Returns
    -------
    """
    #
    etexood.models.create_model(1, 1, 1, 1, "", {})


@pytest.mark.xfail(raises=RuntimeError)
def test_loss_unknown() -> None:
    R"""
    Test unknown loss function.

    Args
    ----

    Returns
    -------
    """
    #
    etexood.models.get_loss(1, 1, 1, 1, "", {})


@pytest.mark.xfail(raises=RuntimeError)
def test_kernel_unknown() -> None:
    R"""
    Test unknown DSSGNN kernel.

    Args
    ----

    Returns
    -------
    """
    #
    etexood.models.create_model(
        1,
        1,
        1,
        1,
        "dssgnn",
        {"activate": "relu", "dropout": 0.0, "kernel": "", "train_eps": False},
    )


@pytest.mark.xfail(raises=RuntimeError)
def test_kernel_unknown2() -> None:
    R"""
    Test unknown DSSGNN kernel on initialization.

    Args
    ----

    Returns
    -------
    """
    #
    model = etexood.models.create_model(
        1,
        1,
        1,
        1,
        "dssgnn",
        {"activate": "relu", "dropout": 0.0, "kernel": "gin", "train_eps": True},
    )
    setattr(getattr(model, "convs")[0], "kernel", "")
    model.reset_parameters(torch.Generator("cpu").manual_seed(42))


@pytest.mark.parametrize(("n", "srcs", "dsts", "num_hops", "bidirect", "name"), BENCHMARKS)
def test_forward(*, n: int, srcs: NPINTS, dsts: NPINTS, num_hops: int, bidirect: bool, name: str) -> None:
    R"""
    Test model forwarding.

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
    - name
        Model name.

    Returns
    -------
    """
    #
    num_nodes = n
    num_edges = max(len(srcs), len(dsts))
    num_relations = int(math.floor(math.sqrt(float(num_edges))))
    if bidirect:
        #
        adjs = torch.from_numpy(onp.concatenate((onp.stack((srcs, dsts)), onp.stack((dsts, srcs))), axis=1))
        rels = torch.from_numpy(
            onp.concatenate(
                (onp.arange(num_edges) % num_relations, onp.arange(num_edges) % num_relations + num_relations),
            ),
        )
    else:
        #
        adjs = torch.from_numpy(onp.stack((srcs, dsts)))
        rels = torch.from_numpy(onp.arange(num_edges) % num_relations)
    # \\:if name == "grail":
    # \\:    #
    # \\:    vfts = torch.zeros(num_nodes, 2, dtype=torch.int64)
    # \\:    vpts = torch.tile(torch.tensor([[0, num_nodes]]), (len(rels), 1))
    # \\:    epts = torch.stack((torch.arange(len(rels)), torch.arange(len(rels)) + 1), dim=1)
    # \\:else:
    # \\:    #
    # \\:    vfts = torch.arange(num_nodes)
    # \\:    heus = torch.full((len(rels), 2), num_hops + 1, dtype=torch.int64)
    vfts = torch.arange(num_nodes)
    heus = torch.full((len(rels), 2), num_hops + 1, dtype=torch.int64)
    lbls = torch.ones(len(rels))

    #
    num_hiddens = 4
    kwargs = {"activate": "relu", "dropout": 0.0, "num_bases": 4, "kernel": "gin", "train_eps": True}
    model = etexood.models.create_model(
        num_nodes,
        num_relations * (1 + int(bidirect)),
        num_hops,
        num_hiddens,
        name,
        kwargs,
    )
    model.reset_parameters(torch.Generator("cpu").manual_seed(42))
    etexood.models.get_loss(num_nodes, num_relations * (1 + int(bidirect)), num_hops, num_hiddens, name, kwargs)
    assert model.get_num_relations() == num_relations * (1 + int(bidirect))

    #
    # \\:if name == "grail":
    # \\:    #
    # \\:    vrps = cast(etexood.models.ModelEnclose, model).forward(vfts, adjs, rels, vpts, epts, rels)
    # \\:elif name == "nbfnet":
    # \\:    #
    # \\:    vrps = cast(etexood.models.ModelBellman, model).forward(vfts, adjs, rels, adjs, rels, lbls)
    # \\:else:
    # \\:    #
    # \\:    vrps = cast(etexood.models.ModelHeuristics, model).forward(vfts, adjs, rels)
    vrps = model.forward(vfts, adjs, rels)
    # \\:assert len(vrps) == len(rels) if name == "nbfnet" else num_nodes
    assert len(vrps) == num_nodes
    assert tuple(vrps.shape[1:]) == tuple(model.get_embedding_shape_entity())

    #
    # \\:if name == "grail":
    # \\:    #
    # \\:    model.loss_function_binary(vrps, adjs, rels, vpts, lbls, sample_negative_rate=0)
    # \\:    model.loss_function_distance(vrps, adjs, rels, vpts, lbls, sample_negative_rate=0, margin=1.0)
    # \\:else:
    # \\:    #
    # \\:    model.loss_function_binary(vrps, adjs, rels, heus, lbls, sample_negative_rate=0)
    # \\:    model.loss_function_distance(vrps, adjs, rels, heus, lbls, sample_negative_rate=0, margin=1.0)
    model.loss_function_binary(vrps, adjs, rels, heus, lbls, sample_negative_rate=0)
    model.loss_function_distance(vrps, adjs, rels, heus, lbls, sample_negative_rate=0, margin=1.0)


# \\:class SpecialGraIL(etexood.models.GraIL):
# \\:    R"""
# \\:    A special model for testing.
# \\:    """
# \\:
# \\:    def measure_score(
# \\:        self: SelfSpecialGraIL,
# \\:        vrps: torch.Tensor,
# \\:        adjs: torch.Tensor,
# \\:        rels: torch.Tensor,
# \\:        heus: torch.Tensor,
# \\:        /,
# \\:    ) -> torch.Tensor:
# \\:        R"""
# \\:        Compute score measurement for given triplets.
# \\:
# \\:        Args
# \\:        ----
# \\:        - vrps
# \\:            Node representations.
# \\:        - adjs
# \\:            Adjacency list to be evaluated.
# \\:        - rels
# \\:            Relations to be evaluated.
# \\:        - heus
# \\:            Placeholder for heuristics of adjacency list.
# \\:            Here, it is specially used as node bounardies of each subgraph.
# \\:
# \\:        Returns
# \\:        -------
# \\:        - measures
# \\:            Measurements.
# \\:        """
# \\:        #
# \\:        subs_given_rel = vrps[adjs[0]].to(rels.device, non_blocking=True)
# \\:        objs_given_rel = vrps[adjs[1]].to(rels.device, non_blocking=True)
# \\:
# \\:        #
# \\:        scores = torch.sum(subs_given_rel * objs_given_rel, dim=1)
# \\:        return scores


class SpecialDSSGNNExcl(etexood.models.DSSGNNExcl):
    R"""
    A special model for testing.
    """

    def measure_score(
        self: SelfSpecialDSSGNNExcl,
        vrps: torch.Tensor,
        adjs: torch.Tensor,
        rels: torch.Tensor,
        heus: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Compute score measurement for given triplets.

        Args
        ----
        - vrps
            Node representations.
        - adjs
            Adjacency list to be evaluated.
        - rels
            Relations to be evaluated.
        - heus
            Heuristics of adjacency list.

        Returns
        -------
        - measures
            Measurements.
        """
        #
        rids = torch.arange(len(rels), device=rels.device)
        subs_given_rel = vrps[adjs[0]][rids, rels].to(rels.device, non_blocking=True)
        objs_given_rel = vrps[adjs[1]][rids, rels].to(rels.device, non_blocking=True)

        #
        scores = torch.sum(subs_given_rel * objs_given_rel, dim=1)
        return scores


def special_create_model(
    num_entities: int,
    num_relations: int,
    num_layers: int,
    num_hiddens: int,
    name: str,
    kwargs: Mapping[str, Any],
    /,
) -> etexood.models.Model:
    R"""
    Special model creation.

    Args
    ----
    - num_entities
        Number of entities.
    - num_relations
        Number of relations.
    - num_layers
        Number of layers.
    - num_hiddens
        Number of hidden embeddings.
    - name
        Model name.
    - kwargs
        Keyword arguments for given model name.

    Returns
    -------
    - model
        Model.
    """
    #
    if name == "dssgnn":
        #
        return SpecialDSSGNNExcl(
            num_entities,
            num_relations,
            num_layers,
            num_hiddens,
            activate=str(kwargs["activate"]),
            dropout=float(kwargs["dropout"]),
            kernel=str(kwargs["kernel"]),
            train_eps=bool(kwargs["train_eps"]),
        )
    # \\:elif name == "grail":
    # \\:    #
    # \\:    return SpecialGraIL(
    # \\:        num_entities,
    # \\:        num_relations,
    # \\:        num_layers,
    # \\:        num_hiddens,
    # \\:        activate=str(kwargs["activate"]),
    # \\:        dropout=float(kwargs["dropout"]),
    # \\:        num_bases=int(kwargs["num_bases"]),
    # \\:    )
    else:
        #
        return etexood.models.create_model(num_entities, num_relations, num_layers, num_hiddens, name, kwargs)


@pytest.mark.parametrize(("n", "srcs", "dsts", "num_hops", "bidirect", "name"), BENCHMARKS)
def test_measure(*, n: int, srcs: NPINTS, dsts: NPINTS, num_hops: int, bidirect: bool, name: str) -> None:
    R"""
    Test model measurements.

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
    - name
        Model name.

    Returns
    -------
    """
    #
    num_nodes = n
    num_edges = max(len(srcs), len(dsts))
    num_relations = int(math.floor(math.sqrt(float(num_edges))))

    #
    num_hiddens = num_nodes
    kwargs = {"activate": "relu", "dropout": 0.0, "num_bases": 4, "kernel": "gin", "train_eps": True}
    model = special_create_model(
        num_nodes,
        num_relations * (1 + int(bidirect)),
        num_hops,
        num_hiddens,
        name,
        kwargs,
    )

    # Enforce node representations for testing.
    if name in ["dssgnn"]:
        #
        vrps = torch.zeros(num_nodes, num_relations * (1 + int(bidirect)), num_nodes)
        vrps[torch.arange(num_nodes - 0), :, torch.arange(num_nodes - 0) + 0] = 1.00
        vrps[torch.arange(num_nodes - 1), :, torch.arange(num_nodes - 1) + 1] = 0.50
        vrps[torch.arange(num_nodes - 2), :, torch.arange(num_nodes - 2) + 2] = 0.25
    elif name == "nbfnet":
        # Can not check for NBFNet.
        return
    else:
        #
        vrps = torch.zeros(num_nodes, num_nodes)
        vrps[torch.arange(num_nodes - 0), torch.arange(num_nodes - 0) + 0] = 1.00
        vrps[torch.arange(num_nodes - 1), torch.arange(num_nodes - 1) + 1] = 0.50
        vrps[torch.arange(num_nodes - 2), torch.arange(num_nodes - 2) + 2] = 0.25
    assert tuple(vrps.shape[1:]) == tuple(model.get_embedding_shape_entity())

    # Enforce parameters unrelated to node representations for testing.
    if name == "dssgnn":
        #
        lin2 = getattr(model, "lin2")
        lin2.weight.data.fill_(0.0)
        lin2.bias.data.fill_(1.0)
    elif name == "grail":
        #
        lin = getattr(model, "lin")
        lin.weight.data.fill_(0.0)
        lin.bias.data.fill_(1.0)
    elif name in ["rgcn", "distmult"]:
        #
        embedding_relation = getattr(model, "embedding_relation")
        for r in range(model.get_num_relations()):
            #
            embedding_relation.data[r].copy_(torch.eye(num_hiddens))
        assert torch.all(getattr(model, "embedding_relation").data == torch.eye(num_hiddens).unsqueeze(0)).item()
    elif name == "transe":
        #
        embedding_relation = getattr(model, "embedding_relation")
        embedding_relation.data.zero_()

    subs_test = torch.tensor([0, 0, 0, 0])
    objs_test = torch.tensor([0, 1, 2, 3])
    adjs_test = torch.stack((subs_test, objs_test))
    rels_test = torch.tensor([0, 0, 0, 0])
    heus_test = torch.full((len(rels_test), 2), num_hops, dtype=torch.int64)
    vpts_test = torch.tile(torch.tensor([[0, num_nodes]]), (len(rels_test), 1))

    #
    dists = model.measure_distance(vrps, adjs_test, rels_test, vpts_test if name == "grail" else heus_test)
    assert tuple(torch.argsort(dists).tolist()) == (0, 1, 2, 3)

    #
    scores = model.measure_score(vrps, adjs_test, rels_test, vpts_test if name == "grail" else heus_test)
    assert tuple(torch.argsort(scores).tolist()) == (3, 2, 1, 0)

    # Ranking case 1.
    sample_negative_rate = 3
    subs_test = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
    objs_test = torch.tensor([1, 2, 0, 2, 3, 0, 1, 3])
    adjs_test = torch.stack((subs_test, objs_test))
    rels_test = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
    lbls_test = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0]).to(torch.get_default_dtype())
    heus_test = torch.full((len(rels_test), 2), num_hops, dtype=torch.int64)
    vpts_test = torch.tile(torch.tensor([[0, num_nodes]]), (len(lbls_test), 1))

    #
    model.is_loss_function_safe(
        vrps,
        adjs_test,
        rels_test,
        vpts_test if name == "grail" else heus_test,
        lbls_test,
        sample_negative_rate=sample_negative_rate,
    )
    (ranks, scores) = model.metric_function_rank(
        vrps,
        adjs_test,
        rels_test,
        vpts_test if name == "grail" else heus_test,
        lbls_test,
        sample_negative_rate=sample_negative_rate,
        ks=(1, 2),
    )
    assert onp.isclose(ranks["MR"], (2.0 + 3.0) * 0.5)
    assert onp.isclose(ranks["MRR"], (1.0 / 2.0 + 1.0 / 3.0) * 0.5)
    assert onp.isclose(ranks["Hit@1"], (0.0 + 0.0) * 0.5)
    assert onp.isclose(ranks["Hit@2"], (1.0 + 0.0) * 0.5)

    # Ranking case 2.
    sample_negative_rate = 2
    subs_test = torch.tensor([0, 0, 0, 0, 0, 0])
    objs_test = torch.tensor([1, 2, 2, 3, 1, 3])
    adjs_test = torch.stack((subs_test, objs_test))
    rels_test = torch.tensor([0, 0, 0, 0, 0, 0])
    lbls_test = torch.tensor([1, 1, 0, 0, 0, 0]).to(torch.get_default_dtype())
    heus_test = torch.full((len(rels_test), 2), num_hops, dtype=torch.int64)
    vpts_test = torch.tile(torch.tensor([[0, num_nodes]]), (len(lbls_test), 1))

    #
    model.is_loss_function_safe(
        vrps,
        adjs_test,
        rels_test,
        vpts_test if name == "grail" else heus_test,
        lbls_test,
        sample_negative_rate=sample_negative_rate,
    )
    (ranks, scores) = model.metric_function_rank(
        vrps,
        adjs_test,
        rels_test,
        vpts_test if name == "grail" else heus_test,
        lbls_test,
        sample_negative_rate=sample_negative_rate,
        ks=(1, 2),
    )
    assert onp.isclose(ranks["MR"], (1.0 + 2.0) * 0.5)
    assert onp.isclose(ranks["MRR"], (1.0 / 1.0 + 1.0 / 2.0) * 0.5)
    assert onp.isclose(ranks["Hit@1"], (1.0 + 0.0) * 0.5)
    assert onp.isclose(ranks["Hit@2"], (1.0 + 1.0) * 0.5)


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
    for test_xfail in [
        test_model_unknown,
        test_loss_unknown,
        test_kernel_unknown,
        test_kernel_unknown2,
    ]:
        #
        try:
            #
            test_xfail()
        except RuntimeError:
            #
            pass
        else:
            #
            raise RuntimeError("Expect failure, but pass successfully.")
    for (n, srcs, dsts, num_hops, bidirect, name) in BENCHMARKS:
        #
        test_forward(n=n, srcs=srcs, dsts=dsts, num_hops=num_hops, bidirect=bidirect, name=name)
    for (n, srcs, dsts, num_hops, bidirect, name) in BENCHMARKS:
        #
        test_measure(n=n, srcs=srcs, dsts=dsts, num_hops=num_hops, bidirect=bidirect, name=name)
    test_fin()


#
if __name__ == "__main__":
    #
    main()
