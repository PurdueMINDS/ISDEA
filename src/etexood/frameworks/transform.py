#
import logging
import torch
import numpy as onp
from typing import TypeVar, Sequence, List, Tuple, Dict, cast
from ..dtypes import NPINTS, NPFLOATS
from ..batches.batch import MinibatchNode, MinibatchEdgeHeuristics, MinibatchEdgeEnclose
from ..batches.hop import ComputationSubsetNode, ComputationSubsetEdge
from ..batches.heuristics import HeuristicsForest1
from ..batches.enclose import Enclose
from ..models import Model, ModelHeuristics


#
SelfTransformer = TypeVar("SelfTransformer", bound="Transformer")
SelfEvaluator = TypeVar("SelfEvaluator", bound="Evaluator")
SelfTrainer = TypeVar("SelfTrainer", bound="Trainer")


class Transformer(object):
    R"""
    Transformation controller.
    """
    #
    HEURISTICS = 0
    ENCLOSE = 1
    BELLMAN = 2

    def __init__(
        self: SelfTransformer,
        logger: logging.Logger,
        num_nodes: int,
        adjs: NPINTS,
        rels: NPINTS,
        cache: str,
        sample: str,
        /,
        *,
        bidirect: bool,
        num_relations: int,
        num_hops: int,
        num_processes: int,
        unit_process: float,
        device: torch.device,
    ) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - logger
            Logging terminal.
        - num_nodes
            Number of nodes to transform.
        - adjs
            Observed adjacency list.
        - rels
            Observed relations.
        - cache
            Cache directory for transformation schedule.
        - sample
            Sampling type.
            It should be either "heuristics" or "enclose".
        - bidirect
            Automatically generate inversed relations in sampling.
            We assume that all given triplet are single direction, and their inversions do not exist.
        - num_relations
            Number of total relations assuming no inversion.
        - num_hops
            Number of avaibale hops in transformation.
        - num_processes
            Number of processes.
        - unit_process
            Reporting time unit (second) for each process.
        - device
            Computation device.

        Returns
        -------
        """
        #
        self._logger = logger
        self._num_nodes = num_nodes
        self._adjs = adjs
        self._rels = rels
        self._cache = cache
        self._sample = {"heuristics": self.HEURISTICS, "enclose": self.ENCLOSE, "bellman": self.BELLMAN}[sample]
        self._bidirect = bidirect
        self._num_relations = num_relations
        self._num_hops = num_hops
        self._num_processes = num_processes
        self._unit_process = unit_process
        self._device = device

        # Safety check.
        if self._bidirect:
            #
            assert len(self._rels) % 2 == 0
            breakpoint = len(self._rels) // 2
            adjs_forward = self._adjs[:, :breakpoint]
            adjs_inverse = self._adjs[:, breakpoint:]
            rels_forward = self._rels[:breakpoint]
            rels_inverse = self._rels[breakpoint:]
            assert onp.all(adjs_forward[0] == adjs_inverse[1]).item()
            assert onp.all(adjs_forward[1] == adjs_inverse[0]).item()
            # \\:assert onp.all(rels_forward == rels_inverse - self._num_relations).item()

        # All transformer requires a node and edge scheduler.
        self._minibatch_node = MinibatchNode(self._logger, self._cache)
        # \\:if self._sample in (self.HEURISTICS, self.BELLMAN):
        # \\:    #
        # \\:    self._minibatch_edge_heuristics = MinibatchEdgeHeuristics(self._logger, self._cache)
        # \\:elif self._sample == self.ENCLOSE:
        # \\:    #
        # \\:    self._minibatch_edge_enclose = MinibatchEdgeEnclose(self._logger, self._cache)
        self._minibatch_edge_heuristics = MinibatchEdgeHeuristics(self._logger, self._cache)

    def generate(
        self: SelfTransformer,
        adjs: NPINTS,
        rels: NPINTS,
        num_epochs: int,
        /,
        *,
        batch_size_node: int,
        batch_size_edge: int,
        negative_rate: int,
        seed: int,
        reusable_edge: bool,
    ) -> None:
        R"""
        Generate focusing target schedule.

        Args
        ----
        - adjs
            Targeting adjacency list.
        - rels
            Targeting relations.
        - num_epochs
            Number of epochs.
        - batch_size_node
            Batch size of node sampling.
            This is used for collecting representations for all nodes.
        - batch_size_edge
            Batch size of edge sampling.
            This is used for computing loss or metric for all given edges with negative samples.
        - negative_rate
            Negative sampling rate.
        - seed
            Random seed for generation.
        - resuable_edge
            If edge sampling can be reused in different epochs.

        Returns
        -------
        """
        # Node batching should always traverse all nodes.
        self._minibatch_node.register(
            ComputationSubsetNode(self._num_nodes, self._adjs, self._rels, num_hops=self._num_hops),
        )

        # Target batching will traverse all given edges with negative samples.
        # Only edge sampling with heuristics can generate.
        if self._sample in (self.HEURISTICS, self.BELLMAN):
            #
            self._minibatch_edge_heuristics.register(
                HeuristicsForest1(
                    self._cache,
                    self._num_nodes,
                    self._adjs,
                    self._rels,
                    num_hops=self._num_hops,
                    num_processes=self._num_processes,
                    unit=self._unit_process,
                ),
                ComputationSubsetEdge(self._num_nodes, self._adjs, self._rels, num_hops=self._num_hops),
            )

        #
        self._minibatch_node.generate(
            "~".join(("node", "e{:d}-b{:d}".format(1, batch_size_node))),
            onp.arange(self._num_nodes),
            batch_size_node,
        )
        if self._sample in (self.HEURISTICS, self.BELLMAN):
            #
            self._minibatch_edge_heuristics.generate(
                "~".join(("edge", "e{:d}-b{:d}-n{:d}-s{:d}".format(num_epochs, batch_size_edge, negative_rate, seed))),
                adjs,
                rels,
                batch_size_edge,
                negative_rate=negative_rate,
                rng=onp.random.RandomState(seed),
                num_epochs=num_epochs,
                reusable=reusable_edge,
            )

    def load(
        self: SelfTransformer,
        num_epochs: int,
        /,
        *,
        batch_size_node: int,
        batch_size_edge: int,
        negative_rate: int,
        seed: int,
        reusable_edge: bool,
    ) -> None:
        R"""
        load focusing target schedule.

        Args
        ----
        - num_epochs
            Number of epochs.
        - batch_size_node
            Batch size of node sampling.
            This is used for collecting representations for all nodes.
        - batch_size_edge
            Batch size of edge sampling.
            This is used for computing loss or metric for all given edges with negative samples.
        - negative_rate
            Negative sampling rate.
        - seed
            Random seed for generation.
        - resuable_edge
            If edge sampling can be reused in different epochs.

        Returns
        -------
        """
        #
        self._minibatch_node.register(
            ComputationSubsetNode(self._num_nodes, self._adjs, self._rels, num_hops=self._num_hops),
        )

        #
        # \\:if self._sample in (self.HEURISTICS, self.BELLMAN):
        # \\:    #
        # \\:    self._minibatch_edge_heuristics.register(
        # \\:        HeuristicsForest1(
        # \\:            self._cache,
        # \\:            self._num_nodes,
        # \\:            self._adjs,
        # \\:            self._rels,
        # \\:            num_hops=self._num_hops,
        # \\:            num_processes=self._num_processes,
        # \\:            unit=self._unit_process,
        # \\:        ),
        # \\:        ComputationSubsetEdge(self._num_nodes, self._adjs, self._rels, num_hops=self._num_hops),
        # \\:    )
        # \\:elif self._sample == self.ENCLOSE:
        # \\:    #
        # \\:    self._minibatch_edge_enclose.register(
        # \\:        Enclose(
        # \\:            self._cache,
        # \\:            self._num_nodes,
        # \\:            self._adjs,
        # \\:            self._rels,
        # \\:            num_hops=self._num_hops,
        # \\:            num_processes=self._num_processes,
        # \\:            unit=self._unit_process,
        # \\:        ),
        # \\:    )
        self._minibatch_edge_heuristics.register(
            HeuristicsForest1(
                self._cache,
                self._num_nodes,
                self._adjs,
                self._rels,
                num_hops=self._num_hops,
                num_processes=self._num_processes,
                unit=self._unit_process,
            ),
            ComputationSubsetEdge(self._num_nodes, self._adjs, self._rels, num_hops=self._num_hops),
        )

        #
        self._minibatch_node.load("~".join(("node", "e{:d}-b{:d}".format(1, batch_size_node))))
        # \\:if self._sample in (self.HEURISTICS, self.BELLMAN):
        # \\:    #
        # \\:    self._minibatch_edge_heuristics.load(
        # \\:        "~".join(("edge", "e{:d}-b{:d}-n{:d}-s{:d}".format(num_epochs, batch_size_edge, negative_rate, seed))),
        # \\:    )
        # \\:    self._minibatch_edge_heuristics.reusable = reusable_edge
        # \\:elif self._sample == self.ENCLOSE:
        # \\:    #
        # \\:    self._minibatch_edge_enclose.load(
        # \\:        "~".join(("edge", "e{:d}-b{:d}-n{:d}-s{:d}".format(num_epochs, batch_size_edge, negative_rate, seed))),
        # \\:    )
        # \\:    self._minibatch_edge_enclose.reusable = reusable_edge
        self._minibatch_edge_heuristics.load(
            "~".join(("edge", "e{:d}-b{:d}-n{:d}-s{:d}".format(num_epochs, batch_size_edge, negative_rate, seed))),
        )
        self._minibatch_edge_heuristics.reusable = reusable_edge

    @torch.no_grad()
    def embed(self: SelfTransformer, model: Model, /) -> torch.Tensor:
        R"""
        Collect node embeddings.

        Args
        ----
        - model
            Model used to generate embeddings.

        Returns
        -------
        - vrps
            Node embeddings.
        """
        #
        assert (
            self._sample == self.HEURISTICS
        ), "Only model utilizing heuristics-like input can provide full node embeddings."
        # \\:model = cast(ModelHeuristics, model)
        model.eval()

        #
        updates: Sequence[List[int]]

        #
        vrps = torch.zeros(self._num_nodes, *model.get_embedding_shape_entity(), device="cpu")
        updates = [[] for _ in range(len(vrps))]

        #
        self._minibatch_node.epoch(1, 1)
        for bid in range(self._minibatch_node.num_batches()):
            #
            (ucenters_numpy, uids_numpy, vids_numpy, adjs_numpy, rels_numpy) = self._minibatch_node.batch(bid)
            ucenters_torch = torch.from_numpy(ucenters_numpy).to(self._device)
            uids_torch = torch.from_numpy(uids_numpy).to(self._device)
            vids_torch = torch.from_numpy(vids_numpy).to(self._device)
            adjs_torch = torch.from_numpy(adjs_numpy).to(self._device)
            rels_torch = torch.from_numpy(rels_numpy).to(self._device)
            vrps.index_copy_(
                0,
                vids_torch[ucenters_torch],
                model.forward(vids_torch, adjs_torch, rels_torch)[ucenters_torch],
            )
            for v in vids_numpy[ucenters_numpy].tolist():
                #
                updates[v].append(bid)

        # Full node emebddings must be stored on CPU.
        assert not vrps.is_cuda
        return vrps


class Evaluator(Transformer):
    R"""
    Transformation controller for evaluation.
    """

    def generate(
        self: SelfEvaluator,
        adjs: NPINTS,
        rels: NPINTS,
        num_epochs: int,
        /,
        *,
        batch_size_node: int,
        batch_size_edge: int,
        negative_rate: int,
        seed: int,
        reusable_edge: bool,
    ) -> None:
        R"""
        Generate focusing target schedule.

        Args
        ----
        - adjs
            Targeting adjacency list.
        - rels
            Targeting relations.
        - num_epochs
            Number of epochs.
        - batch_size_node
            Batch size of node sampling.
            This is used for collecting representations for all nodes.
        - batch_size_edge
            Batch size of edge sampling.
            This is used for computing loss or metric for all given edges with negative samples.
        - negative_rate
            Negative sampling rate.
        - seed
            Random seed for generation.
        - resuable_edge
            If edge sampling can be reused in different epochs.

        Returns
        -------
        """
        #
        assert num_epochs == 1 and reusable_edge
        Transformer.generate(
            self,
            adjs,
            rels,
            num_epochs,
            batch_size_node=batch_size_node,
            batch_size_edge=batch_size_edge,
            negative_rate=negative_rate,
            seed=seed,
            reusable_edge=reusable_edge,
        )

    @torch.no_grad()
    def test(
        self: SelfEvaluator,
        model: Model,
        name_loss: str,
        /,
        *,
        ks: Sequence[int],
        negative_rate: int,
        margin: float,
        eind: int,
        emax: int,
    ) -> Tuple[Dict[str, float], Sequence[NPFLOATS]]:
        R"""
        Tune model parameters by optimizer.

        Args
        ----
        - model
            Model used to generate embeddings.
        - name_loss
            Loss function name.
        - ks
            All k value to compute hit-at-k metrics.
        - negative_rate
            Negative rate.
        - margin
            Distance margin.
        - eind
            Epoch ID.
        - emax
            Epoch maximum.

        Returns
        -------
        - metrics
            Tested metrics.
        - scores
            Scores for each batch.
        """
        #
        buf = []
        name_loss2 = {"binary": "BCE", "distance": "Dist"}[name_loss]
        values = {name_loss2: 0.0, "MR": 0.0, "MRR": 0.0, **{"Hit@{:d}".format(k): 0.0 for k in ks}}
        counts = {name_loss2: 0, "MR": 0, "MRR": 0, **{"Hit@{:d}".format(k): 0 for k in ks}}

        #
        ranger = {
            name_loss2: lambda _, n_samples: n_samples,
            "MR": lambda n_pairs, _: n_pairs,
            "MRR": lambda n_pairs, _: n_pairs,
            **{"Hit@{:d}".format(k): lambda n_pairs, _: n_pairs for k in ks},
        }

        #
        # \\:if self._sample in (self.HEURISTICS, self.BELLMAN):
        # \\:    #
        # \\:    self._minibatch_edge_heuristics.epoch(eind, emax)
        # \\:    num_batches = self._minibatch_edge_heuristics.num_batches()
        # \\:elif self._sample == self.ENCLOSE:
        # \\:    #
        # \\:    self._minibatch_edge_enclose.epoch(eind, emax)
        # \\:    num_batches = self._minibatch_edge_enclose.num_batches()
        self._minibatch_edge_heuristics.epoch(eind, emax)
        num_batches = self._minibatch_edge_heuristics.num_batches()
        for bid in range(num_batches):
            #
            if self._sample == self.HEURISTICS:
                #
                (ranks, scores, labels) = self.evaluate_heuristics(
                    bid,
                    model,
                    name_loss,
                    ks=ks,
                    negative_rate=negative_rate,
                    margin=margin,
                )
            # \\:elif self._sample == self.ENCLOSE:
            # \\:    #
            # \\:    (ranks, scores, labels) = self.evaluate_enclose(
            # \\:        bid,
            # \\:        model,
            # \\:        name_loss,
            # \\:        ks=ks,
            # \\:        negative_rate=negative_rate,
            # \\:        margin=margin,
            # \\:    )
            # \\:elif self._sample == self.BELLMAN:
            # \\:    #
            # \\:    (ranks, scores, labels) = self.evaluate_bellman(
            # \\:        bid,
            # \\:        model,
            # \\:        name_loss,
            # \\:        ks=ks,
            # \\:        negative_rate=negative_rate,
            # \\:        margin=margin,
            # \\:    )

            #
            buf.append(scores.data.cpu().numpy())
            n_pairs = onp.sum(labels == 1).item()
            n_samples = len(labels)
            for key in set(values.keys()) | set(ranks.keys()):
                #
                values[key] += ranks[key] * float(ranger[key](n_pairs, n_samples))
                counts[key] += int(ranger[key](n_pairs, n_samples))

        #
        metrics = {key: values[key] / float(counts[key]) for key in values}
        return (metrics, buf)

    def evaluate_heuristics(
        self: SelfEvaluator,
        bid: int,
        model: Model,
        name_loss: str,
        /,
        *,
        ks: Sequence[int],
        negative_rate: int,
        margin: float,
    ) -> Tuple[Dict[str, float], torch.Tensor, NPINTS]:
        R"""
        Evaluate on heuristics-like input.

        Args
        ----
        - bid
            Evaluating batch ID in current epoch.
        - model
            Model used to generate embeddings.
        - name_loss
            Loss function name.
        - ks
            All k value to compute hit-at-k metrics.
        - negative_rate
            Negative rate.
        - margin
            Distance margin.

        Returns
        -------
        - ranks
            Evaluation metrics (mostly rank-based) of current batch.
        - scores
            Scores of current batch.
        - labels
            Binary classification labels of current batch.
        """
        #
        assert self._sample == self.HEURISTICS
        # \\:model = cast(ModelHeuristics, model)
        model.eval()

        #
        (
            adjs_target_numpy,
            rels_target_numpy,
            heus_target_numpy,
            lbls_target_numpy,
            uids_observe_numpy,
            vids_observe_numpy,
            adjs_observe_numpy,
            rels_observe_numpy,
        ) = self._minibatch_edge_heuristics.batch(bid)
        adjs_target_torch = torch.from_numpy(adjs_target_numpy).to(self._device)
        rels_target_torch = torch.from_numpy(rels_target_numpy).to(self._device)
        heus_target_torch = torch.from_numpy(heus_target_numpy).to(self._device)
        lbls_target_torch = torch.from_numpy(lbls_target_numpy).to(torch.get_default_dtype()).to(self._device)
        uids_observe_torch = torch.from_numpy(uids_observe_numpy).to(self._device)
        vids_observe_torch = torch.from_numpy(vids_observe_numpy).to(self._device)
        adjs_observe_torch = torch.from_numpy(adjs_observe_numpy).to(self._device)
        rels_observe_torch = torch.from_numpy(rels_observe_numpy).to(self._device)

        # Achieve node representations.
        vrps = model.forward(vids_observe_torch, adjs_observe_torch, rels_observe_torch)

        # Get naive training loss as one of the metrics.
        # In the evaluation of knowledge graph, negative samples should come strictly with corresponding positive
        # samples.
        model.is_loss_function_safe(
            vrps,
            adjs_target_torch,
            rels_target_torch,
            heus_target_torch,
            lbls_target_torch,
            sample_negative_rate=negative_rate,
        )
        assert name_loss in ("binary", "distance")
        if name_loss == "binary":
            #
            loss = model.loss_function_binary(
                vrps,
                adjs_target_torch,
                rels_target_torch,
                heus_target_torch,
                lbls_target_torch,
                sample_negative_rate=negative_rate,
            )
        elif name_loss == "distance":
            #
            loss = model.loss_function_distance(
                vrps,
                adjs_target_torch,
                rels_target_torch,
                heus_target_torch,
                lbls_target_torch,
                sample_negative_rate=negative_rate,
                margin=margin,
            )

        #
        (ranks, scores) = model.metric_function_rank(
            vrps,
            adjs_target_torch,
            rels_target_torch,
            heus_target_torch,
            lbls_target_torch,
            sample_negative_rate=negative_rate,
            ks=ks,
        )
        ranks[{"binary": "BCE", "distance": "Dist"}[name_loss]] = loss.item()

        #
        return (ranks, scores, lbls_target_numpy)


# \\:    def evaluate_enclose(
# \\:        self: SelfEvaluator,
# \\:        bid: int,
# \\:        model: Model,
# \\:        name_loss: str,
# \\:        /,
# \\:        *,
# \\:        ks: Sequence[int],
# \\:        negative_rate: int,
# \\:        margin: float,
# \\:    ) -> Tuple[Dict[str, float], torch.Tensor, NPINTS]:
# \\:        R"""
# \\:        Evaluate on enclosed-subgraph-like input.
# \\:
# \\:        Args
# \\:        ----
# \\:        - bid
# \\:            Evaluating batch ID in current epoch.
# \\:        - model
# \\:            Model used to generate embeddings.
# \\:        - name_loss
# \\:            Loss function name.
# \\:        - ks
# \\:            All k value to compute hit-at-k metrics.
# \\:        - negative_rate
# \\:            Negative rate.
# \\:        - margin
# \\:            Distance margin.
# \\:
# \\:        Returns
# \\:        -------
# \\:        - ranks
# \\:            Evaluation metrics (mostly rank-based) of current batch.
# \\:        - scores
# \\:            Scores of current batch.
# \\:        - labels
# \\:            Binary classification labels of current batch.
# \\:        """
# \\:        #
# \\:        assert self._sample == self.ENCLOSE
# \\:        model = cast(ModelEnclose, model)
# \\:
# \\:        #
# \\:        (
# \\:            adjs_target_numpy,
# \\:            rels_target_numpy,
# \\:            lbls_target_numpy,
# \\:            vpts_observe_numpy,
# \\:            epts_observe_numpy,
# \\:            vids_observe_numpy,
# \\:            vfts_observe_numpy,
# \\:            adjs_observe_numpy,
# \\:            rels_observe_numpy,
# \\:        ) = self._minibatch_edge_enclose.batch(bid)
# \\:        adjs_target_torch = torch.from_numpy(adjs_target_numpy).to(self._device)
# \\:        rels_target_torch = torch.from_numpy(rels_target_numpy).to(self._device)
# \\:        lbls_target_torch = torch.from_numpy(lbls_target_numpy).to(torch.get_default_dtype()).to(self._device)
# \\:        vpts_observe_torch = torch.from_numpy(vpts_observe_numpy).to(self._device)
# \\:        epts_observe_torch = torch.from_numpy(epts_observe_numpy).to(self._device)
# \\:        vids_observe_torch = torch.from_numpy(vids_observe_numpy).to(self._device)
# \\:        vfts_observe_torch = torch.from_numpy(vfts_observe_numpy).to(self._device)
# \\:        adjs_observe_torch = torch.from_numpy(adjs_observe_numpy).to(self._device)
# \\:        rels_observe_torch = torch.from_numpy(rels_observe_numpy).to(self._device)
# \\:
# \\:        #
# \\:        vpts_observe_torch = torch.stack((vpts_observe_torch[:-1], vpts_observe_torch[1:]), dim=1)
# \\:        epts_observe_torch = torch.stack((epts_observe_torch[:-1], epts_observe_torch[1:]), dim=1)
# \\:
# \\:        # Achieve node representations.
# \\:        vrps = model.forward(
# \\:            vfts_observe_torch,
# \\:            adjs_observe_torch,
# \\:            rels_observe_torch,
# \\:            vpts_observe_torch,
# \\:            epts_observe_torch,
# \\:            rels_target_torch,
# \\:        )
# \\:
# \\:        # Get naive training loss as one of the metrics.
# \\:        # In the evaluation of knowledge graph, negative samples should come strictly with corresponding positive
# \\:        # samples.
# \\:        model.is_loss_function_safe(
# \\:            vrps,
# \\:            adjs_target_torch,
# \\:            rels_target_torch,
# \\:            vpts_observe_torch,
# \\:            lbls_target_torch,
# \\:            sample_negative_rate=negative_rate,
# \\:        )
# \\:        assert name_loss in ("binary", "distance")
# \\:        if name_loss == "binary":
# \\:            #
# \\:            loss = model.loss_function_binary(
# \\:                vrps,
# \\:                adjs_target_torch,
# \\:                rels_target_torch,
# \\:                vpts_observe_torch,
# \\:                lbls_target_torch,
# \\:                sample_negative_rate=negative_rate,
# \\:            )
# \\:        elif name_loss == "distance":
# \\:            #
# \\:            loss = model.loss_function_distance(
# \\:                vrps,
# \\:                adjs_target_torch,
# \\:                rels_target_torch,
# \\:                vpts_observe_torch,
# \\:                lbls_target_torch,
# \\:                sample_negative_rate=negative_rate,
# \\:                margin=margin,
# \\:            )
# \\:
# \\:        #
# \\:        (ranks, scores) = model.metric_function_rank(
# \\:            vrps,
# \\:            adjs_target_torch,
# \\:            rels_target_torch,
# \\:            vpts_observe_torch,
# \\:            lbls_target_torch,
# \\:            sample_negative_rate=negative_rate,
# \\:            ks=ks,
# \\:        )
# \\:        ranks[{"binary": "BCE", "distance": "Dist"}[name_loss]] = loss.item()
# \\:
# \\:        #
# \\:        return (ranks, scores, lbls_target_numpy)

# \\:    def evaluate_bellman(
# \\:        self: SelfEvaluator,
# \\:        bid: int,
# \\:        model: Model,
# \\:        name_loss: str,
# \\:        /,
# \\:        *,
# \\:        ks: Sequence[int],
# \\:        negative_rate: int,
# \\:        margin: float,
# \\:    ) -> Tuple[Dict[str, float], torch.Tensor, NPINTS]:
# \\:        R"""
# \\:        Evaluate on NBFNet input.
# \\:
# \\:        Args
# \\:        ----
# \\:        - bid
# \\:            Evaluating batch ID in current epoch.
# \\:        - model
# \\:            Model used to generate embeddings.
# \\:        - name_loss
# \\:            Loss function name.
# \\:        - ks
# \\:            All k value to compute hit-at-k metrics.
# \\:        - negative_rate
# \\:            Negative rate.
# \\:        - margin
# \\:            Distance margin.
# \\:
# \\:        Returns
# \\:        -------
# \\:        - ranks
# \\:            Evaluation metrics (mostly rank-based) of current batch.
# \\:        - scores
# \\:            Scores of current batch.
# \\:        - labels
# \\:            Binary classification labels of current batch.
# \\:        """
# \\:        #
# \\:        assert self._sample == self.BELLMAN
# \\:        model = cast(ModelBellman, model)
# \\:
# \\:        #
# \\:        (
# \\:            adjs_target_numpy,
# \\:            rels_target_numpy,
# \\:            heus_target_numpy,
# \\:            lbls_target_numpy,
# \\:            uids_observe_numpy,
# \\:            vids_observe_numpy,
# \\:            adjs_observe_numpy,
# \\:            rels_observe_numpy,
# \\:        ) = self._minibatch_edge_heuristics.batch(bid)
# \\:        adjs_target_torch = torch.from_numpy(adjs_target_numpy).to(self._device)
# \\:        rels_target_torch = torch.from_numpy(rels_target_numpy).to(self._device)
# \\:        heus_target_torch = torch.from_numpy(heus_target_numpy).to(self._device)
# \\:        lbls_target_torch = torch.from_numpy(lbls_target_numpy).to(torch.get_default_dtype()).to(self._device)
# \\:        uids_observe_torch = torch.from_numpy(uids_observe_numpy).to(self._device)
# \\:        vids_observe_torch = torch.from_numpy(vids_observe_numpy).to(self._device)
# \\:        adjs_observe_torch = torch.from_numpy(adjs_observe_numpy).to(self._device)
# \\:        rels_observe_torch = torch.from_numpy(rels_observe_numpy).to(self._device)
# \\:
# \\:        # Achieve node representations.
# \\:        vrps = model.forward(
# \\:            vids_observe_torch,
# \\:            adjs_observe_torch,
# \\:            rels_observe_torch,
# \\:            adjs_target_torch,
# \\:            rels_target_torch,
# \\:            lbls_target_torch,
# \\:        )
# \\:
# \\:        # Get naive training loss as one of the metrics.
# \\:        # In the evaluation of knowledge graph, negative samples should come strictly with corresponding positive
# \\:        # samples.
# \\:        model.is_loss_function_safe(
# \\:            vrps,
# \\:            adjs_target_torch,
# \\:            rels_target_torch,
# \\:            heus_target_torch,
# \\:            lbls_target_torch,
# \\:            sample_negative_rate=negative_rate,
# \\:        )
# \\:        assert name_loss in ("binary", "distance")
# \\:        if name_loss == "binary":
# \\:            #
# \\:            loss = model.loss_function_binary(
# \\:                vrps,
# \\:                adjs_target_torch,
# \\:                rels_target_torch,
# \\:                heus_target_torch,
# \\:                lbls_target_torch,
# \\:                sample_negative_rate=negative_rate,
# \\:            )
# \\:        elif name_loss == "distance":
# \\:            #
# \\:            loss = model.loss_function_distance(
# \\:                vrps,
# \\:                adjs_target_torch,
# \\:                rels_target_torch,
# \\:                heus_target_torch,
# \\:                lbls_target_torch,
# \\:                sample_negative_rate=negative_rate,
# \\:                margin=margin,
# \\:            )
# \\:
# \\:        #
# \\:        (ranks, scores) = model.metric_function_rank(
# \\:            vrps,
# \\:            adjs_target_torch,
# \\:            rels_target_torch,
# \\:            heus_target_torch,
# \\:            lbls_target_torch,
# \\:            sample_negative_rate=negative_rate,
# \\:            ks=ks,
# \\:        )
# \\:        ranks[{"binary": "BCE", "distance": "Dist"}[name_loss]] = loss.item()
# \\:
# \\:        #
# \\:        return (ranks, scores, lbls_target_numpy)


class Trainer(Transformer):
    R"""
    Transformation controller for training.
    """

    def generate(
        self: SelfTrainer,
        adjs: NPINTS,
        rels: NPINTS,
        num_epochs: int,
        /,
        *,
        batch_size_node: int,
        batch_size_edge: int,
        negative_rate: int,
        seed: int,
        reusable_edge: bool,
    ) -> None:
        R"""
        Generate focusing target schedule.

        Args
        ----
        - adjs
            Targeting adjacency list.
        - rels
            Targeting relations.
        - num_epochs
            Number of epochs.
        - batch_size_node
            Batch size of node sampling.
            This is used for collecting representations for all nodes.
        - batch_size_edge
            Batch size of edge sampling.
            This is used for computing loss or metric for all given edges with negative samples.
        - negative_rate
            Negative sampling rate.
        - seed
            Random seed for generation.
        - resuable_edge
            If edge sampling can be reused in different epochs.

        Returns
        -------
        """
        #
        assert num_epochs > 1 and not reusable_edge
        Transformer.generate(
            self,
            adjs,
            rels,
            num_epochs,
            batch_size_node=batch_size_node,
            batch_size_edge=batch_size_edge,
            negative_rate=negative_rate,
            seed=seed,
            reusable_edge=reusable_edge,
        )

    def tune(
        self: SelfTrainer,
        model: Model,
        name_loss: str,
        optimizer: torch.optim.Optimizer,
        /,
        *,
        negative_rate: int,
        margin: float,
        clip_grad_norm: float,
        eind: int,
        emax: int,
    ) -> float:
        R"""
        Tune model parameters by optimizer.

        Args
        ----
        - model
            Model used to generate embeddings.
        - name_loss
            Loss function name.
        - optimizer
            Optimizer.
        - negative_rate
            Negative rate.
        - margin
            Distance margin.
        - clip_grad_norm
            Gradient clip norm.
        - eind
            Epoch ID.
        - emax
            Epoch maximum.

        Returns
        -------
        - loss
            Loss.
        """
        #
        value = 0.0
        count = 0

        #
        # \\:if self._sample in (self.HEURISTICS, self.BELLMAN):
        # \\:    #
        # \\:    self._minibatch_edge_heuristics.epoch(eind, emax)
        # \\:    num_batches = self._minibatch_edge_heuristics.num_batches()
        # \\:elif self._sample == self.ENCLOSE:
        # \\:    #
        # \\:    self._minibatch_edge_enclose.epoch(eind, emax)
        # \\:    num_batches = self._minibatch_edge_enclose.num_batches()
        self._minibatch_edge_heuristics.epoch(eind, emax)
        num_batches = self._minibatch_edge_heuristics.num_batches()
        for bid in range(num_batches):
            #
            optimizer.zero_grad()
            if self._sample == self.HEURISTICS:
                #
                (loss, labels) = self.train_heuristics(
                    bid,
                    model,
                    name_loss,
                    negative_rate=negative_rate,
                    margin=margin,
                )
            # \\:elif self._sample == self.ENCLOSE:
            # \\:    #
            # \\:    (loss, labels) = self.train_enclose(
            # \\:        bid,
            # \\:        model,
            # \\:        name_loss,
            # \\:        negative_rate=negative_rate,
            # \\:        margin=margin,
            # \\:    )
            # \\:elif self._sample == self.BELLMAN:
            # \\:    #
            # \\:    (loss, labels) = self.train_bellman(
            # \\:        bid,
            # \\:        model,
            # \\:        name_loss,
            # \\:        negative_rate=negative_rate,
            # \\:        margin=margin,
            # \\:    )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

            #
            value += loss.item() * float(len(labels))
            count += int(len(labels))

        #
        return value / float(count)

    def train_heuristics(
        self: SelfTrainer,
        bid: int,
        model: Model,
        name_loss: str,
        /,
        *,
        negative_rate: int,
        margin: float,
    ) -> Tuple[torch.Tensor, NPINTS]:
        R"""
        Evaluate on heuristics-like input.

        Args
        ----
        - bid
            Evaluating batch ID in current epoch.
        - model
            Model used to generate embeddings.
        - name_loss
            Loss function name.
        - negative_rate
            Negative rate.
        - margin
            Distance margin.

        Returns
        -------
        - loss
            Loss.
        - labels
            Binary classification labels of current batch.
        """
        #
        assert self._sample == self.HEURISTICS
        # \\:model = cast(ModelHeuristics, model)
        model.train()

        # Get all data for current batch.
        (
            adjs_target_numpy,
            rels_target_numpy,
            heus_target_numpy,
            lbls_target_numpy,
            uids_observe_numpy,
            vids_observe_numpy,
            adjs_observe_numpy,
            rels_observe_numpy,
        ) = self._minibatch_edge_heuristics.batch(bid)
        adjs_target_torch = torch.from_numpy(adjs_target_numpy).to(self._device)
        rels_target_torch = torch.from_numpy(rels_target_numpy).to(self._device)
        heus_target_torch = torch.from_numpy(heus_target_numpy).to(self._device)
        lbls_target_torch = torch.from_numpy(lbls_target_numpy).to(torch.get_default_dtype()).to(self._device)
        uids_observe_torch = torch.from_numpy(uids_observe_numpy).to(self._device)
        vids_observe_torch = torch.from_numpy(vids_observe_numpy).to(self._device)
        adjs_observe_torch = torch.from_numpy(adjs_observe_numpy).to(self._device)
        rels_observe_torch = torch.from_numpy(rels_observe_numpy).to(self._device)

        #
        vrps = model.forward(vids_observe_torch, adjs_observe_torch, rels_observe_torch)

        #
        assert name_loss in ("binary", "distance")
        if name_loss == "binary":
            #
            loss = model.loss_function_binary(
                vrps,
                adjs_target_torch,
                rels_target_torch,
                heus_target_torch,
                lbls_target_torch,
                sample_negative_rate=negative_rate,
            )
        elif name_loss == "distance":
            #
            loss = model.loss_function_distance(
                vrps,
                adjs_target_torch,
                rels_target_torch,
                heus_target_torch,
                lbls_target_torch,
                sample_negative_rate=negative_rate,
                margin=margin,
            )
        return (loss, lbls_target_numpy)


# \\:    def train_enclose(
# \\:        self: SelfTrainer,
# \\:        bid: int,
# \\:        model: Model,
# \\:        name_loss: str,
# \\:        /,
# \\:        *,
# \\:        negative_rate: int,
# \\:        margin: float,
# \\:    ) -> Tuple[torch.Tensor, NPINTS]:
# \\:        R"""
# \\:        Evaluate on heuristics-like input.
# \\:
# \\:        Args
# \\:        ----
# \\:        - bid
# \\:            Evaluating batch ID in current epoch.
# \\:        - model
# \\:            Model used to generate embeddings.
# \\:        - name_loss
# \\:            Loss function name.
# \\:        - negative_rate
# \\:            Negative rate.
# \\:        - margin
# \\:            Distance margin.
# \\:
# \\:        Returns
# \\:        -------
# \\:        - loss
# \\:            Loss.
# \\:        - labels
# \\:            Binary classification labels of current batch.
# \\:        """
# \\:        #
# \\:        assert self._sample == self.ENCLOSE
# \\:        model = cast(ModelEnclose, model)
# \\:
# \\:        # Get all data for current batch.
# \\:        (
# \\:            adjs_target_numpy,
# \\:            rels_target_numpy,
# \\:            lbls_target_numpy,
# \\:            vpts_observe_numpy,
# \\:            epts_observe_numpy,
# \\:            vids_observe_numpy,
# \\:            vfts_observe_numpy,
# \\:            adjs_observe_numpy,
# \\:            rels_observe_numpy,
# \\:        ) = self._minibatch_edge_enclose.batch(bid)
# \\:        adjs_target_torch = torch.from_numpy(adjs_target_numpy).to(self._device)
# \\:        rels_target_torch = torch.from_numpy(rels_target_numpy).to(self._device)
# \\:        lbls_target_torch = torch.from_numpy(lbls_target_numpy).to(torch.get_default_dtype()).to(self._device)
# \\:        vpts_observe_torch = torch.from_numpy(vpts_observe_numpy).to(self._device)
# \\:        epts_observe_torch = torch.from_numpy(epts_observe_numpy).to(self._device)
# \\:        vids_observe_torch = torch.from_numpy(vids_observe_numpy).to(self._device)
# \\:        vfts_observe_torch = torch.from_numpy(vfts_observe_numpy).to(self._device)
# \\:        adjs_observe_torch = torch.from_numpy(adjs_observe_numpy).to(self._device)
# \\:        rels_observe_torch = torch.from_numpy(rels_observe_numpy).to(self._device)
# \\:
# \\:        #
# \\:        vpts_observe_torch = torch.stack((vpts_observe_torch[:-1], vpts_observe_torch[1:]), dim=1)
# \\:        epts_observe_torch = torch.stack((epts_observe_torch[:-1], epts_observe_torch[1:]), dim=1)
# \\:
# \\:        #
# \\:        vrps = model.forward(
# \\:            vfts_observe_torch,
# \\:            adjs_observe_torch,
# \\:            rels_observe_torch,
# \\:            vpts_observe_torch,
# \\:            epts_observe_torch,
# \\:            rels_target_torch,
# \\:        )
# \\:
# \\:        #
# \\:        assert name_loss in ("binary", "distance")
# \\:        if name_loss == "binary":
# \\:            #
# \\:            loss = model.loss_function_binary(
# \\:                vrps,
# \\:                adjs_target_torch,
# \\:                rels_target_torch,
# \\:                vpts_observe_torch,
# \\:                lbls_target_torch,
# \\:                sample_negative_rate=negative_rate,
# \\:            )
# \\:        elif name_loss == "distance":
# \\:            #
# \\:            loss = model.loss_function_distance(
# \\:                vrps,
# \\:                adjs_target_torch,
# \\:                rels_target_torch,
# \\:                vpts_observe_torch,
# \\:                lbls_target_torch,
# \\:                sample_negative_rate=negative_rate,
# \\:                margin=margin,
# \\:            )
# \\:        return (loss, lbls_target_numpy)

# \\:    def train_bellman(
# \\:        self: SelfTrainer,
# \\:        bid: int,
# \\:        model: Model,
# \\:        name_loss: str,
# \\:        /,
# \\:        *,
# \\:        negative_rate: int,
# \\:        margin: float,
# \\:    ) -> Tuple[torch.Tensor, NPINTS]:
# \\:        R"""
# \\:        Evaluate on NBFNet input.
# \\:
# \\:        Args
# \\:        ----
# \\:        - bid
# \\:            Evaluating batch ID in current epoch.
# \\:        - model
# \\:            Model used to generate embeddings.
# \\:        - name_loss
# \\:            Loss function name.
# \\:        - negative_rate
# \\:            Negative rate.
# \\:        - margin
# \\:            Distance margin.
# \\:
# \\:        Returns
# \\:        -------
# \\:        - loss
# \\:            Loss.
# \\:        - labels
# \\:            Binary classification labels of current batch.
# \\:        """
# \\:        #
# \\:        assert self._sample == self.BELLMAN
# \\:        model = cast(ModelBellman, model)
# \\:
# \\:        # Get all data for current batch.
# \\:        (
# \\:            adjs_target_numpy,
# \\:            rels_target_numpy,
# \\:            heus_target_numpy,
# \\:            lbls_target_numpy,
# \\:            uids_observe_numpy,
# \\:            vids_observe_numpy,
# \\:            adjs_observe_numpy,
# \\:            rels_observe_numpy,
# \\:        ) = self._minibatch_edge_heuristics.batch(bid)
# \\:        adjs_target_torch = torch.from_numpy(adjs_target_numpy).to(self._device)
# \\:        rels_target_torch = torch.from_numpy(rels_target_numpy).to(self._device)
# \\:        heus_target_torch = torch.from_numpy(heus_target_numpy).to(self._device)
# \\:        lbls_target_torch = torch.from_numpy(lbls_target_numpy).to(torch.get_default_dtype()).to(self._device)
# \\:        uids_observe_torch = torch.from_numpy(uids_observe_numpy).to(self._device)
# \\:        vids_observe_torch = torch.from_numpy(vids_observe_numpy).to(self._device)
# \\:        adjs_observe_torch = torch.from_numpy(adjs_observe_numpy).to(self._device)
# \\:        rels_observe_torch = torch.from_numpy(rels_observe_numpy).to(self._device)
# \\:
# \\:        #
# \\:        vrps = model.forward(
# \\:            vids_observe_torch,
# \\:            adjs_observe_torch,
# \\:            rels_observe_torch,
# \\:            adjs_target_torch,
# \\:            rels_target_torch,
# \\:            lbls_target_torch,
# \\:        )
# \\:
# \\:        #
# \\:        assert name_loss in ("binary", "distance")
# \\:        if name_loss == "binary":
# \\:            #
# \\:            loss = model.loss_function_binary(
# \\:                vrps,
# \\:                adjs_target_torch,
# \\:                rels_target_torch,
# \\:                heus_target_torch,
# \\:                lbls_target_torch,
# \\:                sample_negative_rate=negative_rate,
# \\:            )
# \\:        elif name_loss == "distance":
# \\:            #
# \\:            loss = model.loss_function_distance(
# \\:                vrps,
# \\:                adjs_target_torch,
# \\:                rels_target_torch,
# \\:                heus_target_torch,
# \\:                lbls_target_torch,
# \\:                sample_negative_rate=negative_rate,
# \\:                margin=margin,
# \\:            )
# \\:        return (loss, lbls_target_numpy)
