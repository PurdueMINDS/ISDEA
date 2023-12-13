#
import abc
import logging
import os
import numpy as onp
import time
import more_itertools as xitertools
from typing import TypeVar, Tuple, Union
from .hop import ComputationSubsetNode, ComputationSubsetEdge
from .heuristics import HeuristicsForest0, HeuristicsForest1
from .enclose import Enclose
from ..dtypes import NPINTS


#
SelfMinibatch = TypeVar("SelfMinibatch", bound="Minibatch")
SelfMinibatchNode = TypeVar("SelfMinibatchNode", bound="MinibatchNode")
SelfMinibatchEdgeHeuristics = TypeVar(
    "SelfMinibatchEdgeHeuristics", bound="MinibatchEdgeHeuristics"
)
SelfMinibatchEdgeEnclose = TypeVar(
    "SelfMinibatchEdgeEnclose", bound="MinibatchEdgeEnclose"
)


class Minibatch(abc.ABC):
    R"""
    Minibatch scheduler.
    """

    def __init__(self: SelfMinibatch, logger: logging.Logger, cache: str, /) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - logger
            Logging terminal.
        - cache
            Cache directory to store minibatch schedule.

        Returns
        -------
        """
        #
        self._logger = logger
        self._cache = cache

        #
        assert os.path.isdir(cache)

    @abc.abstractmethod
    def load(self: SelfMinibatch, title: str, /) -> None:
        R"""
        Load a minibatch schedule.

        Args
        ----
        - title
            Title of schedule.

        Returns
        -------
        """

    @abc.abstractmethod
    def num_batches(self: SelfMinibatch, /) -> int:
        R"""
        Get the number of minibatches of current epoch.

        Args
        ----

        Returns
        -------
        - num
            Number of batches.
        """

    def epoch(self: SelfMinibatch, eind: int, emax: int, /) -> None:
        R"""
        Set working epoch of the schedule.

        Args
        ----
        - eind
            Epoch ID.
        - emax
            Epoch maximum.

        Returns
        -------
        """
        #
        self._eind = eind
        self._emax = emax

        #
        self._maxlen_epoch = len(str(self._emax - 1))
        self._maxlen_batch = len(str(self.num_batches() - 1))


class MinibatchNode(Minibatch):
    R"""
    Minibatch scheduler for nodes.
    """

    def register(
        self: SelfMinibatchNode, subset: ComputationSubsetNode, /
    ) -> SelfMinibatchNode:
        R"""
        Register subset sampler.

        Args
        ----
        - subset
            Computation subset generator centering at nodes.

        Returns
        -------
        """
        #
        self._subset = subset
        return self

    def generate(
        self: SelfMinibatchNode,
        title: str,
        nodes: NPINTS,
        batch_size: int,
        /,
    ) -> None:
        R"""
        Generate a minibatch schedule.

        Args
        ----
        - title
            Title of schedule.
        - nodes
            Nodes to be schedules.
        - batch_size
            Batch size.

        Returns
        -------
        """
        #
        path = os.path.join(self._cache, "{:s}.npy".format(title))
        self._logger.info('Generate node minibatch schedule to "{:s}".'.format(path))

        #
        buf = []
        pointer = 0
        while True:
            # Stop signal.
            if pointer == len(nodes):
                #
                break
            assert not pointer > len(nodes)

            #
            bgn = pointer
            end = min(bgn + batch_size, len(nodes))
            buf.append(nodes[bgn:end])

            #
            pointer = end

        #
        with open(path, "wb") as file:
            #
            onp.save(file, onp.array([len(array) for array in buf]))
            for array in buf:
                #
                onp.save(file, array)

    def load(self: SelfMinibatchNode, title: str, /) -> None:
        R"""
        Load a minibatch schedule.

        Args
        ----
        - title
            Title of schedule.

        Returns
        -------
        """
        #
        path = os.path.join(self._cache, "{:s}.npy".format(title))
        self._logger.info('Load node minibatch schedule from "{:s}".'.format(path))

        #
        buf = []
        with open(path, "rb") as file:
            #
            sizes = onp.load(file)
            for size in sizes.tolist():
                #
                array = onp.load(file)
                buf.append(array)
                assert array.size > 0
                assert len(array) == size
        self._schedule = tuple(buf)

    def num_batches(self: SelfMinibatchNode, /) -> int:
        R"""
        Get the number of minibatches of current epoch.

        Args
        ----

        Returns
        -------
        - num
            Number of batches.
        """
        # Node schedule is fixed and reused by every epoch.
        return len(self._schedule)

    def batch(
        self: SelfMinibatchNode, bid: int, /
    ) -> Tuple[NPINTS, NPINTS, NPINTS, NPINTS, NPINTS]:
        R"""
        Get minibatch of given ID of current epoch.

        Args
        ----
        - bid
            Batch ID.

        Returns
        -------
        - ucenters
            Center node IDs in minibatch.
        - uids
            Node IDs of full graph in minibatch.
        - vids
            Node IDs of minibatch in full graph.
        - adjs
            Adjacency list of minibatch.
        - rels
            Relations of minibatch.
        """
        # Pay attention that batch ID should start from 1.
        title = "[{:>0{:d}d}/{:>0{:d}d}] [{:>0{:d}d}/{:>0{:d}d}]:".format(
            self._eind,
            self._maxlen_epoch,
            self._emax,
            self._maxlen_epoch,
            bid + 1,
            self._maxlen_batch,
            self.num_batches(),
            self._maxlen_batch,
        )
        self._logger.info("{:s} Sample node minibatch.".format(title))

        #
        elapsed = time.time()

        #
        nodes = self._schedule[bid]
        (uids, vids, adjs, rels) = self._subset.sample(
            "graph",
            nodes,
            self._subset.masks_edge_accept(
                onp.zeros((2, 0), dtype=self._subset._adjs.dtype)
            ),
        )

        #
        elapsed = time.time() - elapsed

        #
        self._logger.debug(
            "{:s} Get node minbatch of {:d} centers.".format(
                " " * len(title), len(nodes)
            )
        )
        self._logger.debug(
            "{:s} Get node minbatch of {:d} nodes, {:d} edges.".format(
                " " * len(title), len(vids), len(rels)
            ),
        )
        self._logger.debug(
            "{:s} Get node minbatch in {:.3f} seconds.".format(
                " " * len(title), elapsed
            )
        )
        return (uids[nodes], uids, vids, adjs, rels)


class MinibatchEdgeHeuristics(Minibatch):
    R"""
    Minibatch scheduler for training edges with heuristics.
    """

    def register(
        self: SelfMinibatchEdgeHeuristics,
        heuristics: Union[HeuristicsForest0, HeuristicsForest1],
        subset: ComputationSubsetEdge,
        /,
        *,
        num_relations: int,
    ) -> SelfMinibatchEdgeHeuristics:
        R"""
        Register heuristics collector and subset sampler.

        Args
        ----
        - heuristics
            Edge heuristics collector.
        - subset
            Computation tree sampler.

        Returns
        -------
        """
        #
        self._heuristics = heuristics
        self._subset = subset
        self._num_nodes = self._heuristics._num_nodes
        self._num_relations = num_relations

        # Heuristics collector and subset sampler should work on the same graph.
        assert onp.all(self._heuristics._adjs == self._subset._adjs).all()
        assert onp.all(self._heuristics._rels == self._subset._rels).all()
        return self

    def generate(
        self: SelfMinibatchEdgeHeuristics,
        title: str,
        adjs: NPINTS,
        rels: NPINTS,
        batch_size: int,
        /,
        *,
        negative_rate: int,
        num_neg_rels: int,
        rng: onp.random.RandomState,
        num_epochs: int,
        reusable: bool,
    ) -> None:
        R"""
        Generate a minibatch schedule.

        Args
        ----
        - title
            Title of schedule.
        - adjs
            Adjacency list to be scheduled.
        - rels
            Relations to be scehduled
        - batch_size
            Batch size.
        - negative_rate
            Negative samples per positive training edges.
        - num_neg_rels
            Negative relation samples per positive training edges.
        - rng
            Random state.
        - num_epochs
            Number of epochs.
        - reusable
            A fixed schedule is reused for every epoch.

        Returns
        -------
        """
        #
        path = os.path.join(self._cache, "{:s}.npy".format(title))
        self._logger.info('Generate edge minibatch schedule to "{:s}".'.format(path))

        # Batch size should take negative sampling into consideration.
        assert batch_size % (1 + negative_rate + num_neg_rels) == 0
        batch_size_pos = batch_size // (1 + negative_rate + num_neg_rels)
        batch_size_neg = batch_size_pos * (negative_rate + num_neg_rels)

        #
        assert not reusable or num_epochs == 1
        self.reusable = reusable

        #
        full = []
        for _ in range(num_epochs):
            # Generate negative samples ahead of each epoch
            adjs_ps = adjs
            rels_ps = rels
            if negative_rate > 0:
                #
                (adjs_ng, rels_ng) = self.negative(
                    adjs_ps, rels_ps, negative_rate=negative_rate, rng=rng
                )
            if num_neg_rels > 0:
                #
                (adjs_nr, rels_nr) = self.negative_relations(
                    adjs_ps, rels_ps, num=num_neg_rels, rng=rng
                )

            #
            buf = []
            pointer = 0
            indices = rng.permutation(len(rels))
            while True:
                # Stop signal.
                if pointer == len(rels):
                    #
                    break
                assert not pointer > len(rels)

                # Raw data sampling happens only for positive samples.
                bgn = pointer
                end = min(bgn + batch_size_pos, len(rels))

                #
                indices_pos = indices[bgn:end]
                adjs_pos = adjs_ps[:, indices_pos]
                rels_pos = rels_ps[indices_pos]
                lbls_pos = onp.ones_like(rels_pos)
                buf_adjs = [adjs_pos]
                buf_rels = [rels_pos]
                buf_lbls = [lbls_pos]

                # Generate negative samples correspond to positive samples.
                if negative_rate > 0:
                    #
                    indices_neg_base = (
                        onp.repeat(indices_pos, negative_rate) * negative_rate
                    )
                    indices_neg_bias = onp.tile(
                        onp.arange(negative_rate), (len(indices_pos),)
                    )
                    indices_neg = indices_neg_base + indices_neg_bias
                    adjs_neg = adjs_ng[:, indices_neg]
                    rels_neg = rels_ng[indices_neg]
                    lbls_neg = onp.zeros_like(rels_neg)
                    buf_adjs.append(adjs_neg)
                    buf_rels.append(rels_neg)
                    buf_lbls.append(lbls_neg)

                # Generate negative relation samples correspond to positive samples.
                if num_neg_rels > 0:
                    #
                    indices_ngr_base = (
                        onp.repeat(indices_pos, num_neg_rels) * num_neg_rels
                    )
                    indices_ngr_bias = onp.tile(
                        onp.arange(num_neg_rels), (len(indices_pos),)
                    )
                    indices_ngr = indices_ngr_base + indices_ngr_bias
                    adjs_ngr = adjs_nr[:, indices_ngr]
                    rels_ngr = rels_nr[indices_ngr]
                    lbls_ngr = onp.zeros_like(rels_ngr)
                    buf_adjs.append(adjs_ngr)
                    buf_rels.append(rels_ngr)
                    buf_lbls.append(lbls_ngr)

                #
                buf.append(
                    onp.concatenate(
                        (
                            onp.concatenate(buf_adjs, axis=1),
                            onp.expand_dims(onp.concatenate(buf_rels), 0),
                            onp.expand_dims(onp.concatenate(buf_lbls), 0),
                        ),
                    ),
                )

                #
                pointer = end

            #
            full.append(buf)

        #
        with open(path, "wb") as file:
            #
            onp.save(file, onp.array([len(buf) for buf in full]))
            for buf in full:
                #
                onp.save(file, onp.array([len(array.T) for array in buf]))
                for array in buf:
                    #
                    onp.save(file, array)

        # Heuristics must be generated for all related positive and negative edges.
        self._logger.info(
            "-- Generate positive and negative heuristics for full minibatch schedule."
        )

        # Collect unique adjacency list and their heuristics.
        eids = onp.unique(
            onp.concatenate(
                list(
                    xitertools.flatten(
                        [
                            [array[0] * self._num_nodes + array[1] for array in buf]
                            for buf in full
                        ]
                    )
                ),
            ),
        )
        adjs = onp.stack((eids // self._num_nodes, eids % self._num_nodes))
        self._heuristics.forest(adjs)
        self._heuristics.collect(adjs)

    def negative(
        self: SelfMinibatchEdgeHeuristics,
        adjs_pos: NPINTS,
        rels_pos: NPINTS,
        /,
        *,
        negative_rate: int,
        rng: onp.random.RandomState,
    ) -> Tuple[NPINTS, NPINTS]:
        R"""
        Generate negative sample.

        Args
        ----
        - adjs_pos
            Positive adjacency list.
        - rels_pos
            Positive relations,
        - negative_rate
            Negative sampling rate.
        - rng
            Random state.

        Returns
        -------
        - adjs_neg
            Negative adjacency list.
        - rels_neg
            Negative relations,
        """
        # Given observe edges in initialization should be treated as exclusive positive edges.
        # Pay attention that given positive edges in arguments may not be exclusive, e.g., training.
        adjs_def = self._subset._adjs
        rels_def = self._subset._rels

        #
        num_pos = len(rels_pos)
        adjs_pos = onp.repeat(adjs_pos, negative_rate, axis=1)
        rels_pos = onp.repeat(rels_pos, negative_rate)
        adjs_neg = adjs_pos.copy()
        rels_neg = rels_pos.copy()

        # Collect exclusive relational edge IDs.
        # Since all nodes will be considered, node ID max is total number of nodes.
        # Since only given relations will be considered, relation ID max is the maximum relation ID plus 1.
        vmax = self._num_nodes
        rmax = max(onp.max(rels_def).item(), onp.max(rels_def).item()) + 1
        assert vmax**2 * rmax < 1e12

        #
        eids_def = (adjs_def[0] * vmax + adjs_def[1]) * rmax + rels_def

        # For a negative sample of each positive sample, randomly corrupt its subject or object.
        if negative_rate > 0:
            # Select negative sampling:
            # 1. regular one.
            # 2. NBFNet one.
            # \\:# Uniformly decide if subject or object should be corrupted for each negative sample.
            # \\:probs = rng.uniform(0.0, 1.0, (len(rels_neg),))
            # \\:corrupt_sub = probs < 0.5
            # \\:corrupt_obj = onp.logical_not(corrupt_sub)
            # Only corrupt objects when dataset are augmented with inversions to be robust with NBFNet.
            corrupt_sub = onp.zeros((len(rels_neg),), dtype=onp.bool_)
            corrupt_obj = onp.ones((len(rels_neg),), dtype=onp.bool_)

            # Generate corrupted node IDs for the first time.
            corrupts = rng.choice(self._num_nodes, (len(rels_neg),), replace=True)
            adjs_neg[0, corrupt_sub] = corrupts[corrupt_sub]
            adjs_neg[1, corrupt_obj] = corrupts[corrupt_obj]

            # Repeat negative sampling 10 times to ensure no observation is used as negative.
            for _ in range(100):
                # Get corrupted node IDs conflicting with observed data or itself, and sample again.
                masks = onp.logical_or(
                    onp.isin(
                        (adjs_neg[0] * vmax + adjs_neg[1]) * rmax + rels_neg, eids_def
                    ),
                    onp.logical_and(
                        adjs_neg[0] == adjs_pos[0], adjs_neg[1] == adjs_pos[1]
                    ),
                )
                if onp.any(masks).item():
                    #
                    masks_sub = onp.logical_and(corrupt_sub, masks)
                    masks_obj = onp.logical_and(corrupt_obj, masks)
                else:
                    #
                    break

                #
                corrupts = rng.choice(self._num_nodes, (len(rels_neg),), replace=True)
                adjs_neg[0, masks_sub] = corrupts[masks_sub]
                adjs_neg[1, masks_obj] = corrupts[masks_obj]

            # Check if we still have invalid negative samples.
            masks_reshaped = onp.reshape(
                onp.logical_or(
                    onp.isin(
                        (adjs_neg[0] * vmax + adjs_neg[1]) * rmax + rels_neg, eids_def
                    ),
                    onp.logical_and(
                        adjs_neg[0] == adjs_pos[0], adjs_neg[1] == adjs_pos[1]
                    ),
                ),
                (num_pos, negative_rate),
            )
            adjs_neg_reshaped = onp.reshape(adjs_neg, (2, num_pos, negative_rate))

            # Reuse negative samples to fill.
            for i in range(num_pos):
                #
                if not onp.any(masks_reshaped[i]).item():
                    #
                    continue

                # If negative is still not enough, repeat used negative samples.
                (array_miss,) = onp.nonzero(masks_reshaped[i])
                (array_fill,) = onp.nonzero(onp.logical_not(masks_reshaped[i]))
                ids_miss = array_miss.tolist()
                ids_fill = array_fill.tolist()
                adjs_neg_reshaped[:, i, ids_miss] = adjs_neg_reshaped[
                    :,
                    i,
                    ids_fill * (len(ids_miss) // len(ids_fill))
                    + ids_fill[: len(ids_miss) % len(ids_fill)],
                ]
            adjs_neg = onp.reshape(adjs_neg_reshaped, (2, num_pos * negative_rate))

            # If it still has cases where observation is used as negative, report as an error.
            masks = onp.logical_or(
                onp.isin(
                    (adjs_neg[0] * vmax + adjs_neg[1]) * rmax + rels_neg, eids_def
                ),
                onp.logical_and(adjs_neg[0] == adjs_pos[0], adjs_neg[1] == adjs_pos[1]),
            )
            assert not onp.any(
                masks
            ).item(), "Observed edges are sampled as negative which is invalid."

        #
        return (adjs_neg, rels_neg)

    def negative_relations(
        self: SelfMinibatchEdgeHeuristics,
        adjs_pos: NPINTS,
        rels_pos: NPINTS,
        /,
        *,
        num: int,
        rng: onp.random.RandomState,
    ) -> Tuple[NPINTS, NPINTS]:
        R"""
        Generate negative sample.

        Args
        ----
        - adjs_pos
            Positive adjacency list.
        - rels_pos
            Positive relations,
        - num
            Number of negative relation samples.
        - rng
            Random state.

        Returns
        -------
        - adjs_neg
            Negative adjacency list.
        - rels_neg
            Negative relations,
        """
        # Given observe edges in initialization should be treated as exclusive positive edges.
        # Pay attention that given positive edges in arguments may not be exclusive, e.g., training.
        adjs_def = self._subset._adjs
        rels_def = self._subset._rels

        #
        num_pos = len(rels_pos)
        adjs_pos = onp.repeat(adjs_pos, num, axis=1)
        rels_pos = onp.repeat(rels_pos, num)
        adjs_neg = adjs_pos.copy()
        rels_neg = rels_pos.copy()

        # Collect exclusive relational edge IDs.
        # Since all nodes will be considered, node ID max is total number of nodes.
        # Since only given relations will be considered, relation ID max is the maximum relation ID plus 1.
        vmax = self._num_nodes
        rmax = max(onp.max(rels_def).item(), onp.max(rels_def).item()) + 1
        assert vmax**2 * rmax < 1e12

        #
        eids_def = (adjs_def[0] * vmax + adjs_def[1]) * rmax + rels_def

        # For a negative sample of each positive sample, randomly corrupt its relation.
        if num > 0:
            # Generate corrupted node IDs for the first time.
            corrupts = rng.choice(self._num_relations, (len(rels_neg),), replace=True)
            rels_neg = corrupts

            # Repeat negative sampling 10 times to ensure no observation is used as negative.
            for _ in range(10):
                # Get corrupted node IDs conflicting with observed data or itself, and sample again.
                masks = onp.logical_or(
                    onp.isin(
                        (adjs_neg[0] * vmax + adjs_neg[1]) * rmax + rels_neg, eids_def
                    ),
                    rels_neg == rels_pos,
                )
                if not onp.any(masks).item():
                    #
                    break

                #
                corrupts = rng.choice(
                    self._num_relations, (len(rels_neg),), replace=True
                )
                rels_neg[masks] = corrupts[masks]

            # Check if we still have invalid negative samples.
            masks_reshaped = onp.reshape(
                onp.logical_or(
                    onp.isin(
                        (adjs_neg[0] * vmax + adjs_neg[1]) * rmax + rels_neg, eids_def
                    ),
                    rels_neg == rels_pos,
                ),
                (num_pos, num),
            )
            rels_neg_reshaped = onp.reshape(rels_neg, (num_pos, num))

            # Reuse negative samples to fill.
            for i in range(num_pos):
                #
                if not onp.any(masks_reshaped[i]).item():
                    #
                    continue

                # If negative is still not enough, repeat used negative samples.
                (array_miss,) = onp.nonzero(masks_reshaped[i])
                (array_fill,) = onp.nonzero(onp.logical_not(masks_reshaped[i]))
                ids_miss = array_miss.tolist()
                ids_fill = array_fill.tolist()
                rels_neg_reshaped[i, ids_miss] = rels_neg_reshaped[
                    i,
                    ids_fill * (len(ids_miss) // len(ids_fill))
                    + ids_fill[: len(ids_miss) % len(ids_fill)],
                ]
            rels_neg = onp.reshape(rels_neg_reshaped, (num_pos * num,))

            # If it still has cases where observation is used as negative, report as an error.
            masks = onp.logical_or(
                onp.isin(
                    (adjs_neg[0] * vmax + adjs_neg[1]) * rmax + rels_neg, eids_def
                ),
                rels_neg == rels_pos,
            )
            assert not onp.any(
                masks
            ).item(), "Observed edges are sampled as negative which is invalid."

        #
        return (adjs_neg, rels_neg)

    def load(self: SelfMinibatchEdgeHeuristics, title: str, /) -> None:
        R"""
        Load a minibatch schedule.

        Args
        ----
        - title
            Title of schedule.

        Returns
        -------
        """
        #
        path = os.path.join(self._cache, "{:s}.npy".format(title))
        self._logger.info('Load edge minibatch schedule from "{:s}".'.format(path))

        #
        full = []
        with open(path, "rb") as file:
            #
            epochs = onp.load(file)
            for num in epochs.tolist():
                #
                buf = []
                sizes = onp.load(file)
                for size in sizes.tolist():
                    #
                    array = onp.load(file)
                    buf.append(array)
                    assert array.size > 0
                    assert array.shape == (4, size)
                full.append(tuple(buf))
                assert len(buf) == num
        self._schedule = tuple(full)

    def num_batches(self: SelfMinibatchEdgeHeuristics, /) -> int:
        R"""
        Get the number of minibatches of current epoch.

        Args
        ----

        Returns
        -------
        - num
            Number of batches.
        """
        # Node schedule is fixed and reused by every epoch.
        assert self.reusable or self._eind > 0
        return len(self._schedule[0 if self.reusable else self._eind - 1])

    def batch(
        self: SelfMinibatchEdgeHeuristics,
        bid: int,
        /,
    ) -> Tuple[NPINTS, NPINTS, NPINTS, NPINTS, NPINTS, NPINTS, NPINTS, NPINTS]:
        R"""
        Get minibatch of given ID of current epoch.

        Args
        ----
        - bid
            Batch ID.

        Returns
        -------
        - adjs_target
            Targeting adjacency list of minibatch.
        - rels_target
            Targeting relations of minibatch.
        - heus_target
            Targeting heuristics of minibatch.
        - lbls_target
            Targeting labels of minibatch.
        - uids_observed
            Observed node IDs of full graph in minibatch.
        - vids_observed
            Observed node IDs of minibatch in full graph.
        - adjs_observed
            Observed adjacency list of minibatch.
        - rels_observed
            Observed relations of minibatch.
        """
        # Pay attention that batch ID should start from 1.
        title = "[{:>0{:d}d}/{:>0{:d}d}] [{:>0{:d}d}/{:>0{:d}d}]:".format(
            self._eind,
            self._maxlen_epoch,
            self._emax,
            self._maxlen_epoch,
            bid + 1,
            self._maxlen_batch,
            self.num_batches(),
            self._maxlen_batch,
        )
        self._logger.info("{:s} Sample edge minibatch.".format(title))

        #
        elapsed = time.time()

        #
        assert self.reusable or self._eind > 0
        samples = self._schedule[0 if self.reusable else self._eind - 1][bid]
        adjs_target = samples[:2]
        rels_target = samples[2]
        lbls_target = samples[3]
        heus_target = self._heuristics.load(adjs_target)

        # Sampling observed graph with training edges being removed for inductive learning.
        (uids_observe, vids_observe, adjs_observe, rels_observe) = self._subset.sample(
            "graph",
            adjs_target.T,
            self._subset.masks_edge_accept(adjs_target),
        )
        adjs_target = uids_observe[adjs_target]

        #
        elapsed = time.time() - elapsed

        #
        self._logger.debug(
            "{:s} Get edge minbatch of {:d} positives and {:d} negatives.".format(
                " " * len(title),
                onp.sum(lbls_target == 1).item(),
                onp.sum(lbls_target == 0).item(),
            ),
        )
        self._logger.debug(
            "{:s} Get edge minbatch of {:d} nodes, {:d} edges.".format(
                " " * len(title),
                len(vids_observe),
                len(rels_observe),
            ),
        )
        self._logger.debug(
            "{:s} Get edge minbatch in {:.3f} seconds.".format(
                " " * len(title), elapsed
            )
        )
        return (
            adjs_target,
            rels_target,
            heus_target,
            lbls_target,
            uids_observe,
            vids_observe,
            adjs_observe,
            rels_observe,
        )


class MinibatchEdgeEnclose(Minibatch):
    R"""
    Minibatch scheduler for training edges with enclosed subgraphs.
    """

    def __annotate__(self: SelfMinibatchEdgeEnclose, /) -> None:
        R"""
        Annotations.

        Args
        ----

        Returns
        -------
        """
        #
        self.reusable: bool

    def register(
        self: SelfMinibatchEdgeEnclose, enclose: Enclose, /
    ) -> SelfMinibatchEdgeEnclose:
        R"""
        Register heuristics collector and subset sampler.

        Args
        ----
        - enclose
            Edge enclosed subgraph translator.

        Returns
        -------
        """
        #
        self._enclose = enclose
        self._num_nodes = self._enclose._num_nodes

        # Directly translate on registration.
        self._logger.info(
            'Translate all forests of heuristics collection (version 1) from "{:s}".'.format(
                self._enclose._cache
            ),
        )
        self._enclose.translate()
        return self

    def load(self: SelfMinibatchEdgeEnclose, title: str, /) -> None:
        R"""
        Load a minibatch schedule.

        Args
        ----
        - title
            Title of schedule.

        Returns
        -------
        """
        #
        path = os.path.join(self._cache, "{:s}.npy".format(title))
        self._logger.info('Load edge minibatch schedule from "{:s}".'.format(path))

        #
        full = []
        with open(path, "rb") as file:
            #
            epochs = onp.load(file)
            for num in epochs.tolist():
                #
                buf = []
                sizes = onp.load(file)
                for size in sizes.tolist():
                    #
                    array = onp.load(file)
                    buf.append(array)
                    assert array.size > 0
                    assert array.shape == (4, size)
                full.append(tuple(buf))
                assert len(buf) == num
        self._schedule = tuple(full)

    def num_batches(self: SelfMinibatchEdgeEnclose, /) -> int:
        R"""
        Get the number of minibatches of current epoch.

        Args
        ----

        Returns
        -------
        - num
            Number of batches.
        """
        # Node schedule is fixed and reused by every epoch.
        assert self.reusable or self._eind > 0
        return len(self._schedule[0 if self.reusable else self._eind - 1])

    def batch(
        self: SelfMinibatchEdgeEnclose,
        bid: int,
        /,
    ) -> Tuple[NPINTS, NPINTS, NPINTS, NPINTS, NPINTS, NPINTS, NPINTS, NPINTS, NPINTS]:
        R"""
        Get minibatch of given ID of current epoch.

        Args
        ----
        - bid
            Batch ID.

        Returns
        -------
        - adjs_target
            Targeting adjacency list of minibatch.
        - rels_target
            Targeting relations of minibatch.
        - lbls_target
            Targeting labels of minibatch.
        - vpts_observed
            Observed node data boundaries of each subgraph.
        - epts_observed
            Observed edge data boundaries of each subgraph.
        - vids_observed
            Observed node IDs of full graph in minibatch.
        - vfts_observed
            Observed node features of minibatch in full graph.
        - adjs_observed
            Observed adjacency list of minibatch.
        - rels_observed
            Observed relations of minibatch.
        """
        # Pay attention that batch ID should start from 1.
        title = "[{:>0{:d}d}/{:>0{:d}d}] [{:>0{:d}d}/{:>0{:d}d}]:".format(
            self._eind,
            self._maxlen_epoch,
            self._emax,
            self._maxlen_epoch,
            bid + 1,
            self._maxlen_batch,
            self.num_batches(),
            self._maxlen_batch,
        )
        self._logger.info("{:s} Sample edge minibatch.".format(title))

        #
        elapsed = time.time()

        #
        assert self.reusable or self._eind > 0
        samples = self._schedule[0 if self.reusable else self._eind - 1][bid]
        adjs_target = samples[:2]
        rels_target = samples[2]
        lbls_target = samples[3]

        # Sampling enclosed subgraphs with training edges being removed for inductive learning.
        # Bias node IDs to merge multiple subgraphs into a single graph.
        # Update targeting adjacency list to match with merged version.
        (
            vpts_observe,
            epts_observe,
            vids_observe,
            vfts_observe,
            adjs_observe,
            rels_observe,
        ) = self._enclose.load(
            adjs_target,
        )
        bias = 0
        buf = []
        for i in range(len(lbls_target)):
            # Get focusing data of each subgraph.
            vids_subgraph = vids_observe[vpts_observe[i] : vpts_observe[i + 1]]
            adjs_subgraph = adjs_observe[:, epts_observe[i] : epts_observe[i + 1]]

            #
            (uids,) = onp.nonzero(vids_subgraph == adjs_target[0, i])
            src = uids.item() + bias
            (uids,) = onp.nonzero(vids_subgraph == adjs_target[1, i])
            dst = uids.item() + bias
            buf.append((src, dst))
            adjs_subgraph += bias
            bias += len(vids_observe[vpts_observe[i] : vpts_observe[i + 1]])
        adjs_target = onp.array(buf).T

        #
        elapsed = time.time() - elapsed

        #
        self._logger.debug(
            "{:s} Get edge minbatch of {:d} positives and {:d} negatives.".format(
                " " * len(title),
                onp.sum(lbls_target == 1).item(),
                onp.sum(lbls_target == 0).item(),
            ),
        )
        self._logger.debug(
            "{:s} Get edge minbatch of {:d} nodes, {:d} edges.".format(
                " " * len(title),
                len(vids_observe),
                len(rels_observe),
            ),
        )
        self._logger.debug(
            "{:s} Get edge minbatch in {:.3f} seconds.".format(
                " " * len(title), elapsed
            )
        )
        return (
            adjs_target,
            rels_target,
            lbls_target,
            vpts_observe,
            epts_observe,
            vids_observe,
            vfts_observe,
            adjs_observe,
            rels_observe,
        )
