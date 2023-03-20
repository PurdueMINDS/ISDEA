#
import abc
import torch
import torch.nn.functional as F
import math
from typing import TypeVar, Type, Sequence, Tuple, Dict, Sequence, Type


#
SelfModel = TypeVar("SelfModel", bound="Model")


class Model(abc.ABC, torch.nn.Module):
    R"""
    Model.
    """

    @abc.abstractmethod
    def get_embedding_shape_entity(self: SelfModel, /) -> Sequence[int]:
        R"""
        Entity embedding shape.

        Args
        ----

        Returns
        -------
        - shape
            Shape.
        """

    @abc.abstractmethod
    def get_num_relations(self: SelfModel, /) -> int:
        R"""
        Entity number of relations.

        Args
        ----

        Returns
        -------
        - num
            Number of relations.
        """

    @abc.abstractmethod
    def reset_parameters(self: SelfModel, rng: torch.Generator, /) -> SelfModel:
        R"""
        Reset parameters.

        Args
        ----
        - rng
            Random state.

        Returns
        -------
        - self
            Instance itself.
        """

    @classmethod
    def reset_zeros(cls: Type[SelfModel], rng: torch.Generator, tensor: torch.Tensor, /) -> torch.Tensor:
        R"""
        Reset given tensor by zeros.

        Args
        ----
        - rng
            Random state.
        - tensor
            Tensor.

        Returns
        -------
        - tensor
            Input tensor.
        """
        #
        tensor.zero_()
        return tensor

    @classmethod
    def reset_ones(cls: Type[SelfModel], rng: torch.Generator, tensor: torch.Tensor, /) -> torch.Tensor:
        R"""
        Reset given tensor by ones.

        Args
        ----
        - rng
            Random state.
        - tensor
            Tensor.

        Returns
        -------
        - tensor
            Input tensor.
        """
        #
        tensor.fill_(1)
        return tensor

    @classmethod
    def reset_glorot(
        cls: Type[SelfModel],
        rng: torch.Generator,
        tensor: torch.Tensor,
        /,
        *,
        fanin: int,
        fanout: int,
    ) -> torch.Tensor:
        R"""
        Reset given tensor by zeros.

        Args
        ----
        - rng
            Random state.
        - tensor
            Tensor.
        - fanin
            Input dimension for function using the tensor.
        - fanout
            Output dimension for function using the tensor..

        Returns
        -------
        - tensor
            Input tensor.
        """
        # This ensures no information loss between input and output by applying linear transformation defined by given
        # tensor from distribution view.
        stdv = math.sqrt(6.0 / float(fanin + fanout))
        tensor.uniform_(-stdv, stdv, generator=rng)
        return tensor

    @abc.abstractmethod
    def measure_distance(
        self: SelfModel,
        vrps: torch.Tensor,
        adjs: torch.Tensor,
        rels: torch.Tensor,
        heus: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Compute distance measurement for given triplets.

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

    @abc.abstractmethod
    def measure_score(
        self: SelfModel,
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

    @classmethod
    @torch.no_grad()
    def is_loss_function_safe(
        cls: Type[SelfModel],
        vrps: torch.Tensor,
        adjs: torch.Tensor,
        rels: torch.Tensor,
        heus: torch.Tensor,
        lbls: torch.Tensor,
        /,
        *,
        sample_negative_rate: int,
    ) -> None:
        R"""
        Ensure loss function inputs are proper.

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
        - lbls
            Binary labels to be evaluated.
        - sample_negative_rate
            Negative sampling rate.

        Returns
        -------
        """
        # Label is used for binary logit, thus should be floating.
        assert lbls.dtype == torch.get_default_dtype(), "Improper label dtype."

        #
        num = len(lbls) // (1 + sample_negative_rate)

        #
        assert len(lbls) % (1 + sample_negative_rate) == 0, "Improper number of total samples."
        assert torch.all(lbls[:num] == 1).item(), "Headings must be positive samples, which is not."
        assert torch.all(lbls[num:] == 0).item(), "Tailings must be negative samples, which is not."

        #
        adjs_pos = torch.reshape(adjs[:, :num], (2, num, 1))
        adjs_neg = torch.reshape(adjs[:, num:], (2, num, sample_negative_rate))
        rels_pos = torch.reshape(rels[:num], (num, 1))
        rels_neg = torch.reshape(rels[num:], (num, sample_negative_rate))

        #
        assert torch.all(rels_pos == rels_neg).item(), "Negative samples should not change relation type."
        assert torch.all(
            torch.logical_or(adjs_pos[0] == adjs_neg[0], adjs_pos[1] == adjs_neg[1])
        ).item(), "Negative samples should not change both subject and object."

    def loss_function_distance(
        self: SelfModel,
        vrps: torch.Tensor,
        adjs: torch.Tensor,
        rels: torch.Tensor,
        heus: torch.Tensor,
        lbls: torch.Tensor,
        /,
        *,
        sample_negative_rate: int,
        margin: float,
    ) -> torch.Tensor:
        R"""
        Loss function for embedding distance.

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
        - lbls
            Binary labels to be evaluated.
        - sample_negative_rate
            Negative sampling rate.
        - margin
            Margin value (gamma).

        Returns
        -------
        - loss
            Loss.
        """
        #
        dists = self.measure_distance(vrps, adjs, rels, heus)

        # Compute mean over postive-against-negative pairs to match with binary classification loss function.
        # To use SGD, we take negative distance measurement as loss.
        num = len(lbls) // (1 + sample_negative_rate)
        dists_pos = torch.reshape(dists[:num], (num, 1))
        dists_neg = torch.reshape(dists[num:], (num, sample_negative_rate))
        loss = -torch.mean(F.relu(dists_neg - dists_pos + margin))
        return loss

    def loss_function_binary(
        self: SelfModel,
        vrps: torch.Tensor,
        adjs: torch.Tensor,
        rels: torch.Tensor,
        heus: torch.Tensor,
        lbls: torch.Tensor,
        /,
        *,
        sample_negative_rate: int,
    ) -> torch.Tensor:
        R"""
        Loss function for binary classification.

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
        - lbls
            Binary labels to be evaluated.
        - sample_negative_rate
            Negative sampling rate.

        Returns
        -------
        - loss
            Loss.
        """
        #
        scrs = self.measure_score(vrps, adjs, rels, heus)

        #
        weights = torch.tensor([sample_negative_rate], dtype=scrs.dtype, device=scrs.device)
        loss = F.binary_cross_entropy_with_logits(scrs, lbls, pos_weight=weights)
        return loss

    def metric_function_rank(
        self: SelfModel,
        vrps: torch.Tensor,
        adjs: torch.Tensor,
        rels: torch.Tensor,
        heus: torch.Tensor,
        lbls: torch.Tensor,
        /,
        *,
        sample_negative_rate: int,
        ks: Sequence[int],
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        R"""
        Rank metric function.
        For each triplet, we only consider its rank against all entities as triplet object.

        Args
        ----
        - vrps
            Node representations.
        - adjs
            Adjacency list to be evaluated.
            Against-pairs should already be included.
        - rels
            Relations to be evaluated.
            Against-pairs should already be included.
        - heus
            Heuristics of each pair of source node in adjacency list and all entities.
            Against-pairs should already be included.
        - sample_negative_rate
            Negative sampling rate.
        - ks
            All k value to compute hit-at-k metrics.

        Returns
        -------
        - metrics
            All metrics.
        - scores
            Scores of all rank computations.
        """
        # Achieve scores on flatten data.
        scores = self.measure_score(vrps, adjs, rels, heus)

        # Ensure rankable data shape is easy for compare positive with negative.
        num = len(lbls) // (1 + sample_negative_rate)
        scores_pos = torch.reshape(scores[:num], (num, 1))
        scores_neg = torch.reshape(scores[num:], (num, sample_negative_rate))

        #
        # \\:rankables = torch.cat((scores_pos, scores_neg), dim=1)
        # \\:ranks = torch.argsort(rankables, dim=1, descending=True)
        # \\:(_, ranks) = torch.nonzero(ranks == 0).T
        # \\:ranks = (ranks + 1).to(lbls.dtype)
        ranks = torch.sum(scores_pos <= scores_neg, dim=1)
        ranks = (ranks + 1).to(lbls.dtype)

        #
        mr = torch.mean(ranks).item()
        mrr = torch.mean(1.0 / ranks).item()
        hit_at_ks = {k: torch.mean((ranks <= k).to(lbls.dtype)).item() for k in ks}

        #
        return ({"MR": mr, "MRR": mrr, **{"Hit@{:d}".format(k): hit_at_ks[k] for k in ks}}, scores)
