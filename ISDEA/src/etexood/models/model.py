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
    def reset_zeros(
        cls: Type[SelfModel], rng: torch.Generator, tensor: torch.Tensor, /
    ) -> torch.Tensor:
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
    def reset_ones(
        cls: Type[SelfModel], rng: torch.Generator, tensor: torch.Tensor, /
    ) -> torch.Tensor:
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
        sample_num_neg_rels: int,
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
        - sample_num_neg_rels
            Number of negative relation samples.

        Returns
        -------
        """
        # Label is used for binary logit, thus should be floating.
        assert lbls.dtype == torch.get_default_dtype(), "Improper label dtype."

        #
        num = len(lbls) // (1 + sample_negative_rate + sample_num_neg_rels)

        #
        assert (
            len(lbls) % (1 + sample_negative_rate + sample_num_neg_rels) == 0
        ), "Improper number of total samples."
        assert torch.all(
            lbls[:num] == 1
        ).item(), "Headings must be positive samples, which is not."
        assert torch.all(
            lbls[num:] == 0
        ).item(), "Tailings must be negative samples, which is not."

        #
        adjs_pos = torch.reshape(adjs[:, :num], (2, num, 1))
        rels_pos = torch.reshape(rels[:num], (num, 1))
        ptr = num
        if sample_negative_rate > 0:
            #
            adjs_neg = torch.reshape(
                adjs[:, ptr : ptr + num * sample_negative_rate],
                (2, num, sample_negative_rate),
            )
            rels_neg = torch.reshape(
                rels[ptr : ptr + num * sample_negative_rate],
                (num, sample_negative_rate),
            )
            ptr += num * sample_negative_rate
        if sample_num_neg_rels > 0:
            #
            adjs_ngr = torch.reshape(
                adjs[:, ptr : ptr + num * sample_num_neg_rels],
                (2, num, sample_num_neg_rels),
            )
            rels_ngr = torch.reshape(
                rels[ptr : ptr + num * sample_num_neg_rels], (num, sample_num_neg_rels)
            )
            ptr += num * sample_num_neg_rels

        #
        assert ptr == len(
            rels
        ), "More than positive and negative, which should not have."
        if sample_negative_rate > 0:
            #
            assert torch.all(
                torch.logical_or(adjs_pos[0] == adjs_neg[0], adjs_pos[1] == adjs_neg[1])
            ).item(), "Negative samples should not change both subject and object."
            assert torch.all(
                rels_pos == rels_neg
            ).item(), "Negative samples should not change relation type."
        if sample_num_neg_rels > 0:
            #
            assert torch.all(
                torch.logical_and(
                    adjs_pos[0] == adjs_ngr[0], adjs_pos[1] == adjs_ngr[1]
                )
            ).item(), "Negative relation samples should not change subject or object."

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
        sample_num_neg_rels: int,
        margin: float,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        - sample_num_neg_rels
            Number of negative relation samples.
        - margin
            Margin value (gamma).

        Returns
        -------
        - loss
            Loss.
        - auxiliaries
            Auxiliary losses.
        """
        #
        dists = self.measure_distance(vrps, adjs, rels, heus)

        #
        num = len(lbls) // (1 + sample_negative_rate + sample_num_neg_rels)
        dists_pos = torch.reshape(dists[:num], (num, 1))
        buf_dists_neg = []
        ptr = num
        if sample_negative_rate > 0:
            #
            dists_nge = torch.reshape(
                dists[ptr : ptr + num * sample_negative_rate],
                (num, sample_negative_rate),
            )
            ptr += num * sample_negative_rate
            buf_dists_neg.append(dists_nge)
        if sample_num_neg_rels > 0:
            #
            dists_ngr = torch.reshape(
                dists[ptr : ptr + num * sample_num_neg_rels], (num, sample_num_neg_rels)
            )
            ptr += num * sample_num_neg_rels
            buf_dists_neg.append(dists_ngr)

        # \\:# Compute mean over postive-against-negative pairs to match with binary classification loss function.
        # \\:# To use SGD, we take negative distance measurement as loss.
        # \\:if sample_negative_rate > 0:
        # \\:    #
        # \\:    loss_ent = -torch.mean(F.relu(dists_neg - dists_pos + margin))
        # \\:else:
        # \\:    #
        # \\:    loss_ent = torch.tensor(0.0, dtype=dists.dtype, device=dists.device)

        # \\:#
        # \\:if sample_num_neg_rels > 0:
        # \\:    #
        # \\:    loss_rel = -torch.mean(F.relu(dists_ngr - dists_pos + margin))
        # \\:else:
        # \\:    #
        # \\:    loss_rel = torch.tensor(0.0, dtype=dists.dtype, device=dists.device)

        # \\:#
        # \\:loss = loss_ent + loss_rel

        # Entity and relation negative samples are treated as same kind of negative samples.
        dists_neg = torch.concatenate(buf_dists_neg)
        loss = -torch.mean(F.relu(dists_neg - dists_pos + margin))
        return loss, (0.0, 0.0)

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
        sample_num_neg_rels: int,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        - sample_num_neg_rels
            Number of negative relation samples.

        Returns
        -------
        - loss
            Loss.
        - auxiliaries
            Auxiliary losses.
        """
        #
        scrs = self.measure_score(vrps, adjs, rels, heus)

        #
        num = len(lbls) // (1 + sample_negative_rate + sample_num_neg_rels)
        scrs_pos = torch.reshape(scrs[:num], (num, 1))
        lbls_pos = torch.reshape(lbls[:num], (num, 1))
        buf_scrs_neg = []
        buf_lbls_neg = []
        ptr = num
        if sample_negative_rate > 0:
            #
            scrs_nge = torch.reshape(
                scrs[ptr : ptr + num * sample_negative_rate],
                (num, sample_negative_rate),
            )
            lbls_nge = torch.reshape(
                lbls[ptr : ptr + num * sample_negative_rate],
                (num, sample_negative_rate),
            )
            ptr += num * sample_negative_rate
            buf_scrs_neg.append(scrs_nge)
            buf_lbls_neg.append(lbls_nge)
        if sample_num_neg_rels > 0:
            #
            scrs_ngr = torch.reshape(
                scrs[ptr : ptr + num * sample_num_neg_rels], (num, sample_num_neg_rels)
            )
            lbls_ngr = torch.reshape(
                lbls[ptr : ptr + num * sample_num_neg_rels], (num, sample_num_neg_rels)
            )
            ptr += num * sample_num_neg_rels
            buf_scrs_neg.append(scrs_ngr)
            buf_lbls_neg.append(lbls_ngr)

        # \\:#
        # \\:if sample_negative_rate > 0:
        # \\:    #
        # \\:    weights_ent = torch.tensor(
        # \\:        [sample_negative_rate], dtype=scrs.dtype, device=scrs.device
        # \\:    )
        # \\:    loss_ent = F.binary_cross_entropy_with_logits(
        # \\:        torch.concatenate(
        # \\:            (
        # \\:                torch.reshape(scrs_pos, (num,)),
        # \\:                torch.reshape(scrs_neg, (num * sample_negative_rate,)),
        # \\:            ),
        # \\:        ),
        # \\:        torch.concatenate(
        # \\:            (
        # \\:                torch.reshape(lbls_pos, (num,)),
        # \\:                torch.reshape(lbls_neg, (num * sample_negative_rate,)),
        # \\:            ),
        # \\:        ),
        # \\:        pos_weight=weights_ent,
        # \\:    )
        # \\:else:
        # \\:    #
        # \\:    loss_ent = torch.tensor(0.0, dtype=scrs.dtype, device=scrs.device)

        # \\:#
        # \\:if sample_num_neg_rels > 0:
        # \\:    #
        # \\:    weights_rel = torch.tensor(
        # \\:        [sample_num_neg_rels], dtype=scrs.dtype, device=scrs.device
        # \\:    )
        # \\:    loss_rel = F.binary_cross_entropy_with_logits(
        # \\:        torch.concatenate(
        # \\:            (
        # \\:                torch.reshape(scrs_pos, (num,)),
        # \\:                torch.reshape(scrs_ngr, (num * sample_num_neg_rels,)),
        # \\:            ),
        # \\:        ),
        # \\:        torch.concatenate(
        # \\:            (
        # \\:                torch.reshape(lbls_pos, (num,)),
        # \\:                torch.reshape(lbls_ngr, (num * sample_num_neg_rels,)),
        # \\:            ),
        # \\:        ),
        # \\:        pos_weight=weights_rel,
        # \\:    )
        # \\:else:
        # \\:    #
        # \\:    loss_rel = torch.tensor(0.0, dtype=scrs.dtype, device=scrs.device)

        # \\:#
        # \\:loss = loss_ent + loss_rel

        # Entity and relation negative samples are treated as same kind of negative samples.
        scrs_neg = torch.concatenate(buf_scrs_neg, dim=1)
        lbls_neg = torch.concatenate(buf_lbls_neg, dim=1)
        weights = torch.tensor(
            [sample_negative_rate + sample_num_neg_rels],
            dtype=scrs.dtype,
            device=scrs.device,
        )
        loss = F.binary_cross_entropy_with_logits(
            torch.concatenate(
                (
                    torch.reshape(scrs_pos, (num,)),
                    torch.reshape(
                        scrs_neg, (num * (sample_negative_rate + sample_num_neg_rels),)
                    ),
                ),
            ),
            torch.concatenate(
                (
                    torch.reshape(lbls_pos, (num,)),
                    torch.reshape(
                        lbls_neg, (num * (sample_negative_rate + sample_num_neg_rels),)
                    ),
                ),
            ),
            pos_weight=weights,
        )
        return loss, (0.0, 0.0)

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
        sample_num_neg_rels: int,
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
        - sample_num_neg_rels
            Number of negative relation samples.
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
        num = len(lbls) // (1 + sample_negative_rate + sample_num_neg_rels)
        scores_pos = torch.reshape(scores[:num], (num, 1))
        buf_scores_neg = []
        ptr = num
        if sample_negative_rate > 0:
            #
            scores_nge = torch.reshape(
                scores[ptr : ptr + num * sample_negative_rate],
                (num, sample_negative_rate),
            )
            ptr += num * sample_negative_rate
            buf_scores_neg.append(scores_nge)
        if sample_num_neg_rels > 0:
            #
            scores_ngr = torch.reshape(
                scores[ptr : ptr + num * sample_num_neg_rels],
                (num, sample_num_neg_rels),
            )
            ptr += num * sample_num_neg_rels
            buf_scores_neg.append(scores_ngr)

        #
        # \\:rankables = torch.cat((scores_pos, scores_neg), dim=1)
        # \\:ranks = torch.argsort(rankables, dim=1, descending=True)
        # \\:(_, ranks) = torch.nonzero(ranks == 0).T
        # \\:ranks = (ranks + 1).to(lbls.dtype)

        # Entity and relation negative samples are treated as same kind of negative samples.
        scores_neg = torch.concat(buf_scores_neg, dim=1)
        # ranks = torch.sum(scores_pos <= scores_neg, dim=1)
        # Uniformly rank over scores that has the same values
        ranks = torch.sum(scores_pos < scores_neg, dim=1)
        num_same = torch.sum(scores_pos == scores_neg, dim=1)
        rand_prop = torch.rand(len(ranks)).to(num_same.device)
        rand_val = torch.round(num_same * rand_prop).int()
        ranks += rand_val
        ranks = (ranks + 1).to(lbls.dtype)

        #
        mr = torch.mean(ranks).item()
        mrr = torch.mean(1.0 / ranks).item()
        hit_at_ks = {k: torch.mean((ranks <= k).to(lbls.dtype)).item() for k in ks}

        #
        return (
            {"MR": mr, "MRR": mrr, **{"Hit@{:d}".format(k): hit_at_ks[k] for k in ks}},
            scores,
        )
