#
import torch
import torch_geometric as thgeo
from typing import TypeVar, Sequence, Callable, cast, Type
from .model import Model


#
SelfGraILConv = TypeVar("SelfGraILConv", bound="GraILConv")
SelfGraIL = TypeVar("SelfGraIL", bound="GraIL")


class GraILConv(torch.nn.Module):
    R"""
    GraIL convolution layer.
    """

    def __init__(
        self: SelfGraILConv,
        num_inputs_node: int,
        num_inputs_edge: int,
        num_outputs: int,
        num_relations: int,
        /,
        *,
        activate: str,
        dropout: float,
        num_bases: int,
    ):
        R"""
        Initialize the class.

        Args
        ----
        - num_inputs_node
            Number of input dimensions on nodes.
        - num_inputs_edge
            Number of input dimensions on edges.
        - num_outputs
            Number of output dimensions.
        - num_relations
            Number of relations.
        - activate
            Activation.
        - dropout
            Dropout.
        - num_bases
            Number of bases.

        Returns
        -------
        """
        #
        torch.nn.Module.__init__(self)

        #
        self.num_inputs_node = num_inputs_node
        self.num_inputs_edge = num_inputs_edge
        self.num_outputs = num_outputs
        self.num_relations = num_relations
        self.num_bases = num_bases

        #
        self.activate = {"relu": torch.nn.ReLU(), "tanh": torch.nn.Tanh()}[activate]

        #
        self.weight = torch.nn.Parameter(torch.zeros(self.num_bases, self.num_inputs_node, self.num_outputs))
        self.comp = torch.nn.Parameter(torch.zeros(self.num_relations, self.num_bases))

        #
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(self.num_inputs_node * 2 + self.num_inputs_edge * 2, self.num_outputs),
            self.activate,
            torch.nn.Linear(self.num_outputs, 1),
            torch.nn.Sigmoid(),
        )
        self.update = torch.nn.Parameter(torch.zeros(self.num_inputs_node, self.num_outputs))

    def reset_parameters(self: SelfGraILConv, rng: torch.Generator, /) -> SelfGraILConv:
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
        #
        Model.reset_glorot(rng, self.weight.data, fanin=self.num_inputs_node, fanout=self.num_outputs)
        Model.reset_glorot(rng, self.comp.data, fanin=self.num_bases, fanout=self.num_relations)

        #
        Model.reset_glorot(
            rng,
            self.attention[0].weight.data,
            fanin=self.num_inputs_node * 2 + self.num_inputs_edge * 2,
            fanout=self.num_outputs,
        )
        Model.reset_zeros(rng, self.attention[0].bias.data)
        Model.reset_glorot(rng, self.attention[2].weight.data, fanin=self.num_outputs, fanout=1)
        Model.reset_zeros(rng, self.attention[2].bias.data)

        #
        Model.reset_glorot(rng, self.update.data, fanin=self.num_inputs_node, fanout=self.num_outputs)

        #
        assert sum(parameter.numel() for parameter in self.parameters()) == (
            self.weight.data.numel()
            + self.comp.data.numel()
            + self.attention[0].weight.data.numel()
            + self.attention[0].bias.data.numel()
            + self.attention[2].weight.data.numel()
            + self.attention[2].bias.data.numel()
            + self.update.data.numel()
        )

        #
        return self

    def forward(
        self: SelfGraILConv,
        vfts: torch.Tensor,
        adjs: torch.Tensor,
        rels: torch.Tensor,
        embed_rels: torch.Tensor,
        embed_rels_target: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.

        Args
        ----
        - vfts
            Node features.
        - adjs
            Adjacency list.
        - rels
            Relations.
        - embed_rels
            Embeddings of relations.
        - embed_rels
            Embeddings of relations as target.

        Returns
        -------
        - vrps
            Node representations.
        """
        #
        erps = torch.concatenate((vfts[adjs[0]], vfts[adjs[1]], embed_rels, embed_rels_target), dim=1)
        alphas = self.attention.forward(erps)

        # Get source node representations.
        weight = torch.reshape(
            torch.mm(self.comp, torch.reshape(self.weight, (self.num_bases, self.num_inputs_node * self.num_outputs))),
            (self.num_relations, self.num_inputs_node, self.num_outputs),
        )
        srcs = torch.matmul(torch.reshape(vfts, (1, len(vfts), self.num_inputs_node)), weight)
        srcs = alphas[rels] * srcs[rels, adjs[0]]

        #
        vrps = torch.zeros(len(vfts), self.num_outputs, dtype=vfts.dtype, device=vfts.device)
        vrps.index_add_(0, adjs[1], srcs)

        #
        dsts = torch.mm(vfts, self.update)
        vrps.index_add_(0, adjs[1], dsts[adjs[1]])
        return cast(torch.Tensor, self.activate.forward(vrps))


class GraIL(Model):
    R"""
    GraIL.
    """

    def __init__(
        self: SelfGraIL,
        num_entities: int,
        num_relations: int,
        num_layers: int,
        num_hiddens: int,
        /,
        *,
        activate: str,
        dropout: float,
        num_bases: int,
    ) -> None:
        R"""
        Initialize the class.

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
        - activate
            Activation.
        - dropout
            Dropout rate.

        Returns
        -------
        """
        #
        torch.nn.Module.__init__(self)

        #
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.num_hiddens = num_hiddens

        #
        self.activate = cast(
            Callable[[torch.Tensor], torch.Tensor],
            {"relu": torch.relu, "tanh": torch.tanh}[activate],
        )

        #
        self.embedding_shortest = torch.nn.Parameter(torch.zeros(self.num_layers + 2, self.num_hiddens))
        self.embedding_relation = torch.nn.Parameter(torch.zeros(self.num_relations, self.num_hiddens))

        #
        self.convs = torch.nn.ModuleList()
        for (fanin, fanout) in [
            (self.num_hiddens * 2, self.num_hiddens),
            *((self.num_hiddens, self.num_hiddens) for _ in range(self.num_layers - 1)),
        ]:
            #
            self.convs.append(
                GraILConv(
                    fanin,
                    self.num_hiddens,
                    fanout,
                    self.num_relations,
                    activate=activate,
                    dropout=dropout,
                    num_bases=num_bases,
                ),
            )

        #
        self.lin = torch.nn.Linear(self.num_hiddens * 4, 1)

    def get_embedding_shape_entity(self: SelfGraIL, /) -> Sequence[int]:
        R"""
        Entity embedding shape.

        Args
        ----

        Returns
        -------
        - shape
            Shape.
        """
        #
        return (self.num_hiddens,)

    def get_num_relations(self: SelfGraIL, /) -> int:
        R"""
        Entity number of relations.

        Args
        ----

        Returns
        -------
        - num
            Number of relations.
        """
        #
        return self.num_relations

    def reset_parameters(self: SelfGraIL, rng: torch.Generator, /) -> SelfGraIL:
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
        #
        self.reset_glorot(rng, self.embedding_shortest.data, fanin=1, fanout=self.num_hiddens)
        self.reset_glorot(rng, self.embedding_relation.data, fanin=1, fanout=self.num_hiddens)

        #
        for i in range(self.num_layers):
            #
            self.convs[i].reset_parameters(rng)

        #
        self.reset_glorot(rng, self.lin.weight.data, fanin=self.num_hiddens * 4, fanout=1)
        self.reset_zeros(rng, self.lin.bias.data)

        #
        assert sum(parameter.numel() for parameter in self.parameters()) == (
            self.embedding_shortest.data.numel()
            + self.embedding_relation.data.numel()
            + sum(sum(parameter.numel() for parameter in self.convs[i].parameters()) for i in range(self.num_layers))
            + self.lin.weight.data.numel()
            + self.lin.bias.data.numel()
        )

        #
        return self

    def forward(
        self: SelfGraIL,
        vfts: torch.Tensor,
        adjs: torch.Tensor,
        rels: torch.Tensor,
        vpts: torch.Tensor,
        epts: torch.Tensor,
        rels_target: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.

        Args
        ----
        - vfts
            Node features.
            It should be node entity ID only.
        - adjs
            Adjacency lists.
        - rels
            Relations.
        - vpts
            Node data boundaries of each subgraph.
        - epts
            Node data boundaries of each subgraph.
        - rels_target
            Targeting relations of each subgraph.


        Returns
        -------
        - vrps
            Node representations.
        """
        # Expand targeting relations.
        assert epts.ndim == 2 and tuple(epts.shape) == (len(rels_target), 2)
        rels2 = torch.zeros_like(rels)
        for i in range(len(rels_target)):
            #
            rels2[int(epts[i, 0].item()) : int(epts[i, 1].item())] = rels_target[i]

        #
        embed_rels = self.embedding_relation[rels]
        embed_rels2 = self.embedding_relation[rels2]

        #
        vrps = torch.cat((self.embedding_shortest[vfts[:, 0]], self.embedding_shortest[vfts[:, 1]]), dim=1)
        for i in range(self.num_layers - 1):
            #
            vrps = self.activate(self.convs[i].forward(vrps, adjs, rels, embed_rels, embed_rels2))
        vrps = self.convs[self.num_layers - 1].forward(vrps, adjs, rels, embed_rels, embed_rels2)
        return vrps

    def measure_distance(
        self: SelfGraIL,
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
        # Assume distance and score are the same measurement.
        return -self.measure_score(vrps, adjs, rels, heus)

    def measure_score(
        self: SelfGraIL,
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
            Placeholder for heuristics of adjacency list.
            Here, it is specially used as node bounardies of each subgraph.

        Returns
        -------
        - measures
            Measurements.
        """
        #
        vpts = heus

        #
        subgraphs = torch.zeros(len(rels), self.num_hiddens, dtype=vrps.dtype, device=vrps.device)
        for i in range(len(rels)):
            #
            subgraphs[i] = torch.mean(vrps[int(vpts[i, 0].item()) : int(vpts[i, 1].item())], dim=0)

        #
        subjects = vrps[adjs[0]].to(rels.device, non_blocking=True)
        objects = vrps[adjs[1]].to(rels.device, non_blocking=True)
        relations = self.embedding_relation[rels]

        #
        erps = torch.concatenate((subgraphs, subjects, objects, relations), dim=1)

        #
        scores = self.lin.forward(erps)
        scores = torch.reshape(scores, (len(rels),))
        return scores

    @classmethod
    @torch.no_grad()
    def is_loss_function_safe(
        cls: Type[SelfGraIL],
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
            Placeholder for heuristics of adjacency list.
            Here, it is specially used as node bounardies of each subgraph.
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
        rels_pos = torch.reshape(rels[:num], (num, 1))
        rels_neg = torch.reshape(rels[num:], (num, sample_negative_rate))

        # GraIL takes only enclosed-subgraph-like input, thus it will not check adjacency list.
        assert torch.all(rels_pos == rels_neg).item(), "Negative samples should not change relation type."

        # Heuristics argument is used as interface for node boundaries of each subgraph.
        vpts = heus
        assert vpts.ndim == 2 and tuple(vpts.shape) == (len(rels), 2)
