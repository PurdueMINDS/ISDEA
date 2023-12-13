#
import torch
import torch_geometric as thgeo
from typing import TypeVar, Sequence, Callable, cast
from .model import Model


#
SelfRGCN = TypeVar("SelfRGCN", bound="RGCN")


class RGCN(Model):
    R"""
    R-GCN.
    """

    def __init__(
        self: SelfRGCN,
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
        - num_bases
            Number of bases.

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
        self.num_bases = num_bases

        #
        self.activate = cast(
            Callable[[torch.Tensor], torch.Tensor],
            {"relu": torch.relu, "tanh": torch.tanh}[activate],
        )

        #
        self.embedding_entity = torch.nn.Parameter(torch.zeros(self.num_entities, self.num_hiddens))
        self.embedding_relation = torch.nn.Parameter(
            torch.zeros(self.num_relations, self.num_hiddens, self.num_hiddens),
        )
        self.convs = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            #
            self.convs.append(
                thgeo.nn.RGCNConv(self.num_hiddens, self.num_hiddens, self.num_relations, num_bases=self.num_bases),
            )

    def get_embedding_shape_entity(self: SelfRGCN, /) -> Sequence[int]:
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

    def get_num_relations(self: SelfRGCN, /) -> int:
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

    def reset_parameters(self: SelfRGCN, rng: torch.Generator, /) -> SelfRGCN:
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
        self.reset_glorot(rng, self.embedding_entity.data, fanin=1, fanout=self.num_hiddens)
        self.reset_glorot(rng, self.embedding_relation.data, fanin=self.num_hiddens, fanout=self.num_hiddens)

        #
        for i in range(self.num_layers):
            #
            self.reset_glorot(rng, self.convs[i].weight.data, fanin=self.num_hiddens, fanout=self.num_hiddens)
            self.reset_glorot(rng, self.convs[i].comp.data, fanin=self.num_bases, fanout=self.num_relations)
            self.reset_glorot(rng, self.convs[i].root.data, fanin=1, fanout=self.num_hiddens)
            self.reset_zeros(rng, self.convs[i].bias.data)

        #
        assert sum(parameter.numel() for parameter in self.parameters()) == (
            self.embedding_entity.data.numel()
            + self.embedding_relation.data.numel()
            + sum(
                self.convs[i].weight.data.numel()
                + self.convs[i].comp.data.numel()
                + self.convs[i].root.data.numel()
                + self.convs[i].bias.data.numel()
                for i in range(self.num_layers)
            )
        )

        #
        return self

    def forward(self: SelfRGCN, vfts: torch.Tensor, adjs: torch.Tensor, rels: torch.Tensor, /) -> torch.Tensor:
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

        Returns
        -------
        - vrps
            Node representations.
        """
        #
        vrps = self.embedding_entity[vfts]
        for i in range(self.num_layers - 1):
            #
            vrps = self.activate(self.convs[i].forward(vrps, adjs, rels))
        vrps = self.convs[self.num_layers - 1].forward(vrps, adjs, rels)
        return vrps

    def measure_distance(
        self: SelfRGCN,
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
        self: SelfRGCN,
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
        subjects = vrps[adjs[0]].to(rels.device, non_blocking=True)
        relations = self.embedding_relation[rels].to(rels.device, non_blocking=True)
        objects = vrps[adjs[1]].to(rels.device, non_blocking=True)

        # Formalize shape.
        n = len(rels)
        subjects = torch.reshape(subjects, (n, 1, self.num_hiddens))
        relations = torch.reshape(relations, (n, self.num_hiddens, self.num_hiddens))
        objects = torch.reshape(objects, (n, self.num_hiddens, 1))

        #
        scores = torch.bmm(torch.bmm(subjects, relations), objects)
        scores = torch.reshape(scores, (n,))
        return scores
