#
import torch
import torch_geometric as thgeo
from typing import TypeVar, Sequence, Callable, cast
from .model import Model


#
SelfDecompose = TypeVar("SelfDecompose", bound="Decompose")


class Decompose(Model):
    R"""
    Decompose.
    """

    def __init__(
        self: SelfDecompose,
        num_entities: int,
        num_relations: int,
        num_layers: int,
        num_hiddens: int,
        /,
        *,
        activate: str,
        dropout: float,
        kernel: str,
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
        - kernel
            Decomposition kernel.

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
        self.kernel = kernel

        #
        assert self.kernel in ("distmult", "transe", "complex", "rotate")

        #
        self.embedding_entity = torch.nn.Parameter(torch.zeros(self.num_entities, self.num_hiddens))
        if self.kernel in ("distmult",):
            #
            self.embedding_relation = torch.nn.Parameter(
                torch.zeros(self.num_relations, self.num_hiddens, self.num_hiddens),
            )
        elif self.kernel in ("transe"):
            #
            self.embedding_relation = torch.nn.Parameter(torch.zeros(self.num_relations, self.num_hiddens))

    def get_embedding_shape_entity(self: SelfDecompose, /) -> Sequence[int]:
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

    def get_num_relations(self: SelfDecompose, /) -> int:
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

    def reset_parameters(self: SelfDecompose, rng: torch.Generator, /) -> SelfDecompose:
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
        if self.kernel in ("distmult",):
            #
            self.reset_glorot(rng, self.embedding_relation.data, fanin=self.num_hiddens, fanout=self.num_hiddens)
        elif self.kernel in ("transe"):
            #
            self.reset_glorot(rng, self.embedding_relation.data, fanin=self.num_hiddens, fanout=self.num_hiddens)

        #
        assert sum(parameter.numel() for parameter in self.parameters()) == (
            self.embedding_entity.data.numel() + self.embedding_relation.data.numel()
        )

        #
        return self

    def forward(self: SelfDecompose, vfts: torch.Tensor, adjs: torch.Tensor, rels: torch.Tensor, /) -> torch.Tensor:
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
        return vrps

    def measure_distance(
        self: SelfDecompose,
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
        self: SelfDecompose,
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

        #
        if self.kernel in ("distmult",):
            #
            scores = torch.bmm(
                torch.bmm(torch.reshape(subjects, (len(subjects), 1, self.num_hiddens)), relations),
                torch.reshape(objects, (len(objects), self.num_hiddens, 1)),
            )
        elif self.kernel in ("transe"):
            #
            scores = -torch.norm(subjects + relations - objects, dim=1)
        scores = torch.reshape(scores, (len(rels),))
        return scores
