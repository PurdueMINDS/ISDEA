#
import torch
import torch_geometric as thgeo
from typing import TypeVar, Sequence, Callable, cast
from nbfnet.models import NBFNet  # type: ignore
from torch_geometric.data import Data
from .model import Model


#
SelfNBFNetWrap = TypeVar("SelfNBFNetWrap", bound="NBFNetWrap")


class NBFNetWrap(Model):
    R"""
    Wrap NBFNet with project interface.
    """

    def __init__(
        self: SelfNBFNetWrap,
        num_entities: int,
        num_relations: int,
        num_layers: int,
        num_hiddens: int,
        /,
        *,
        activate: str,
        dropout: float,
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
        self.activate = activate

        #
        self.nbfnet = NBFNet(
            self.num_hiddens,
            [self.num_hiddens for _ in range(self.num_layers)],
            self.num_relations,
            activation=activate,
        )

    def get_embedding_shape_entity(self: SelfNBFNetWrap, /) -> Sequence[int]:
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
        return (1,)

    def get_num_relations(self: SelfNBFNetWrap, /) -> int:
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

    def reset_parameters(self: SelfNBFNetWrap, rng: torch.Generator, /) -> SelfNBFNetWrap:
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
        for i in range(self.num_layers):
            #
            self.reset_glorot(
                rng,
                self.nbfnet.layers[i].linear.weight.data,
                fanin=self.num_hiddens * 13,
                fanout=self.num_hiddens,
            )
            self.reset_zeros(rng, self.nbfnet.layers[i].linear.bias.data)
            self.reset_glorot(
                rng,
                self.nbfnet.layers[i].relation_linear.weight.data,
                fanin=self.num_hiddens,
                fanout=self.num_hiddens * self.num_relations,
            )
            self.reset_zeros(rng, self.nbfnet.layers[i].relation_linear.bias.data)

        #
        self.reset_glorot(rng, self.nbfnet.query.weight.data, fanin=1, fanout=self.num_hiddens)

        #
        self.reset_glorot(rng, self.nbfnet.mlp[0].weight.data, fanin=self.num_hiddens * 2, fanout=self.num_hiddens * 2)
        self.reset_zeros(rng, self.nbfnet.mlp[0].bias.data)
        self.reset_glorot(rng, self.nbfnet.mlp[2].weight.data, fanin=self.num_hiddens * 2, fanout=self.num_hiddens * 2)
        self.reset_zeros(rng, self.nbfnet.mlp[2].bias.data)

        #
        assert sum(parameter.numel() for parameter in self.parameters()) == (
            sum(
                self.nbfnet.layers[i].linear.weight.data.numel()
                + self.nbfnet.layers[i].linear.bias.data.numel()
                + self.nbfnet.layers[i].relation_linear.weight.data.numel()
                + self.nbfnet.layers[i].relation_linear.bias.data.numel()
                for i in range(self.num_layers)
            )
            + self.nbfnet.query.weight.data.numel()
            + self.nbfnet.mlp[0].weight.data.numel()
            + self.nbfnet.mlp[0].bias.data.numel()
            + self.nbfnet.mlp[2].weight.data.numel()
            + self.nbfnet.mlp[2].bias.data.numel()
        )

        #
        return self

    def forward(
        self: SelfNBFNetWrap,
        vfts_observe: torch.Tensor,
        adjs_observe: torch.Tensor,
        rels_observe: torch.Tensor,
        adjs_target: torch.Tensor,
        rels_target: torch.Tensor,
        lbls_target: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.

        Args
        ----
        - vfts_observe
            Observed node features.
            It should be node entity ID only.
        - adjs_observe
            Observed adjacency lists.
        - rels_observe
            Observed relations.
        - adjs_target
            Targeting adjacency lists.
        - rels_target
            Targeting relations.
        - lbls_target
            Targeting labels.

        Returns
        -------
        - vrps
            Node representations.
        """
        # Translate to NBFNet input.
        data = Data(edge_index=adjs_observe, edge_type=rels_observe, num_nodes=len(vfts_observe))

        #
        num = int(torch.sum(lbls_target == 1).item())
        negative_rate = int(torch.sum(lbls_target == 0).item()) // num
        assert num * (1 + negative_rate) == len(lbls_target)

        #
        srcs_target_pos = torch.reshape(adjs_target[0, :num], (num, 1))
        dsts_target_pos = torch.reshape(adjs_target[1, :num], (num, 1))
        rels_target_pos = torch.reshape(rels_target[:num], (num, 1))
        srcs_target_neg = torch.reshape(adjs_target[0, num:], (num, negative_rate))
        dsts_target_neg = torch.reshape(adjs_target[1, num:], (num, negative_rate))
        rels_target_neg = torch.reshape(rels_target[num:], (num, negative_rate))
        batch = torch.concatenate(
            (
                torch.stack((srcs_target_pos, dsts_target_pos, rels_target_pos), dim=2),
                torch.stack((srcs_target_neg, dsts_target_neg, rels_target_neg), dim=2),
            ),
            dim=1,
        )
        return cast(torch.Tensor, self.nbfnet.forward(data, batch))

    def measure_distance(
        self: SelfNBFNetWrap,
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
        self: SelfNBFNetWrap,
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
        return torch.reshape(vrps, (len(rels),))
