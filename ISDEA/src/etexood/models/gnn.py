#
import torch
import torch_geometric as thgeo
from typing import TypeVar, Sequence, Callable, cast
from .model import Model


class GNN(Model):
    R"""
    DSS GNN whose second GNN uses adjacency matrix of self-exclusive each relation types.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        num_layers: int,
        num_hiddens: int,
        /,
        *,
        activate: str,
        dropout: float,
        train_eps: bool,
        ablate: str,
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
            Dropout.
        - kernel
            Kernel convolution name.
        - train_eps
            Training epislon of GIN.
        - dss_aggr
            DSS aggregation.
        - ablate
            Ablation study.

        Returns
        -------
        """
        raise NotImplementedError

    def get_embedding_shape_entity(self, /) -> Sequence[int]:
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
        return (self.num_relations, self.num_hiddens)

    def get_num_relations(self, /) -> int:
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

    def reset_parameters(self, rng: torch.Generator, /):
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
        raise NotImplementedError

    def forward(self, vfts: torch.Tensor, adjs: torch.Tensor, rels: torch.Tensor, /) -> torch.Tensor:
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
            vrps = self.activate(self.convs[i].forward(vrps, adjs))
        vrps = self.convs[self.num_layers - 1].forward(vrps, adjs)
        
        return vrps

    def measure_distance(
        self,
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
        self,
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
        subs_given_rel = vrps[adjs[0]].to(rels.device, non_blocking=True)
        objs_given_rel = vrps[adjs[1]].to(rels.device, non_blocking=True)

        #
        dists_sub_to_obj = self.embedding_shortest[heus[:, 0]]
        dists_obj_to_sub = self.embedding_shortest[heus[:, 1]]
        if self.ablate == "dss":
            #
            erps = torch.concatenate((subs_given_rel, objs_given_rel, subs_given_rel, objs_given_rel), dim=1)
        elif self.ablate == "dist":
            #
            erps = torch.concatenate((dists_sub_to_obj, dists_obj_to_sub, dists_sub_to_obj, dists_obj_to_sub), dim=1)
        else:
            #
            erps = torch.concatenate((subs_given_rel, objs_given_rel, dists_sub_to_obj, dists_obj_to_sub), dim=1)

        #
        scores = self.lin2.forward(self.activate(self.lin1.forward(erps)))
        scores = torch.reshape(scores, (len(rids),))
        return scores


class GIN(GNN):
    R"""
    DSS GNN whose second GNN uses adjacency matrix of self-exclusive each relation types.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        num_layers: int,
        num_hiddens: int,
        /,
        *,
        activate: str,
        dropout: float,
        train_eps: bool,
        ablate: str,
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
            Dropout.
        - kernel
            Kernel convolution name.
        - train_eps
            Training epislon of GIN.
        - dss_aggr
            DSS aggregation.
        - ablate
            Ablation study.

        Returns
        -------
        """
        #
        torch.nn.Module.__init__(self)

        #

        #
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.num_hiddens = num_hiddens
        self.ablate = ablate
        #
        self.activate = cast(
            torch.nn.Module,
            {"relu": torch.nn.ReLU(), "tanh": torch.nn.Tanh()}[activate],
        )

        # DSSGNN is strutural, thus entity embedding is none.
        # We use 1 as embedding of all entities, and freeze them from learning.
        self.embedding_entity = torch.nn.Parameter(torch.zeros(self.num_entities, 1))
        self.embedding_entity.requires_grad = False

        self.train_eps = train_eps
        # DSSGNN should be joint representation which is simplified into a structrual representation along with some
        # heuristic representation.
        self.embedding_shortest = torch.nn.Parameter(torch.zeros(self.num_layers + 2, self.num_hiddens))

        #
        self.convs = torch.nn.ModuleList()
        for fanin, fanout in (
            (1, self.num_hiddens),
            *((self.num_hiddens, self.num_hiddens) for _ in range(self.num_layers - 1)),
        ):
            #
            self.convs.append(
                thgeo.nn.GINConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(fanin, fanout),
                        self.activate,
                        torch.nn.Linear(fanout, fanout),
                    ),
                    train_eps=self.train_eps,
                )
            )

        #
        # self.dsslin1 = DSSLinearExcl(self.num_hiddens, self.num_hiddens, self.num_relations, self.dss_aggr)
        # self.dsslin2 = DSSLinearExcl(self.num_hiddens, self.num_hiddens, self.num_relations, self.dss_aggr)
        self.lin1 = torch.nn.Linear(self.num_hiddens * (2 + 2), self.num_hiddens)
        self.lin2 = torch.nn.Linear(self.num_hiddens, 1)

    def reset_parameters(self, rng: torch.Generator, /):
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
        self.reset_ones(rng, self.embedding_entity.data)
        self.reset_glorot(rng, self.embedding_shortest.data, fanin=1, fanout=self.num_hiddens)

        #
        for i, (fanin, fanout) in enumerate([
            (1, self.num_hiddens),
            *((self.num_hiddens, self.num_hiddens) for _ in range(self.num_layers - 1))
        ]):
            Model.reset_glorot(rng, self.convs[i].nn[0].weight.data, fanin=fanin, fanout=fanout)
            Model.reset_zeros(rng, self.convs[i].nn[0].bias.data)
            Model.reset_glorot(rng, self.convs[i].nn[2].weight.data, fanin=fanout, fanout=fanout)
            Model.reset_zeros(rng, self.convs[i].nn[2].bias.data)
            if self.train_eps:
                #
                Model.reset_zeros(rng, self.convs[i].eps.data)

        #
        # self.dsslin1.reset_parameters(rng)
        # self.dsslin2.reset_parameters(rng)

        #
        self.reset_glorot(rng, self.lin1.weight.data, fanin=self.num_hiddens * (2 + 2), fanout=self.num_hiddens)
        self.reset_zeros(rng, self.lin1.bias.data)
        self.reset_glorot(rng, self.lin2.weight.data, fanin=self.num_hiddens, fanout=1)
        self.reset_zeros(rng, self.lin2.bias.data)

        #
        # assert sum(parameter.numel() for parameter in self.parameters()) == (
        #     self.embedding_entity.data.numel()
        #     + self.embedding_shortest.data.numel()
        #     + sum(sum(parameter.numel() for parameter in self.convs[i].parameters()) for i in range(self.num_layers))
        #     + sum(parameter.numel() for parameter in self.dsslin1.parameters())
        #     + sum(parameter.numel() for parameter in self.dsslin2.parameters())
        #     + self.lin1.weight.data.numel()
        #     + self.lin1.bias.data.numel()
        #     + self.lin2.weight.data.numel()
        #     + self.lin2.bias.data.numel()
        # )

        #
        return self
    
class GAT(GNN):
    R"""
    DSS GNN whose second GNN uses adjacency matrix of self-exclusive each relation types.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        num_layers: int,
        num_hiddens: int,
        /,
        *,
        activate: str,
        dropout: float,
        train_eps: bool,
        ablate: str,
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
            Dropout.
        - kernel
            Kernel convolution name.
        - train_eps
            Training epislon of GIN.
        - dss_aggr
            DSS aggregation.
        - ablate
            Ablation study.

        Returns
        -------
        """
        #
        torch.nn.Module.__init__(self)

        #

        #
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.num_hiddens = num_hiddens
        self.ablate = ablate
        #
        self.activate = cast(
            torch.nn.Module,
            {"relu": torch.nn.ReLU(), "tanh": torch.nn.Tanh()}[activate],
        )

        # DSSGNN is strutural, thus entity embedding is none.
        # We use 1 as embedding of all entities, and freeze them from learning.
        self.embedding_entity = torch.nn.Parameter(torch.zeros(self.num_entities, 1))
        self.embedding_entity.requires_grad = False

        self.num_heads = 2
        # DSSGNN should be joint representation which is simplified into a structrual representation along with some
        # heuristic representation.
        self.embedding_shortest = torch.nn.Parameter(torch.zeros(self.num_layers + 2, self.num_hiddens))

        #
        self.convs = torch.nn.ModuleList()
        for fanin, fanout in (
            (1, self.num_hiddens),
            *((self.num_hiddens, self.num_hiddens) for _ in range(self.num_layers - 1)),
        ):
            #
            self.convs.append(
                thgeo.nn.GATConv(
                    fanin,
                    fanout//self.num_heads,
                    heads=self.num_heads
                )
            )

        #
        # self.dsslin1 = DSSLinearExcl(self.num_hiddens, self.num_hiddens, self.num_relations, self.dss_aggr)
        # self.dsslin2 = DSSLinearExcl(self.num_hiddens, self.num_hiddens, self.num_relations, self.dss_aggr)
        self.lin1 = torch.nn.Linear(self.num_hiddens * (2 + 2), self.num_hiddens)
        self.lin2 = torch.nn.Linear(self.num_hiddens, 1)

    def reset_parameters(self, rng: torch.Generator, /):
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
        self.reset_ones(rng, self.embedding_entity.data)
        self.reset_glorot(rng, self.embedding_shortest.data, fanin=1, fanout=self.num_hiddens)

        #
        for i, (fanin, fanout) in enumerate([
            (1, self.num_hiddens),
            *((self.num_hiddens, self.num_hiddens) for _ in range(self.num_layers - 1))
        ]):
            Model.reset_glorot(rng, self.convs[i].lin_src.weight.data, fanin=fanin, fanout=fanout)
            Model.reset_glorot(rng, self.convs[i].att_src.data, fanin=fanin, fanout=fanout//self.num_heads)
            Model.reset_glorot(rng, self.convs[i].att_dst.data, fanin=fanin, fanout=fanout//self.num_heads)
            
        #
        # self.dsslin1.reset_parameters(rng)
        # self.dsslin2.reset_parameters(rng)

        #
        self.reset_glorot(rng, self.lin1.weight.data, fanin=self.num_hiddens * (2 + 2), fanout=self.num_hiddens)
        self.reset_zeros(rng, self.lin1.bias.data)
        self.reset_glorot(rng, self.lin2.weight.data, fanin=self.num_hiddens, fanout=1)
        self.reset_zeros(rng, self.lin2.bias.data)

        #
        # assert sum(parameter.numel() for parameter in self.parameters()) == (
        #     self.embedding_entity.data.numel()
        #     + self.embedding_shortest.data.numel()
        #     + sum(sum(parameter.numel() for parameter in self.convs[i].parameters()) for i in range(self.num_layers))
        #     + sum(parameter.numel() for parameter in self.dsslin1.parameters())
        #     + sum(parameter.numel() for parameter in self.dsslin2.parameters())
        #     + self.lin1.weight.data.numel()
        #     + self.lin1.bias.data.numel()
        #     + self.lin2.weight.data.numel()
        #     + self.lin2.bias.data.numel()
        # )

        #
        return self
    
class SAGE(GNN):
    R"""
    DSS GNN whose second GNN uses adjacency matrix of self-exclusive each relation types.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        num_layers: int,
        num_hiddens: int,
        /,
        *,
        activate: str,
        dropout: float,
        train_eps: bool,
        ablate: str,
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
            Dropout.
        - kernel
            Kernel convolution name.
        - train_eps
            Training epislon of GIN.
        - dss_aggr
            DSS aggregation.
        - ablate
            Ablation study.

        Returns
        -------
        """
        #
        torch.nn.Module.__init__(self)

        #

        #
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.num_hiddens = num_hiddens
        self.ablate = ablate
        #
        self.activate = cast(
            torch.nn.Module,
            {"relu": torch.nn.ReLU(), "tanh": torch.nn.Tanh()}[activate],
        )

        # DSSGNN is strutural, thus entity embedding is none.
        # We use 1 as embedding of all entities, and freeze them from learning.
        self.embedding_entity = torch.nn.Parameter(torch.zeros(self.num_entities, 1))
        self.embedding_entity.requires_grad = False

        self.num_heads = 2
        # DSSGNN should be joint representation which is simplified into a structrual representation along with some
        # heuristic representation.
        self.embedding_shortest = torch.nn.Parameter(torch.zeros(self.num_layers + 2, self.num_hiddens))

        #
        self.convs = torch.nn.ModuleList()
        for fanin, fanout in (
            (1, self.num_hiddens),
            *((self.num_hiddens, self.num_hiddens) for _ in range(self.num_layers - 1)),
        ):
            #
            self.convs.append(
                thgeo.nn.SAGEConv(
                    fanin,
                    fanout,
                    aggr='sum',
                    normalize=True,
                )
            )

        #
        # self.dsslin1 = DSSLinearExcl(self.num_hiddens, self.num_hiddens, self.num_relations, self.dss_aggr)
        # self.dsslin2 = DSSLinearExcl(self.num_hiddens, self.num_hiddens, self.num_relations, self.dss_aggr)
        self.lin1 = torch.nn.Linear(self.num_hiddens * (2 + 2), self.num_hiddens)
        self.lin2 = torch.nn.Linear(self.num_hiddens, 1)

    def reset_parameters(self, rng: torch.Generator, /):
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
        self.reset_ones(rng, self.embedding_entity.data)
        self.reset_glorot(rng, self.embedding_shortest.data, fanin=1, fanout=self.num_hiddens)

        #
        for i, (fanin, fanout) in enumerate([
            (1, self.num_hiddens),
            *((self.num_hiddens, self.num_hiddens) for _ in range(self.num_layers - 1))
        ]):
            Model.reset_glorot(rng, self.convs[i].lin_l.weight.data, fanin=fanin, fanout=fanout)
            Model.reset_glorot(rng, self.convs[i].lin_r.weight.data, fanin=fanin, fanout=fanout)
            self.reset_zeros(rng, self.convs[i].lin_l.bias.data)
            
        #
        # self.dsslin1.reset_parameters(rng)
        # self.dsslin2.reset_parameters(rng)

        #
        self.reset_glorot(rng, self.lin1.weight.data, fanin=self.num_hiddens * (2 + 2), fanout=self.num_hiddens)
        self.reset_zeros(rng, self.lin1.bias.data)
        self.reset_glorot(rng, self.lin2.weight.data, fanin=self.num_hiddens, fanout=1)
        self.reset_zeros(rng, self.lin2.bias.data)

        #
        # assert sum(parameter.numel() for parameter in self.parameters()) == (
        #     self.embedding_entity.data.numel()
        #     + self.embedding_shortest.data.numel()
        #     + sum(sum(parameter.numel() for parameter in self.convs[i].parameters()) for i in range(self.num_layers))
        #     + sum(parameter.numel() for parameter in self.dsslin1.parameters())
        #     + sum(parameter.numel() for parameter in self.dsslin2.parameters())
        #     + self.lin1.weight.data.numel()
        #     + self.lin1.bias.data.numel()
        #     + self.lin2.weight.data.numel()
        #     + self.lin2.bias.data.numel()
        # )

        #
        return self
    