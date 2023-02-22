#
import torch
import torch_geometric as thgeo
from typing import TypeVar, Sequence, Callable, cast
from .model import Model


#
SelfDSSLinearExcl = TypeVar("SelfDSSLinearExcl", bound="DSSLinearExcl")
SelfDSSConvExcl = TypeVar("SelfDSSConvExcl", bound="DSSConvExcl")
SelfDSSGNNExcl = TypeVar("SelfDSSGNNExcl", bound="DSSGNNExcl")


class DSSLinearExcl(torch.nn.Module):
    R"""
    DSS Linear whose second linear uses embedding of self-exclusive each relation types.
    """

    def __init__(
        self: SelfDSSLinearExcl,
        num_inputs: int,
        num_outputs: int,
        num_relations: int,
        /,
    ) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - num_inputs
            Number of input dimensions.
        - num_outputs
            Number of output dimensions.
        - num_relations
            Number of relations.

        Returns
        -------
        """
        #
        torch.nn.Module.__init__(self)

        #
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_relations = num_relations

        #
        self.lin1 = torch.nn.Linear(self.num_inputs, self.num_outputs)
        self.lin2 = torch.nn.Linear(self.num_inputs, self.num_outputs)

    def reset_parameters(self: SelfDSSLinearExcl, rng: torch.Generator, /) -> SelfDSSLinearExcl:
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
        Model.reset_glorot(rng, self.lin1.weight.data, fanin=self.num_inputs, fanout=self.num_outputs)
        Model.reset_zeros(rng, self.lin1.bias.data)

        #
        Model.reset_glorot(rng, self.lin2.weight.data, fanin=self.num_inputs, fanout=self.num_outputs)
        Model.reset_zeros(rng, self.lin2.bias.data)

        #
        assert sum(parameter.numel() for parameter in self.parameters()) == (
            self.lin1.weight.data.numel()
            + self.lin1.bias.data.numel()
            + self.lin2.weight.data.numel()
            + self.lin2.bias.data.numel()
        )

        #
        return self

    def forward(self: SelfDSSLinearExcl, vfts: torch.Tensor, /) -> torch.Tensor:
        R"""
        Forward.

        Args
        ----
        - vfts
            Node features.

        Returns
        -------
        - vrps
            Node representations.
        """
        #
        num_nodes = len(vfts)
        vrps = torch.zeros(num_nodes, self.num_relations, self.num_outputs, dtype=vfts.dtype, device=vfts.device)

        #
        for r in range(self.num_relations):
            #
            nrs = [nr for nr in range(self.num_relations) if nr != r]

            #
            vrps1 = self.lin1.forward(vfts[:, r])
            vrps2 = self.lin2.forward(torch.sum(vfts[:, nrs], dim=1))
            vrps[:, r] = vrps1 + vrps2
        return vrps


class DSSConvExcl(torch.nn.Module):
    R"""
    DSS GIN whose second GNN uses adjacency matrix of self-exclusive each relation types.
    """

    def __init__(
        self: SelfDSSConvExcl,
        num_inputs: int,
        num_outputs: int,
        num_relations: int,
        /,
        *,
        activate: str,
        dropout: float,
        kernel: str,
        train_eps: bool,
    ) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - num_inputs
            Number of input dimensions.
        - num_outputs
            Number of output dimensions.
        - num_relations
            Number of relations.
        - activate
            Activation.
        - dropout
            Dropout.
        - kernel
            Kernel convolution name.
        - train_eps
            Training epislon of GIN.

        Returns
        -------
        """
        #
        torch.nn.Module.__init__(self)

        #
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_relations = num_relations
        self.kernel = kernel
        self.train_eps = train_eps

        #
        self.activate = {"relu": torch.nn.ReLU(), "tanh": torch.nn.Tanh()}[activate]

        #
        if self.kernel == "gin":
            #
            self.conv1 = thgeo.nn.GINConv(
                torch.nn.Sequential(
                    torch.nn.Linear(self.num_inputs, self.num_outputs),
                    self.activate,
                    torch.nn.Linear(self.num_outputs, self.num_outputs),
                ),
                train_eps=self.train_eps,
            )
            self.conv2 = thgeo.nn.GINConv(
                torch.nn.Sequential(
                    torch.nn.Linear(self.num_inputs, self.num_outputs),
                    self.activate,
                    torch.nn.Linear(self.num_outputs, self.num_outputs),
                ),
                train_eps=self.train_eps,
            )
        else:
            #
            raise RuntimeError('Unknown convolution kernel "{:s}".'.format(self.kernel))

    def reset_parameters(self: SelfDSSConvExcl, rng: torch.Generator, /) -> SelfDSSConvExcl:
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
        if self.kernel == "gin":
            #
            Model.reset_glorot(rng, self.conv1.nn[0].weight.data, fanin=self.num_inputs, fanout=self.num_outputs)
            Model.reset_zeros(rng, self.conv1.nn[0].bias.data)
            Model.reset_glorot(rng, self.conv1.nn[2].weight.data, fanin=self.num_outputs, fanout=self.num_outputs)
            Model.reset_zeros(rng, self.conv1.nn[2].bias.data)
            if self.train_eps:
                #
                Model.reset_zeros(rng, self.conv1.eps.data)

            #
            Model.reset_glorot(rng, self.conv2.nn[0].weight.data, fanin=self.num_inputs, fanout=self.num_outputs)
            Model.reset_zeros(rng, self.conv2.nn[0].bias.data)
            Model.reset_glorot(rng, self.conv2.nn[2].weight.data, fanin=self.num_outputs, fanout=self.num_outputs)
            Model.reset_zeros(rng, self.conv2.nn[2].bias.data)
            if self.train_eps:
                #
                Model.reset_zeros(rng, self.conv2.eps.data)

            #
            assert sum(parameter.numel() for parameter in self.parameters()) == (
                self.conv1.nn[0].weight.data.numel()
                + self.conv1.nn[0].bias.data.numel()
                + self.conv1.nn[2].weight.data.numel()
                + self.conv1.nn[2].bias.data.numel()
                + int(self.train_eps)
                + self.conv2.nn[0].weight.data.numel()
                + self.conv2.nn[0].bias.data.numel()
                + self.conv2.nn[2].weight.data.numel()
                + self.conv2.nn[2].bias.data.numel()
                + int(self.train_eps)
            )
        else:
            #
            raise RuntimeError('Unknown convolution kernel "{:s}".'.format(self.kernel))

        #
        return self

    def forward(
        self: SelfDSSConvExcl,
        vfts: torch.Tensor,
        adjs: torch.Tensor,
        rels: torch.Tensor,
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

        Returns
        -------
        - vrps
            Node representations.
        """
        #
        num_nodes = len(vfts)
        vmax = torch.max(adjs) + 1
        vrps = torch.zeros(num_nodes, self.num_relations, self.num_outputs, dtype=vfts.dtype, device=vfts.device)

        #
        for r in range(self.num_relations):
            #
            nrs = [nr for nr in range(self.num_relations) if nr != r]

            # Get adjacency list of enumerating relation.
            adjs_r = adjs[:, rels == r]

            # Get adjacency list of non-enumerating relation with replacement.
            adjs_nrs = adjs[:, rels != r]

            #
            if vfts.ndim == 3:
                #
                vrps1 = self.conv1.forward(vfts[:, r], adjs_r)
                vrps2 = self.conv2.forward(torch.sum(vfts[:, nrs], dim=1), adjs_nrs)
            else:
                #
                vrps1 = self.conv1.forward(vfts, adjs_r)
                vrps2 = self.conv2.forward(vfts, adjs_nrs)
            vrps[:, r] = vrps1 + vrps2
        return vrps


class DSSGNNExcl(Model):
    R"""
    DSS GNN whose second GNN uses adjacency matrix of self-exclusive each relation types.
    """

    def __init__(
        self: SelfDSSGNNExcl,
        num_entities: int,
        num_relations: int,
        num_layers: int,
        num_hiddens: int,
        /,
        *,
        activate: str,
        dropout: float,
        kernel: str,
        train_eps: bool,
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

        # DSSGNN is strutural, thus entity embedding is none.
        # We use 1 as embedding of all entities, and freeze them from learning.
        self.embedding_entity = torch.nn.Parameter(torch.zeros(self.num_entities, 1))
        self.embedding_entity.requires_grad = False

        # DSSGNN should be joint representation which is simplified into a structrual representation along with some
        # heuristic representation.
        self.embedding_shortest = torch.nn.Parameter(torch.zeros(self.num_layers + 2, self.num_hiddens))

        #
        self.convs = torch.nn.ModuleList()
        for (fanin, fanout) in (
            (1, self.num_hiddens),
            *((self.num_hiddens, self.num_hiddens) for _ in range(self.num_layers - 1)),
        ):
            #
            self.convs.append(
                DSSConvExcl(
                    fanin,
                    fanout,
                    self.num_relations,
                    activate=activate,
                    dropout=dropout,
                    kernel=kernel,
                    train_eps=train_eps,
                )
            )

        #
        self.dsslin1 = DSSLinearExcl(self.num_hiddens, self.num_hiddens, self.num_relations)
        self.dsslin2 = DSSLinearExcl(self.num_hiddens, self.num_hiddens, self.num_relations)
        self.lin1 = torch.nn.Linear(self.num_hiddens * (2 + 2), self.num_hiddens)
        # self.lin1 = torch.nn.Linear(self.num_hiddens, self.num_hiddens)
        self.lin2 = torch.nn.Linear(self.num_hiddens, 1)

    def get_embedding_shape_entity(self: SelfDSSGNNExcl, /) -> Sequence[int]:
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

    def get_num_relations(self: SelfDSSGNNExcl, /) -> int:
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

    def reset_parameters(self: SelfDSSGNNExcl, rng: torch.Generator, /) -> SelfDSSGNNExcl:
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
        for i in range(self.num_layers):
            #
            self.convs[i].reset_parameters(rng)

        #
        self.dsslin1.reset_parameters(rng)
        self.dsslin2.reset_parameters(rng)

        #
        self.reset_glorot(rng, self.lin1.weight.data, fanin=self.num_hiddens * (2 + 2), fanout=self.num_hiddens)
        #self.reset_glorot(rng, self.lin1.weight.data, fanin=self.num_hiddens, fanout=self.num_hiddens)
        self.reset_zeros(rng, self.lin1.bias.data)
        self.reset_glorot(rng, self.lin2.weight.data, fanin=self.num_hiddens, fanout=1)
        self.reset_zeros(rng, self.lin2.bias.data)

        #
        assert sum(parameter.numel() for parameter in self.parameters()) == (
            self.embedding_entity.data.numel()
            + self.embedding_shortest.data.numel()
            + sum(sum(parameter.numel() for parameter in self.convs[i].parameters()) for i in range(self.num_layers))
            + sum(parameter.numel() for parameter in self.dsslin1.parameters())
            + sum(parameter.numel() for parameter in self.dsslin2.parameters())
            + self.lin1.weight.data.numel()
            + self.lin1.bias.data.numel()
            + self.lin2.weight.data.numel()
            + self.lin2.bias.data.numel()
        )

        #
        return self

    def forward(self: SelfDSSGNNExcl, vfts: torch.Tensor, adjs: torch.Tensor, rels: torch.Tensor, /) -> torch.Tensor:
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

        #
        vrps = self.dsslin2.forward(self.activate(self.dsslin1.forward(vrps)))
        return vrps

    def measure_distance(
        self: SelfDSSGNNExcl,
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
        self: SelfDSSGNNExcl,
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
        dists_sub_to_obj = self.embedding_shortest[heus[:, 0]]
        dists_obj_to_sub = self.embedding_shortest[heus[:, 1]]
        erps = torch.concatenate((subs_given_rel, objs_given_rel, dists_sub_to_obj, dists_obj_to_sub), dim=1)
        # erps = self.embedding_shortest[torch.minimum(heus[:, 0], heus[:, 1])]

        #
        scores = self.lin2.forward(self.activate(self.lin1.forward(erps)))
        scores = torch.reshape(scores, (len(rids),))
        return scores
