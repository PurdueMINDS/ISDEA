#
from typing import Mapping, Any, Union
from .dssgnn import DSSGNNExcl
from .gnn import GIN, GAT, SAGE


#
ModelHeuristics = DSSGNNExcl
Model = ModelHeuristics


def create_model(
    num_entities: int,
    num_relations: int,
    num_hops: int,
    num_layers: int,
    num_hiddens: int,
    name: str,
    kwargs: Mapping[str, Any],
    /,
) -> Model:
    R"""
    Create a model.

    Args
    ----
    - num_entities
        Number of entities.
    - num_relations
        Number of relations.
    - num_hops
        Number of distance hops to create heuristics embeddings for.
    - num_layers
        Number of layers.
    - num_hiddens
        Number of hidden embeddings.
    - name
        Model name.
    - kwargs
        Keyword arguments for given model name.

    Returns
    -------
    - model
        Model.
    """
    #
    if name == "dssgnn":
        #
        return DSSGNNExcl(
            num_entities,
            num_relations,
            num_hops,
            num_layers,
            num_hiddens,
            activate=str(kwargs["activate"]),
            dropout=float(kwargs["dropout"]),
            kernel=str(kwargs["kernel"]),
            train_eps=bool(kwargs["train_eps"]),
            dss_aggr=str(kwargs["dss_aggr"]),
            ablate=str(kwargs["ablate"]),
        )
    elif name == 'gin':
        return GIN(
            num_entities,
            num_relations,
            num_layers,
            num_hiddens,
            activate=str(kwargs["activate"]),
            dropout=float(kwargs["dropout"]),
            train_eps=bool(kwargs["train_eps"]),
            ablate=str(kwargs["ablate"]),
        )
    elif name == 'gat':
        return GAT(
            num_entities,
            num_relations,
            num_layers,
            num_hiddens,
            activate=str(kwargs["activate"]),
            dropout=float(kwargs["dropout"]),
            train_eps=bool(kwargs["train_eps"]),
            ablate=str(kwargs["ablate"]),
        )
    elif name == 'sage':
        return SAGE(
            num_entities,
            num_relations,
            num_layers,
            num_hiddens,
            activate=str(kwargs["activate"]),
            dropout=float(kwargs["dropout"]),
            train_eps=bool(kwargs["train_eps"]),
            ablate=str(kwargs["ablate"]),
        )
    # \\:elif name in ("distmult", "transe", "complex", "rotate"):
    # \\:    #
    # \\:    return Decompose(
    # \\:        num_entities,
    # \\:        num_relations,
    # \\:        num_layers,
    # \\:        num_hiddens,
    # \\:        activate=str(kwargs["activate"]),
    # \\:        dropout=float(kwargs["dropout"]),
    # \\:        kernel=name,
    # \\:    )
    # \\:if name == "rgcn":
    # \\:    #
    # \\:    return RGCN(
    # \\:        num_entities,
    # \\:        num_relations,
    # \\:        num_layers,
    # \\:        num_hiddens,
    # \\:        activate=str(kwargs["activate"]),
    # \\:        dropout=float(kwargs["dropout"]),
    # \\:        num_bases=int(kwargs["num_bases"]),
    # \\:    )
    # \\:elif name == "grail":
    # \\:    #
    # \\:    return GraIL(
    # \\:        num_entities,
    # \\:        num_relations,
    # \\:        num_layers,
    # \\:        num_hiddens,
    # \\:        activate=str(kwargs["activate"]),
    # \\:        dropout=float(kwargs["dropout"]),
    # \\:        num_bases=int(kwargs["num_bases"]),
    # \\:    )
    # \\:elif name == "nbfnet":
    # \\:    #
    # \\:    return NBFNetWrap(
    # \\:        num_entities,
    # \\:        num_relations,
    # \\:        num_layers,
    # \\:        num_hiddens,
    # \\:        activate=str(kwargs["activate"]),
    # \\:        dropout=float(kwargs["dropout"]),
    # \\:    )
    else:
        #
        raise RuntimeError('Unsupported model name "{:s}".'.format(name))


def get_loss(
    num_entities: int,
    num_relations: int,
    num_layers: int,
    num_hiddens: int,
    name: str,
    kwargs: Mapping[str, Any],
    /,
) -> str:
    R"""
    Get loss function name corresponding to model.

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
    - name
        Model name.
    - kwargs
        Keyword arguments for given model name.

    Returns
    -------
    - loss
        Loss function name.
    """
    #
    if name in ["dssgnn", "gin", "gat", "sage", "gcn"]:
        #
        return "binary"
    # \\:elif name in ("distmult", "transe", "complex", "rotate"):
    # \\:    #
    # \\:    return "distance"
    # \\:elif name == "rgcn":
    # \\:    #
    # \\:    return "binary"
    # \\:elif name == "grail":
    # \\:    #
    # \\:    return "binary"
    # \\:elif name == "nbfnet":
    # \\:    #
    # \\:    return "binary"
    else:
        #
        raise RuntimeError('Unsupported model name "{:s}".'.format(name))
