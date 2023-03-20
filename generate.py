#
import math
import os
import shutil
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as onp
from networkx.drawing.nx_pydot import graphviz_layout
from matplotlib.lines import Line2D
from typing import Tuple, Mapping, Sequence, List, Dict
from types import MappingProxyType


#
DEPTH = 6


def fa_mo_tree(
    root_gender: int,
    depth: int,
    /,
    *,
    rel_fa: int,
    rel_mo: int,
) -> Tuple[Sequence[int], Mapping[int, Sequence[Tuple[Sequence[int], int]]]]:
    R"""
    Generate a father-mother tree from a person to his ancestors.

    Args
    ----
    - root_gender
        Gender of root node.
    - depth
        Tree depth.
    - rel_fa
        Relation ID of father.
    - rel_mo
        Relation ID of mother.

    Returns
    -------
    - vfts
        Node features (genders).
    - adjs
        Adjacency dictionary.
    """
    #
    adjs: Dict[int, List[Tuple[Sequence[int], int]]]

    #
    num_nodes = 1
    vfts = [root_gender]
    adjs = {}

    #
    queue = [(0, 0)]
    while len(queue) > 0:
        #
        (node_ch, level_ch) = queue.pop(0)
        if level_ch == depth:
            # Stop expansion at leaf node.
            continue

        # New nodes.
        num_nodes += 2
        vfts.append(-1)
        vfts.append(-1)
        node_fa = num_nodes - 2
        vfts[node_fa] = 0
        node_mo = num_nodes - 1
        vfts[node_mo] = 1
        level_fa = level_ch + 1
        level_mo = level_ch + 1
        queue.append((node_fa, level_fa))
        queue.append((node_mo, level_mo))

        # New edges.
        for node in (node_ch, node_fa, node_mo):
            #
            if node not in adjs:
                #
                adjs[node] = []
        adjs[node_ch].append(((rel_fa,), node_fa))
        adjs[node_ch].append(((rel_mo,), node_mo))
    assert all(vft >= 0 for vft in vfts)

    #
    return (vfts, MappingProxyType({key: tuple(val) for (key, val) in adjs.items()}))


def gr_tree(
    vfts: Sequence[int],
    adjs_fa_mo: Mapping[int, Sequence[Tuple[Sequence[int], int]]],
    /,
    *,
    rel_fa: int,
    rel_mo: int,
    rel_gr0: int,
    rel_gr1: int,
) -> Mapping[int, Sequence[Tuple[Sequence[int], int]]]:
    R"""
    Generate a grand tree from father-mother tree.

    Args
    ----
    - num_nodes
        Number of nodes.
    - adjs_fa_mo
        Father-mother adjacency dictionary.
    - rel_fa
        Relation ID of father.
    - rel_mo
        Relation ID of mother.
    - rel_gr0
        Relation ID of grand 0 (father-father or mother-mother).
    - rel_gr1
        Relation ID of grand 1 (father-mother or mother-father).

    Returns
    -------
    - adjs_gr
        Grand adjacency dictionary.
    """
    #
    adjs_gr_: Dict[int, List[Tuple[Sequence[int], int]]]

    #
    adjs_gr_ = {}
    for (node_ch, neighbors_pa1) in adjs_fa_mo.items():
        #
        for (rels_pa1, node_pa1) in neighbors_pa1:
            #
            if node_pa1 in adjs_fa_mo:
                #
                neighbors_pa2 = adjs_fa_mo[node_pa1]
                for (rels_pa2, node_pa2) in neighbors_pa2:
                    #
                    for node in (node_ch, node_pa2):
                        #
                        if node not in adjs_gr_:
                            #
                            adjs_gr_[node] = []

                    # Get relations between grand parent and grand child.
                    cnt = 0
                    if rel_fa in rels_pa1 and rel_fa in rels_pa2:
                        #
                        rel_gr = rel_gr0
                        rel_pa = rel_fa
                        cnt += 1
                    if rel_mo in rels_pa1 and rel_mo in rels_pa2:
                        #
                        rel_gr = rel_gr0
                        rel_pa = rel_mo
                        cnt += 1
                    if rel_fa in rels_pa1 and rel_mo in rels_pa2:
                        #
                        rel_gr = rel_gr1
                        rel_pa = rel_mo
                        cnt += 1
                    if rel_mo in rels_pa1 and rel_fa in rels_pa2:
                        #
                        rel_gr = rel_gr1
                        rel_pa = rel_fa
                        cnt += 1
                    assert cnt == 1

                    #
                    adjs_gr_[node_ch].append(((rel_pa, rel_gr), node_pa2))

    #
    adjs_gr: Dict[int, Sequence[Tuple[Sequence[int], int]]]

    #
    adjs_gr = {node_ch: tuple(neighbors_gr) for (node_ch, neighbors_gr) in adjs_gr_.items()}
    return MappingProxyType(adjs_gr)


def merge_and_lock(
    adjs1: Mapping[int, Sequence[Tuple[Sequence[int], int]]],
    adjs2: Mapping[int, Sequence[Tuple[Sequence[int], int]]],
    /,
) -> Mapping[int, Sequence[Tuple[Sequence[int], int]]]:
    R"""
    Merge first and second order adjacency list.

    Args
    ----
    - adjs1
        Adjacency list 1.
    - adjs2
        Adjacency list 2.

    Returns
    -------
    - adjs
        Merged adjacency list.
    """
    #
    adjs_: Dict[int, List[Tuple[Sequence[int], int]]]

    # Merge all-order relations.
    adjs_ = {}
    for adjs0 in (adjs1, adjs2):
        #
        for (node, neighbors) in adjs0.items():
            #
            if node not in adjs_:
                #
                adjs_[node] = []
            adjs_[node].extend(neighbors)

    #
    adjs: Dict[int, Sequence[Tuple[Sequence[int], int]]]

    #
    adjs = {node: tuple(neighbors_) for (node, neighbors_) in adjs_.items()}
    return MappingProxyType(adjs)


def render(
    path: str,
    vfts: Sequence[int],
    adjs: Mapping[int, Sequence[Tuple[Sequence[int], int]]],
    /,
    *,
    bidirect: bool,
    node_palette: Sequence[Sequence[float]],
    edge_texture: Sequence[Tuple[int, Sequence[int]]],
    edge_palette: Sequence[Sequence[float]],
    figsize: Tuple[float, float],
    rels: Sequence[str],
) -> None:
    R"""
    Render adjacency list with relations as multigraph.

    Args
    ----
    - path
        Figure path.
    - vfts
        Node features (genders).
    - adjs
        Adjacency list.
    - bidirect
        Render as a bidirected multigraph.
    - node_palette
        Color of node features.
    - edge_texture
        Style of edge relations.
    - edge_palette
        Color of edge relations.
    - figsize
        Figure size.
    - rels
        Relation names.

    Returns
    -------
    """
    #
    num_relations = max(len(node_palette), len(edge_texture), len(edge_palette), len(rels))

    #
    rbuf: List[int]

    #
    graph = nx.MultiDiGraph()
    rbuf = []
    for (s, ns) in adjs.items():
        #
        for (rs, o) in ns:
            #
            for r in sorted(rs):
                #
                graph.add_edge(s, o, relation=r)
                if bidirect:
                    #
                    graph.add_edge(o, s, relation=r + num_relations)
                assert s >= 0 and r >= 0 and o >= 0
            rbuf.extend(rs)
    rset = set(rbuf)

    #
    node_colors = [node_palette[vfts[v]] for v in graph.nodes]
    edge_styles = [edge_texture[graph.get_edge_data(u, v, k)["relation"] % num_relations] for (u, v, k) in graph.edges]
    edge_colors = [edge_palette[graph.get_edge_data(u, v, k)["relation"] % num_relations] for (u, v, k) in graph.edges]

    #
    (fig, ax) = plt.subplots(1, 1, figsize=figsize)
    nx.draw(
        graph,
        graphviz_layout(graph, prog="twopi"),
        ax=ax,
        with_labels=True,
        node_color=node_colors,
        linewidths=2.0,
        width=2.0,
        style=edge_styles,
        edge_color=edge_colors,
        connectionstyle="arc3,rad=0.1",
        font_size=8.0,
    )
    ax.legend(
        handles=[Line2D([0], [0], linewidth=2.0, linestyle=edge_texture[i], color=edge_palette[i]) for i in rset],
        labels=[rels[i] for i in rset],
    )
    fig.tight_layout(pad=0.0)

    #
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save(
    path: str,
    vfts: Sequence[int],
    adjs: Mapping[int, Sequence[Tuple[Sequence[int], int]]],
    node: str,
    rels: Sequence[str],
    /,
) -> None:
    R"""
    Render adjacency list with relations as multigraph.

    Args
    ----
    - path
        Figure path.
    - vfts
        Node features (genders).
    - adjs
        Adjacency list.
    - node
        Node prefix.
    - rels
        Relation names.

    Returns
    -------
    """
    #
    maxlen_s = 0
    maxlen_o = 0
    maxlen_r = 0
    for (s, ns) in adjs.items():
        #
        maxlen_s = max(len(node) + len(str(s)), maxlen_s)
        for (rs, o) in ns:
            #
            maxlen_o = max(len(node) + len(str(o)), maxlen_o)
            for r in sorted(rs):
                #
                maxlen_r = max(len(rels[r]), maxlen_r)
                assert s >= 0 and r >= 0 and o >= 0

    #
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        #
        for s in sorted(adjs.keys()):
            #
            for (rs, o) in sorted(adjs[s], key=lambda neighbor: neighbor[1]):
                #
                for r in sorted(rs):
                    #
                    file.write(
                        "{:>{:d}s} {:>{:d}s} {:>{:d}s}\n".format(
                            node + str(s),
                            maxlen_s,
                            rels[r],
                            maxlen_r,
                            node + str(o),
                            maxlen_o,
                        ),
                    )


def family_diagram(
    *,
    rel_fa: int,
    rel_mo: int,
    rel_gr0: int,
    rel_gr1: int,
) -> Tuple[Sequence[int], Mapping[int, Sequence[Tuple[Sequence[int], int]]]]:
    R"""
    Generate a full family digram.

    Args
    ----
    - rel_fa
        Relation ID of father.
    - rel_mo
        Relation ID of mother.
    - rel_gr0
        Relation ID of grand 0 (father-father or mother-mother).
    - rel_gr1
        Relation ID of grand 1 (father-mother or mother-father).

    Returns
    -------
    - vfts
        Node features (genders).
    - adjs
        Adjacency dictionary.
    """
    #
    root_gender = 0
    depth = DEPTH

    # Get first-order relations.
    (vfts, adjs1) = fa_mo_tree(root_gender, depth, rel_fa=rel_fa, rel_mo=rel_mo)

    # Get second-order relations.
    adjs2 = gr_tree(vfts, adjs1, rel_fa=rel_fa, rel_mo=rel_mo, rel_gr0=rel_gr0, rel_gr1=rel_gr1)

    #
    return (vfts, merge_and_lock(adjs1, adjs2))


def family_diagram1_trans_observe(
    *,
    rel_fa: int,
    rel_mo: int,
    rel_gr0: int,
    rel_gr1: int,
) -> Tuple[Sequence[int], Mapping[int, Sequence[Tuple[Sequence[int], int]]]]:
    R"""
    Generate transductive observation part of family digram 1.

    Args
    ----
    - rel_fa
        Relation ID of father.
    - rel_mo
        Relation ID of mother.
    - rel_gr0
        Relation ID of grand 0 (father-father or mother-mother).
    - rel_gr1
        Relation ID of grand 1 (father-mother or mother-father).

    Returns
    -------
    - vfts
        Node features (genders).
    - adjs
        Adjacency dictionary.
    """
    #
    (vfts, adjs_full) = family_diagram(rel_fa=rel_fa, rel_mo=rel_mo, rel_gr0=rel_gr0, rel_gr1=rel_gr1)

    #
    adjs: Mapping[int, Sequence[Tuple[Sequence[int], int]]]

    #
    adjs = {}
    for (source, neighbors) in adjs_full.items():
        #
        buf = []
        for (relations, destination) in neighbors:
            #
            if len(relations) == 1:
                #
                buf.append((relations, destination))
        if len(buf) > 0:
            #
            adjs[source] = tuple(buf)
    return (vfts, MappingProxyType(adjs))


def family_diagram1_trans_train(
    *,
    rel_fa: int,
    rel_mo: int,
    rel_gr0: int,
    rel_gr1: int,
    seed: int,
) -> Tuple[Sequence[int], Mapping[int, Sequence[Tuple[Sequence[int], int]]]]:
    R"""
    Generate transductive training part of family digram 1.

    Args
    ----
    - rel_fa
        Relation ID of father.
    - rel_mo
        Relation ID of mother.
    - rel_gr0
        Relation ID of grand 0 (father-father or mother-mother).
    - rel_gr1
        Relation ID of grand 1 (father-mother or mother-father).
    - seed
        Random seed.

    Returns
    -------
    - vfts
        Node features (genders).
    - adjs
        Adjacency dictionary.
    """
    #
    (vfts, adjs_full) = family_diagram(rel_fa=rel_fa, rel_mo=rel_mo, rel_gr0=rel_gr0, rel_gr1=rel_gr1)

    # Achieve essential allocation.
    cnt = 0
    for (source, neighbors) in adjs_full.items():
        #
        for (relations, destination) in neighbors:
            #
            if len(relations) == 2 and rel_mo not in relations:
                #
                cnt += 1
    indices = onp.random.RandomState(seed).permutation(cnt)
    indices = indices[: int(math.ceil(float(len(indices)) * 0.5))]
    masks = onp.zeros((cnt,), dtype=onp.bool_)
    masks[indices] = True

    #
    adjs: Mapping[int, Sequence[Tuple[Sequence[int], int]]]

    #
    cnt = 0
    adjs = {}
    for (source, neighbors) in adjs_full.items():
        #
        buf = []
        for (relations, destination) in neighbors:
            #
            if len(relations) == 2 and rel_mo not in relations:
                #
                if masks[cnt].item():
                    #
                    buf.append((relations, destination))
                cnt += 1
        if len(buf) > 0:
            #
            adjs[source] = tuple(buf)
    return (vfts, MappingProxyType(adjs))


def family_diagram1_trans_valid(
    *,
    rel_fa: int,
    rel_mo: int,
    rel_gr0: int,
    rel_gr1: int,
    seed: int,
) -> Tuple[Sequence[int], Mapping[int, Sequence[Tuple[Sequence[int], int]]]]:
    R"""
    Generate transductive validation part of family digram 1.

    Args
    ----
    - rel_fa
        Relation ID of father.
    - rel_mo
        Relation ID of mother.
    - rel_gr0
        Relation ID of grand 0 (father-father or mother-mother).
    - rel_gr1
        Relation ID of grand 1 (father-mother or mother-father).
    - seed
        Random seed.

    Returns
    -------
    - vfts
        Node features (genders).
    - adjs
        Adjacency dictionary.
    """
    #
    (vfts, adjs_full) = family_diagram(rel_fa=rel_fa, rel_mo=rel_mo, rel_gr0=rel_gr0, rel_gr1=rel_gr1)

    # Achieve essential allocation.
    cnt = 0
    for (source, neighbors) in adjs_full.items():
        #
        for (relations, destination) in neighbors:
            #
            if len(relations) == 2 and rel_mo not in relations:
                #
                cnt += 1
    indices = onp.random.RandomState(seed).permutation(cnt)
    indices = indices[int(math.ceil(float(len(indices)) * 0.5)) :]
    masks = onp.zeros((cnt,), dtype=onp.bool_)
    masks[indices] = True

    #
    adjs: Mapping[int, Sequence[Tuple[Sequence[int], int]]]

    #
    cnt = 0
    adjs = {}
    for (source, neighbors) in adjs_full.items():
        #
        buf = []
        for (relations, destination) in neighbors:
            #
            if len(relations) == 2 and rel_mo not in relations:
                #
                if masks[cnt].item():
                    #
                    buf.append((relations, destination))
                cnt += 1
        if len(buf) > 0:
            #
            adjs[source] = tuple(buf)
    return (vfts, MappingProxyType(adjs))


def family_diagram1_ind_observe(
    *,
    rel_fa: int,
    rel_mo: int,
    rel_gr0: int,
    rel_gr1: int,
) -> Tuple[Sequence[int], Mapping[int, Sequence[Tuple[Sequence[int], int]]]]:
    R"""
    Generate inductive observation part of family digram 1.

    Args
    ----
    - rel_fa
        Relation ID of father.
    - rel_mo
        Relation ID of mother.
    - rel_gr0
        Relation ID of grand 0 (father-father or mother-mother).
    - rel_gr1
        Relation ID of grand 1 (father-mother or mother-father).

    Returns
    -------
    - vfts
        Node features (genders).
    - adjs
        Adjacency dictionary.
    """
    #
    (vfts, adjs_full) = family_diagram(rel_fa=rel_fa, rel_mo=rel_mo, rel_gr0=rel_gr0, rel_gr1=rel_gr1)

    #
    adjs: Mapping[int, Sequence[Tuple[Sequence[int], int]]]

    #
    adjs = {}
    for (source, neighbors) in adjs_full.items():
        #
        buf = []
        for (relations, destination) in neighbors:
            #
            if len(relations) == 1:
                #
                buf.append((relations, destination))
        if len(buf) > 0:
            #
            adjs[source] = tuple(buf)
    return (vfts, MappingProxyType(adjs))


def family_diagram1_ind_test(
    *,
    rel_fa: int,
    rel_mo: int,
    rel_gr0: int,
    rel_gr1: int,
) -> Tuple[Sequence[int], Mapping[int, Sequence[Tuple[Sequence[int], int]]]]:
    R"""
    Generate inductive test part of family digram 1.

    Args
    ----
    - rel_fa
        Relation ID of father.
    - rel_mo
        Relation ID of mother.
    - rel_gr0
        Relation ID of grand 0 (father-father or mother-mother).
    - rel_gr1
        Relation ID of grand 1 (father-mother or mother-father).

    Returns
    -------
    - vfts
        Node features (genders).
    - adjs
        Adjacency dictionary.
    """
    #
    (vfts, adjs_full) = family_diagram(rel_fa=rel_fa, rel_mo=rel_mo, rel_gr0=rel_gr0, rel_gr1=rel_gr1)

    #
    adjs: Mapping[int, Sequence[Tuple[Sequence[int], int]]]

    #
    adjs = {}
    for (source, neighbors) in adjs_full.items():
        #
        buf = []
        for (relations, destination) in neighbors:
            #
            if len(relations) == 2 and rel_mo in relations:
                #
                buf.append((relations, destination))
        if len(buf) > 0:
            #
            adjs[source] = tuple(buf)
    return (vfts, MappingProxyType(adjs))


def fd1() -> None:
    R"""
    Generate family diagram 1.

    Args
    ----

    Returns
    -------
    """
    #
    data = "data"
    figure = "figure"
    name = "FD1"
    node = "node"
    rels = ["father", "mother", "grand", "grand"]
    node_palette = [(*sns.color_palette()[cid], 1.0) for cid in [0, 2]]
    edge_texture = [(0, ()), (0, ()), (0, (3, 3)), (0, (3, 3))]
    edge_palette = [(*sns.color_palette()[cid], 1.0) for cid in [0, 2, 1, 3]]
    if os.path.isdir(os.path.join(data, name)):
        #
        shutil.rmtree(os.path.join(data, name))
    (vfts_trans, adjs_trans_observe) = family_diagram1_trans_observe(rel_fa=0, rel_mo=1, rel_gr0=2, rel_gr1=2)
    (_, adjs_trans_train) = family_diagram1_trans_train(rel_fa=0, rel_mo=1, rel_gr0=2, rel_gr1=2, seed=42)
    (_, adjs_trans_valid) = family_diagram1_trans_valid(rel_fa=0, rel_mo=1, rel_gr0=2, rel_gr1=2, seed=42)
    (vfts_ind, adjs_ind_observe) = family_diagram1_ind_observe(rel_fa=0, rel_mo=1, rel_gr0=2, rel_gr1=2)
    (_, adjs_ind_test) = family_diagram1_ind_test(rel_fa=0, rel_mo=1, rel_gr0=2, rel_gr1=2)
    save(os.path.join(data, "{:s}-trans".format(name), "observe.txt"), vfts_trans, adjs_trans_observe, node, rels)
    save(os.path.join(data, "{:s}-trans".format(name), "train.txt"), vfts_trans, adjs_trans_train, node, rels)
    save(os.path.join(data, "{:s}-trans".format(name), "valid.txt"), vfts_trans, adjs_trans_valid, node, rels)
    save(os.path.join(data, "{:s}-ind".format(name), "observe.txt"), vfts_ind, adjs_ind_observe, node, rels)
    save(os.path.join(data, "{:s}-ind".format(name), "test.txt"), vfts_ind, adjs_ind_test, node, rels)
    for bidirect in (False, True):
        #
        render(
            os.path.join(figure, "{:s}-trans~dx{:d}".format(name, 1 + int(bidirect)), "observe.png"),
            vfts_trans,
            adjs_trans_observe,
            bidirect=bidirect,
            node_palette=node_palette,
            edge_texture=edge_texture,
            edge_palette=edge_palette,
            figsize=(10.0, 8.0),
            rels=rels,
        )
        render(
            os.path.join(figure, "{:s}-trans~dx{:d}".format(name, 1 + int(bidirect)), "train.png"),
            vfts_trans,
            adjs_trans_train,
            bidirect=bidirect,
            node_palette=node_palette,
            edge_texture=edge_texture,
            edge_palette=edge_palette,
            figsize=(10.0, 8.0),
            rels=rels,
        )
        render(
            os.path.join(figure, "{:s}-trans~dx{:d}".format(name, 1 + int(bidirect)), "valid.png"),
            vfts_trans,
            adjs_trans_valid,
            bidirect=bidirect,
            node_palette=node_palette,
            edge_texture=edge_texture,
            edge_palette=edge_palette,
            figsize=(10.0, 8.0),
            rels=rels,
        )
        render(
            os.path.join(figure, "{:s}-ind~dx{:d}".format(name, 1 + int(bidirect)), "observe.png"),
            vfts_ind,
            adjs_ind_observe,
            bidirect=bidirect,
            node_palette=node_palette,
            edge_texture=edge_texture,
            edge_palette=edge_palette,
            figsize=(10.0, 8.0),
            rels=rels,
        )
        render(
            os.path.join(figure, "{:s}-ind~dx{:d}".format(name, 1 + int(bidirect)), "test.png"),
            vfts_ind,
            adjs_ind_test,
            bidirect=bidirect,
            node_palette=node_palette,
            edge_texture=edge_texture,
            edge_palette=edge_palette,
            figsize=(10.0, 8.0),
            rels=rels,
        )


def family_diagram2_trans_observe(
    *,
    rel_fa: int,
    rel_mo: int,
) -> Tuple[Sequence[int], Mapping[int, Sequence[Tuple[Sequence[int], int]]]]:
    R"""
    Generate transductive observation part of family digram 2.

    Args
    ----
    - rel_fa
        Relation ID of father.
    - rel_mo
        Relation ID of mother.

    Returns
    -------
    - vfts
        Node features (genders).
    - adjs
        Adjacency dictionary.
    """
    #
    (vfts, adjs_full) = family_diagram(rel_fa=rel_fa, rel_mo=rel_mo, rel_gr0=-1, rel_gr1=-1)

    #
    adjs: Mapping[int, Sequence[Tuple[Sequence[int], int]]]

    #
    adjs = {}
    for (source, neighbors) in adjs_full.items():
        #
        buf = []
        for (relations, destination) in neighbors:
            #
            if len(relations) == 1:
                #
                buf.append((relations, destination))
        if len(buf) > 0:
            #
            adjs[source] = tuple(buf)
    return (vfts, MappingProxyType(adjs))


def family_diagram2_trans_train(
    *,
    rel_fa: int,
    rel_mo: int,
    seed: int,
) -> Tuple[Sequence[int], Mapping[int, Sequence[Tuple[Sequence[int], int]]]]:
    R"""
    Generate transductive training part of family digram 2.

    Args
    ----
    - rel_fa
        Relation ID of father.
    - rel_mo
        Relation ID of mother.
    - seed
        Random seed.

    Returns
    -------
    - vfts
        Node features (genders).
    - adjs
        Adjacency dictionary.
    """
    #
    (vfts, adjs_full) = family_diagram(rel_fa=rel_fa, rel_mo=rel_mo, rel_gr0=-1, rel_gr1=-1)

    # Achieve essential allocation.
    cnt = 0
    for (source, neighbors) in adjs_full.items():
        #
        for (relations, destination) in neighbors:
            #
            if len(relations) == 2:
                #
                cnt += 1
    indices = onp.random.RandomState(seed).permutation(cnt)
    indices = indices[: int(math.ceil(float(len(indices)) * 0.5))]
    masks = onp.zeros((cnt,), dtype=onp.bool_)
    masks[indices] = True

    #
    adjs: Mapping[int, Sequence[Tuple[Sequence[int], int]]]

    #
    cnt = 0
    adjs = {}
    for (source, neighbors) in adjs_full.items():
        #
        buf = []
        for (relations, destination) in neighbors:
            #
            if len(relations) == 2:
                #
                (r2, r1) = (min(relations), max(relations))
                assert r2 == -1, (source, relations, destination)
                if masks[cnt].item():
                    #
                    buf.append(((r1,), destination))
                cnt += 1
        if len(buf) > 0:
            #
            adjs[source] = tuple(buf)
    return (vfts, MappingProxyType(adjs))


def family_diagram2_trans_valid(
    *,
    rel_fa: int,
    rel_mo: int,
    seed: int,
) -> Tuple[Sequence[int], Mapping[int, Sequence[Tuple[Sequence[int], int]]]]:
    R"""
    Generate transductive validation part of family digram 2.

    Args
    ----
    - rel_fa
        Relation ID of father.
    - rel_mo
        Relation ID of mother.
    - seed
        Random seed.

    Returns
    -------
    - vfts
        Node features (genders).
    - adjs
        Adjacency dictionary.
    """
    #
    (vfts, adjs_full) = family_diagram(rel_fa=rel_fa, rel_mo=rel_mo, rel_gr0=-1, rel_gr1=-1)

    # Achieve essential allocation.
    cnt = 0
    for (source, neighbors) in adjs_full.items():
        #
        for (relations, destination) in neighbors:
            #
            if len(relations) == 2:
                #
                cnt += 1
    indices = onp.random.RandomState(seed).permutation(cnt)
    indices = indices[int(math.ceil(float(len(indices)) * 0.5)) :]
    masks = onp.zeros((cnt,), dtype=onp.bool_)
    masks[indices] = True

    #
    adjs: Mapping[int, Sequence[Tuple[Sequence[int], int]]]

    #
    cnt = 0
    adjs = {}
    for (source, neighbors) in adjs_full.items():
        #
        buf = []
        for (relations, destination) in neighbors:
            #
            if len(relations) == 2:
                #
                (r2, r1) = (min(relations), max(relations))
                assert r2 == -1, (source, relations, destination)
                if masks[cnt].item():
                    #
                    buf.append(((r1,), destination))
                cnt += 1
        if len(buf) > 0:
            #
            adjs[source] = tuple(buf)
    return (vfts, MappingProxyType(adjs))


def family_diagram2_ind_observe(
    *,
    rel_fa0: int,
    rel_mo0: int,
    rel_fa1: int,
    rel_mo1: int,
) -> Tuple[Sequence[int], Mapping[int, Sequence[Tuple[Sequence[int], int]]]]:
    R"""
    Generate inductive observation part of family digram 2.

    Args
    ----
    - rel_fa0
        Relation ID of father in component 0.
    - rel_mo0
        Relation ID of mother in component 0.
    - rel_fa1
        Relation ID of father in component 1.
    - rel_mo1
        Relation ID of mother in component 1.

    Returns
    -------
    - vfts
        Node features (genders).
    - adjs
        Adjacency dictionary.
    """
    #
    (vfts0, adjs_full0) = family_diagram(rel_fa=rel_fa0, rel_mo=rel_mo0, rel_gr0=-1, rel_gr1=-1)
    (vfts1, adjs_full1) = family_diagram(rel_fa=rel_fa1, rel_mo=rel_mo1, rel_gr0=-1, rel_gr1=-1)

    #
    adjs: Mapping[int, Sequence[Tuple[Sequence[int], int]]]

    #
    adjs = {}
    for (bias, adjs_full) in ((0, adjs_full0), (len(vfts0), adjs_full1)):
        #
        for (source, neighbors) in adjs_full.items():
            #
            buf = []
            for (relations, destination) in neighbors:
                #
                if len(relations) == 1:
                    #
                    buf.append((relations, destination + bias))
            if len(buf) > 0:
                #
                adjs[source + bias] = tuple(buf)

    #
    return ([*vfts0, *vfts1], MappingProxyType(adjs))


def family_diagram2_ind_test(
    *,
    rel_fa0: int,
    rel_mo0: int,
    rel_fa1: int,
    rel_mo1: int,
) -> Tuple[Sequence[int], Mapping[int, Sequence[Tuple[Sequence[int], int]]]]:
    R"""
    Generate inductive test part of family digram 2.

    Args
    ----
    - rel_fa0
        Relation ID of father in component 0.
    - rel_mo0
        Relation ID of mother in component 0.
    - rel_fa1
        Relation ID of father in component 1.
    - rel_mo1
        Relation ID of mother in component 1.

    Returns
    -------
    - vfts
        Node features (genders).
    - adjs
        Adjacency dictionary.
    """
    #
    (vfts0, adjs_full0) = family_diagram(rel_fa=rel_fa0, rel_mo=rel_mo0, rel_gr0=-1, rel_gr1=-1)
    (vfts1, adjs_full1) = family_diagram(rel_fa=rel_fa1, rel_mo=rel_mo1, rel_gr0=-1, rel_gr1=-1)

    #
    adjs: Mapping[int, Sequence[Tuple[Sequence[int], int]]]

    #
    adjs = {}
    for (bias, adjs_full) in ((0, adjs_full0), (len(vfts0), adjs_full1)):
        #
        for (source, neighbors) in adjs_full.items():
            #
            buf = []
            for (relations, destination) in neighbors:
                #
                if len(relations) == 2:
                    #
                    (r2, r1) = (min(relations), max(relations))
                    assert r2 == -1, (source, relations, destination)
                    buf.append(((r1,), destination + bias))
            if len(buf) > 0:
                #
                adjs[source + bias] = tuple(buf)
    return ([*vfts0, *vfts1], MappingProxyType(adjs))


def fd2() -> None:
    R"""
    Generate family diagram 2.

    Args
    ----

    Returns
    -------
    """
    #
    data = "data"
    figure = "figure"
    name = "FD2"
    node = "v"
    rels = ["r/1", "r/2", "r/a", "r/b", "r/x", "r/y"]
    node_palette = [(*sns.color_palette()[cid], 1.0) for cid in [0, 2]]
    edge_texture = [(0, ()), (0, ()), (0, ()), (0, ()), (0, ()), (0, ())]
    edge_palette = [(*sns.color_palette()[cid], 1.0) for cid in [0, 2, 1, 3, 8, 9]]
    if os.path.isdir(os.path.join(data, name)):
        #
        shutil.rmtree(os.path.join(data, name))
    (vfts_trans, adjs_trans_observe) = family_diagram2_trans_observe(rel_fa=0, rel_mo=1)
    (_, adjs_trans_train) = family_diagram2_trans_train(rel_fa=0, rel_mo=1, seed=42)
    (_, adjs_trans_valid) = family_diagram2_trans_valid(rel_fa=0, rel_mo=1, seed=42)
    (vfts_ind, adjs_ind_observe) = family_diagram2_ind_observe(rel_fa0=2, rel_mo0=3, rel_fa1=4, rel_mo1=5)
    (_, adjs_ind_test) = family_diagram2_ind_test(rel_fa0=2, rel_mo0=3, rel_fa1=4, rel_mo1=5)
    save(os.path.join(data, "{:s}-trans".format(name), "observe.txt"), vfts_trans, adjs_trans_observe, node, rels)
    save(os.path.join(data, "{:s}-trans".format(name), "train.txt"), vfts_trans, adjs_trans_train, node, rels)
    save(os.path.join(data, "{:s}-trans".format(name), "valid.txt"), vfts_trans, adjs_trans_valid, node, rels)
    save(os.path.join(data, "{:s}-ind".format(name), "observe.txt"), vfts_ind, adjs_ind_observe, node, rels)
    save(os.path.join(data, "{:s}-ind".format(name), "test.txt"), vfts_ind, adjs_ind_test, node, rels)
    for bidirect in (False, True):
        #
        render(
            os.path.join(figure, "{:s}-trans~dx{:d}".format(name, 1 + int(bidirect)), "observe.png"),
            vfts_trans,
            adjs_trans_observe,
            bidirect=bidirect,
            node_palette=node_palette,
            edge_texture=edge_texture,
            edge_palette=edge_palette,
            figsize=(10.0, 8.0),
            rels=rels,
        )
        render(
            os.path.join(figure, "{:s}-trans~dx{:d}".format(name, 1 + int(bidirect)), "train.png"),
            vfts_trans,
            adjs_trans_train,
            bidirect=bidirect,
            node_palette=node_palette,
            edge_texture=edge_texture,
            edge_palette=edge_palette,
            figsize=(10.0, 8.0),
            rels=rels,
        )
        render(
            os.path.join(figure, "{:s}-trans~dx{:d}".format(name, 1 + int(bidirect)), "valid.png"),
            vfts_trans,
            adjs_trans_valid,
            bidirect=bidirect,
            node_palette=node_palette,
            edge_texture=edge_texture,
            edge_palette=edge_palette,
            figsize=(10.0, 8.0),
            rels=rels,
        )
        render(
            os.path.join(figure, "{:s}-ind~dx{:d}".format(name, 1 + int(bidirect)), "observe.png"),
            vfts_ind,
            adjs_ind_observe,
            bidirect=bidirect,
            node_palette=node_palette,
            edge_texture=edge_texture,
            edge_palette=edge_palette,
            figsize=(16.0, 12.0),
            rels=rels,
        )
        render(
            os.path.join(figure, "{:s}-ind~dx{:d}".format(name, 1 + int(bidirect)), "test.png"),
            vfts_ind,
            adjs_ind_test,
            bidirect=bidirect,
            node_palette=node_palette,
            edge_texture=edge_texture,
            edge_palette=edge_palette,
            figsize=(16.0, 12.0),
            rels=rels,
        )


#
if __name__ == "__main__":
    #
    # \\:fd1()
    fd2()
