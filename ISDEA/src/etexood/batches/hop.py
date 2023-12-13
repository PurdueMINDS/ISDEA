#
import abc
import numpy as onp
import scipy.sparse as osparse
from typing import TypeVar, Tuple, cast
from ..dtypes import NPINTS, NPBOOLS


#
SelfComputationSubset = TypeVar("SelfComputationSubset", bound="ComputationSubset")
SelfComputationSubsetNode = TypeVar("SelfComputationSubsetNode", bound="ComputationSubsetNode")
SelfComputationSubsetEdge = TypeVar("SelfComputationSubsetEdge", bound="ComputationSubsetEdge")


class ComputationSubset(abc.ABC):
    R"""
    Computation subset generator.
    """

    def __init__(self: SelfComputationSubset, n: int, adjs: NPINTS, rels: NPINTS, /, *, num_hops: int) -> None:
        R"""
        Reset.

        Args
        ----
        - n
            Number of nodes.
        - adjs
            Adjecency list.
        - rels
            Relations.
        - num_hops
            Number of hops.

        Returns
        -------
        """
        #
        self._num_nodes = n
        self._num_hops = num_hops

        #
        self._adjs = adjs
        (self._srcs, self._dsts) = adjs
        self._rels = rels
        (_, self._num_edges) = self._adjs.shape
        assert not self._adjs.flags.writeable
        assert not self._rels.flags.writeable
        assert not self._srcs.flags.writeable
        assert not self._dsts.flags.writeable

        #
        self._eids = self._srcs * self._num_nodes + self._dsts
        self._eids.setflags(write=False)

        # Reusable masks.
        # It may be used by other instances, thus we will lock them by default.
        self.masks_node_active = onp.zeros((self._num_hops + 1, self._num_nodes), dtype=onp.bool_)
        self.masks_edge_active = onp.zeros((self._num_hops, self._num_edges), dtype=onp.bool_)
        self.masks_node_active.setflags(write=False)
        self.masks_edge_active.setflags(write=False)

        # Computation sampling translation.
        self.translate = {"tree": self.tree, "graph": self.graph}

    def reset(self: SelfComputationSubset, /) -> None:
        R"""
        Clean cache.

        Args
        ----

        Returns
        -------
        """
        #
        self.masks_node_active.setflags(write=True)
        self.masks_edge_active.setflags(write=True)
        self.masks_node_active.fill(False)
        self.masks_edge_active.fill(False)
        self.masks_node_active.setflags(write=False)
        self.masks_edge_active.setflags(write=False)

    def update(self: SelfComputationSubset, nodes: NPINTS, masks_edge_accept: NPBOOLS, /) -> None:
        R"""
        Update computation tree masks from give nodes.

        Args
        ----
        - nodes
            Nodes.
        - masks_edge_accept
            Edge accessibilities.

        Returns
        -------
        """
        #
        self.reset()

        # Create node and edge masks for each layer separately.
        self.masks_node_active.setflags(write=True)
        self.masks_edge_active.setflags(write=True)
        onp.put(self.masks_node_active[0], nodes, True)
        for l in range(self._num_hops):
            #
            onp.take(self.masks_node_active[l], self._dsts, out=self.masks_edge_active[l])
            onp.logical_and(self.masks_edge_active[l], masks_edge_accept, out=self.masks_edge_active[l])
            onp.put(self.masks_node_active[l + 1], self._srcs[self.masks_edge_active[l]], True)
        self.masks_node_active.setflags(write=False)
        self.masks_edge_active.setflags(write=False)

    def tree(self: SelfComputationSubset, /) -> Tuple[NPINTS, NPINTS, NPINTS, NPINTS]:
        R"""
        Translate computation tree masks into reduced tree.

        Args
        ----

        Returns
        -------
        - uids
            Full graph node IDs in computation tree.
            Negative means no mapping.
        - vids
            Computation tree node IDs in full graph.
        - adjs
            Computation tree adjacency list.
        - rels
            Computation tree relations.
        """
        #
        buf_vid = []
        buf_adj = []
        buf_rel = []

        # Get reduced graph node IDs in full graph.
        # Get full graph node IDs in reduced graph for each hop layer separately.
        uids = onp.full((self._num_hops + 1, self._num_nodes), -1, dtype=self._adjs.dtype)
        total = 0
        for l in range(self._num_hops + 1):
            #
            (vids,) = onp.nonzero(self.masks_node_active[l])
            onp.put(uids[l], vids, onp.arange(len(vids), dtype=vids.dtype) + total)
            buf_vid.append(vids)
            total += len(vids)

        # Reduce edges between different hop layers independently.
        for l in range(self._num_hops):
            #
            masks = self.masks_edge_active[l]
            (srcs, dsts) = self._adjs[:, masks]
            rels = self._rels[masks]
            buf_adj.append(onp.stack((uids[l + 1][srcs], uids[l][dsts])))
            buf_rel.append(rels)

        #
        vids = onp.concatenate(buf_vid)
        adjs = onp.concatenate(buf_adj, axis=1)
        rels = onp.concatenate(buf_rel)
        return (uids, vids, adjs, rels)

    def graph(self: SelfComputationSubset, /) -> Tuple[NPINTS, NPINTS, NPINTS, NPINTS]:
        R"""
        Translate computation tree masks into reduced graph.

        Args
        ----

        Returns
        -------
        - uids
            Full graph node IDs in computation tree.
            Negative means no mapping.
        - vids
            Computation tree node IDs in full graph.
        - adjs
            Computation tree adjacency list.
        - rels
            Computation tree relations.
        """
        # Get reduced graph node IDs in full graph.
        # Get all active nodes together.
        uids = onp.full((self._num_nodes,), -1, dtype=self._adjs.dtype)
        masks_node_active = onp.any(self.masks_node_active, axis=0)
        (vids,) = onp.nonzero(masks_node_active)
        onp.put(uids, vids, onp.arange(len(vids), dtype=vids.dtype))

        # Reduce edges between different hop layers independently.
        masks_edge_active = onp.logical_and(masks_node_active[self._adjs[0]], masks_node_active[self._adjs[1]])
        adjs = uids[self._adjs[:, masks_edge_active]]
        rels = self._rels[masks_edge_active].copy()

        #
        return (uids, vids, adjs, rels)

    @abc.abstractmethod
    def sample(
        self: SelfComputationSubset,
        form: str,
        centers: NPINTS,
        masks_edge_accept: NPBOOLS,
        /,
    ) -> Tuple[NPINTS, NPINTS, NPINTS, NPINTS]:
        R"""
        Sample computation tree centering at given centers.

        Args
        ----
        - form
            Sampling translation form.
        - centers
            Centers.
        - masks_edge_accept
            Edge accessibilities.

        Returns
        -------
        - uids
            Full graph node IDs in computation tree.
            Negative means no mapping.
        - vids
            Computation tree node IDs in full graph.
        - adjs
            Computation tree adjacency list.
        - rels
            Computation tree relations.
        """

    def masks_edge_accept(self: SelfComputationSubset, xadjs: NPINTS, /) -> NPBOOLS:
        R"""
        Get masks to filter given rejected adjacency list for sampling.

        Args
        ----
        - xadjs
            Rejecting adjacency list.

        Returns
        -------
        - masks
            Masks of adjacency list.
        """
        #
        (xsrcs, xdsts) = xadjs
        xeids = xsrcs * self._num_nodes + xdsts
        return onp.logical_not(onp.isin(self._eids, xeids))


class ComputationSubsetNode(ComputationSubset):
    R"""
    Computation subset generator centered at nodes.
    """

    def sample(
        self: SelfComputationSubsetNode,
        form: str,
        centers: NPINTS,
        masks_edge_accept: NPBOOLS,
        /,
    ) -> Tuple[NPINTS, NPINTS, NPINTS, NPINTS]:
        R"""
        Sample computation tree.

        Args
        ----
        - form
            Sampling translation form.
        - centers
            Centers.
        - masks_edge_accept
            Edge accessibilities.

        Returns
        -------
        - uids
            Full graph node IDs in computation tree.
            Negative means no mapping.
        - vids
            Computation tree node IDs in full graph.
        - adjs
            Computation tree adjacency list.
        - rels
            Computation tree relations.
        """
        #
        assert centers.ndim == 1
        nodes = centers
        self.update(nodes, masks_edge_accept)
        return self.translate[form]()


class ComputationSubsetEdge(ComputationSubset):
    R"""
    Computation subset generator centered at edges.
    """

    def sample(
        self: SelfComputationSubsetEdge,
        form: str,
        centers: NPINTS,
        masks_edge_accept: NPBOOLS,
        /,
    ) -> Tuple[NPINTS, NPINTS, NPINTS, NPINTS]:
        R"""
        Sample computation tree.

        Args
        ----
        - form
            Sampling translation form.
        - centers
            Centers.
        - masks_edge_accept
            Edge accessibilities.

        Returns
        -------
        - uids
            Full graph node IDs in computation tree.
            Negative means no mapping.
        - vids
            Computation tree node IDs in full graph.
        - adjs
            Computation tree adjacency list.
        - rels
            Computation tree relations.
        """
        #
        assert centers.ndim == 2 and len(centers.T) == 2
        nodes = onp.unique(onp.reshape(centers, (len(centers) * 2,)))
        self.update(nodes, masks_edge_accept)
        return self.translate[form]()


def shortest(n: int, adjs: NPINTS, xadjs: NPINTS, /, *, num_hops: int) -> NPINTS:
    R"""
    Brutal-force compution for shortest distance between all pairs of given nodes within given number of hops.

    Args
    ----
    - n
        Number of nodes.
    - adjs
        Adjacency list.
    - xadjs
        Rejecting adjacency list.
    - num_hops
        Number of hops.

    Returns
    -------
    - shortests
        All-pair shortest distances.
    """
    # We only care about shortest distances within going down in the computation tree.
    # Thus, shortest distance longer than number of hops should be filtered by an arbitrary constant.
    # Since we are working with knowledge graph, we need to filter edges for uniqueness first.
    matrix = onp.zeros((n, n), dtype=onp.int64)
    matrix[adjs[0], adjs[1]] = 1
    matrix[xadjs[0], xadjs[1]] = 0
    shortests = osparse.csgraph.shortest_path(csgraph=matrix, directed=True)
    shortests[shortests > num_hops] = num_hops + 1
    shortests = shortests.astype(onp.int64)
    return cast(NPINTS, shortests)


def computation_tree(
    nodes: NPINTS,
    adjs: NPINTS,
    rels: NPINTS,
    xadjs: NPINTS,
    /,
    *,
    bidirect: bool,
    num_relations: int,
    num_nodes: int,
    num_hops: int,
) -> Tuple[NPINTS, NPINTS, NPINTS, NPINTS]:
    R"""
    Brutal-force sampling computation tree centering at given centers.

    Args
    ----
    - nodes
        Centering nodes.
    - adjs
        Adjacency matrix.
    - rels
        Relations.
    - xadjs
        Rejecting adjacency list.
        Pay attention that in this application, rejecting adjacency list will never be bidirected.
    - bidirect
        Automatically generate inversed relations in sampling.
        We assume that all given triplet are single direction, and their inversions do not exist.
    - num_relations
        Number of total relations assuming no inversion.
    - num_nodes
        Number of total nodes.
    - num_hops
        Number of hops.

    Returns
    -------
    - uids
        Full graph node IDs in reduced computation tree.
    - vids
        Reduced computation tree node IDs in full graph.
    - adjs
        Reduced computation tree adjacency list.
    - rels
        Reduced computation tree relations.
    """
    # Get adjacency matrix for brutal-force computation.
    matrix = onp.zeros((num_nodes, num_nodes), dtype=onp.int64)
    matrix[adjs[0], adjs[1]] = 1
    if bidirect:
        #
        matrix[adjs[1], adjs[0]] = 1
    matrix[xadjs[0], xadjs[1]] = 0

    # Get edge ID in full graph.
    eids = adjs[0] * num_nodes + adjs[1]

    #
    reduce_row = onp.full((num_nodes,), -1, dtype=onp.int64)
    reduce_col = onp.full((num_nodes,), -1, dtype=onp.int64)
    uids = onp.full((num_hops + 1, num_nodes), -1, dtype=onp.int64)
    buf_vid = []
    buf_adj = []
    buf_rel = []

    #
    total = 0
    starts = onp.unique(nodes)
    buf_vid.append(starts)
    for l in range(num_hops):
        # Update destination node ID reduction for previous hop layer.
        reduce_col.fill(-1)
        onp.put(reduce_col, starts, onp.arange(len(starts), dtype=onp.int64) + total)
        uids[l] = reduce_col

        #
        (rows, cols) = onp.nonzero(matrix[:, starts])
        total += len(starts)
        ends = onp.unique(rows)

        # Update source node ID reduction for current hop layer.
        reduce_row.fill(-1)
        onp.put(reduce_row, ends, onp.arange(len(ends), dtype=onp.int64) + total)
        uids[l + 1] = reduce_row

        # Get edges that are active at current layer.
        # Pay attention that destination nodes are already reduced for current layer, thus we need to recover them.
        masks = onp.isin(eids, rows * num_nodes + starts[cols])

        # Update buffer.
        buf_vid.append(ends)
        buf_adj.append(onp.stack((reduce_row[adjs[0][masks]], reduce_col[adjs[1][masks]])))
        buf_rel.append(rels[masks])

        #
        starts = ends

    #
    vids = onp.concatenate(buf_vid)
    adjs = onp.concatenate(buf_adj, axis=1)
    rels = onp.concatenate(buf_rel)

    #
    return (uids, vids, adjs, rels)
