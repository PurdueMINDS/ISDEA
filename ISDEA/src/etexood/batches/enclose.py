#
import numpy as onp
import time
import multiprocessing as mp
import os
import math
import itertools
from typing import TypeVar, Tuple, Sequence, cast
from multiprocessing.shared_memory import SharedMemory
from ..dtypes import NPINTS
from .hop import shortest


#
SelfEnclose = TypeVar("SelfEnclose", bound="Enclose")


def load(pairs: NPINTS, num_nodes: int, directory: str, /) -> Tuple[NPINTS, NPINTS, NPINTS, NPINTS, NPINTS, NPINTS]:
    R"""
    Load enclosed subgraphs.

    Args
    ----
    - adjs
        Collecting node pairs.
    - num_nodes
        Number of nodes.
    - directory
        Loading directory.

    Returns
    -------
    - vpts
        Node data boundaries of each subgraph in merged graph.
    - epts
        Edge data boundaries of each subgraph in merged graph.
    - vids
        Node IDs of merged graph in full graph.
    - vfts
        Node features of merged graph.
    - adjs
        Adjacency list of merged graph.
    - rels
        Relations of merged graph.
    """
    #
    maxlen = len(str(num_nodes**2 - 1))
    buf_vid = []
    buf_vft = []
    buf_adj = []
    buf_rel = []
    bias = 0
    for (src, dst) in pairs.T.tolist():
        #
        eid = src * num_nodes + dst
        with open(os.path.join(directory, "{:>0{:d}d}.npy".format(eid, maxlen)), "rb") as file:
            #
            vids = onp.load(file)
            vfts = onp.load(file)
            adjs = onp.load(file)
            rels = onp.load(file)

            #
            buf_vid.append(vids)
            buf_vft.append(vfts)
            buf_adj.append(adjs)
            buf_rel.append(rels)
            bias += len(vids)

    #
    vpts = onp.array([0] + list(itertools.accumulate(len(vfts) for vfts in buf_vft)))
    epts = onp.array([0] + list(itertools.accumulate(len(rels) for rels in buf_rel)))
    vids = onp.concatenate(buf_vid)
    vfts = onp.concatenate(buf_vft)
    adjs = onp.concatenate(buf_adj, axis=1)
    rels = onp.concatenate(buf_rel)
    return (vpts, epts, vids, vfts, adjs, rels)


class Enclose(object):
    R"""
    Enclose subgraph generator.
    It will utilize heuristics cache.
    """
    #

    def __init__(
        self: SelfEnclose,
        cache: str,
        n: int,
        adjs: NPINTS,
        rels: NPINTS,
        /,
        *,
        num_hops: int,
        num_processes: int,
        unit: float,
    ) -> None:
        R"""
        Reset.

        Args
        ----
        - cache
            Cache directory.
        - n
            Number of nodes.
        - adjs
            Adjecency list.
        - rels
            Relations.
        - num_hops
            Number of hops.
        - num_processes
            Number of processes for collection.
        - unit
            Time unit (second) to report progress per process.

        Returns
        -------
        """
        # Cache directory should be manully created to avoid duplicates.
        assert os.path.isdir(cache)

        #
        self._cache = cache
        self._num_nodes = n
        self._adjs = adjs
        self._rels = rels
        self._num_hops = num_hops
        self._num_processes = num_processes
        self._unit = unit

    def translate(self: SelfEnclose, /) -> None:
        R"""
        Translate heuristics reachability masks into enclosed subgraphs.

        Args
        ----

        Returns
        -------
        """
        #
        directory_heuristics = os.path.join(self._cache, "hop{:d}.forest".format(self._num_hops))
        buf = []
        maxlen = 0
        for filename in os.listdir(directory_heuristics):
            #
            (filename, extension) = os.path.splitext(filename)
            assert extension == ".npy", extension
            buf.append(int(filename))
            maxlen = len(filename)

        #
        directory_enclose = os.path.join(self._cache, "hop{:d}.enclose".format(self._num_hops))
        os.makedirs(directory_enclose, exist_ok=True)
        buf = [
            eid
            for eid in buf
            if not os.path.isfile(
                os.path.join(directory_enclose, "{:>0{:d}d}.npy".format(eid, maxlen)),
            )
        ]
        pairs = onp.unique(onp.array(buf))
        pairs.setflags(write=False)
        if pairs.nbytes == 0:
            #
            return

        #
        num_pairs = len(pairs)
        num_processes = min(self._num_processes, num_pairs)

        # Initialization.
        (
            (shadjs, shape_shadjs, dtype_shadjs),
            (shrels, shape_shrels, dtype_shrels),
            (shpairs, shape_shpairs, dtype_shpairs),
        ) = self.translate_ini(self._adjs, self._rels, pairs)

        #
        if num_processes > 1:
            #
            print("Generate {:d} enclosed subgraphs by {:d} processes.".format(num_pairs, num_processes))

            # Multiprocessing.
            num_pairs_per_process = int(math.ceil(float(num_pairs) / float(num_processes)))
            jobs = [
                (
                    directory_heuristics,
                    i * num_pairs_per_process,
                    min((i + 1) * num_pairs_per_process, num_pairs),
                    self._num_nodes,
                    (shadjs.name, shape_shadjs, dtype_shadjs),
                    (shrels.name, shape_shrels, dtype_shrels),
                    (shpairs.name, shape_shpairs, dtype_shpairs),
                    self._num_hops,
                    directory_enclose,
                    maxlen,
                    "{:>0{:d}d}".format(i, len(str(num_processes - 1))),
                    self._unit,
                )
                for i in range(num_processes)
            ]
            with mp.Pool(num_processes) as pool:
                #
                pool.starmap(self.translate_work, jobs)
        else:
            #
            print("Generate {:d} enclosed subgraphs.".format(num_pairs))

            # Call the process kernel directly.
            self.translate_work(
                directory_heuristics,
                0,
                num_pairs,
                self._num_nodes,
                (shadjs.name, shape_shadjs, dtype_shadjs),
                (shrels.name, shape_shrels, dtype_shrels),
                (shpairs.name, shape_shpairs, dtype_shpairs),
                self._num_hops,
                directory_enclose,
                maxlen,
                "-",
                self._unit,
            )

        # Finalization.
        self.translate_fin(shadjs, shrels, shpairs)

    @staticmethod
    def translate_ini(
        adjs: NPINTS,
        rels: NPINTS,
        pairs: NPINTS,
        /,
    ) -> Tuple[
        Tuple[SharedMemory, Sequence[int], type],
        Tuple[SharedMemory, Sequence[int], type],
        Tuple[SharedMemory, Sequence[int], type],
    ]:
        R"""
        Initialization for translation.

        Args
        ----
        - adjs
            Adjacency list.
        - rels
            Relations.
        - pairs
            Collecting pair IDs.

        Returns
        -------
        - shmadjs
            Sharing memory descriptor of adjacency list.
        - shmrels
            Sharing memory descriptor of relations.
        - shmpairs
            Sharing memory descriptor of collecting pair IDs.
        """
        #
        assert not adjs.flags.writeable and not rels.flags.writeable and not pairs.flags.writeable
        print("Sharing adjacency list, relations and collecting pair IDs.")

        #
        shadjs = SharedMemory(create=True, size=adjs.nbytes)
        shrels = SharedMemory(create=True, size=rels.nbytes)
        shpairs = SharedMemory(create=True, size=pairs.nbytes)

        # Sharing memory interface as NumPy array.
        onp.copyto(onp.ndarray(adjs.shape, dtype=adjs.dtype, buffer=shadjs.buf), adjs)
        onp.copyto(onp.ndarray(rels.shape, dtype=rels.dtype, buffer=shrels.buf), rels)
        onp.copyto(onp.ndarray(pairs.shape, dtype=pairs.dtype, buffer=shpairs.buf), pairs)

        #
        return (
            (shadjs, adjs.shape, cast(type, adjs.dtype)),
            (shrels, rels.shape, cast(type, rels.dtype)),
            (shpairs, pairs.shape, cast(type, pairs.dtype)),
        )

    @staticmethod
    def translate_fin(shadjs: SharedMemory, shrels: SharedMemory, shpairs: SharedMemory, /) -> None:
        R"""
        Finalization for translation.

        Args
        ----
        - shadjs
            Shared adjacency list.
        - shrels
            Shared of relations.
        - shpairs
            Shared collecting pair IDs.

        Returns
        -------
        """
        #
        print("Releasing adjacency list, relations and shared collecting pair IDs.")

        #
        shadjs.close()
        shrels.close()
        shpairs.close()

        #
        shadjs.unlink()
        shrels.unlink()
        shpairs.unlink()

    @staticmethod
    def translate_work(
        directory_heuristics: str,
        bgn: int,
        end: int,
        num_nodes: int,
        shmadjs: Tuple[str, Sequence[int], type],
        shmrels: Tuple[str, Sequence[int], type],
        shmpairs: Tuple[str, Sequence[int], type],
        num_hops: int,
        directory_enclose: str,
        maxlen: int,
        title: str,
        unit: float,
        /,
    ) -> None:
        R"""
        Worker for translation.

        Args
        ----
        - directory_heuriscstics
            Saving directory for heuristics.
        - bgn
            Beginning node of current worker.
        - end
            Ending node of current worker.
        - num_nodes
            Number of nodes.
        - shmadjs
            Sharing memory descriptor of adjacency list.
        - shmrels
            Sharing memory descriptor of relations.
        - shmnodes
            Sharing memory descriptor of generating nodes.
        - num_hops
            Number of hops.
        - directory_enclose
            Load directory for enclosed subgraph.
        - maxlen
            Maximum length of filename.
        - title
            Title of worker.
        - unit
            Time unit (second) to report progress.

        Returns
        -------
        """
        #
        adjs: NPINTS
        rels: NPINTS
        pairs: NPINTS

        # Reconstruct adjacency list and relations.
        (url_adjs, shape_adjs, dtype_adjs) = shmadjs
        (url_rels, shape_rels, dtype_rels) = shmrels
        (url_pairs, shape_pairs, dtype_pairs) = shmpairs
        shadjs = SharedMemory(name=url_adjs)
        shrels = SharedMemory(name=url_rels)
        shpairs = SharedMemory(name=url_pairs)
        adjs = onp.ndarray(shape_adjs, dtype=dtype_adjs, buffer=shadjs.buf)
        rels = onp.ndarray(shape_rels, dtype=dtype_rels, buffer=shrels.buf)
        pairs = onp.ndarray(shape_pairs, dtype=dtype_pairs, buffer=shpairs.buf)
        adjs.setflags(write=False)
        rels.setflags(write=False)
        pairs.setflags(write=False)

        #
        Enclose.translate_work_(
            directory_heuristics,
            bgn,
            end,
            num_nodes,
            adjs,
            rels,
            pairs,
            num_hops,
            directory_enclose,
            maxlen,
            title,
            unit,
        )

        #
        shadjs.close()
        shrels.close()
        shpairs.close()

    @staticmethod
    def translate_work_(
        directory_heuristics: str,
        bgn: int,
        end: int,
        num_nodes: int,
        adjs_full: NPINTS,
        rels_full: NPINTS,
        pairs: NPINTS,
        num_hops: int,
        directory_enclose: str,
        maxlen: int,
        title: str,
        unit: float,
        /,
    ) -> None:
        R"""
        Worker for translation.

        Args
        ----
        - directory_heuriscstics
            Saving directory for heuristics.
        - bgn
            Beginning pair ID of current worker.
        - end
            Ending pair ID of current worker.
        - num_nodes
            Number of nodes.
        - adjs_full
            Full adjacency list.
        - rels_full
            Full relations.
        - pairs
            IDs of collecting node pairs.
        - num_hops
            Number of hops.
        - directory_enclose
            Load directory for enclosed subgraph.
        - maxlen
            Maximum length of filename.
        - title
            Title of worker.
        - unit
            Time unit (second) to report progress.

        Returns
        -------
        """
        #
        pid = os.getpid()
        print('Process "{:s}" ({:d}) is dealing with pairs [{:d}, {:d}).'.format(title, pid, bgn, end))

        # Allocate reusable memory.
        masks_src = onp.ones((num_hops + 2, num_nodes), dtype=onp.bool_)
        masks_dst = onp.ones((num_hops + 2, num_nodes), dtype=onp.bool_)
        masks = onp.zeros((num_hops + 2, num_nodes), dtype=onp.bool_)
        masks_node_active = onp.zeros((num_nodes,), dtype=onp.bool_)
        masks_edge_active = onp.zeros((len(rels_full),), dtype=onp.bool_)
        uids = onp.zeros((num_nodes,), dtype=onp.int64)

        #
        elapsed = time.time()
        for i in range(bgn, end):
            #
            if time.time() - elapsed > unit or i == bgn:
                #
                print('Process "{:s}" ({:d}) has done with pairs [{:d}, {:d}/{:d}).'.format(title, pid, bgn, i, end))
                elapsed = time.time()

            # Load heuristics cached reachability.
            path = os.path.join(directory_heuristics, "{:>0{:d}d}.npy".format(pairs[i].item(), maxlen))
            with open(path, "rb") as file:
                #
                (masks_src_, masks_dst_) = onp.load(file)
            masks_src[:-1] = masks_src_
            masks_dst[:-1] = masks_dst_

            # Ensure safety.
            (src_,) = onp.nonzero(masks_src[0])
            (dst_,) = onp.nonzero(masks_dst[0])
            (src, dst) = (src_.item(), dst_.item())
            assert pairs[i].item() == src * num_nodes + dst

            # Collect active nodes and active edges.
            onp.logical_or(masks_src, masks_dst, out=masks)
            masks_node_active.fill(False)
            masks_edge_active.fill(False)
            onp.any(masks[:-1], axis=0, out=masks_node_active)
            onp.logical_and(masks_node_active[adjs_full[0]], masks_node_active[adjs_full[1]], out=masks_edge_active)

            # Remove current pair itself from active edges.
            onp.logical_and(
                masks_edge_active,
                onp.logical_not(
                    onp.logical_or(
                        onp.logical_and(adjs_full[0] == src, adjs_full[1] == dst),
                        onp.logical_and(adjs_full[0] == dst, adjs_full[1] == src),
                    ),
                ),
                out=masks_edge_active,
            )

            # Collect DRNL (node labeling) as node features.
            (nodes,) = onp.nonzero(masks_node_active)
            dists_src = onp.argmax(masks_src, axis=0)[nodes]
            dists_dst = onp.argmax(masks_dst, axis=0)[nodes]
            assert onp.all(onp.minimum(dists_src, dists_dst) <= num_hops).item()

            # Reduce graph.
            # Pay attention that this subgraph can be an empty graph.
            uids.fill(-1)
            uids[nodes] = onp.arange(len(nodes))
            vfts = onp.stack((dists_src, dists_dst), axis=1)
            adjs = uids[adjs_full[:, masks_edge_active]]
            rels = rels_full[masks_edge_active]
            assert vfts.ndim == 2 and vfts.shape[1] == 2
            assert adjs.ndim == 2 and adjs.shape[0] == 2
            assert rels.ndim == 1
            assert onp.all(adjs >= 0).item() and onp.all(adjs < len(nodes)).item()

            #
            path = os.path.join(directory_enclose, "{:>0{:d}d}.npy".format(pairs[i].item(), maxlen))
            with open(path, "wb") as file:
                #
                onp.save(file, nodes)
                onp.save(file, vfts)
                onp.save(file, adjs)
                onp.save(file, rels)
            with open(path, "rb") as file:
                #
                onp.load(file)
                onp.load(file)
                onp.load(file)
                onp.load(file)

        #
        print('Process "{:s}" ({:d}) is done with pairs [{:d}, {:d}).'.format(title, pid, bgn, end))

    def load(self: SelfEnclose, pairs: NPINTS, /) -> Tuple[NPINTS, NPINTS, NPINTS, NPINTS, NPINTS, NPINTS]:
        R"""
        Load heuristics for given node pairs.

        Args
        ----
        - pairs
            Node pairs to collect enclosed subgraphs.

        Returns
        -------
        - vpts
            Node data boundaries of each subgraph in merged graph.
        - epts
            Edge data boundaries of each subgraph in merged graph.
        - vids
            Node Ids of merged graph in full graph.
        - vfts
            Node features of merged graph.
        - adjs
            Adjacency list of merged graph.
        - rels
            Relations of merged graph.
        """
        #
        return load(pairs, self._num_nodes, os.path.join(self._cache, "hop{:d}.enclose".format(self._num_hops)))


def enclose(
    num_nodes: int,
    adjs: NPINTS,
    rels: NPINTS,
    src: int,
    dst: int,
    /,
    *,
    num_hops: int,
) -> Tuple[NPINTS, NPINTS, NPINTS, NPINTS]:
    R"""
    Brutal-force enclosed subgraph sampling.

    Args
    ----
    - num_nodes
        Number of nodes
    - adjs
        Adjacency list
    - rels
        Relations.
    - src
        Source node of enclosed subgraph sampling.
    - dst
        Destination node of enclose subgraph sampling.
    - num_hops
        Number of hops.

    Returns
    -------
    - vids
        Node IDs of enclosed subgraph in full graph.
    - vfts
        Node features of enclosed subgraph.
    - adjs
        Adjacency list of enclosed subgraph.
    - rels
        Relations of enclosed subgraph.
    """
    #
    xadjs = onp.array([(src, dst), (dst, src)]).T
    shortests = shortest(num_nodes, adjs, xadjs, num_hops=num_hops)
    shortests_src = shortests[:, src]
    shortests_dst = shortests[:, dst]
    (nodes_src,) = onp.nonzero(shortests_src <= num_hops)
    (nodes_dst,) = onp.nonzero(shortests_dst <= num_hops)
    nodes = onp.unique(onp.concatenate((nodes_src, nodes_dst)))

    #
    masks = onp.logical_and(
        onp.logical_and(onp.isin(adjs[0], nodes), onp.isin(adjs[1], nodes)),
        onp.logical_not(
            onp.logical_or(
                onp.logical_and(adjs[0] == src, adjs[1] == dst),
                onp.logical_and(adjs[0] == dst, adjs[1] == src),
            ),
        ),
    )

    #
    uids = onp.full((num_nodes,), -1, dtype=onp.int64)
    uids[nodes] = onp.arange(len(nodes))
    vfts = onp.stack((shortests_src[nodes], shortests_dst[nodes]), axis=1)
    adjs = uids[adjs[:, masks]]
    rels = rels[masks]
    return (nodes, vfts, adjs, rels)
