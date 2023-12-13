#
import numpy as onp
import time
import multiprocessing as mp
import os
import math
from typing import TypeVar, Tuple, Sequence, cast
from multiprocessing.shared_memory import SharedMemory
from .hop import ComputationSubsetNode
from ..dtypes import NPINTS, NPBOOLS


#
SelfHeuristicsForest0 = TypeVar("SelfHeuristicsForest0", bound="HeuristicsForest0")
SelfHeuristicsForest1 = TypeVar("SelfHeuristicsForest1", bound="HeuristicsForest1")


def collect(src: int, dst: int, masks_src: NPBOOLS, masks_dst: NPBOOLS, /) -> NPINTS:
    R"""
    Collect heuristics from reachable masks.

    Args
    ----
    - src
        Source node.
    - dst
        Destination node.
    - masks_src
        Reachable masks from source node.
    - masks_dst
        Reachable masks from destination node.

    Returns
    -------
    - heuristics
        Heuristics.
    """
    #
    assert onp.all(masks_src[-1]).item() and onp.all(masks_dst[-1]).item()
    return onp.array(
        [
            onp.argmax(masks_src[:, dst]).item(),
            onp.argmax(masks_dst[:, src]).item(),
            onp.sum(onp.logical_and(masks_src[1], masks_dst[1])).item(),
            onp.sum(onp.logical_or(masks_src[1], masks_dst[1])).item(),
        ],
    )


def load(pairs: NPINTS, num_nodes: int, directory: str, /) -> NPINTS:
    R"""
    Load heuristics.

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
    - heuristics
        Heuristics.
    """
    #
    maxlen = len(str(num_nodes**2 - 1))
    buf = []
    for src, dst in pairs.T.tolist():
        #
        eid = src * num_nodes + dst
        with open(os.path.join(directory, "{:>0{:d}d}.npy".format(eid, maxlen)), "rb") as file:
            #
            buf.append(onp.load(file))
    return onp.stack(buf)


class HeuristicsForest0(object):
    R"""
    Heuristics generator.
    This version will fix the observed graph on collection.
    """
    #
    NUM_HEURISTICS = 4

    def __init__(
        self: SelfHeuristicsForest0,
        cache: str,
        n: int,
        adjs: NPINTS,
        rels: NPINTS,
        /,
        *,
        num_hops: int,
        num_processes: int,
        unit: float,
        pesudo: bool,
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
        - pseudo
            If True, no collection will be performed even if the collection is explicitly called.

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
        self._pseudo = pesudo

        #
        # assert self._num_hops > 1, "Number of hops should be greater than 1 for heuristics to be useful."

    def forest(self: SelfHeuristicsForest0, nodes: NPINTS, /) -> None:
        R"""
        Generate node reachability at each layer of hop for all nodes.

        Args
        ----
        - nodes
            Nodes for generation.

        Returns
        -------
        """
        #
        if self._pseudo:
            #
            print("-" * 80)
            print("Node forests are skipped.")
            print("-" * 80)
            return

        # Masks of reachable nodes at each layer of hop for all nodes.
        # Node pairs are independetly stored under the same directory for memory efficient loading.
        directory = os.path.join(self._cache, "hop{:d}.forest".format(self._num_hops))
        os.makedirs(directory, exist_ok=True)

        # Ensure uniqueness.
        nodes = onp.unique(nodes)

        #
        maxlen = len(str(self._num_nodes - 1))
        masks = onp.array(
            [not os.path.isfile(os.path.join(directory, "{:>0{:d}d}.npy".format(v, maxlen))) for v in nodes.tolist()],
        )

        # Get pairs to be generated.
        nodes = nodes[masks]
        nodes.setflags(write=False)
        if nodes.nbytes == 0:
            #
            return

        # Logging from here may involve multiprocessing, thus we use naive print for logging.
        num_items = len(nodes)
        num_processes = min(self._num_processes, num_items)

        # Initialization.
        (
            (shadjs, shape_shadjs, dtype_shadjs),
            (shrels, shape_shrels, dtype_shrels),
            (shnodes, shape_shnodes, dtype_shnodes),
        ) = self.forest_ini(self._adjs, self._rels, nodes)

        #
        if num_processes > 1:
            #
            print("-" * 80)
            print("Generate {:d} node forests by {:d} processes.".format(num_items, num_processes))
            print("-" * 80)

            # Multiprocessing.
            num_nodes_per_process = int(math.ceil(float(num_items) / float(num_processes)))
            jobs = [
                (
                    directory,
                    i * num_nodes_per_process,
                    min((i + 1) * num_nodes_per_process, num_items),
                    self._num_nodes,
                    (shadjs.name, shape_shadjs, dtype_shadjs),
                    (shrels.name, shape_shrels, dtype_shrels),
                    (shnodes.name, shape_shnodes, dtype_shnodes),
                    self._num_hops,
                    "{:>0{:d}d}".format(i, len(str(num_processes - 1))),
                    self._unit,
                )
                for i in range(num_processes)
            ]
            with mp.Pool(num_processes) as pool:
                #
                pool.starmap(self.forest_work, jobs)
        else:
            #
            print("-" * 80)
            print("Generate {:d} node forests.".format(num_items))
            print("-" * 80)

            # Call the process kernel directly.
            self.forest_work(
                directory,
                0,
                num_items,
                self._num_nodes,
                (shadjs.name, shape_shadjs, dtype_shadjs),
                (shrels.name, shape_shrels, dtype_shrels),
                (shnodes.name, shape_shnodes, dtype_shnodes),
                self._num_hops,
                "-",
                self._unit,
            )

        # Finalization.
        self.forest_fin(shadjs, shrels, shnodes)
        print("-" * 80)
        print("Node forests are collected.")
        print("-" * 80)

    @staticmethod
    def forest_ini(
        adjs: NPINTS,
        rels: NPINTS,
        nodes: NPINTS,
        /,
    ) -> Tuple[
        Tuple[SharedMemory, Sequence[int], type],
        Tuple[SharedMemory, Sequence[int], type],
        Tuple[SharedMemory, Sequence[int], type],
    ]:
        R"""
        Initialization for forest generation.

        Args
        ----
        - adjs
            Adjacency list.
        - rels
            Relations.
        - nodes
            Generating nodes.

        Returns
        -------
        - shmadjs
            Sharing memory descriptor of adjacency list.
        - shmrels
            Sharing memory descriptor of relations.
        - shmnodes
            Sharing memory descriptor of generating nodes.
        """
        #
        assert not adjs.flags.writeable and not rels.flags.writeable and not nodes.flags.writeable
        print("Sharing adjacency list, relations and generating nodes.")

        #
        shadjs = SharedMemory(create=True, size=adjs.nbytes)
        shrels = SharedMemory(create=True, size=rels.nbytes)
        shnodes = SharedMemory(create=True, size=nodes.nbytes)

        # Sharing memory interface as NumPy array.
        onp.copyto(onp.ndarray(adjs.shape, dtype=adjs.dtype, buffer=shadjs.buf), adjs)
        onp.copyto(onp.ndarray(rels.shape, dtype=rels.dtype, buffer=shrels.buf), rels)
        onp.copyto(onp.ndarray(nodes.shape, dtype=nodes.dtype, buffer=shnodes.buf), nodes)

        #
        return (
            (shadjs, adjs.shape, cast(type, adjs.dtype)),
            (shrels, rels.shape, cast(type, rels.dtype)),
            (shnodes, nodes.shape, cast(type, nodes.dtype)),
        )

    @staticmethod
    def forest_fin(shadjs: SharedMemory, shrels: SharedMemory, shnodes: SharedMemory, /) -> None:
        R"""
        Finalization for forest generation.

        Args
        ----
        - shadjs
            Shared adjacency list.
        - shrels
            Shared relations.
        - shnodes
            Shared gnerating nodes.

        Returns
        -------
        """
        #
        print("Releasing shared adjacency list, relations and generating nodes.")

        #
        shadjs.close()
        shrels.close()
        shnodes.close()

        #
        shadjs.unlink()
        shrels.unlink()
        shnodes.unlink()

    @staticmethod
    def forest_work(
        directory: str,
        bgn: int,
        end: int,
        num_nodes: int,
        shmadjs: Tuple[str, Sequence[int], type],
        shmrels: Tuple[str, Sequence[int], type],
        shmnodes: Tuple[str, Sequence[int], type],
        num_hops: int,
        title: str,
        unit: float,
        /,
    ) -> None:
        R"""
        Worker for forest generation.

        Args
        ----
        - directory
            Saving directory.
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
        nodes: NPINTS

        # Reconstruct adjacency list and relations.
        (url_adjs, shape_adjs, dtype_adjs) = shmadjs
        (url_rels, shape_rels, dtype_rels) = shmrels
        (url_nodes, shape_nodes, dtype_nodes) = shmnodes
        shadjs = SharedMemory(name=url_adjs)
        shrels = SharedMemory(name=url_rels)
        shnodes = SharedMemory(name=url_nodes)
        adjs = onp.ndarray(shape_adjs, dtype=dtype_adjs, buffer=shadjs.buf)
        rels = onp.ndarray(shape_rels, dtype=dtype_rels, buffer=shrels.buf)
        nodes = onp.ndarray(shape_nodes, dtype=dtype_nodes, buffer=shnodes.buf)
        adjs.setflags(write=False)
        rels.setflags(write=False)
        nodes.setflags(write=False)

        #
        HeuristicsForest0.forest_work_(directory, bgn, end, num_nodes, adjs, rels, nodes, num_hops, title, unit)

        #
        shadjs.close()
        shrels.close()
        shnodes.close()

    @staticmethod
    def forest_work_(
        directory: str,
        bgn: int,
        end: int,
        num_nodes: int,
        adjs: NPINTS,
        rels: NPINTS,
        nodes: NPINTS,
        num_hops: int,
        title: str,
        unit: float,
        /,
    ) -> None:
        R"""
        Worker for forest generation.

        Args
        ----
        - directory
            Saving directory.
        - bgn
            Beginning node of current worker.
        - end
            Ending node of current worker.
        - num_nodes
            Number of nodes.
        - adjs
            Adjacency list.
        - rels
            Relations.
        - nodes
            Generating nodes.
        - num_hops
            Number of hops.
        - title
            Title of worker.
        - unit
            Time unit (second) to report progress.

        Returns
        -------
        """
        #
        pid = os.getpid()
        print('Process "{:s}" ({:d}) is dealing with nodes [{:d}, {:d}).'.format(title, pid, bgn, end))

        #
        maxlen = len(str(num_nodes - 1))
        xadjs = onp.zeros((2, 0), dtype=adjs.dtype)

        #
        elapsed = time.time()
        elapsed_start = elapsed
        tree = ComputationSubsetNode(num_nodes, adjs, rels, num_hops=num_hops)
        for i in range(bgn, end):
            #
            if time.time() - elapsed > unit or i == bgn:
                #
                print(
                    'Process "{:s}" ({:d}) has done with nodes [{:d}|{:d}|{:d}) in {:d} secs.'.format(
                        title,
                        pid,
                        bgn,
                        i,
                        end,
                        int(math.ceil(time.time() - elapsed_start)),
                    ),
                )
                elapsed = time.time()

            #
            tree.reset()
            tree.sample("tree", onp.array([nodes[i]]), tree.masks_edge_accept(xadjs))

            #
            masks = tree.masks_node_active
            assert masks.size > 0
            with open(os.path.join(directory, "{:>0{:d}d}.npy".format(nodes[i], maxlen)), "wb") as file:
                #
                onp.save(file, masks)
            with open(os.path.join(directory, "{:>0{:d}d}.npy".format(nodes[i], maxlen)), "rb") as file:
                #
                onp.load(file)

        #
        print('Process "{:s}" ({:d}) is done with nodes [{:d}, {:d}).'.format(title, pid, bgn, end))

    def collect(self: SelfHeuristicsForest0, pairs: NPINTS, /) -> None:
        R"""
        Collect heuristics for given node pairs.

        Args
        ----
        - pairs
            Node pairs to collect heuristics.

        Returns
        -------
        - heuristics
            Collected heuristics.
        """
        #
        if self._pseudo:
            #
            print("-" * 80)
            print("Heuristics are skipped.")
            print("-" * 80)
            return

        # Ensure uniqueness.
        eids = onp.unique(pairs[0] * self._num_nodes + pairs[1])
        pairs = onp.stack((eids // self._num_nodes, eids % self._num_nodes))

        # Heuristics are independetly stored under the same directory for memory efficient loading.
        directory_heuristics = os.path.join(self._cache, "hop{:d}.heuristics".format(self._num_hops))
        os.makedirs(directory_heuristics, exist_ok=True)
        maxlen = len(str(self._num_nodes**2 - 1))
        masks = onp.array(
            [
                not os.path.isfile(
                    os.path.join(directory_heuristics, "{:>0{:d}d}.npy".format(src * self._num_nodes + dst, maxlen)),
                )
                for (src, dst) in pairs.T.tolist()
            ],
        )
        pairs = pairs[:, masks]
        pairs.setflags(write=False)
        if pairs.nbytes == 0:
            #
            return

        #
        (_, num_pairs) = pairs.shape
        num_processes = min(self._num_processes, num_pairs)
        directory_forest = os.path.join(self._cache, "hop{:d}.forest".format(self._num_hops))
        os.makedirs(directory_forest, exist_ok=True)

        # Initialization.
        (shpairs, shape_shpairs, dtype_shpairs) = self.collect_ini(pairs)

        # Logging from here may involve multiprocessing, thus we use naive print for logging.
        if num_processes > 1:
            #
            print("-" * 80)
            print("Collect {:d} heuristics by {:d} processes.".format(num_pairs, num_processes))
            print("-" * 80)

            # Multiprocessing.
            num_pairs_per_process = int(math.ceil(float(num_pairs) / float(num_processes)))
            jobs = [
                (
                    directory_heuristics,
                    i * num_pairs_per_process,
                    min((i + 1) * num_pairs_per_process, num_pairs),
                    self._num_nodes,
                    (shpairs.name, shape_shpairs, dtype_shpairs),
                    directory_forest,
                    "{:>0{:d}d}".format(i, len(str(num_processes - 1))),
                    self._unit,
                )
                for i in range(num_processes)
            ]
            with mp.Pool(num_processes) as pool:
                #
                pool.starmap(self.collect_work, jobs)
        else:
            #
            print("-" * 80)
            print("Collect {:d} heuristics.".format(num_pairs))
            print("-" * 80)

            # Call the process kernel directly.
            self.collect_work(
                directory_heuristics,
                0,
                num_pairs,
                self._num_nodes,
                (shpairs.name, shape_shpairs, dtype_shpairs),
                directory_forest,
                "-",
                self._unit,
            )

        # Finalization.
        self.collect_fin(shpairs)
        print("-" * 80)
        print("Heuristics are collected.".format(num_pairs))
        print("-" * 80)

    @staticmethod
    def collect_ini(pairs: NPINTS, /) -> Tuple[SharedMemory, Sequence[int], type]:
        R"""
        Initialization for heuristics collection.

        Args
        ----
        - pairs
            Collecting pairs.

        Returns
        -------
        - shmpairs
            Sharing memory descriptor of collecting pairs.
        """
        #
        assert not pairs.flags.writeable
        print("Sharing collecting pairs.")

        #
        shpairs = SharedMemory(create=True, size=pairs.nbytes)

        # Sharing memory interface as NumPy array.
        onp.copyto(onp.ndarray(pairs.shape, dtype=pairs.dtype, buffer=shpairs.buf), pairs)

        #
        return (shpairs, pairs.shape, cast(type, pairs.dtype))

    @staticmethod
    def collect_fin(shpairs: SharedMemory, /) -> None:
        R"""
        Finalization for heurisctics collection.

        Args
        ----
        - shpairs
            Shared collecting pairs.

        Returns
        -------
        """
        #
        print("Releasing shared collecting pairs.")

        #
        shpairs.close()

        #
        shpairs.unlink()

    @staticmethod
    def collect_work(
        directory_heuristics: str,
        bgn: int,
        end: int,
        num_nodes: int,
        shmpairs: Tuple[str, Sequence[int], type],
        directory_forest: str,
        title: str,
        unit: float,
        /,
    ) -> None:
        R"""
        Worker for heuristics collection.

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
        - shmpairs
            Sharing memory descriptor of collecting pairs.
        - directory_forest
            Load directory for reachable masks.
        - title
            Title of worker.
        - unit
            Time unit (second) to report progress.

        Returns
        -------
        """
        #
        pairs: NPINTS

        # Reconstruct adjacency list and relations.
        (url_pairs, shape_pairs, dtype_pairs) = shmpairs
        shpairs = SharedMemory(name=url_pairs)
        pairs = onp.ndarray(shape_pairs, dtype=dtype_pairs, buffer=shpairs.buf)
        pairs.setflags(write=False)

        #
        HeuristicsForest0.collect_work_(
            directory_heuristics,
            bgn,
            end,
            num_nodes,
            pairs,
            directory_forest,
            title,
            unit,
        )

        #
        shpairs.close()

    @staticmethod
    def collect_work_(
        directory_heuristics: str,
        bgn: int,
        end: int,
        num_nodes: int,
        pairs: NPINTS,
        directory_forest: str,
        title: str,
        unit: float,
        /,
    ) -> None:
        R"""
        Worker for heuristics collection.

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
        - pairs
            Collecting node pairs.
        - directory_forest
            Load directory for reachable masks.
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

        #
        maxlen_node = len(str(num_nodes - 1))
        maxlen_pair = len(str(num_nodes**2 - 1))

        #
        elapsed = time.time()
        elapsed_start = elapsed
        for i in range(bgn, end):
            #
            if time.time() - elapsed > unit or i == bgn:
                #
                print(
                    'Process "{:s}" ({:d}) has done with pairs [{:d}|{:d}|{:d}) in {:d} secs.'.format(
                        title,
                        pid,
                        bgn,
                        i,
                        end,
                        int(math.ceil(time.time() - elapsed_start)),
                    ),
                )
                elapsed = time.time()

            #
            (src, dst) = pairs[:, i]
            with open(os.path.join(directory_forest, "{:>0{:d}d}.npy".format(src, maxlen_node)), "rb") as file:
                #
                masks_src = onp.load(file)
            with open(os.path.join(directory_forest, "{:>0{:d}d}.npy".format(dst, maxlen_node)), "rb") as file:
                #
                masks_dst = onp.load(file)
            masks_src = onp.concatenate((masks_src, onp.ones((1, num_nodes), dtype=onp.bool_)))
            masks_dst = onp.concatenate((masks_dst, onp.ones((1, num_nodes), dtype=onp.bool_)))

            #
            eid = src * num_nodes + dst
            heuristics = collect(src, dst, masks_src, masks_dst)
            assert heuristics.size > 0
            with open(os.path.join(directory_heuristics, "{:>0{:d}d}.npy".format(eid, maxlen_pair)), "wb") as file:
                #
                onp.save(file, heuristics)
            with open(os.path.join(directory_heuristics, "{:>0{:d}d}.npy".format(eid, maxlen_pair)), "rb") as file:
                #
                onp.load(file)

        #
        print('Process "{:s}" ({:d}) is done with pairs [{:d}, {:d}).'.format(title, pid, bgn, end))

    def load(self: SelfHeuristicsForest0, pairs: NPINTS, /) -> NPINTS:
        R"""
        Load heuristics for given node pairs.

        Args
        ----
        - pairs
            Node pairs to collect heuristics.

        Returns
        -------
        - heuristics
            Collected heuristics.
        """
        #
        if self._pseudo:
            #
            return onp.zeros((pairs.shape[1], 4), dtype=onp.int64)
        else:
            #
            return load(pairs, self._num_nodes, os.path.join(self._cache, "hop{:d}.heuristics".format(self._num_hops)))


class HeuristicsForest1(object):
    R"""
    Heuristics generator.
    This version will remove each collecting pair independently from the observed graph on collection.
    """
    #
    NUM_HEURISTICS = 4

    def __init__(
        self: SelfHeuristicsForest1,
        cache: str,
        n: int,
        adjs: NPINTS,
        rels: NPINTS,
        /,
        *,
        num_hops: int,
        num_processes: int,
        unit: float,
        pseudo: bool,
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
        - pseudo
            If True, no collection will be performed even if the collection is explicitly called.

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
        self._pseudo = pseudo

        #
        # assert self._num_hops > 1, "Number of hops should be greater than 1 for heuristics to be useful."

    def forest(self: SelfHeuristicsForest1, pairs: NPINTS, /) -> None:
        R"""
        Generate node reachability at each layer of hop for give node pairs.

        Args
        ----
        - pairs
            Collecting pairs.

        Returns
        -------
        """
        #
        if self._pseudo:
            #
            print("-" * 80)
            print("Cached forest is skipped.")
            print("-" * 80)
            return

        # Ensure uniqueness.
        eids = onp.unique(pairs[0] * self._num_nodes + pairs[1])
        pairs = onp.stack((eids // self._num_nodes, eids % self._num_nodes))

        # Masks of reachable nodes at each layer of hop for all node pairs.
        # Node pairs are independetly stored under the same directory for memory efficient loading.
        directory = os.path.join(self._cache, "hop{:d}.forest".format(self._num_hops))
        os.makedirs(directory, exist_ok=True)

        #
        eids = pairs[0] * self._num_nodes + pairs[1]
        maxlen = len(str(self._num_nodes**2 - 1))
        masks = onp.array(
            [not os.path.isfile(os.path.join(directory, "{:>0{:d}d}.npy".format(eid, maxlen))) for eid in eids],
        )

        # Get pairs that really requires generation.
        pairs = pairs[:, masks]
        pairs.setflags(write=False)
        if pairs.nbytes == 0:
            #
            return

        #
        (_, num_pairs) = pairs.shape
        num_processes = min(self._num_processes, num_pairs)

        # Initialization.
        (
            (shadjs, shape_shadjs, dtype_shadjs),
            (shrels, shape_shrels, dtype_shrels),
            (shpairs, shape_shpairs, dtype_shpairs),
        ) = self.forest_ini(self._adjs, self._rels, pairs)

        # Logging from here may involve multiprocessing, thus we use naive print for logging.
        if num_processes > 1:
            #
            print("-" * 80)
            print("Cached forest is missing, and will be generated by {:d} processes.".format(num_processes))
            print("-" * 80)

            # Multiprocessing.
            num_pairs_per_process = int(math.ceil(float(num_pairs) / float(num_processes)))
            jobs = [
                (
                    directory,
                    i * num_pairs_per_process,
                    min((i + 1) * num_pairs_per_process, num_pairs),
                    self._num_nodes,
                    (shadjs.name, shape_shadjs, dtype_shadjs),
                    (shrels.name, shape_shrels, dtype_shrels),
                    (shpairs.name, shape_shpairs, dtype_shpairs),
                    self._num_hops,
                    "{:>0{:d}d}".format(i, len(str(num_processes - 1))),
                    self._unit,
                )
                for i in range(num_processes)
            ]
            with mp.Pool(num_processes) as pool:
                #
                pool.starmap(self.forest_work, jobs)
        else:
            #
            print("-" * 80)
            print("Cached forest is missing, and will be generated.")
            print("-" * 80)

            # Call the process kernel directly.
            self.forest_work(
                directory,
                0,
                num_pairs,
                self._num_nodes,
                (shadjs.name, shape_shadjs, dtype_shadjs),
                (shrels.name, shape_shrels, dtype_shrels),
                (shpairs.name, shape_shpairs, dtype_shpairs),
                self._num_hops,
                "-",
                self._unit,
            )

        # Finalization.
        self.forest_fin(shadjs, shrels, shpairs)
        print("-" * 80)
        print("Cached forest is collected.")
        print("-" * 80)

    @staticmethod
    def forest_ini(
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
        Initialization for forest generation.

        Args
        ----
        - adjs
            Adjacency list.
        - rels
            Relations.
        - pairs
            Collecting pairs.

        Returns
        -------
        - shmadjs
            Sharing memory descriptor of adjacency list.
        - shmrels
            Sharing memory descriptor of relations.
        - shmpairs
            Sharing memory descriptor of collecting pairs.
        """
        #
        assert not adjs.flags.writeable and not rels.flags.writeable and not pairs.flags.writeable
        print("Sharing adjacency list, relations and collecting pairs.")

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
    def forest_fin(shadjs: SharedMemory, shrels: SharedMemory, shpairs: SharedMemory, /) -> None:
        R"""
        Finalization for forest generation.

        Args
        ----
        - shadjs
            Shared adjacency list.
        - shrels
            Shared relations.
        - shpairs
            Shared collecting pairs.

        Returns
        -------
        """
        #
        print("Releasing shared adjacency list, relations and collecting pairs.")

        #
        shadjs.close()
        shrels.close()
        shpairs.close()

        #
        shadjs.unlink()
        shrels.unlink()
        shpairs.unlink()

    @staticmethod
    def forest_work(
        directory: str,
        bgn: int,
        end: int,
        num_nodes: int,
        shmadjs: Tuple[str, Sequence[int], type],
        shmrels: Tuple[str, Sequence[int], type],
        shmpairs: Tuple[str, Sequence[int], type],
        num_hops: int,
        title: str,
        unit: float,
        /,
    ) -> None:
        R"""
        Worker for forest generation.

        Args
        ----
        - directory
            Saving directory.
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
        - shmpairs
            Sharing memory descriptor of collecting pairs.
        - num_hops
            Number of hops.
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
        HeuristicsForest1.forest_work_(directory, bgn, end, num_nodes, adjs, rels, pairs, num_hops, title, unit)

        #
        shadjs.close()
        shrels.close()
        shpairs.close()

    @staticmethod
    def forest_work_(
        directory: str,
        bgn: int,
        end: int,
        num_nodes: int,
        adjs: NPINTS,
        rels: NPINTS,
        pairs: NPINTS,
        num_hops: int,
        title: str,
        unit: float,
        /,
    ) -> None:
        R"""
        Worker for forest generation.

        Args
        ----
        - directory
            Saving directory.
        - bgn
            Beginning node of current worker.
        - end
            Ending node of current worker.
        - num_nodes
            Number of nodes.
        - adjs
            Adjacency list.
        - rels
            Relations.
        - pairs
            Collecting pairs.
        - num_hops
            Number of hops.
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

        #
        tree_src = ComputationSubsetNode(num_nodes, adjs, rels, num_hops=num_hops)
        tree_dst = ComputationSubsetNode(num_nodes, adjs, rels, num_hops=num_hops)
        maxlen = len(str(num_nodes**2 - 1))

        #
        elapsed = time.time()
        elapsed_start = elapsed
        for i in range(bgn, end):
            #
            if time.time() - elapsed > unit or i == bgn:
                #
                print(
                    'Process "{:s}" ({:d}) has done with pairs [{:d}|{:d}|{:d}) in {:d} secs.'.format(
                        title,
                        pid,
                        bgn,
                        i,
                        end,
                        int(math.ceil(time.time() - elapsed_start)),
                    ),
                )
                elapsed = time.time()

            # Get source and destination computation tree with bidirected self rejection.
            (src, dst) = pairs[:, i].tolist()
            xadjs = onp.array([(src, dst), (dst, src)]).T
            tree_src.reset()
            tree_dst.reset()
            tree_src.sample("tree", onp.array([src]), tree_src.masks_edge_accept(xadjs))
            tree_dst.sample("tree", onp.array([dst]), tree_dst.masks_edge_accept(xadjs))

            # Concatenate.
            eid = src * num_nodes + dst
            masks = onp.stack((tree_src.masks_node_active, tree_dst.masks_node_active))
            assert masks.size > 0
            with open(os.path.join(directory, "{:>0{:d}d}.npy".format(eid, maxlen)), "wb") as file:
                #
                onp.save(file, masks)
            with open(os.path.join(directory, "{:>0{:d}d}.npy".format(eid, maxlen)), "rb") as file:
                #
                onp.load(file)

        #
        print('Process "{:s}" ({:d}) is done with pairs [{:d}, {:d}).'.format(title, pid, bgn, end))

    def collect(self: SelfHeuristicsForest1, pairs: NPINTS, /) -> None:
        R"""
        Collect heuristics for given node pairs.

        Args
        ----
        - pairs
            Node pairs to collect heuristics.

        Returns
        -------
        - heuristics
            Collected heuristics.
        """
        #
        if self._pseudo:
            #
            print("-" * 80)
            print("Heuristics are skipped.")
            print("-" * 80)
            return

        # Ensure uniqueness.
        eids = onp.unique(pairs[0] * self._num_nodes + pairs[1])
        pairs = onp.stack((eids // self._num_nodes, eids % self._num_nodes))

        # Heuristics are independetly stored under the same directory for memory efficient loading.
        directory_heuristics = os.path.join(self._cache, "hop{:d}.heuristics".format(self._num_hops))
        os.makedirs(directory_heuristics, exist_ok=True)
        maxlen = len(str(self._num_nodes**2 - 1))
        masks = onp.array(
            [
                not os.path.isfile(
                    os.path.join(directory_heuristics, "{:>0{:d}d}.npy".format(src * self._num_nodes + dst, maxlen)),
                )
                for (src, dst) in pairs.T.tolist()
            ],
        )
        pairs = pairs[:, masks]
        pairs.setflags(write=False)
        if pairs.nbytes == 0:
            #
            return

        #
        (_, num_pairs) = pairs.shape
        num_processes = min(self._num_processes, num_pairs)
        directory_forest = os.path.join(self._cache, "hop{:d}.forest".format(self._num_hops))
        os.makedirs(directory_forest, exist_ok=True)

        # Initialization.
        (shpairs, shape_shpairs, dtype_shpairs) = self.collect_ini(pairs)

        # Logging from here may involve multiprocessing, thus we use naive print for logging.
        if num_processes > 1:
            #
            print("-" * 80)
            print("Collect {:d} heuristics by {:d} processes.".format(num_pairs, num_processes))
            print("-" * 80)

            # Multiprocessing.
            num_pairs_per_process = int(math.ceil(float(num_pairs) / float(num_processes)))
            jobs = [
                (
                    directory_heuristics,
                    i * num_pairs_per_process,
                    min((i + 1) * num_pairs_per_process, num_pairs),
                    self._num_nodes,
                    (shpairs.name, shape_shpairs, dtype_shpairs),
                    directory_forest,
                    "{:>0{:d}d}".format(i, len(str(num_processes - 1))),
                    self._unit,
                )
                for i in range(num_processes)
            ]
            with mp.Pool(num_processes) as pool:
                #
                pool.starmap(self.collect_work, jobs)
        else:
            #
            print("-" * 80)
            print("Collect {:d} heuristics.".format(num_pairs))
            print("-" * 80)

            # Call the process kernel directly.
            self.collect_work(
                directory_heuristics,
                0,
                num_pairs,
                self._num_nodes,
                (shpairs.name, shape_shpairs, dtype_shpairs),
                directory_forest,
                "-",
                self._unit,
            )

        # Finalization.
        self.collect_fin(shpairs)
        print("-" * 80)
        print("Heuristics are collected.".format(num_pairs))
        print("-" * 80)

    @staticmethod
    def collect_ini(pairs: NPINTS, /) -> Tuple[SharedMemory, Sequence[int], type]:
        R"""
        Initialization for heuristics collection.

        Args
        ----
        - pairs
            Collecting pairs.

        Returns
        -------
        - shmpairs
            Sharing memory descriptor of collecting pairs.
        """
        #
        assert not pairs.flags.writeable
        print("Sharing collecting pairs.")

        #
        shpairs = SharedMemory(create=True, size=pairs.nbytes)

        # Sharing memory interface as NumPy array.
        onp.copyto(onp.ndarray(pairs.shape, dtype=pairs.dtype, buffer=shpairs.buf), pairs)

        #
        return (shpairs, pairs.shape, cast(type, pairs.dtype))

    @staticmethod
    def collect_fin(shpairs: SharedMemory, /) -> None:
        R"""
        Finalization for heurisctics collection.

        Args
        ----
        - shpairs
            Shared collecting pairs.

        Returns
        -------
        """
        #
        print("Releasing shared collecting pairs.")

        #
        shpairs.close()

        #
        shpairs.unlink()

    @staticmethod
    def collect_work(
        directory_heuristics: str,
        bgn: int,
        end: int,
        num_nodes: int,
        shmpairs: Tuple[str, Sequence[int], type],
        directory_forest: str,
        title: str,
        unit: float,
        /,
    ) -> None:
        R"""
        Worker for heuristics collection.

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
        - shmpairs
            Sharing memory descriptor of collecting pairs.
        - directory_forest
            Load directory for reachable masks.
        - title
            Title of worker.
        - unit
            Time unit (second) to report progress.

        Returns
        -------
        """
        #
        pairs: NPINTS

        # Reconstruct adjacency list and relations.
        (url_pairs, shape_pairs, dtype_pairs) = shmpairs
        shpairs = SharedMemory(name=url_pairs)
        pairs = onp.ndarray(shape_pairs, dtype=dtype_pairs, buffer=shpairs.buf)
        pairs.setflags(write=False)

        #
        HeuristicsForest1.collect_work_(
            directory_heuristics,
            bgn,
            end,
            num_nodes,
            pairs,
            directory_forest,
            title,
            unit,
        )

        #
        shpairs.close()

    @staticmethod
    def collect_work_(
        directory_heuristics: str,
        bgn: int,
        end: int,
        num_nodes: int,
        pairs: NPINTS,
        directory_forest: str,
        title: str,
        unit: float,
        /,
    ) -> None:
        R"""
        Worker for heuristics collection.

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
        - pairs
            Collecting node pairs.
        - directory_forest
            Load directory for reachable masks.
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

        #
        maxlen = len(str(num_nodes**2 - 1))

        #
        elapsed = time.time()
        elapsed_start = elapsed
        for i in range(bgn, end):
            #
            if time.time() - elapsed > unit or i == bgn:
                #
                print(
                    'Process "{:s}" ({:d}) has done with pairs [{:d}|{:d}|{:d}) in {:d} secs.'.format(
                        title,
                        pid,
                        bgn,
                        i,
                        end,
                        int(math.ceil(time.time() - elapsed_start)),
                    ),
                )
                elapsed = time.time()

            #
            (src, dst) = pairs[:, i]
            eid = src * num_nodes + dst
            with open(os.path.join(directory_forest, "{:>0{:d}d}.npy".format(eid, maxlen)), "rb") as file:
                #
                (masks_src, masks_dst) = onp.load(file)
            masks_src = onp.concatenate((masks_src, onp.ones((1, num_nodes), dtype=onp.bool_)))
            masks_dst = onp.concatenate((masks_dst, onp.ones((1, num_nodes), dtype=onp.bool_)))

            # Release tree file for more space.
            os.remove(os.path.join(directory_forest, "{:>0{:d}d}.npy".format(eid, maxlen)))

            #
            heuristics = collect(src, dst, masks_src, masks_dst)
            assert heuristics.size > 0
            with open(os.path.join(directory_heuristics, "{:>0{:d}d}.npy".format(eid, maxlen)), "wb") as file:
                #
                onp.save(file, heuristics)
            with open(os.path.join(directory_heuristics, "{:>0{:d}d}.npy".format(eid, maxlen)), "rb") as file:
                #
                onp.load(file)

        #
        print('Process "{:s}" ({:d}) is done with pairs [{:d}, {:d}).'.format(title, pid, bgn, end))

    def load(self: SelfHeuristicsForest1, pairs: NPINTS, /) -> NPINTS:
        R"""
        Load heuristics for given node pairs.

        Args
        ----
        - pairs
            Node pairs to collect heuristics.

        Returns
        -------
        - heuristics
            Collected heuristics.
        """
        #
        if self._pseudo:
            #
            return onp.zeros((pairs.shape[1], 4), dtype=onp.int64)
        else:
            #
            return load(pairs, self._num_nodes, os.path.join(self._cache, "hop{:d}.heuristics".format(self._num_hops)))
