#
import re
import os
import more_itertools as xitertools
import math
from typing import Mapping, Sequence, Tuple, Dict, List
from types import MappingProxyType


def load(path: str, /) -> Mapping[str, Sequence[Tuple[str, str]]]:
    R"""
    Load raw data.

    Args
    ----
    - path
        Path.

    Returns
    -------
    - adjs
        Adjacency list.
    """
    #
    adjs: Dict[str, List[Tuple[str, str]]]

    #
    adjs = {}
    with open(path, "r") as file:
        #
        for line in file:
            #
            (sub, rel, obj) = re.split(r"\s+", line.strip())
            if sub in adjs:
                #
                adjs[sub].append((rel, obj))
            else:
                #
                adjs[sub] = [(rel, obj)]
    return MappingProxyType({sub: tuple(neighbors) for (sub, neighbors) in adjs.items()})


def save(path: str, adjs: Mapping[str, Sequence[Tuple[str, str]]], /) -> None:
    R"""
    Save regulated data.

    Args
    ----
    - path
        Path.
    - adjs
        Adjacency list.

    Returns
    -------
    """
    #
    maxlen_sub = max(len(sub) for sub in adjs.keys())
    maxlen_rel = max(len(rel) for (rel, _) in xitertools.flatten(adjs.values()))
    maxlen_obj = max(len(obj) for (_, obj) in xitertools.flatten(adjs.values()))

    #
    with open(path, "w") as file:
        #
        for sub in sorted(adjs.keys()):
            #
            for rel, obj in sorted(adjs[sub], key=lambda neighbor: (neighbor[-1], neighbor[0])):
                #
                file.write("{:>{:d}s} {:>{:d}s} {:>{:d}s}\n".format(sub, maxlen_sub, rel, maxlen_rel, obj, maxlen_obj))


def save_discard(path: str, adjs: Mapping[str, Sequence[Tuple[str, str]]], discards: Sequence[str], /) -> None:
    R"""
    Save regulated data.

    Args
    ----
    - path
        Path.
    - adjs
        Adjacency list.
    - discards
        Discard relations.

    Returns
    -------
    """
    #
    maxlen_sub = max(len(sub) for sub in adjs.keys())
    maxlen_rel = max(len(rel) for (rel, _) in xitertools.flatten(adjs.values()))
    maxlen_obj = max(len(obj) for (_, obj) in xitertools.flatten(adjs.values()))

    #
    with open(path, "w") as file:
        #
        for sub in sorted(adjs.keys()):
            #
            for rel, obj in sorted(adjs[sub], key=lambda neighbor: (neighbor[-1], neighbor[0])):
                #
                if rel not in discards:
                    #
                    file.write(
                        "{:>{:d}s} {:>{:d}s} {:>{:d}s}\n".format(sub, maxlen_sub, rel, maxlen_rel, obj, maxlen_obj),
                    )


def save_select(path: str, adjs: Mapping[str, Sequence[Tuple[str, str]]], selects: Sequence[str], /) -> None:
    R"""
    Save regulated data.

    Args
    ----
    - path
        Path.
    - adjs
        Adjacency list.
    - selects
        Selected relations.

    Returns
    -------
    """
    #
    maxlen_sub = max(len(sub) for sub in adjs.keys())
    maxlen_rel = max(len(rel) for (rel, _) in xitertools.flatten(adjs.values()))
    maxlen_obj = max(len(obj) for (_, obj) in xitertools.flatten(adjs.values()))

    #
    with open(path, "w") as file:
        #
        for sub in sorted(adjs.keys()):
            #
            for rel, obj in sorted(adjs[sub], key=lambda neighbor: (neighbor[-1], neighbor[0])):
                #
                if rel in selects:
                    #
                    file.write(
                        "{:>{:d}s} {:>{:d}s} {:>{:d}s}\n".format(sub, maxlen_sub, rel, maxlen_rel, obj, maxlen_obj),
                    )


def copy_trans(source: str, destination: str, /) -> None:
    R"""
    Copy transductive task data from source to destination directory.

    Args
    ----
    - source
        Source directory.
    - destination
        Desination directory.

    Returns
    -------
    """
    #
    for src, dst in (("train", "train"), ("valid", "valid")):
        #
        adjs = load(os.path.join(source, "{:s}.txt".format(src)))
        save(os.path.join(destination, "{:s}.txt".format(dst)), adjs)


def copy_trans_discard(source: str, destination: str, rate: float, /) -> Sequence[str]:
    R"""
    Copy transductive task data from source to destination directory with discard.

    Args
    ----
    - source
        Source directory.
    - destination
        Desination directory.
    - rate
        Discard rate.

    Returns
    -------
    - discards
        Discard relations.
    """
    #
    raws = {}
    for src in ("train", "valid"):
        #
        raws[src] = load(os.path.join(source, "{:s}.txt".format(src)))
    counts = {}
    for adjs in raws.values():
        #
        for neighbors in adjs.values():
            #
            for rel, _ in neighbors:
                #
                if rel in counts:
                    #
                    counts[rel] += 1
                else:
                    #
                    counts[rel] = 1
    buf = list(sorted(((rel, cnt) for (rel, cnt) in counts.items()), key=lambda pair: pair[1]))
    discard_ids = [i * 10 for i in range(len(buf) // 10)]
    discard_rels = [buf[i][0] for i in discard_ids]
    print(discard_rels)
    for src, dst in (("train", "train"), ("valid", "valid")):
        #
        save_discard(os.path.join(destination, "{:s}.txt".format(dst)), raws[src], discard_rels)
    return discard_rels


def copy_ind(source: str, destination: str, /) -> None:
    R"""
    Copy inductive task data from source to destination directory.

    Args
    ----
    - source
        Source directory.
    - destination
        Desination directory.

    Returns
    -------
    """
    #
    for src, dst in (("train", "observe"), ("test", "test")):
        #
        adjs = load(os.path.join(source, "{:s}.txt".format(src)))
        save(os.path.join(destination, "{:s}.txt".format(dst)), adjs)


def copy_ind_discard(source: str, destination: str, discards: Sequence[str], /) -> None:
    R"""
    Copy inductive task data from source to destination directory with discard.

    Args
    ----
    - source
        Source directory.
    - destination
        Desination directory.
    - discards
        Discard relations.

    Returns
    -------
    """
    #
    for src, dst in [("train", "observe")]:
        #
        adjs = load(os.path.join(source, "{:s}.txt".format(src)))
        save(os.path.join(destination, "{:s}.txt".format(dst)), adjs)
    for src, dst in [("test", "test")]:
        #
        adjs = load(os.path.join(source, "{:s}.txt".format(src)))
        save_select(os.path.join(destination, "{:s}.txt".format(dst)), adjs, discards)


#
RAW = os.path.join("clone", "GraIL", "data")


def wn18rr(version: int, /) -> None:
    R"""
    Task of WN18RR.

    Args
    ----
    - version
        Version ID.

    Returns
    -------
    """
    #
    source = "WN18RR_v{:d}".format(version)
    destination = "WN18RR{:d}".format(version)
    os.makedirs(os.path.join("data", "-".join((destination, "trans"))), exist_ok=True)
    os.makedirs(os.path.join("data", "-".join((destination, "ind"))), exist_ok=True)
    copy_trans(os.path.join(RAW, source), os.path.join("data", "-".join((destination, "trans"))))
    copy_ind(os.path.join(RAW, "{:s}_ind".format(source)), os.path.join("data", "-".join((destination, "ind"))))


def nell995(version: int, /) -> None:
    R"""
    Task of NELL995.

    Args
    ----
    - version
        Version ID.

    Returns
    -------
    """
    #
    source = "nell_v{:d}".format(version)
    destination = "NELL995{:d}".format(version)
    os.makedirs(os.path.join("data", "-".join((destination, "trans"))), exist_ok=True)
    os.makedirs(os.path.join("data", "-".join((destination, "ind"))), exist_ok=True)
    copy_trans(os.path.join(RAW, source), os.path.join("data", "-".join((destination, "trans"))))
    copy_ind(os.path.join(RAW, "{:s}_ind".format(source)), os.path.join("data", "-".join((destination, "ind"))))


def fb237(version: int, /) -> None:
    R"""
    Task of FB237.

    Args
    ----
    - version
        Version ID.

    Returns
    -------
    """
    #
    source = "fb237_v{:d}".format(version)
    destination = "FB237{:d}".format(version)
    os.makedirs(os.path.join("data", "-".join((destination, "trans"))), exist_ok=True)
    os.makedirs(os.path.join("data", "-".join((destination, "ind"))), exist_ok=True)
    copy_trans(os.path.join(RAW, source), os.path.join("data", "-".join((destination, "trans"))))
    copy_ind(os.path.join(RAW, "{:s}_ind".format(source)), os.path.join("data", "-".join((destination, "ind"))))


def discard_wn18rr(version: int, rate: float, /) -> None:
    R"""
    Task of WN18RR.

    Args
    ----
    - version
        Version ID.

    Returns
    -------
    """
    #
    source = "WN18RR_v{:d}".format(version)
    destination = "WN18RR{:d}Dis".format(version)
    os.makedirs(os.path.join("data", "-".join((destination, "trans"))), exist_ok=True)
    os.makedirs(os.path.join("data", "-".join((destination, "ind"))), exist_ok=True)
    discards = copy_trans_discard(
        os.path.join(RAW, source),
        os.path.join("data", "-".join((destination, "trans"))),
        rate,
    )
    copy_ind_discard(
        os.path.join(RAW, "{:s}_ind".format(source)),
        os.path.join("data", "-".join((destination, "ind"))),
        discards,
    )


def discard_nell995(version: int, rate: float, /) -> None:
    R"""
    Task of NELL995.

    Args
    ----
    - version
        Version ID.

    Returns
    -------
    """
    #
    source = "nell_v{:d}".format(version)
    destination = "NELL995{:d}Dis".format(version)
    os.makedirs(os.path.join("data", "-".join((destination, "trans"))), exist_ok=True)
    os.makedirs(os.path.join("data", "-".join((destination, "ind"))), exist_ok=True)
    discards = copy_trans_discard(
        os.path.join(RAW, source),
        os.path.join("data", "-".join((destination, "trans"))),
        rate,
    )
    copy_ind_discard(
        os.path.join(RAW, "{:s}_ind".format(source)),
        os.path.join("data", "-".join((destination, "ind"))),
        discards,
    )


def discard_fb237(version: int, rate: float, /) -> None:
    R"""
    Task of dicarded FB237.

    Args
    ----
    - version
        Version ID.
    - rate
        Discard rate.

    Returns
    -------
    """
    #
    source = "fb237_v{:d}".format(version)
    destination = "FB237{:d}Dis".format(version)
    os.makedirs(os.path.join("data", "-".join((destination, "trans"))), exist_ok=True)
    os.makedirs(os.path.join("data", "-".join((destination, "ind"))), exist_ok=True)
    discards = copy_trans_discard(
        os.path.join(RAW, source),
        os.path.join("data", "-".join((destination, "trans"))),
        rate,
    )
    copy_ind_discard(
        os.path.join(RAW, "{:s}_ind".format(source)),
        os.path.join("data", "-".join((destination, "ind"))),
        discards,
    )


#
if __name__ == "__main__":
    #
    # \\:wn18rr(1)
    # \\:wn18rr(2)
    # \\:wn18rr(3)
    # \\:wn18rr(4)
    # \\:nell995(1)
    # \\:nell995(2)
    # \\:nell995(3)
    # \\:nell995(4)
    # \\:fb237(1)
    # \\:fb237(2)
    # \\:fb237(3)
    # \\:fb237(4)
    discard_wn18rr(1, 0.1)
    discard_wn18rr(2, 0.1)
    discard_wn18rr(3, 0.1)
    discard_wn18rr(4, 0.1)
    discard_nell995(1, 0.1)
    discard_nell995(2, 0.1)
    discard_nell995(3, 0.1)
    discard_nell995(4, 0.1)
    discard_fb237(1, 0.1)
    discard_fb237(2, 0.1)
    discard_fb237(3, 0.1)
    discard_fb237(4, 0.1)
