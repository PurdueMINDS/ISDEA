#
import re
import os
import more_itertools as xitertools
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
            for (rel, obj) in sorted(adjs[sub], key=lambda neighbor: (neighbor[-1], neighbor[0])):
                #
                file.write("{:>{:d}s} {:>{:d}s} {:>{:d}s}\n".format(sub, maxlen_sub, rel, maxlen_rel, obj, maxlen_obj))


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
    for (src, dst) in (("train", "train"), ("valid", "valid")):
        #
        adjs = load(os.path.join(source, "{:s}.txt".format(src)))
        save(os.path.join(destination, "{:s}.txt".format(dst)), adjs)


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
    for (src, dst) in (("train", "observe"), ("test", "test")):
        #
        adjs = load(os.path.join(source, "{:s}.txt".format(src)))
        save(os.path.join(destination, "{:s}.txt".format(dst)), adjs)


#
RAW = os.path.join("clone", "GraIL", "data")


def wn18rr_v1() -> None:
    R"""
    Task of WN18RR v1.

    Args
    ----

    Returns
    -------
    """
    #
    source = "WN18RR_v1"
    destination = "WN18RR1"
    os.makedirs(os.path.join("data", "-".join((destination, "trans"))), exist_ok=True)
    os.makedirs(os.path.join("data", "-".join((destination, "ind"))), exist_ok=True)
    copy_trans(os.path.join(RAW, source), os.path.join("data", "-".join((destination, "trans"))))
    copy_ind(os.path.join(RAW, "{:s}_ind".format(source)), os.path.join("data", "-".join((destination, "ind"))))


def wn18rr_v4() -> None:
    R"""
    Task of WN18RR v4.

    Args
    ----

    Returns
    -------
    """
    #
    source = "WN18RR_v4"
    destination = "WN18RR4"
    os.makedirs(os.path.join("data", "-".join((destination, "trans"))), exist_ok=True)
    os.makedirs(os.path.join("data", "-".join((destination, "ind"))), exist_ok=True)
    copy_trans(os.path.join(RAW, source), os.path.join("data", "-".join((destination, "trans"))))
    copy_ind(os.path.join(RAW, "{:s}_ind".format(source)), os.path.join("data", "-".join((destination, "ind"))))


def wn18rr_v2() -> None:
    R"""
    Task of WN18RR v2.

    Args
    ----

    Returns
    -------
    """
    #
    source = "WN18RR_v2"
    destination = "WN18RR2"
    os.makedirs(os.path.join("data", "-".join((destination, "trans"))), exist_ok=True)
    os.makedirs(os.path.join("data", "-".join((destination, "ind"))), exist_ok=True)
    copy_trans(os.path.join(RAW, source), os.path.join("data", "-".join((destination, "trans"))))
    copy_ind(os.path.join(RAW, "{:s}_ind".format(source)), os.path.join("data", "-".join((destination, "ind"))))


def wn18rr_v3() -> None:
    R"""
    Task of WN18RR v3.

    Args
    ----

    Returns
    -------
    """
    #
    source = "WN18RR_v3"
    destination = "WN18RR3"
    os.makedirs(os.path.join("data", "-".join((destination, "trans"))), exist_ok=True)
    os.makedirs(os.path.join("data", "-".join((destination, "ind"))), exist_ok=True)
    copy_trans(os.path.join(RAW, source), os.path.join("data", "-".join((destination, "trans"))))
    copy_ind(os.path.join(RAW, "{:s}_ind".format(source)), os.path.join("data", "-".join((destination, "ind"))))


def nell995_v1() -> None:
    R"""
    Task of NELL995 v1.

    Args
    ----

    Returns
    -------
    """
    #
    source = "nell_v1"
    destination = "NELL9951"
    os.makedirs(os.path.join("data", "-".join((destination, "trans"))), exist_ok=True)
    os.makedirs(os.path.join("data", "-".join((destination, "ind"))), exist_ok=True)
    copy_trans(os.path.join(RAW, source), os.path.join("data", "-".join((destination, "trans"))))
    copy_ind(os.path.join(RAW, "{:s}_ind".format(source)), os.path.join("data", "-".join((destination, "ind"))))


#
if __name__ == "__main__":
    #
    wn18rr_v1()
    wn18rr_v4()
    wn18rr_v2()
    wn18rr_v3()
    nell995_v1()
