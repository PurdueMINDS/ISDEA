#
import os
import re
import argparse
import numpy as onp
from typing import Dict, Sequence


def load_and_build(
    prefix: str, /, *, suffices_load: Sequence[str], suffices_save: Sequence[str]
) -> None:
    R"""
    Load triplets from file and build mapping from entities and relations to IDs.

    Args
    ----
    - data
        Data directory prefix.
    - suffices_load
        Suffix of each data directory.
    - suffices_save
        Suffix of each data directory.

    Returns
    -------
    """
    #
    entity2id: Dict[str, int]
    relation2id: Dict[str, int]

    #
    entity2id = {}
    relation2id = {}

    #
    print("-- Build mappings:")
    triplets = []
    for suffix in suffices_load:
        #
        for section in ("observe", "train", "valid"):
            #
            path = os.path.join("-".join((prefix, suffix)), "{:s}.txt".format(section))
            if os.path.isfile(path):
                #
                print('Load "{:s}".'.format(path))
            else:
                #
                print('Skip "{:s}".'.format(path))
                continue
            with open(path, "r") as file:
                #
                for line in file:
                    #
                    (sub, rel, obj) = re.split(r"\s+", line.strip())

                    # Subject or object as entity.
                    for ent in (sub, obj):
                        #
                        if ent not in entity2id:
                            #
                            entity2id[ent] = len(entity2id)

                    # Relation.
                    if rel not in relation2id:
                        #
                        relation2id[rel] = len(relation2id)

                    #
                    triplets.append((entity2id[sub], entity2id[obj], relation2id[rel]))

    #
    print("-- Generate sampled mappings:")
    id2entity = {i: name for (name, i) in entity2id.items()}
    id2relation = {i: name for (name, i) in relation2id.items()}

    #
    for suffix in suffices_save:
        #
        os.makedirs("-".join((prefix, suffix)), exist_ok=True)
        if suffix == "ind-perm":
            #
            continue
        for name2id, title in ((entity2id, "entities"), (relation2id, "relations")):
            #
            maxlen_id = max(len(str(val)) for val in name2id.values())
            maxlen_name = max(len(key) for key in name2id.keys())
            with open(
                os.path.join("-".join((prefix, suffix)), "{:s}.dict".format(title)), "w"
            ) as file:
                #
                for key, val in name2id.items():
                    #
                    file.write(
                        "{:<{:d}d} {:>{:d}s}\n".format(val, maxlen_id, key, maxlen_name)
                    )

    #
    perm_entity = onp.random.RandomState(42).permutation(len(entity2id)).tolist()
    perm_relation = onp.random.RandomState(42).permutation(len(relation2id)).tolist()
    entity2id_perm = {id2entity[perm_entity[i]]: i for i in entity2id.values()}
    relation2id_perm = {id2relation[perm_relation[i]]: i for i in relation2id.values()}

    #
    if "ind-perm" in suffices_save:
        #
        suffix = "ind-perm"
        os.makedirs("-".join((prefix, suffix)), exist_ok=True)
        for name2id, title in (
            (entity2id_perm, "entities"),
            (relation2id_perm, "relations"),
        ):
            #
            maxlen_id = max(len(str(val)) for val in name2id.values())
            maxlen_name = max(len(key) for key in name2id.keys())
            with open(
                os.path.join("-".join((prefix, suffix)), "{:s}.dict".format(title)), "w"
            ) as file:
                #
                for key, val in name2id.items():
                    #
                    file.write(
                        "{:<{:d}d} {:>{:d}s}\n".format(val, maxlen_id, key, maxlen_name)
                    )


def load_and_apply(data: str, /) -> None:
    R"""
    Load triplets and mappings from file and try to map from entities and relations to IDs.

    Args
    ----
    - data
        Data directory.

    Returns
    -------
    """
    #
    entity2id = {}
    with open(os.path.join(data, "entities.dict"), "r") as file:
        #
        for line in file:
            #
            (msg_id, msg_name) = re.split(r"\s+", line.strip())
            entity2id[msg_name] = int(msg_id)

    #
    relation2id = {}
    with open(os.path.join(data, "relations.dict"), "r") as file:
        #
        for line in file:
            #
            (msg_id, msg_name) = re.split(r"\s+", line.strip())
            relation2id[msg_name] = int(msg_id)

    #
    print("-- Apply mappings:")
    for section in ("observe", "train", "valid", "test"):
        #
        path = os.path.join(data, "{:s}.txt".format(section))
        if os.path.isfile(path):
            #
            print('Load "{:s}".'.format(path))
        else:
            #
            print('Skip "{:s}".'.format(path))
            continue
        with open(path, "r") as file:
            #
            for line in file:
                #
                (sub, rel, obj) = re.split(r"\s+", line.strip())
                assert entity2id[sub] >= 0
                assert entity2id[obj] >= 0
                assert relation2id[rel] >= 0


def main() -> None:
    R"""
    Main execution.

    Args
    ----

    Returns
    -------
    """
    #
    parser = argparse.ArgumentParser(description="Generate Name-to-ID mapping.")
    parser.add_argument("--data", type=str, required=True, help="Dataset name.")
    args = parser.parse_args()

    #
    prefix = os.path.join("data", args.data)
    load_and_build(prefix, suffices_load=["trans"], suffices_save=["trans"])
    load_and_build(
        prefix, suffices_load=["trans", "ind"], suffices_save=["ind", "ind-perm"]
    )
    for suffix in ["trans", "ind"]:
        #
        load_and_apply("-".join((prefix, suffix)))


if __name__ == "__main__":
    #
    main()
