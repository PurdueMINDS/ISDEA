#
import os
import numpy as onp
import time
import argparse


def validate_forest(directory: str, num_hops: int, unit: float, /) -> None:
    R"""
    Validate forest data.

    Args
    ----
    - directory
        Directory.
    - num_hops
        Number of hops.
    - unit
        Report unit.

    Returns
    -------
    """
    #
    print("-- Validating forest data:")
    subdirectory = os.path.join(directory, "hop{:d}.forest".format(num_hops))
    if os.path.isdir(subdirectory):
        #
        filenames = list(os.listdir(subdirectory))
        elapsed = time.time()
        maxlen = len(str(len(filenames) + 1))
        for (i, filename) in enumerate(filenames):
            #
            if i == 0 or time.time() - elapsed > unit:
                #
                print("[{:>{:d}d}/{:>{:d}d}]".format(i + 1, maxlen, len(filenames), maxlen))
                elapsed = time.time()

            #
            (_, extension) = os.path.splitext(filename)
            assert extension == ".npy"

            #
            path = os.path.join(subdirectory, filename)
            try:
                #
                with open(path, "rb") as file:
                    #
                    masks = onp.load(file)
                assert masks.ndim == 3
                assert masks.shape[0] == 2
                assert masks.shape[1] == num_hops + 1
            except:
                #
                print('invalid "{:s}" is removed.'.format(path))
                os.remove(path)
        print("[{:>{:d}d}/{:>{:d}d}]".format(len(filenames), maxlen, len(filenames), maxlen))


def validate_heuristics(directory: str, num_hops: int, unit: float, /) -> None:
    R"""
    Validate heuristics data.

    Args
    ----
    - directory
        Directory.
    - num_hops
        Number of hops.
    - unit
        Report unit.

    Returns
    -------
    """
    #
    print("-- Validating heuristics data:")
    subdirectory = os.path.join(directory, "hop{:d}.heuristics".format(num_hops))
    if os.path.isdir(subdirectory):
        #
        filenames = list(os.listdir(subdirectory))
        elapsed = time.time()
        maxlen = len(str(len(filenames) + 1))
        for (i, filename) in enumerate(filenames):
            #
            if i == 0 or time.time() - elapsed > unit:
                #
                print("[{:>{:d}d}/{:>{:d}d}]".format(i + 1, maxlen, len(filenames), maxlen))
                elapsed = time.time()

            #
            (_, extension) = os.path.splitext(filename)
            assert extension == ".npy"

            #
            path = os.path.join(subdirectory, filename)
            #
            with open(path, "rb") as file:
                #
                heuristics = onp.load(file)
            assert len(heuristics) == 4
        print("[{:>{:d}d}/{:>{:d}d}]".format(len(filenames), maxlen, len(filenames), maxlen))


def validate_enclose(directory: str, num_hops: int, unit: float, /) -> None:
    R"""
    Validate enclosed subgraph data.

    Args
    ----
    - directory
        Directory.
    - num_hops
        Number of hops.
    - unit
        Report unit.

    Returns
    -------
    """
    #
    print("-- Validating enclosed subgraph data:")
    subdirectory = os.path.join(directory, "hop{:d}.enclose".format(num_hops))
    if os.path.isdir(subdirectory):
        #
        filenames = list(os.listdir(subdirectory))
        elapsed = time.time()
        maxlen = len(str(len(filenames) + 1))
        for (i, filename) in enumerate(filenames):
            #
            if i == 0 or time.time() - elapsed > unit:
                #
                print("[{:>{:d}d}/{:>{:d}d}]".format(i + 1, maxlen, len(filenames), maxlen))
                elapsed = time.time()

            #
            (_, extension) = os.path.splitext(filename)
            assert extension == ".npy"

            #
            path = os.path.join(subdirectory, filename)
            #
            with open(path, "rb") as file:
                #
                vids = onp.load(file)
                vfts = onp.load(file)
                adjs = onp.load(file)
                rels = onp.load(file)
            assert len(vids) == len(vfts)
            assert len(adjs.T) == len(rels)
        print("[{:>{:d}d}/{:>{:d}d}]".format(len(filenames), maxlen, len(filenames), maxlen))


def main() -> None:
    R"""
    Main execution.

    Args
    ----

    Returns
    -------
    """
    #
    parser = argparse.ArgumentParser(description="Clean.")
    parser.add_argument("--task", type=str, required=True, help="Task.")
    parser.add_argument("--num-hops", type=int, required=True, help="Number of hops.")
    args = parser.parse_args()
    task = args.task
    num_hops = args.num_hops
    directory = os.path.join("cache", task)
    unit = 60.0

    #
    validate_forest(directory, num_hops, unit)
    validate_heuristics(directory, num_hops, unit)
    validate_enclose(directory, num_hops, unit)


if __name__ == "__main__":
    #
    main()
