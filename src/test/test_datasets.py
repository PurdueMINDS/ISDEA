#
import os
import pytest
import etexood
import shutil
import numpy as onp


#
ROOT = os.path.join("debug", "log")
DATA = "data"


#
TEST_FROM_FILE = [
    os.path.join(DATA, "FD1-trans"),
    os.path.join(DATA, "FD1-ind"),
    os.path.join(DATA, "FD2-trans"),
    os.path.join(DATA, "FD2-ind"),
    os.path.join(DATA, "WN18RR1-trans"),
    os.path.join(DATA, "WN18RR1-ind"),
    os.path.join(DATA, "WN18RR4-trans"),
    os.path.join(DATA, "WN18RR4-ind"),
    os.path.join(DATA, "WN18RR2-trans"),
    os.path.join(DATA, "WN18RR2-ind"),
    os.path.join(DATA, "WN18RR3-trans"),
    os.path.join(DATA, "WN18RR3-ind"),
    os.path.join(DATA, "NELL9951-trans"),
    os.path.join(DATA, "NELL9951-ind"),
]


def test_ini() -> None:
    R"""
    Test initialization.

    Args
    ----

    Returns
    -------
    """


@pytest.mark.parametrize("path", TEST_FROM_FILE)
def test_from_file(*, path: str) -> None:
    R"""
    Test initialized from file.

    Args
    ----
    - path
        Path.

    Returns
    -------
    """
    #
    prefix = "~".join(["datasets", "from-file"])
    suffix = os.path.basename(path)
    unique = etexood.loggings.create_framework_directory(ROOT, prefix, "", suffix, sleep=1.1, max_waits=11)
    logger = etexood.loggings.create_logger(unique, os.path.basename(unique), level_file=None, level_console=None)

    #
    (
        triplets_train,
        triplets_valid,
        triplets_test,
        observe,
        entity2id,
        relation2id,
    ) = etexood.datasets.DatasetTriplet.from_file_(logger, path)

    #
    dataset = etexood.datasets.DatasetTriplet.from_file(logger, path)

    # Triplets use naive content as file defined.
    assert onp.all(dataset.triplets_train == triplets_train).item()
    assert onp.all(dataset.triplets_valid == triplets_valid).item()
    assert onp.all(dataset.triplets_test == triplets_test).item()

    # Observation triplets is a heading subset of training.
    assert len(dataset.triplets_observe) == observe and len(dataset.triplets_train) >= observe
    assert onp.all(dataset.triplets_observe == dataset.triplets_train[:observe]).item()

    # Entity and relation mapping uses naive content.
    for (name2id0, name2id1) in ((dataset._entity2id, entity2id), (dataset._relation2id, relation2id)):
        #
        assert len(name2id0) == len(name2id1)
        for ((name0, id0), (name1, id1)) in zip(name2id0.items(), name2id1.items()):
            #
            assert name0 == name1 and id0 == id1


def test_fin() -> None:
    R"""
    Test finalization.

    Args
    ----

    Returns
    -------
    """


def main():
    R"""
    Main execution.

    Args
    ----

    Returns
    -------
    """
    #
    test_ini()
    for path in TEST_FROM_FILE:
        #
        test_from_file(path=path)
    test_fin()


#
if __name__ == "__main__":
    #
    main()
