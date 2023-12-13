#
import os
import pytest
import etexood
import shutil


#
ROOT = os.path.join("debug", "log")


def test_ini() -> None:
    R"""
    Test initialization.

    Args
    ----

    Returns
    -------
    """
    #
    if os.path.isdir(ROOT):
        #
        shutil.rmtree(ROOT)
    while os.path.isdir(ROOT):
        #
        pass


@pytest.mark.xfail(raises=RuntimeError)
def test_directory_wait() -> None:
    R"""
    Test directory creation with waiting.

    Args
    ----

    Returns
    -------
    """
    # Run twice with constant identifier to produce conflict.
    etexood.loggings.create_framework_directory(ROOT, "loggings", "_", "directory-wait", sleep=0.05, max_waits=1)
    etexood.loggings.create_framework_directory(ROOT, "loggings", "_", "directory-wait", sleep=0.05, max_waits=2)


def test_fin() -> None:
    R"""
    Test finalization.

    Args
    ----

    Returns
    -------
    """
    #
    if os.path.isdir(ROOT):
        #
        shutil.rmtree(ROOT)
    while os.path.isdir(ROOT):
        #
        pass


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
    for test_xfail in [test_directory_wait]:
        #
        try:
            #
            test_xfail()
        except RuntimeError:
            #
            pass
        else:
            #
            raise RuntimeError("Expect failure, but pass successfully.")
    test_fin()


#
if __name__ == "__main__":
    #
    main()
