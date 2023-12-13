#
import os
import datetime
import time
import logging
from typing import Optional


def create_framework_directory(
    root: str,
    prefix: str,
    identifier: str,
    suffix: str,
    /,
    *,
    sleep: float,
    max_waits: int,
) -> str:
    R"""
    Create unique directory for given framework.

    Args
    ----
    - root
        Root directory.
    - prefix
        Prefix of framework name.
        If empty, it will use an underline character.
    - identifier
        Identifier used to ensure uniqueness.
        If empty, it will use creation time.
    - suffix
        Suffix of framework name.
        If empty, it will use an underline character.
    - sleep
        Sleep time unit to avoid naming conflict by creation time.
    - max_waits
        Maximum number of waiting times before raising error.

    Returns
    -------
    - unique
        Unique directory.
    """
    #
    for _ in range(max_waits):
        #
        now = datetime.datetime.now()
        unique = os.path.join(
            root,
            ":".join(
                [
                    prefix if prefix else "_",
                    identifier
                    if identifier
                    else "{:>04d}{:>02d}{:>02d}{:>02d}{:>02d}{:>02d}".format(
                        now.year,
                        now.month,
                        now.day,
                        now.hour,
                        now.minute,
                        now.second,
                    ),
                    suffix if suffix else "_",
                ],
            ),
        )
        if os.path.isdir(unique):
            #
            time.sleep(sleep)
        else:
            #
            break
    if os.path.isdir(unique):
        #
        raise RuntimeError(
            'Fail to create unque directory in format "{:s}:${{identifier}}:{:s}" in {:d} trials.'.format(
                prefix if prefix else "_",
                suffix if suffix else "_",
                max_waits,
            )
        )

    #
    os.makedirs(unique)
    return unique


def create_logger(path: str, title: str, /, level_file: Optional[int], level_console: Optional[int]) -> logging.Logger:
    R"""
    Create logging terminal.

    Args
    ----
    - path
        Unique directory for given framework.
    - title
        Framework name.
        It should be basename of framework directory.
    - level_file
        Logging level for file handler.
    - level_console
        Logging level for console handler.

    Returns
    -------
    - logger
        Logging terminal.
    """
    #
    logger = logging.getLogger(title)
    logger.setLevel(
        min(
            logging.DEBUG if level_file is None else level_file,
            logging.INFO if level_console is None else level_console,
        ),
    )

    # Create file handler.
    handler_file = logging.FileHandler(os.path.join(path, "logging.txt"))
    handler_file.setLevel(logging.DEBUG if level_file is None else level_file)
    formatter_file = logging.Formatter("[%(levelname)8s] %(message)s @%(filename)s:%(lineno)d")
    handler_file.setFormatter(formatter_file)

    # Create console handler.
    handler_console = logging.StreamHandler()
    handler_console.setLevel(logging.INFO if level_console is None else level_console)
    formatter_console = logging.Formatter("[%(levelname)8s] \x1b[97m%(message)s\x1b[0m @%(filename)s:%(lineno)d")
    handler_console.setFormatter(formatter_console)

    #
    logger.addHandler(handler_file)
    logger.addHandler(handler_console)
    return logger
