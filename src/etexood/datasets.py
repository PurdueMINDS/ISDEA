#
import os
import logging
import re
import numpy as onp
from typing import TypeVar, Type, Dict, Tuple, List, Mapping
from types import MappingProxyType
from .dtypes import NPINTS


#
SelfDatasetTriplet = TypeVar("SelfDatasetTriplet", bound="DatasetTriplet")


class DatasetTriplet(object):
    R"""
    Triplet dataset.
    """
    #
    READABLES = (
        "entities.dict",
        "relations.dict",
        "observe.txt",
        "train.txt",
        "valid.txt",
        "test.txt",
        "LICENSE",
        "README",
    )

    #
    OBSERVE = 0
    TRAIN = 1
    VALID = 2
    TEST = 3

    def __init__(
        self: SelfDatasetTriplet,
        logger: logging.Logger,
        triplets_train: NPINTS,
        triplets_valid: NPINTS,
        triplets_test: NPINTS,
        observe: int,
        entity2id: Dict[str, int],
        relation2id: Dict[str, int],
        /,
    ) -> None:
        R"""
        Initialize the class.

        Args
        ----
        - logger
            Logging terminal.
        - triplets_train
            Training triplets.
            If observed triplets exist, it will be a subset of this.
        - triplets_valid
            Validation triplets.
        - triplets_test
            Test triplets.
        - observe
            Number of heading observed triplets in training.
        - entity2id
            Translation from entity names to consecutive IDs.
        - relation2id
            Translation from relation names to consecutive IDs.

        Returns
        -------
        """
        #
        self._logger = logger

        #
        self._bgn_observe = 0
        self._len_observe = observe
        self._end_observe = self._bgn_observe + self._len_observe

        #
        self._bgn_train = 0
        self._len_train = len(triplets_train)
        self._end_train = self._bgn_train + self._len_train

        #
        self._bgn_valid = self._end_train
        self._len_valid = len(triplets_valid)
        self._end_valid = self._bgn_valid + self._len_valid

        #
        self._bgn_test = self._end_valid
        self._len_test = len(triplets_test)
        self._end_test = self._bgn_test + self._len_test

        #
        self._triplets = onp.concatenate((triplets_train, triplets_valid, triplets_test))
        self._entity2id = MappingProxyType(entity2id)
        self._relation2id = MappingProxyType(relation2id)

        # Lock data.
        self._triplets.setflags(write=False)

        #
        maxlen = max(
            len(str(length)) for length in (self._len_observe, self._len_train, self._len_valid, self._len_test)
        )

        #
        self._logger.critical("-- Dataset Summary:")
        self._logger.critical("[  Observed]: {:>{:d}d} triplets.".format(len(self.triplets_observe), maxlen))
        self._logger.critical("[  Training]: {:>{:d}d} triplets.".format(len(self.triplets_train), maxlen))
        self._logger.critical("[Validation]: {:>{:d}d} triplets.".format(len(self.triplets_valid), maxlen))
        self._logger.critical("[      Test]: {:>{:d}d} triplets.".format(len(self.triplets_test), maxlen))

    @property
    def triplets_observe(self: SelfDatasetTriplet, /) -> NPINTS:
        R"""
        Observed triplets.

        Args
        ----

        Returns
        -------
        - triplets
            Triplets.
        """
        #
        return self._triplets[self._bgn_observe : self._end_observe]

    @property
    def triplets_train(self: SelfDatasetTriplet, /) -> NPINTS:
        R"""
        Training triplets.

        Args
        ----

        Returns
        -------
        - triplets
            Triplets.
        """
        #
        return self._triplets[self._bgn_train : self._end_train]

    @property
    def triplets_valid(self: SelfDatasetTriplet, /) -> NPINTS:
        R"""
        Validation triplets.

        Args
        ----

        Returns
        -------
        - triplets
            Triplets.
        """
        #
        return self._triplets[self._bgn_valid : self._end_valid]

    @property
    def triplets_test(self: SelfDatasetTriplet, /) -> NPINTS:
        R"""
        Test triplets.

        Args
        ----

        Returns
        -------
        - triplets
            Triplets.
        """
        #
        return self._triplets[self._bgn_test : self._end_test]

    @classmethod
    def from_file(cls: Type[SelfDatasetTriplet], logger: logging.Logger, path: str, /) -> SelfDatasetTriplet:
        R"""
        Initialize the class from file.

        Args
        ----
        - logger
            Logging terminal.
        - path
            Path.

        Returns
        -------
        - self
            Instaniated class.
        """
        #
        return cls(logger, *cls.from_file_(logger, path))

    @classmethod
    def from_file_(
        cls: Type[SelfDatasetTriplet],
        logger: logging.Logger,
        path: str,
        /,
    ) -> Tuple[NPINTS, NPINTS, NPINTS, int, Dict[str, int], Dict[str, int]]:
        R"""
        Load all essential data from file.

        Args
        ----
        - logger
            Logging terminal.
        - path
            Path.

        Returns
        -------
        - triplets_train
            Training triplets.
        - triplets_valid
            Validation triplets.
        - triplets_test
            Test triplets.
        - observe
            Number of heading observed triplets in training.
        - entity2id
            Translation from entity names to consecutive IDs.
        - relation2id
            Translation from relation names to consecutive IDs.
        """
        # Ensure dataset directory is strictly clean.
        for name in os.listdir(path):
            #
            assert name in cls.READABLES, 'Unexpected content "{:s}" in data directory "{:s}".'.format(name, path)

        #
        logger.info("-- Entity mapping:")
        entity2id = cls.from_file_name2id(logger, os.path.join(path, "entities.dict"))

        #
        logger.info("-- Relation mapping:")
        relation2id = cls.from_file_name2id(logger, os.path.join(path, "relations.dict"))

        #
        logger.info("-- Observed triplets:")
        triplets_observe = cls.from_file_triplet(logger, os.path.join(path, "observe.txt"), entity2id, relation2id)

        #
        logger.info("-- Training triplets:")
        triplets_train = cls.from_file_triplet(logger, os.path.join(path, "train.txt"), entity2id, relation2id)

        #
        logger.info("-- Validation triplets:")
        triplets_valid = cls.from_file_triplet(logger, os.path.join(path, "valid.txt"), entity2id, relation2id)

        #
        logger.info("-- Test triplets:")
        triplets_test = cls.from_file_triplet(logger, os.path.join(path, "test.txt"), entity2id, relation2id)

        #
        logger.info("-- Merge observation ahead of training:")
        triplets_train = onp.concatenate((triplets_observe, triplets_train))
        logger.info(
            "Observation size changes from {:d} to {:d}.".format(
                len(triplets_observe),
                len(triplets_observe) if len(triplets_observe) else len(triplets_train),
            ),
        )
        logger.info(
            "Training size changes from {:d} to {:d}.".format(
                len(triplets_train) - len(triplets_observe),
                len(triplets_train),
            ),
        )

        #
        return (
            triplets_train,
            triplets_valid,
            triplets_test,
            len(triplets_observe) if len(triplets_observe) else len(triplets_train),
            entity2id,
            relation2id,
        )

    @staticmethod
    def from_file_name2id(logger: logging.Logger, path: str, /) -> Dict[str, int]:
        R"""
        Load mapping from string name to integer ID from file.

        Args
        ----
        - logger
            Logging terminal.
        - path
            Path.

        Returns
        -------
        - name2id
            Mapping from name to ID.
        """
        #
        with open(path, "r") as file:
            #
            name2id = dict()
            for line in file:
                #
                (id_, name) = re.split(r"\s+", line.strip())
                name2id[name] = int(id_)
        logger.info("Load {:,d} mappings from string name to integer ID.".format(len(name2id)))
        return name2id

    @staticmethod
    def from_file_triplet(
        logger: logging.Logger,
        path: str,
        entity2id: Dict[str, int],
        relation2id: Dict[str, int],
        /,
    ) -> NPINTS:
        R"""
        Load triplets from file.

        Args
        ----
        - logger
            Logging terminal.
        - path
            Path.
        - entity2id
            Translation from entity names to consecutive IDs.
        - relation2id
            Translation from relation names to consecutive IDs.

        Returns
        -------
        - triplets
            All tripelts as an array.
        """
        #
        triplets = []
        if os.path.isfile(path):
            #
            with open(path, "r") as file:
                #
                for line in file:
                    #
                    (sub_, rel_, obj_) = re.split(r"\s+", line.strip())
                    if sub_ in entity2id:
                        #
                        sub = entity2id[sub_]
                    if rel_ in relation2id:
                        #
                        rel = relation2id[rel_]
                    if obj_ in entity2id:
                        #
                        obj = entity2id[obj_]
                    triplets.append((sub, obj, rel))
        logger.info("Load {:,d} triplet(s).".format(len(triplets)))

        #
        return onp.reshape(onp.array(triplets, dtype=onp.int64), (len(triplets), 3))
