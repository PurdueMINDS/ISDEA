#
import argparse
import os
import etexood
import numpy as onp
import torch


def main() -> None:
    R"""
    Main execution.

    Args
    ----

    Returns
    -------
    """
    #
    parser = argparse.ArgumentParser(description="Enclose.")
    parser.add_argument("--data", type=str, required=True, help="Data root directory")
    parser.add_argument("--cache", type=str, required=True, help="Cache root directory")
    parser.add_argument("--task", type=str, required=True, help="Task prefix.")
    parser.add_argument("--bidirect", action="store_true", help="Treat observation graph as bidirected.")
    parser.add_argument("--overfit", action="store_true", help="Overfit on both training and validation.")
    parser.add_argument("--num-hops", type=int, required=True, help="Number of hops.")
    parser.add_argument("--num-processes", type=int, required=True, help="Number of processes.")
    parser.add_argument("--unit-process", type=float, required=True, help="Report unit of process.")
    args = parser.parse_args()

    # Allocate caching disk space.
    task_trans = "-".join((args.task, "trans"))
    task_ind = "-".join((args.task, "ind"))
    path_dataset_trans = os.path.join(args.data, task_trans)
    path_dataset_ind = os.path.join(args.data, task_ind)
    path_cache_trans = os.path.join(args.cache, "~".join((task_trans, "dx{:d}".format(1 + int(args.bidirect)))))
    path_cache_ind = os.path.join(args.cache, "~".join((task_ind, "dx{:d}".format(1 + int(args.bidirect)))))
    os.makedirs(path_cache_trans, exist_ok=True)
    os.makedirs(path_cache_ind, exist_ok=True)

    # Allocate logging disk space.
    prefix = "~".join((args.task, "dx{:d}".format(1 + int(args.bidirect))))
    suffix = "_"
    unique = etexood.loggings.create_framework_directory(
        os.path.join("logs", "enclose"),
        prefix,
        "_",
        suffix,
        sleep=1.1,
        max_waits=11,
    )

    # Prepare logging terminal.
    logger = etexood.loggings.create_logger(unique, os.path.basename(unique), level_file=None, level_console=None)

    # Load dataset.
    dataset_trans = etexood.datasets.DatasetTriplet.from_file(logger, path_dataset_trans)
    dataset_ind = etexood.datasets.DatasetTriplet.from_file(logger, path_dataset_ind)

    # Prepare tuning observed graph.
    num_nodes_trans = len(dataset_trans._entity2id)
    num_relations_trans = len(dataset_trans._relation2id)
    tripelts_trans_observe = dataset_trans.triplets_observe
    adjs_trans_observe = tripelts_trans_observe[:, :2].T
    rels_trans_observe = tripelts_trans_observe[:, 2]

    #
    if args.bidirect:
        #
        logger.info("-- Augment observation by inversion.")
        adjs_trans_observe = onp.concatenate((adjs_trans_observe[[0, 1]], adjs_trans_observe[[1, 0]]), axis=1)
        rels_trans_observe = onp.concatenate((rels_trans_observe, rels_trans_observe + num_relations_trans))

    # Prepare tuning training edges.
    if len(dataset_trans.triplets_train) == len(dataset_trans.triplets_observe):
        #
        tripelts_trans_train = dataset_trans.triplets_train
        adjs_trans_train = tripelts_trans_train[:, :2].T
        rels_trans_train = tripelts_trans_train[:, 2]
    else:
        #
        tripelts_trans_train = dataset_trans.triplets_train[len(dataset_trans.triplets_observe) :]
        adjs_trans_train = tripelts_trans_train[:, :2].T
        rels_trans_train = tripelts_trans_train[:, 2]

    # Prepare tuning validation edges.
    tripelts_trans_valid = dataset_trans.triplets_valid
    adjs_trans_valid = tripelts_trans_valid[:, :2].T
    rels_trans_valid = tripelts_trans_valid[:, 2]

    # Prepare test observed graph.
    num_nodes_ind = len(dataset_ind._entity2id)
    num_relations_ind = len(dataset_ind._relation2id)
    tripelts_ind_observe = dataset_ind.triplets_observe
    adjs_ind_observe = tripelts_ind_observe[:, :2].T
    rels_ind_observe = tripelts_ind_observe[:, 2]

    #
    if args.bidirect:
        #
        logger.info("-- Augment observation by inversion.")
        adjs_ind_observe = onp.concatenate((adjs_ind_observe[[0, 1]], adjs_ind_observe[[1, 0]]), axis=1)
        rels_ind_observe = onp.concatenate((rels_ind_observe, rels_ind_observe + num_relations_ind))

    # Prepare test test graph.
    tripelts_ind_test = dataset_ind.triplets_test
    adjs_ind_test = tripelts_ind_test[:, :2].T
    rels_ind_test = tripelts_ind_test[:, 2]

    # Lock triplets in memory.
    adjs_trans_observe.setflags(write=False)
    rels_trans_observe.setflags(write=False)
    adjs_trans_train.setflags(write=False)
    rels_trans_train.setflags(write=False)
    adjs_trans_valid.setflags(write=False)
    rels_trans_valid.setflags(write=False)
    adjs_ind_observe.setflags(write=False)
    rels_ind_observe.setflags(write=False)
    adjs_ind_test.setflags(write=False)
    rels_ind_test.setflags(write=False)

    # Check data.
    starter_trans = int(len(dataset_trans.triplets_observe) != len(dataset_trans.triplets_train))
    starter_trans *= len(dataset_trans.triplets_observe)

    #
    assert len(dataset_trans.triplets_observe) * (1 + int(args.bidirect)) == len(rels_trans_observe)
    assert adjs_trans_observe.ndim == 2 and tuple(adjs_trans_observe.shape) == (2, len(rels_trans_observe))
    assert len(rels_trans_observe) > 0
    assert len(dataset_trans.triplets_train[starter_trans:]) == len(rels_trans_train)
    assert adjs_trans_train.ndim == 2 and tuple(adjs_trans_train.shape) == (2, len(rels_trans_train))
    assert len(rels_trans_train) > 0
    assert len(dataset_trans.triplets_valid) == len(rels_trans_valid)
    assert adjs_trans_valid.ndim == 2 and tuple(adjs_trans_valid.shape) == (2, len(rels_trans_valid))
    assert len(rels_trans_valid) > 0
    assert len(dataset_trans.triplets_test) == 0

    #
    assert len(dataset_ind.triplets_observe) * (1 + int(args.bidirect)) == len(rels_ind_observe)
    assert adjs_ind_observe.ndim == 2 and tuple(adjs_ind_observe.shape) == (2, len(rels_ind_observe))
    assert len(rels_ind_observe) > 0
    assert len(dataset_ind.triplets_train) == len(dataset_ind.triplets_observe)
    assert len(dataset_ind.triplets_valid) == 0
    assert len(dataset_ind.triplets_test) == len(rels_ind_test)
    assert adjs_ind_test.ndim == 2 and tuple(adjs_ind_test.shape) == (2, len(rels_ind_test))
    assert len(rels_ind_test) > 0

    #
    logger.info("-- Translate schedule:")
    translator_train = etexood.batches.enclose.Enclose(
        path_cache_trans,
        num_nodes_trans,
        adjs_trans_observe,
        rels_trans_observe,
        num_hops=args.num_hops,
        num_processes=args.num_processes,
        unit=args.unit_process,
    )
    translator_valid = etexood.batches.enclose.Enclose(
        path_cache_trans,
        num_nodes_trans,
        adjs_trans_observe,
        rels_trans_observe,
        num_hops=args.num_hops,
        num_processes=args.num_processes,
        unit=args.unit_process,
    )
    translator_test = etexood.batches.enclose.Enclose(
        path_cache_ind,
        num_nodes_ind,
        adjs_ind_observe,
        rels_ind_observe,
        num_hops=args.num_hops,
        num_processes=args.num_processes,
        unit=args.unit_process,
    )

    #
    logger.critical("-- Translate training schedule:")
    translator_train.translate()
    logger.critical("-- Generate validation schedule:")
    translator_valid.translate()
    logger.critical("-- Generate test schedule:")
    translator_test.translate()


if __name__ == "__main__":
    #
    main()
