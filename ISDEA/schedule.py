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
    parser = argparse.ArgumentParser(description="Schedule.")
    parser.add_argument("--data", type=str, required=True, help="Data root directory")
    parser.add_argument("--cache", type=str, required=True, help="Cache root directory")
    parser.add_argument("--task", type=str, required=True, help="Task prefix.")
    parser.add_argument("--sample", type=str, required=True, help="Sampling data type.")
    parser.add_argument(
        "--bidirect", action="store_true", help="Treat observation graph as bidirected."
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Overfit on both training and validation.",
    )
    parser.add_argument("--num-hops", type=int, required=True, help="Number of hops.")
    parser.add_argument(
        "--num-processes", type=int, required=True, help="Number of processes."
    )
    parser.add_argument(
        "--unit-process", type=float, required=True, help="Report unit of process."
    )
    parser.add_argument(
        "--num-epochs", type=int, required=True, help="Number of epochs."
    )
    parser.add_argument(
        "--batch-size-node",
        type=int,
        required=True,
        help="Batch size of node sampling.",
    )
    parser.add_argument(
        "--batch-size-edge-train",
        type=int,
        required=True,
        help="Batch size of training edge sampling.",
    )
    parser.add_argument(
        "--batch-size-edge-valid",
        type=int,
        required=True,
        help="Batch size of validation edge sampling.",
    )
    parser.add_argument(
        "--batch-size-edge-test",
        type=int,
        required=True,
        help="Batch size of test edge sampling.",
    )
    parser.add_argument(
        "--negative-rate-train",
        type=int,
        required=True,
        help="Negative sampling rate of training.",
    )
    parser.add_argument(
        "--negative-rate-eval",
        type=int,
        required=True,
        help="Negative sampling rate of evaluation.",
    )
    parser.add_argument(
        "--num-neg-rels-train",
        type=int,
        required=True,
        help="Number of negative relation samples of both training.",
    )
    parser.add_argument(
        "--num-neg-rels-eval",
        type=int,
        required=True,
        help="Number of negative relation samples of both evaluation.",
    )
    parser.add_argument(
        "--skip-forest",
        action="store_true",
        help="Skip enclosed subgraph collection and follow-ups.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Seed.")
    args = parser.parse_args()

    # Allocate caching disk space.
    task_trans = "-".join((args.task, "trans"))
    task_ind = "-".join((args.task, "ind"))
    path_dataset_trans = os.path.join(args.data, task_trans)
    path_dataset_ind = os.path.join(args.data, task_ind)
    path_cache_trans = os.path.join(
        args.cache, "~".join((task_trans, "dx{:d}".format(1 + int(args.bidirect))))
    )
    path_cache_ind = os.path.join(
        args.cache, "~".join((task_ind, "dx{:d}".format(1 + int(args.bidirect))))
    )
    os.makedirs(path_cache_trans, exist_ok=True)
    os.makedirs(path_cache_ind, exist_ok=True)

    # Allocate logging disk space.
    prefix = "~".join((args.task, "dx{:d}".format(1 + int(args.bidirect))))
    suffix = "e{:d}-s{:d}".format(args.num_epochs, args.seed)
    unique = etexood.loggings.create_framework_directory(
        os.path.join("logs", "schedule"),
        prefix,
        "_",
        suffix,
        sleep=1.1,
        max_waits=11,
    )

    # Prepare logging terminal.
    logger = etexood.loggings.create_logger(
        unique, os.path.basename(unique), level_file=None, level_console=None
    )

    # Load dataset.
    dataset_trans = etexood.datasets.DatasetTriplet.from_file(
        logger, path_dataset_trans
    )
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
        adjs_trans_observe = onp.concatenate(
            (adjs_trans_observe[[0, 1]], adjs_trans_observe[[1, 0]]), axis=1
        )
        rels_trans_observe = onp.concatenate(
            (rels_trans_observe, rels_trans_observe + num_relations_trans)
        )

    # Prepare tuning training edges.
    if len(dataset_trans.triplets_train) == len(dataset_trans.triplets_observe):
        #
        tripelts_trans_train = dataset_trans.triplets_train
        adjs_trans_train = tripelts_trans_train[:, :2].T
        rels_trans_train = tripelts_trans_train[:, 2]
    else:
        #
        tripelts_trans_train = dataset_trans.triplets_train[
            len(dataset_trans.triplets_observe) :
        ]
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
        adjs_ind_observe = onp.concatenate(
            (adjs_ind_observe[[0, 1]], adjs_ind_observe[[1, 0]]), axis=1
        )
        rels_ind_observe = onp.concatenate(
            (rels_ind_observe, rels_ind_observe + num_relations_ind)
        )

    # Prepare test test graph.
    tripelts_ind_test = dataset_ind.triplets_test
    adjs_ind_test = tripelts_ind_test[:, :2].T
    rels_ind_test = tripelts_ind_test[:, 2]

    # Check data.
    starter_trans = int(
        len(dataset_trans.triplets_observe) != len(dataset_trans.triplets_train)
    )
    starter_trans *= len(dataset_trans.triplets_observe)

    #
    assert len(dataset_trans.triplets_observe) * (1 + int(args.bidirect)) == len(
        rels_trans_observe
    )
    assert adjs_trans_observe.ndim == 2 and tuple(adjs_trans_observe.shape) == (
        2,
        len(rels_trans_observe),
    )
    assert len(rels_trans_observe) > 0
    assert len(dataset_trans.triplets_train[starter_trans:]) == len(rels_trans_train)
    assert adjs_trans_train.ndim == 2 and tuple(adjs_trans_train.shape) == (
        2,
        len(rels_trans_train),
    )
    assert len(rels_trans_train) > 0
    assert len(dataset_trans.triplets_valid) == len(rels_trans_valid)
    assert adjs_trans_valid.ndim == 2 and tuple(adjs_trans_valid.shape) == (
        2,
        len(rels_trans_valid),
    )
    assert len(rels_trans_valid) > 0
    assert len(dataset_trans.triplets_test) == 0

    #
    assert len(dataset_ind.triplets_observe) * (1 + int(args.bidirect)) == len(
        rels_ind_observe
    )
    assert adjs_ind_observe.ndim == 2 and tuple(adjs_ind_observe.shape) == (
        2,
        len(rels_ind_observe),
    )
    assert len(rels_ind_observe) > 0
    assert len(dataset_ind.triplets_train) == len(dataset_ind.triplets_observe)
    assert len(dataset_ind.triplets_valid) == 0
    assert len(dataset_ind.triplets_test) == len(rels_ind_test)
    assert adjs_ind_test.ndim == 2 and tuple(adjs_ind_test.shape) == (
        2,
        len(rels_ind_test),
    )
    assert len(rels_ind_test) > 0

    # To fit with NBFNet design by corrupting only object with inversion augmentation.
    if args.bidirect:
        #
        logger.info("-- Augment training, validation and test by inversion.")
        adjs_trans_train = onp.concatenate(
            (adjs_trans_train[[0, 1]], adjs_trans_train[[1, 0]]), axis=1
        )
        rels_trans_train = onp.concatenate(
            (rels_trans_train, rels_trans_train + num_relations_trans)
        )
        adjs_trans_valid = onp.concatenate(
            (adjs_trans_valid[[0, 1]], adjs_trans_valid[[1, 0]]), axis=1
        )
        rels_trans_valid = onp.concatenate(
            (rels_trans_valid, rels_trans_valid + num_relations_trans)
        )
        adjs_ind_test = onp.concatenate(
            (adjs_ind_test[[0, 1]], adjs_ind_test[[1, 0]]), axis=1
        )
        rels_ind_test = onp.concatenate(
            (rels_ind_test, rels_ind_test + num_relations_ind)
        )

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

    #
    assert args.sample == "heuristics", "Only heuristics version can generate schedule."

    #
    logger.info("-- Preprocess schedule:")
    trainer = etexood.frameworks.transform.Trainer(
        logger,
        num_nodes_trans,
        adjs_trans_observe,
        rels_trans_observe,
        path_cache_trans,
        args.sample,
        bidirect=args.bidirect,
        num_relations=num_relations_trans,
        num_hops=args.num_hops,
        num_processes=args.num_processes,
        unit_process=args.unit_process,
        skip_forest=args.skip_forest,
        device=torch.device("cpu"),
    )
    validator = etexood.frameworks.transform.Evaluator(
        logger,
        num_nodes_trans,
        adjs_trans_observe,
        rels_trans_observe,
        path_cache_trans,
        args.sample,
        bidirect=args.bidirect,
        num_relations=num_relations_trans,
        num_hops=args.num_hops,
        num_processes=args.num_processes,
        unit_process=args.unit_process,
        skip_forest=args.skip_forest,
        device=torch.device("cpu"),
    )
    tester = etexood.frameworks.transform.Evaluator(
        logger,
        num_nodes_ind,
        adjs_ind_observe,
        rels_ind_observe,
        path_cache_ind,
        args.sample,
        bidirect=args.bidirect,
        num_relations=num_relations_ind,
        num_hops=args.num_hops,
        num_processes=args.num_processes,
        unit_process=args.unit_process,
        skip_forest=args.skip_forest,
        device=torch.device("cpu"),
    )

    # Adjust test negative ratio by half since test cases are augmented by twice.
    # For NBFNet negative sampling.
    logger.critical("-- Generate training schedule:")
    trainer.generate(
        adjs_trans_train,
        rels_trans_train,
        args.num_epochs,
        batch_size_node=args.batch_size_node,
        batch_size_edge=args.batch_size_edge_train
        * (1 + args.negative_rate_train + args.num_neg_rels_train),
        negative_rate=args.negative_rate_train,
        num_neg_rels=args.num_neg_rels_train,
        seed=args.seed + 1,
        reusable_edge=False,
    )
    logger.critical("-- Generate validation schedule:")
    validator.generate(
        adjs_trans_valid,
        rels_trans_valid,
        1,
        batch_size_node=args.batch_size_node,
        batch_size_edge=args.batch_size_edge_valid
        * (1 + args.negative_rate_eval + args.num_neg_rels_eval),
        negative_rate=args.negative_rate_eval,
        num_neg_rels=args.num_neg_rels_eval,
        seed=args.seed + 2,
        reusable_edge=True,
    )
    logger.critical("-- Generate test schedule:")
    assert args.negative_rate_eval % 2 == 0
    assert args.num_neg_rels_eval % 2 == 0
    test_negative_rate_eval = args.negative_rate_eval
    test_num_neg_rels = args.num_neg_rels_eval
    tester.generate(
        adjs_ind_test,
        rels_ind_test,
        1,
        batch_size_node=args.batch_size_node,
        batch_size_edge=args.batch_size_edge_test
        * (1 + test_negative_rate_eval + test_num_neg_rels),
        negative_rate=test_negative_rate_eval,
        num_neg_rels=test_num_neg_rels,
        seed=args.seed + 3,
        reusable_edge=True,
    )


if __name__ == "__main__":
    #
    main()
