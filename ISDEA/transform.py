#
import argparse
import os
import etexood
import numpy as onp
import torch
import re
import json
import math


def main() -> None:
    R"""
    Main execution.

    Args
    ----

    Returns
    -------
    """
    #
    parser = argparse.ArgumentParser(description="Transform.")
    parser.add_argument(
        "--resume", type=str, required=True, help="Resuming log directory."
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle node and relation IDs in testing.",
    )
    parser.add_argument("--special", type=str, default="", help="Special test dataset.")
    parser.add_argument(
        "--resume-param", type=str, required=False, help="Resuming parameter directory."
    )
    args = parser.parse_args()
    resume = args.resume
    shuffle = args.shuffle
    path_special = args.special
    resume_param = args.resume_param
    shuffle_seed = 42

    #
    with open(os.path.join(resume, "arguments.json"), "r") as file:
        #
        args = argparse.Namespace(**json.load(file))
    if args.ks == "1,5,10":
        #
        args.ks = "1,3,5,10"

    # Allocate caching disk space.
    task = "-".join((args.task, "ind"))
    if len(path_special) > 0:
        #
        path_dataset = path_special
        path_cache = os.path.join(
            args.cache,
            "~".join(
                (
                    os.path.basename(path_special),
                    "dx{:d}".format(1 + int(args.bidirect)),
                )
            ),
        )
    else:
        #
        path_dataset = os.path.join(args.data, task)
        path_cache = os.path.join(
            args.cache, "~".join((task, "dx{:d}".format(1 + int(args.bidirect))))
        )
    os.makedirs(path_cache, exist_ok=True)

    # Allocate logging disk space.
    if len(path_special) > 0:
        #
        (title_special, _) = os.path.basename(path_special).split("-")
        prefix = "~".join(
            [
                title_special,
                "dx{:d}".format(1 + int(args.bidirect)),
                "X",
                "-".join([args.model, args.dss_aggr, args.ablate]),
            ]
        )
    else:
        #
        prefix = "~".join(
            [
                args.task,
                "dx{:d}".format(1 + int(args.bidirect)),
                str(int(shuffle)),
                "-".join([args.model, args.dss_aggr, args.ablate]),
            ]
        )
    suffix = "~".join(
        (
            "e{:d}-ss{:d}".format(args.num_epochs, args.seed_schedule),
            "l{:d}-sm{:d}".format(int(-math.log10(float(args.lr))), args.seed_model),
        )
    )
    unique = etexood.loggings.create_framework_directory(
        os.path.join("logs", "transform"),
        prefix,
        "_",
        suffix,
        sleep=1.1,
        max_waits=11,
    )

    #
    with open(os.path.join(unique, "arguments.json"), "w") as file:
        #
        json.dump({"resume": resume}, file, indent=4)

    # Prepare logging terminal.
    logger = etexood.loggings.create_logger(
        unique, os.path.basename(unique), level_file=None, level_console=None
    )

    # Load dataset.
    print(path_dataset)
    dataset = etexood.datasets.DatasetTriplet.from_file(logger, path_dataset)

    #
    num_nodes = len(dataset._entity2id)
    num_relations = len(dataset._relation2id)

    # Prepare observed graph.
    tripelts_observe = dataset.triplets_observe
    adjs_observe = tripelts_observe[:, :2].T
    rels_observe = tripelts_observe[:, 2]

    #
    if args.bidirect:
        #
        logger.info("-- Augment observation by inversion.")
        adjs_observe = onp.concatenate(
            (adjs_observe[[0, 1]], adjs_observe[[1, 0]]), axis=1
        )
        rels_observe = onp.concatenate((rels_observe, rels_observe + num_relations))

    # Prepare test edges.
    tripelts_test = dataset.triplets_test
    adjs_test = tripelts_test[:, :2].T
    rels_test = tripelts_test[:, 2]

    # Check size.
    assert len(dataset.triplets_observe) * (1 + int(args.bidirect)) == len(rels_observe)
    assert adjs_observe.ndim == 2 and tuple(adjs_observe.shape) == (
        2,
        len(rels_observe),
    )
    assert len(rels_observe) > 0
    assert len(dataset.triplets_test) == len(rels_test)
    assert adjs_test.ndim == 2 and tuple(adjs_test.shape) == (2, len(rels_test))
    assert len(rels_test) > 0

    #
    assert onp.max(dataset._triplets[:, 2]) < onp.max(dataset._triplets[:, :2])

    # Check content.
    assert onp.all(
        dataset.triplets_observe[:, :2]
        == adjs_observe.T[: len(rels_observe) // (1 + int(args.bidirect))]
    ).item()
    assert onp.all(dataset.triplets_test[:, :2] == adjs_test.T)

    #
    assert onp.all(
        dataset.triplets_observe[:, 2]
        == rels_observe[: len(rels_observe) // (1 + int(args.bidirect))]
    ).item()
    assert onp.all(dataset.triplets_test[:, 2] == rels_test)

    # To fit with NBFNet design by corrupting only object with inversion augmentation.
    if args.bidirect:
        #
        logger.info("-- Augment test by inversion.")
        adjs_test = onp.concatenate((adjs_test[[0, 1]], adjs_test[[1, 0]]), axis=1)
        rels_test = onp.concatenate((rels_test, rels_test + num_relations))

    #
    if shuffle:
        #
        perm_relation = onp.array(
            list(reversed(range(num_relations * (1 + args.bidirect))))
        )
    else:
        #
        perm_relation = onp.array(list(range(num_relations * (1 + args.bidirect))))
    rels_observe = perm_relation[rels_observe]
    rels_test = perm_relation[rels_test]

    # Lock triplets in memory.
    adjs_observe.setflags(write=False)
    rels_observe.setflags(write=False)
    adjs_test.setflags(write=False)
    rels_test.setflags(write=False)

    #
    skip_forest = {"dss": True, "dist": False, "both": False}[args.ablate]

    #
    logger.info("-- Load preprocessed schedule:")
    tester = etexood.frameworks.transform.Evaluator(
        logger,
        num_nodes,
        adjs_observe,
        rels_observe,
        path_cache,
        args.sample,
        bidirect=args.bidirect,
        num_relations=num_relations,
        num_hops=args.num_hops,
        num_processes=args.num_processes,
        unit_process=args.unit_process,
        skip_forest=skip_forest,
        device=torch.device(args.device),
    )

    # Adjust test negative ratio by half since test cases are augmented by twice.
    # For NBFNet negative sampling.
    test_negative_rate_eval = args.negative_rate_eval
    test_num_neg_rels_eval = args.num_neg_rels_eval
    tester.load(
        1,
        batch_size_node=args.batch_size_node,
        batch_size_edge=args.batch_size_edge_test
        * (1 + test_negative_rate_eval + test_num_neg_rels_eval),
        negative_rate=test_negative_rate_eval,
        num_neg_rels=test_num_neg_rels_eval,
        seed=args.seed_schedule + 3,
        reusable_edge=True,
    )
    assert len(tester._minibatch_edge_heuristics._schedule) == 1
    for prep_batch in tester._minibatch_edge_heuristics._schedule[0]:
        #
        assert len(prep_batch) == 4
        prep_batch[2] = perm_relation[prep_batch[2]]

    #
    logger.info("-- Load a pretrained model:")
    model = (
        etexood.models.create_model(
            num_nodes,
            num_relations * (1 + int(args.bidirect)),
            args.num_hops,
            args.hidden,
            args.model,
            {
                "activate": args.activate,
                "dropout": args.dropout,
                "num_bases": args.num_bases,
                "kernel": "gin",
                "train_eps": True,
                "dss_aggr": args.dss_aggr,
                "ablate": args.ablate,
            },
        )
        .reset_parameters(torch.Generator("cpu").manual_seed(args.seed_model))
        .to(torch.device(args.device))
    )
    if resume_param is None:
        #
        state_dict = torch.load(
            os.path.join(resume, "parameters"), map_location=torch.device(args.device)
        )
    else:
        #
        state_dict = torch.load(
            os.path.join(resume_param, "parameters"),
            map_location=torch.device(args.device),
        )
    # \\:if len(model.embedding_entity) != len(state_dict["embedding_entity"]):
    # \\:    #
    # \\:    assert torch.all(model.embedding_entity.data == 1.0).item()
    # \\:    assert torch.all(state_dict["embedding_entity"] == 1.0).item()
    # \\:    state_dict["embedding_entity"] = model.embedding_entity.data
    state_dict["embedding_entity"] = state_dict["embedding_entity"].new_ones(
        (num_nodes, 1)
    )
    model.load_state_dict(state_dict)
    name_loss = etexood.models.get_loss(0, 0, 0, 0, args.model, {})

    #
    ks = [int(msg) for msg in re.split(r"\s*,\s*", args.ks)]

    # Test evaluation.
    (metrics, scores) = tester.test(
        model,
        name_loss,
        ks=ks,
        negative_rate=test_negative_rate_eval,
        num_neg_rels=test_num_neg_rels_eval,
        margin=args.margin,
        eind=args.num_epochs,
        emax=args.num_epochs,
    )
    with open(os.path.join(unique, "metrics.json"), "w") as file:
        #
        json.dump(metrics, file, indent=4)


if __name__ == "__main__":
    #
    main()
