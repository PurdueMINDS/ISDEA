#
import argparse
import os
import etexood
import numpy as onp
import torch
import re
import pandas as pd
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
    parser = argparse.ArgumentParser(description="Fit.")
    parser.add_argument("--data", type=str, required=True, help="Data root directory")
    parser.add_argument("--cache", type=str, required=True, help="Cache root directory")
    parser.add_argument("--task", type=str, required=True, help="Task prefix.")
    parser.add_argument("--sample", type=str, required=True, help="Sampling data type.")
    parser.add_argument("--bidirect", action="store_true", help="Treat observation graph as bidirected.")
    parser.add_argument("--overfit", action="store_true", help="Overfit on both training and validation.")
    parser.add_argument("--num-hops", type=int, required=True, help="Number of hops.")
    parser.add_argument("--num-processes", type=int, required=True, help="Number of processes.")
    parser.add_argument("--unit-process", type=float, required=True, help="Report unit of process.")
    parser.add_argument("--num-epochs", type=int, required=True, help="Number of epochs.")
    parser.add_argument("--batch-size-node", type=int, required=True, help="Batch size of node sampling.")
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
    parser.add_argument("--batch-size-edge-test", type=int, required=True, help="Batch size of test edge sampling.")
    parser.add_argument("--negative-rate-train", type=int, required=True, help="Negative sampling rate of training.")
    parser.add_argument("--negative-rate-eval", type=int, required=True, help="Negative sampling rate of evaluation.")
    parser.add_argument("--seed-schedule", type=int, required=True, help="Schedule seed.")
    parser.add_argument("--device", type=str, required=True, help="Device.")
    parser.add_argument("--model", type=str, required=True, help="Model.")
    parser.add_argument("--hidden", type=int, required=True, help="Hidden layer size.")
    parser.add_argument("--activate", type=str, required=True, help="Activation function.")
    parser.add_argument("--dropout", type=float, required=True, help="Dropout.")
    parser.add_argument("--num-bases", type=int, required=True, help="Number of RGCN bases.")
    parser.add_argument("--clip-grad-norm", type=float, required=True, help="Gradient clipping norm.")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, required=True, help="Weight decay.")
    parser.add_argument("--seed-model", type=int, required=True, help="Model seed.")
    parser.add_argument("--ks", type=str, required=True, help="Evaluating hit at ks separated by comma.")
    parser.add_argument("--margin", type=float, required=True, help="Distance margin.")
    parser.add_argument("--early-stop", type=int, required=True, help="Early stop patience.")
    args = parser.parse_args()

    # Allocate caching disk space.
    task = "-".join((args.task, "trans"))
    path_dataset = os.path.join(args.data, task)
    path_cache = os.path.join(args.cache, "~".join((task, "dx{:d}".format(1 + int(args.bidirect)))))
    os.makedirs(path_cache, exist_ok=True)

    #
    assert 10 ** int(math.log10(float(args.lr))) == args.lr

    # Allocate logging disk space.
    prefix = "~".join([args.task, "dx{:d}".format(1 + int(args.bidirect)), args.model])
    suffix = "~".join(
        (
            "e{:d}-ss{:d}".format(args.num_epochs, args.seed_schedule),
            "l{:d}-sm{:d}".format(int(-math.log10(float(args.lr))), args.seed_model),
        )
    )
    unique = etexood.loggings.create_framework_directory(
        os.path.join("logs", "fit"),
        prefix,
        "_",
        suffix,
        sleep=1.1,
        max_waits=11,
    )

    #
    with open(os.path.join(unique, "arguments.json"), "w") as file:
        #
        json.dump(vars(args), file, indent=4)

    # Prepare logging terminal.
    logger = etexood.loggings.create_logger(unique, os.path.basename(unique), level_file=None, level_console=None)

    # Load dataset.
    dataset = etexood.datasets.DatasetTriplet.from_file(logger, path_dataset)

    # Prepare observed graph.
    num_nodes = len(dataset._entity2id)
    num_relations = len(dataset._relation2id)
    tripelts_observe = dataset.triplets_observe
    adjs_observe = tripelts_observe[:, :2].T
    rels_observe = tripelts_observe[:, 2]

    #
    if args.bidirect:
        #
        logger.info("-- Augment observation by inversion.")
        adjs_observe = onp.concatenate((adjs_observe[[0, 1]], adjs_observe[[1, 0]]), axis=1)
        rels_observe = onp.concatenate((rels_observe, rels_observe + num_relations))

    # Prepare training edges.
    if len(dataset.triplets_train) == len(dataset.triplets_observe):
        #
        triplets_train = dataset.triplets_train
        adjs_train = triplets_train[:, :2].T
        rels_train = triplets_train[:, 2]
    else:
        #
        triplets_train = dataset.triplets_train[len(dataset.triplets_observe) :]
        adjs_train = triplets_train[:, :2].T
        rels_train = triplets_train[:, 2]

    # Prepare validation edges.
    triplets_valid = dataset.triplets_valid
    adjs_valid = triplets_valid[:, :2].T
    rels_valid = triplets_valid[:, 2]

    # Check data.
    starter_trans = int(len(dataset.triplets_observe) != len(dataset.triplets_train))
    starter_trans *= len(dataset.triplets_observe)

    # Check size.
    assert len(dataset.triplets_observe) * (1 + int(args.bidirect)) == len(rels_observe)
    assert adjs_observe.ndim == 2 and tuple(adjs_observe.shape) == (2, len(rels_observe))
    assert len(rels_observe) > 0
    assert len(dataset.triplets_train[starter_trans:]) == len(rels_train)
    assert adjs_train.ndim == 2 and tuple(adjs_train.shape) == (2, len(rels_train))
    assert len(rels_train) > 0
    assert len(dataset.triplets_valid) == len(rels_valid)
    assert adjs_valid.ndim == 2 and tuple(adjs_valid.shape) == (2, len(rels_valid))
    assert len(rels_valid) > 0

    #
    assert onp.max(dataset._triplets[:, 2]) < onp.max(dataset._triplets[:, :2])

    # Check content.
    assert onp.all(
        dataset.triplets_observe[:, :2] == adjs_observe.T[: len(rels_observe) // (1 + int(args.bidirect))]
    ).item()
    assert onp.all(dataset.triplets_train[starter_trans:][:, :2] == adjs_train.T)
    assert onp.all(dataset.triplets_valid[:, :2] == adjs_valid.T)

    #
    assert onp.all(
        dataset.triplets_observe[:, 2] == rels_observe[: len(rels_observe) // (1 + int(args.bidirect))]
    ).item()
    assert onp.all(dataset.triplets_train[starter_trans:][:, 2] == rels_train)
    assert onp.all(dataset.triplets_valid[:, 2] == rels_valid)

    # Merge training and validation if we want to overfit on provided tuning data.
    # It should only be used for debugging.
    if args.overfit:
        #
        triplets_train2 = onp.concatenate((triplets_train, triplets_valid))
        triplets_valid2 = onp.concatenate((triplets_train, triplets_valid))
        triplets_train = triplets_train2
        adjs_train = triplets_train[:, :2].T
        rels_train = triplets_train[:, 2]
        triplets_valid = triplets_valid2
        adjs_valid = triplets_valid[:, :2].T
        rels_valid = triplets_valid[:, 2]

    # Lock triplets in memory.
    adjs_observe.setflags(write=False)
    rels_observe.setflags(write=False)
    adjs_train.setflags(write=False)
    rels_train.setflags(write=False)
    adjs_valid.setflags(write=False)
    rels_valid.setflags(write=False)

    #
    logger.info("-- Load preprocessed schedule:")
    trainer = etexood.frameworks.transform.Trainer(
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
        device=torch.device(args.device),
    )
    validator = etexood.frameworks.transform.Evaluator(
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
        device=torch.device(args.device),
    )

    #
    trainer.load(
        args.num_epochs,
        batch_size_node=args.batch_size_node,
        batch_size_edge=args.batch_size_edge_train * (1 + args.negative_rate_train),
        negative_rate=args.negative_rate_train,
        seed=args.seed_schedule + 1,
        reusable_edge=False,
    )
    validator.load(
        1,
        batch_size_node=args.batch_size_node,
        batch_size_edge=args.batch_size_edge_valid * (1 + args.negative_rate_eval),
        negative_rate=args.negative_rate_eval,
        seed=args.seed_schedule + 2,
        reusable_edge=True,
    )

    #
    logger.info("-- Create a randomly initialized model:")
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
            },
        )
        .reset_parameters(torch.Generator("cpu").manual_seed(args.seed_model))
        .to(torch.device(args.device))
    )
    loss = etexood.models.get_loss(0, 0, 0, 0, args.model, {})
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #
    ks = [int(msg) for msg in re.split(r"\s*,\s*", args.ks)]
    LOSS = {"binary": "BCE", "distance": "Dist"}[loss]

    # Initial evaluation.
    (metrics, _) = validator.test(
        model,
        loss,
        ks=ks,
        negative_rate=args.negative_rate_eval,
        margin=args.margin,
        eind=0,
        emax=args.num_epochs,
    )
    buf = [metrics]
    metric_pair = (-metrics["MR"], metrics["MRR"], *(metrics["Hit@{:d}".format(k)] for k in ks), -metrics[LOSS])
    metric_best = metric_pair
    torch.save(model.state_dict(), os.path.join(unique, "parameters"))
    num_no_improves = 0
    metrics["Improve"] = True
    pd.DataFrame.from_records(
        buf
        + [
            {key: {float: float("nan"), bool: False}[type(val)] for (key, val) in buf[-1].items()}
            for _ in range(args.num_epochs)
        ],
    ).to_csv(os.path.join(unique, "metrics.csv"))

    #
    for eind in range(1, args.num_epochs + 1):
        # Train.
        trainer.tune(
            model,
            loss,
            optimizer,
            negative_rate=args.negative_rate_train,
            margin=args.margin,
            clip_grad_norm=args.clip_grad_norm,
            eind=eind,
            emax=args.num_epochs,
        )

        # Evaluate.
        (metrics, _) = validator.test(
            model,
            loss,
            ks=ks,
            negative_rate=args.negative_rate_eval,
            margin=args.margin,
            eind=eind,
            emax=args.num_epochs,
        )
        buf.append(metrics)
        metric_pair = (-metrics["MR"], metrics["MRR"], *(metrics["Hit@{:d}".format(k)] for k in ks), -metrics[LOSS])
        if metric_best < metric_pair:
            #
            metric_best = metric_pair
            torch.save(model.state_dict(), os.path.join(unique, "parameters"))
            num_no_improves = 0
            metrics["Improve"] = True
        else:
            #
            num_no_improves += 1
            metrics["Improve"] = False
        pd.DataFrame.from_records(
            buf
            + [
                {key: {float: float("nan"), bool: False}[type(val)] for (key, val) in buf[-1].items()}
                for _ in range(args.num_epochs - eind)
            ],
        ).to_csv(os.path.join(unique, "metrics.csv"))

        #
        if num_no_improves == args.early_stop:
            #
            break

    # Collect peak memory cost.
    if args.device != "cpu":
        #
        print(
            "GPU: {:d} MB".format(
                int(math.ceil(float(torch.cuda.max_memory_allocated(device=torch.device(args.device))) / 1024.0**2)),
            ),
        )


if __name__ == "__main__":
    #
    main()
