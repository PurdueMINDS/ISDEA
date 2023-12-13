import argparse
import json

def parse(test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default = "./data/", type = str)
    parser.add_argument('--data_name', default = 'NL-100', type = str)
    parser.add_argument('--exp', default = 'exp', type = str)
    # Add seed
    parser.add_argument('--seed', default = 0, type = int)

    parser.add_argument('-m', '--margin', default = 2, type = float)
    parser.add_argument('-lr', '--learning_rate', default=5e-4, type = float)
    parser.add_argument('-nle', '--num_layer_ent', default = 2, type = int)
    parser.add_argument('-nlr', '--num_layer_rel', default = 2, type = int)
    parser.add_argument('-d_e', '--dimension_entity', default = 32, type = int)
    parser.add_argument('-d_r', '--dimension_relation', default = 32, type = int)
    parser.add_argument('-hdr_e', '--hidden_dimension_ratio_entity', default = 8, type = int)
    parser.add_argument('-hdr_r', '--hidden_dimension_ratio_relation', default = 4, type = int)
    parser.add_argument('-b', '--num_bin', default = 10, type = int)
    parser.add_argument('-e', '--num_epoch', default = 10000, type = int)
    if test:
        parser.add_argument('--mc', default=1, type=int, help="Number of Monte Carlo samples")
        parser.add_argument('--target_epoch', default = 6600, type = int)
        parser.add_argument('--run_hash', default = "", type = str)
        parser.add_argument('--data_name_run_hash', default = "", type = str)  # Supercedes both --data_name and --run_hash
        parser.add_argument('--alt-test-data', default = "", type = str, help="Alternative test data to use")
        parser.add_argument('--full_graph_neg', action = 'store_true', help="Use full graph negative sampling for nodes")
    parser.add_argument('-v', '--validation_epoch', default = 200, type = int)
    parser.add_argument('--num_head', default = 8, type = int)
    parser.add_argument('--num_neg', default = 10, type = int)
    parser.add_argument('--best', action = 'store_true')
    if not test:
        parser.add_argument('--no_write', action = 'store_true')

    # Support Weight & Biases logging
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, required=False, help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, required=False, help="Weights & Biases team name/account username")
    parser.add_argument("--wandb-job-type", type=str, required=False, help="Weights & Biases job type")

    args = parser.parse_args()

    if test and args.best:
        # Check that target hash is specified
        assert args.run_hash != "" or args.data_name_run_hash != "", \
            "Target run hash must be specified by --run_hash when --best is set"
        # If data_name_run_hash is specified, use it instead of data_name and run_hash
        if args.data_name_run_hash != "":
            args.data_name, args.run_hash = args.data_name_run_hash.split("/")
            ### DEBUG ###
            print(f"Specified data_name_run_hash = {args.data_name_run_hash}")
            print(f"Using data_name = {args.data_name}, run_hash = {args.run_hash}")
        # If alt-test-data is specified, also save and later overwrite the data_path
        if args.alt_test_data != "":
            data_path = args.data_path
        remaining_args = []
        with open(f"./ckpt/{args.exp}/{args.data_name}/{args.run_hash}/config.json") as f:
            configs = json.load(f)
        for key in vars(args).keys():
            if key in configs:
                vars(args)[key] = configs[key]
            else:
                remaining_args.append(key)
        # If alt-test-data is specified, overwrite the data_path
        if args.alt_test_data != "":
            args.data_path = data_path
        # Reset args.best to True
        args.best = True

    return args