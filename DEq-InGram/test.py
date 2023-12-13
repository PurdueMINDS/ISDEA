from relgraph import generate_relation_triplets
from dataset import TestNewData
from tqdm import tqdm
import random
from model import InGram
import torch
import numpy as np
from utils import get_rank, get_metrics, wandb_run_name
from my_parser import parse
from evaluation import evaluate_mc
from initialize import initialize
import os
import wandb
import time

# TODO: integrate with Weights & Biases

args = parse(test=True)


# If alt-test-data is specified, check that it exists
if args.alt_test_data != "":
	assert args.alt_test_data in os.listdir(args.data_path), f"{args.alt_test_data} Not Found"
else:
	assert args.data_name in os.listdir(args.data_path), f"{args.data_name} Not Found"

# Initialize Weights & Biases logging, and log the arguments for this run
run_config = vars(args)
run_config["stage"] = "transform"
wandb.init(mode="online" if args.wandb else "disabled",  # Turn on wandb logging only if --wandb is set
		   project=args.wandb_project,
		   entity=args.wandb_entity,
		   job_type=args.wandb_job_type,
		   config=run_config)
wandb.run.name = wandb_run_name(args.run_hash, 'transform')

OMP_NUM_THREADS=8
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
torch.cuda.empty_cache()

# If alt-test-data is specified, use it instead of the default test data
if args.alt_test_data != "":
	path = args.data_path + args.alt_test_data + "/"
else:
	path = args.data_path + args.data_name + "/"
test = TestNewData(path, data_type = "test")

if not args.best:
	file_format = f"lr_{args.learning_rate}_dim_{args.dimension_entity}_{args.dimension_relation}" + \
				f"_bin_{args.num_bin}_total_{args.num_epoch}_every_{args.validation_epoch}" + \
				f"_neg_{args.num_neg}_layer_{args.num_layer_ent}_{args.num_layer_rel}" + \
				f"_hid_{args.hidden_dimension_ratio_entity}_{args.hidden_dimension_ratio_relation}" + \
				f"_head_{args.num_head}_margin_{arg.margin}"

d_e = args.dimension_entity
d_r = args.dimension_relation
hdr_e = args.hidden_dimension_ratio_entity
hdr_r = args.hidden_dimension_ratio_relation
B = args.num_bin
num_neg = args.num_neg

my_model = InGram(dim_ent = d_e, hid_dim_ratio_ent = hdr_e, dim_rel = d_r, hid_dim_ratio_rel = hdr_r,\
				num_bin = B, num_layer_ent = args.num_layer_ent, num_layer_rel = args.num_layer_rel, \
				num_head = args.num_head)
my_model = my_model.cuda()


if not args.best:
	ckpt_path = f"ckpt/{args.exp}/{args.data_name}/{file_format}_{args.target_epoch}.ckpt"
else:
	ckpt_path = f"ckpt/{args.exp}/{args.data_name}/{args.run_hash}/best.ckpt"
my_model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])


print("Test")
test_start_time = time.time()

my_model.eval()
test_msg = test.msg_triplets
test_sup = test.sup_triplets
# test_relation_triplets = generate_relation_triplets(test_msg, test.num_ent, test.num_rel, B)
# test_init_emb_ent = torch.load(ckpt_path)["inf_emb_ent"]
# test_init_emb_rel = torch.load(ckpt_path)["inf_emb_rel"]

# Obtain multiple Monte Carlo samples of the initial random embeddings
test_init_emb_ent_samples, test_init_emb_rel_samples, test_relation_triplets_samples = [], [], []
for _ in range(args.mc):
	test_init_emb_ent, test_init_emb_rel, test_relation_triplets = initialize(test, test_msg, d_e, d_r, B)
	test_relation_triplets = torch.tensor(test_relation_triplets).cuda()
	test_init_emb_ent_samples.append(test_init_emb_ent)
	test_init_emb_rel_samples.append(test_init_emb_rel)
	test_relation_triplets_samples.append(test_relation_triplets)

test_sup = torch.tensor(test_sup).cuda()
test_msg = torch.tensor(test_msg).cuda()

metrics = evaluate_mc(
	my_model, test, 
	test_init_emb_ent_samples, test_init_emb_rel_samples, test_relation_triplets_samples,
	full_graph_neg = args.full_graph_neg)

test_end_time = time.time()
print(f"Test Time: {test_end_time - test_start_time}")

# Log to Weights & Biases
wandb.log(metrics)
