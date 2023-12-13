from relgraph import generate_relation_triplets
from dataset import TrainData, TestNewData
from tqdm import tqdm
import random
from model import InGram
import torch
import numpy as np
from utils import generate_neg, create_hash, wandb_run_name
import os
from evaluation import evaluate
from initialize import initialize
from my_parser import parse
import wandb
import json
import time

# TODO: Integrate with Weights & Bias

args = parse()

OMP_NUM_THREADS = 8
# Use random seed specified in the command line argument
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
torch.cuda.empty_cache()

# Initialize Weights & Biases logging, and log the arguments for this run
run_config = vars(args)
run_hash = create_hash(str(vars(args)))
run_config["run_hash"] = run_hash
run_config["stage"] = "fit"
wandb.init(mode="online" if args.wandb else "disabled",  # Turn on wandb logging only if --wandb is set
		   project=args.wandb_project,
		   entity=args.wandb_entity,
		   job_type=args.wandb_job_type,
		   config=run_config)
# Custom run name: run hash + model name + task/dataset name
wandb.run.name = wandb_run_name(run_hash, "fit")

print(f">>>>> Run Hash: {run_hash} <<<<<")

assert args.data_name in os.listdir(args.data_path), f"{args.data_name} Not Found"
path = args.data_path + args.data_name + "/"
train = TrainData(path)
valid = TestNewData(path, data_type = "valid")

ckpt_path = f"ckpt/{args.exp}/{args.data_name}/{run_hash}"
if not args.no_write:
	os.makedirs(ckpt_path, exist_ok=True)
	# Save config to JSON file
	with open(f"{ckpt_path}/config.json", "w") as f:
		json.dump(vars(args), f)

file_format = f"lr_{args.learning_rate}_dim_{args.dimension_entity}_{args.dimension_relation}" + \
			  f"_bin_{args.num_bin}_total_{args.num_epoch}_every_{args.validation_epoch}" + \
			  f"_neg_{args.num_neg}_layer_{args.num_layer_ent}_{args.num_layer_rel}" + \
			  f"_hid_{args.hidden_dimension_ratio_entity}_{args.hidden_dimension_ratio_relation}" + \
			  f"_head_{args.num_head}_margin_{args.margin}"

d_e = args.dimension_entity
d_r = args.dimension_relation
hdr_e = args.hidden_dimension_ratio_entity
hdr_r = args.hidden_dimension_ratio_relation
B = args.num_bin
epochs = args.num_epoch
valid_epochs = args.validation_epoch
num_neg = args.num_neg

my_model = InGram(dim_ent = d_e, hid_dim_ratio_ent = hdr_e, dim_rel = d_r, hid_dim_ratio_rel = hdr_r, \
				num_bin = B, num_layer_ent = args.num_layer_ent, num_layer_rel = args.num_layer_rel, \
				num_head = args.num_head)
my_model = my_model.cuda()
loss_fn = torch.nn.MarginRankingLoss(margin = args.margin, reduction = 'mean')

optimizer = torch.optim.Adam(my_model.parameters(), lr = args.learning_rate)
pbar = tqdm(range(epochs))

total_loss = 0

# Record best validation metrics and early-stop counter
best_val_mrr = 0
early_stop_counter = 0
BEST_METRIC_KEY = 'mrr_ent'

for epoch in pbar:
	train_time_start = time.time()

	optimizer.zero_grad()
	msg, sup = train.split_transductive(0.75)

	init_emb_ent, init_emb_rel, relation_triplets = initialize(train, msg, d_e, d_r, B)
	msg = torch.tensor(msg).cuda()
	sup = torch.tensor(sup).cuda()

	emb_ent, emb_rel = my_model(init_emb_ent, init_emb_rel, msg, relation_triplets)
	pos_scores = my_model.score(emb_ent, emb_rel, sup)
	neg_scores = my_model.score(emb_ent, emb_rel, generate_neg(sup, train.num_ent, num_neg = num_neg))

	loss = loss_fn(pos_scores.repeat(num_neg), neg_scores, torch.ones_like(neg_scores))

	loss.backward()
	torch.nn.utils.clip_grad_norm_(my_model.parameters(), 0.1, error_if_nonfinite = False)
	optimizer.step()
	total_loss += loss.item()
	pbar.set_description(f"loss {loss.item()}")	

	train_time_end = time.time()
	print(f"Epoch {epoch} train time: {train_time_end - train_time_start}")

	# Log training loss to Weights & Biases
	wandb.log({"loss": loss.item()})

	if ((epoch + 1) % valid_epochs) == 0:
		print("Validation")
		valid_time_start = time.time()

		my_model.eval()
		val_init_emb_ent, val_init_emb_rel, val_relation_triplets = initialize(valid, valid.msg_triplets, \
																				d_e, d_r, B)

		metrics = evaluate(my_model, valid, epoch, val_init_emb_ent, val_init_emb_rel, val_relation_triplets)
		
		valid_time_end = time.time()
		print(f"Epoch {epoch} validation time: {valid_time_end - valid_time_start}")
		
		# Save best checkpoint based on MRR. Best checkpoint is named best.ckpt
		if metrics[BEST_METRIC_KEY] > best_val_mrr:
			best_val_mrr = metrics[BEST_METRIC_KEY]
			early_stop_counter = 0
			if not args.no_write:
				print(">>>>> New best checkpoint! <<<<<")
				torch.save({'model_state_dict': my_model.state_dict(), \
							'optimizer_state_dict': optimizer.state_dict(), \
							'inf_emb_ent': val_init_emb_ent, \
							'inf_emb_rel': val_init_emb_rel}, \
					f"{ckpt_path}/best.ckpt")
		# If MRR does not improve, increment early-stop counter
		else:
			early_stop_counter += 1
			print(f">>>>> No Improvement. Early stop counter: {early_stop_counter} <<<<<")
			# If early-stop counter reaches 10, stop training
			if early_stop_counter == 10:
				print(">>>>> Early stop! <<<<<")
				break

		# Log validation metrics to Weights & Biases
		wandb.log(metrics)

		my_model.train()