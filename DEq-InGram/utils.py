import numpy as np
from datetime import datetime
import hashlib

def remove_duplicate(x):
	return list(dict.fromkeys(x))

def generate_neg(triplets, num_ent, num_neg = 1):
	import torch
	neg_triplets = triplets.unsqueeze(dim=1).repeat(1,num_neg,1)
	rand_result = torch.rand((len(triplets),num_neg)).cuda()
	perturb_head = rand_result < 0.5
	perturb_tail = rand_result >= 0.5
	rand_idxs = torch.randint(low=0, high = num_ent-1, size = (len(triplets),num_neg)).cuda()
	rand_idxs[perturb_head] += rand_idxs[perturb_head] >= neg_triplets[:,:,0][perturb_head]
	rand_idxs[perturb_tail] += rand_idxs[perturb_tail] >= neg_triplets[:,:,2][perturb_tail]
	neg_triplets[:,:,0][perturb_head] = rand_idxs[perturb_head]
	neg_triplets[:,:,2][perturb_tail] = rand_idxs[perturb_tail]
	neg_triplets = torch.cat(torch.split(neg_triplets, 1, dim = 1), dim = 0).squeeze(dim = 1)

	return neg_triplets

def get_rank(triplet, scores, filters, target = 0):
	thres = scores[triplet[0,target]].item()
	scores[filters] = thres - 1
	# rank = (scores > thres).sum() + (scores == thres).sum()//2 + 1
	# Compute the worst case rank when breaking ties, instead of compute an average
	rank = (scores >= thres).sum() + 1
	return rank.item()

def get_metrics(rank):
	rank = np.array(rank, dtype = int)     # Replace np.int with int since np.int is deprecated
	mr = np.mean(rank)
	mrr = np.mean(1 / rank)
	hit10 = np.sum(rank < 11) / len(rank)
	hit5 = np.sum(rank < 6) / len(rank)
	hit4 = np.sum(rank < 5) / len(rank)
	hit3 = np.sum(rank < 4) / len(rank)
	hit2 = np.sum(rank < 3) / len(rank)
	hit1 = np.sum(rank < 2) / len(rank)
	return mr, mrr, hit10, hit5, hit4, hit3, hit2, hit1

def create_hash(params_string: str) -> str:
    R"""
    Create hash from command line argument string. This is mainly for logging purposes.
    """
    hasher = hashlib.md5()
    hasher.update(params_string.encode('utf-8'))
    raw_hash =  hasher.hexdigest()
    hash_str = "{}".format(raw_hash)[:8]
    return hash_str

def wandb_run_name(
    run_hash: str,
    stage_name: str,
) -> str:
    R"""
    Create a run name for Weights & Biases.
    """
    return f"{run_hash} ({stage_name}) @ {datetime.now().strftime('%m%d%Y|%H:%M:%S')}"