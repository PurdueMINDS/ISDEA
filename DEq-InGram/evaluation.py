import torch
from utils import get_rank, get_metrics
from tqdm import tqdm


def evaluate_triplet(triplet, target, my_model, emb_ent, emb_rel, full_graph_neg = False):
    """
    Evaluate a the ranks of a single triplet against negative samples.

    Also allow multiple entity and relation embeddings to be passed in as Monte Carlo samples.
    """
    triplet = triplet.unsqueeze(dim = 0)

    if full_graph_neg:
        # Use full graph negative sampling, i.e., the original setting
        head_corrupt = triplet.repeat(target.num_ent, 3)
        head_corrupt[:,0] = torch.arange(end = target.num_ent)
        
        if isinstance(emb_ent, list):
            head_scores = torch.stack([my_model.score(ee, er, head_corrupt) for ee, er in zip(emb_ent, emb_rel)]).mean(dim = 0)
        else:
            head_scores = my_model.score(emb_ent, emb_rel, head_corrupt)
        head_filters = target.filter_dict[('_', int(triplet[0,1].item()), int(triplet[0,2].item()))]
        head_rank = get_rank(triplet, head_scores, head_filters, target = 0)
    else:
        ### MODIFY: Random sample 50 corrupt heads instead of all ###
        head_corrupt = triplet.repeat(51, 3)   # 51 because 50 + 1 (for the original triplet)
        # head_filters gives us the other positive triplets with the same relation and tail
        head_filters = target.filter_dict[('_', int(triplet[0,1].item()), int(triplet[0,2].item()))]
        # Sample 50 corrupt heads that we know are not in the filter
        neg_heads_all = torch.tensor(list(set(range(target.num_ent)) - set(head_filters)))
        neg_heads_sampled_ids = torch.randint(low=0, high = len(neg_heads_all)-1, size = (50,))
        head_corrupt[1:,0] = neg_heads_all[neg_heads_sampled_ids]
        # Compute the score. Compute multiple scores and then average if emb_ent and emb_rel are lists
        if isinstance(emb_ent, list):
            head_scores_51 = torch.stack([my_model.score(ee, er, head_corrupt) for ee, er in zip(emb_ent, emb_rel)]).mean(dim = 0)
        else:
            head_scores_51 = my_model.score(emb_ent, emb_rel, head_corrupt)
        
        # The original head_scores has score for every head, and get_rank() can only work with that
        # So to work around this, create a head_scores tensor with the same length as the number of entities
        # And for the original head and those sampled entities, fill in the scores from head_scores_51
        # For the rest of the entities, fill in a score of original head score - 1, that is, head_scores_51[0] - 1
        head_scores = torch.ones(target.num_ent).cuda() * (head_scores_51[0] - 1)
        head_scores[head_corrupt[:,0]] = head_scores_51
        head_rank = get_rank(triplet, head_scores, head_filters, target = 0)

    if full_graph_neg:
        # Use full graph negative sampling, i.e., the original setting
        tail_corrupt = triplet.repeat(target.num_ent, 3)
        tail_corrupt[:,2] = torch.arange(end = target.num_ent)

        if isinstance(emb_ent, list):
            tail_scores = torch.stack([my_model.score(ee, er, tail_corrupt) for ee, er in zip(emb_ent, emb_rel)]).mean(dim = 0)
        else:
            tail_scores = my_model.score(emb_ent, emb_rel, tail_corrupt)
        tail_filters = target.filter_dict[(int(triplet[0,0].item()), int(triplet[0,1].item()), '_')]
        tail_rank = get_rank(triplet, tail_scores, tail_filters, target = 2)
    else:
        ### MODIFY: Random sample 50 corrupt tails instead of all ###
        tail_corrupt = triplet.repeat(51, 3)
        tail_filters = target.filter_dict[(int(triplet[0,0].item()), int(triplet[0,1].item()), '_')]
        neg_tails_all = torch.tensor(list(set(range(target.num_ent)) - set(tail_filters)))
        neg_tails_sampled_ids = torch.randint(low=0, high = len(neg_tails_all)-1, size = (50,))
        tail_corrupt[1:,2] = neg_tails_all[neg_tails_sampled_ids]

        # Compute the score. Compute multiple scores and then average if emb_ent and emb_rel are lists
        if isinstance(emb_ent, list):
            tail_scores_51 = torch.stack([my_model.score(ee, er, tail_corrupt) for ee, er in zip(emb_ent, emb_rel)]).mean(dim = 0)
        else:
            tail_scores_51 = my_model.score(emb_ent, emb_rel, tail_corrupt)

        tail_scores = torch.ones(target.num_ent).cuda() * (tail_scores_51[0] - 1)
        tail_scores[tail_corrupt[:,2]] = tail_scores_51
        tail_rank = get_rank(triplet, tail_scores, tail_filters, target = 2)

    ### MODIFY: Random sample 50 corrupt relations###
    rel_corrupt = triplet.repeat(51, 3)
    rel_filters = target.filter_dict[(int(triplet[0,0].item()), '_', int(triplet[0,2].item()))]
    neg_rels_all = torch.tensor(list(set(range(target.num_rel)) - set(rel_filters)))
    neg_rels_sampled_ids = torch.randint(low=0, high = len(neg_rels_all)-1, size = (50,))
    rel_corrupt[1:,1] = neg_rels_all[neg_rels_sampled_ids]

    # Compute the score. Compute multiple scores and then average if emb_ent and emb_rel are lists
    if isinstance(emb_ent, list):
        rel_scores_51 = torch.stack([my_model.score(ee, er, rel_corrupt) for ee, er in zip(emb_ent, emb_rel)]).mean(dim = 0)
    else:
        rel_scores_51 = my_model.score(emb_ent, emb_rel, rel_corrupt)

    rel_scores = torch.ones(target.num_rel).cuda() * (rel_scores_51[0] - 1)
    rel_scores[rel_corrupt[:,1]] = rel_scores_51
    rel_rank = get_rank(triplet, rel_scores, rel_filters, target = 1)

    if full_graph_neg:
        # If in full graph mode, we don't compute dual metrics. Return dummy values.
        dual_rank = 50
    else:
        ### MODIFY: Dual sampling loss, 12 negative heads + 12 negative tails + 26 negative relations ###
        # Use the contents in head_corrupt, tail_corrupt, and rel_corrupt to create the 50 negative triplets
        # The first 12 are negative heads, the next 12 are negative tails, and the last 26 are negative relations
        # Compute the rank against first 12 negative heads
        head_corrupt_13 = head_corrupt[:13]
        head_scores_13 = head_scores_51[:13]
        head_scores_dual = torch.ones(target.num_ent).cuda() * (head_scores_13[0] - 1)
        head_scores_dual[head_corrupt_13[:,0]] = head_scores_13
        head_rank_13 = get_rank(triplet, head_scores_dual, head_filters, target = 0)
        head_rank_dual = head_rank_13 - 1   # This counts how many negative heads are ranked higher than the original head
        # Compute the rank against next 12 negative tails
        tail_corrupt_13 = tail_corrupt[:13]
        tail_scores_13 = tail_scores_51[:13]
        tail_scores_dual = torch.ones(target.num_ent).cuda() * (tail_scores_13[0] - 1)
        tail_scores_dual[tail_corrupt_13[:,2]] = tail_scores_13
        tail_rank_13 = get_rank(triplet, tail_scores_dual, tail_filters, target = 2)
        tail_rank_dual = tail_rank_13 - 1   # This counts how many negative tails are ranked higher than the original tail
        # Compute the rank against last 26 negative relations
        rel_corrupt_27 = rel_corrupt[:27]
        rel_scores_27 = rel_scores_51[:27]
        rel_scores_dual = torch.ones(target.num_rel).cuda() * (rel_scores_27[0] - 1)
        rel_scores_dual[rel_corrupt_27[:,1]] = rel_scores_27
        rel_rank_27 = get_rank(triplet, rel_scores_dual, rel_filters, target = 1)
        rel_rank_dual = rel_rank_27 - 1   # This counts how many negative relations are ranked higher than the original relation
        # Compute the final mixed rank
        dual_rank = head_rank_dual + tail_rank_dual + rel_rank_dual + 1

    return head_rank, tail_rank, rel_rank, dual_rank


# Modify to random sampling on both entities and relations
#     1. Randomly sample 50 head entities, 50 tail entities, and 50 relations
#     2. Results for head and tail entities are aggregated 
def evaluate(my_model, target, epoch, init_emb_ent, init_emb_rel, relation_triplets):
    with torch.no_grad():
        my_model.eval()
        msg = torch.tensor(target.msg_triplets).cuda()
        sup = torch.tensor(target.sup_triplets).cuda()

        emb_ent, emb_rel = my_model(init_emb_ent, init_emb_rel, msg, relation_triplets)

        ent_ranks = []
        rel_ranks = []
        dual_ranks = []
        for triplet in tqdm(sup):
            head_rank, tail_rank, rel_rank, dual_rank = evaluate_triplet(triplet, target, my_model, emb_ent, emb_rel)

            ent_ranks.append(head_rank)
            ent_ranks.append(tail_rank)
            rel_ranks.append(rel_rank)
            dual_ranks.append(dual_rank)

        print("--------Entiti--------")
        mr_ent, mrr_ent, hit10_ent, hit5_ent, hit4_ent, hit3_ent, hit2_ent, hit1_ent = get_metrics(ent_ranks)
        mr_rel, mrr_rel, hit10_rel, hit5_rel, hit4_rel, hit3_rel, hit2_rel, hit1_rel = get_metrics(rel_ranks)
        mr_dual, mrr_dual, hit10_dual, hit5_dual, hit4_rel, hit3_dual, hit2_rel, hit1_dual = get_metrics(dual_ranks)
        print(f"MR_ENT: {mr_ent:.1f}")
        print(f"MRR_ENT: {mrr_ent:.3f}")
        print(f"Hits@10_ENT: {hit10_ent:.3f}")
        print(f"Hits@1_ENT: {hit1_ent:.3f}")
        print("--------Relation--------")
        print(f"MR_REL: {mr_rel:.1f}")
        print(f"MRR_REL: {mrr_rel:.3f}")
        print(f"Hits@10_REL: {hit10_rel:.3f}")
        print(f"Hits@1_REL: {hit1_rel:.3f}")
        print("--------Dual--------")
        print(f"MR_DUAL: {mr_dual:.1f}")
        print(f"MRR_DUAL: {mrr_dual:.3f}")
        print(f"Hits@10_DUAL: {hit10_dual:.3f}")
        print(f"Hits@1_DUAL: {hit1_dual:.3f}")

        # Return the metrics in dict
        return {
            'mr_ent': mr_ent, 'mrr_ent': mrr_ent, 'hit10_ent': hit10_ent, 'hit3_ent': hit3_ent, 'hit1_ent': hit1_ent,
            'mr_rel': mr_rel, 'mrr_rel': mrr_rel, 'hit10_rel': hit10_rel, 'hit3_rel': hit3_rel, 'hit1_rel': hit1_rel,
            'mr_dual': mr_dual, 'mrr_dual': mrr_dual, 'hit10_dual': hit10_dual, 'hit3_dual': hit3_dual, 'hit1_dual': hit1_dual
        }
    

def evaluate_mc(my_model, target, init_emb_ent_samples, init_emb_rel_samples, relation_triplets_samples, full_graph_neg = False):
    """
    Evaluate using multiple Monte Carlo samples of the embeddings.
    """
    with torch.no_grad():
        my_model.eval()
        msg = torch.tensor(target.msg_triplets).cuda()
        sup = torch.tensor(target.sup_triplets).cuda()

        # Compute the entity embedding and relation embeddings for each Monte Carlo sample
        emb_ent_samples, emb_rel_samples = [], []
        for init_emb_ent, init_emb_rel, relation_triplets in \
                tqdm(zip(init_emb_ent_samples, init_emb_rel_samples, relation_triplets_samples)):
            emb_ent, emb_rel = my_model(init_emb_ent, init_emb_rel, msg, relation_triplets)
            emb_ent_samples.append(emb_ent)
            emb_rel_samples.append(emb_rel)

        ent_ranks = []
        rel_ranks = []
        dual_ranks = []
        for triplet in tqdm(sup):
            head_rank, tail_rank, rel_rank, dual_rank = evaluate_triplet(triplet, target, my_model, emb_ent_samples, emb_rel_samples, full_graph_neg = full_graph_neg)

            ent_ranks.append(head_rank)
            ent_ranks.append(tail_rank)
            rel_ranks.append(rel_rank)
            dual_ranks.append(dual_rank)

        print("--------Entiti--------")
        mr_ent, mrr_ent, hit10_ent, hit5_ent, hit4_ent, hit3_ent, hit2_ent, hit1_ent = get_metrics(ent_ranks)
        mr_rel, mrr_rel, hit10_rel, hit5_rel, hit4_rel, hit3_rel, hit2_rel, hit1_rel = get_metrics(rel_ranks)
        mr_dual, mrr_dual, hit10_dual, hit5_dual, hit4_rel, hit3_dual, hit2_rel, hit1_dual = get_metrics(dual_ranks)
        print(f"MR_ENT: {mr_ent:.1f}")
        print(f"MRR_ENT: {mrr_ent:.3f}")
        print(f"Hits@10_ENT: {hit10_ent:.3f}")
        print(f"Hits@1_ENT: {hit1_ent:.3f}")
        print("--------Relation--------")
        print(f"MR_REL: {mr_rel:.1f}")
        print(f"MRR_REL: {mrr_rel:.3f}")
        print(f"Hits@10_REL: {hit10_rel:.3f}")
        print(f"Hits@1_REL: {hit1_rel:.3f}")
        print("--------Dual--------")
        print(f"MR_DUAL: {mr_dual:.1f}")
        print(f"MRR_DUAL: {mrr_dual:.3f}")
        print(f"Hits@10_DUAL: {hit10_dual:.3f}")
        print(f"Hits@1_DUAL: {hit1_dual:.3f}")

        # Return the metrics in dict
        return {
            'mr_ent': mr_ent, 'mrr_ent': mrr_ent, 'hit10_ent': hit10_ent, 'hit5_ent': hit5_ent, 
            'hit4_ent': hit4_ent, 'hit3_ent': hit3_ent, 'hit2_ent': hit2_ent, 'hit1_ent': hit1_ent,
            'mr_rel': mr_rel, 'mrr_rel': mrr_rel, 'hit10_rel': hit10_rel, 'hit5_rel': hit5_rel, 
            'hit4_rel': hit4_rel, 'hit3_rel': hit3_rel, 'hit2_rel': hit2_rel,  'hit1_rel': hit1_rel,
            'mr_dual': mr_dual, 'mrr_dual': mrr_dual, 'hit10_dual': hit10_dual, 'hit5_dual': hit5_dual, 'hit3_dual': hit3_dual, 'hit1_dual': hit1_dual
        }
