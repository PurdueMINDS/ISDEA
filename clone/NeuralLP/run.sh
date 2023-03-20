#
set -e

#
task=FD1
epoch=1

#
rm -rf exps/${task}-trans-*
rm -rf exps/${task}-ind-*
rm -rf exps/${task}-perm-ind-*
. eval/collect_all_facts.sh datasets/${task}-ind
python eval/get_truths.py datasets/${task}-ind
. eval/collect_all_facts.sh datasets/${task}-perm-ind
python eval/get_truths.py datasets/${task}-perm-ind
for seed in 42; do
    #
    python src/main.py --datadir=datasets/${task}-trans --exps_dir=exps/ --exp_name=${task}-trans-${seed} --seed ${seed} --num_step 4 --top_k 50 --query_embed_size 32 --rnn_state_size 32 --vocab_embed_size 32 --max_epoch ${epoch} --no_link_percent 2.0 --gpu "" --resume ""
    python src/main.py --datadir=datasets/${task}-ind --exps_dir=exps/ --exp_name=${task}-ind-${seed} --seed ${seed} --num_step 4 --top_k 50 --query_embed_size 32 --rnn_state_size 32 --vocab_embed_size 32 --max_epoch ${epoch} --no_link_percent 2.0 --gpu "" --resume "exps/${task}-trans-${seed}"
    python eval/evaluate.py --preds=exps/${task}-ind-${seed}/test_predictions.txt --truths=datasets/${task}-ind/truths.pckl --top_k 10 --raw | tee exps/${task}-ind-${seed}/top10.txt
    python eval/evaluate.py --preds=exps/${task}-ind-${seed}/test_predictions.txt --truths=datasets/${task}-ind/truths.pckl --top_k 5 --raw | tee exps/${task}-ind-${seed}/top5.txt
    python eval/evaluate.py --preds=exps/${task}-ind-${seed}/test_predictions.txt --truths=datasets/${task}-ind/truths.pckl --top_k 3 --raw | tee exps/${task}-ind-${seed}/top3.txt
    python eval/evaluate.py --preds=exps/${task}-ind-${seed}/test_predictions.txt --truths=datasets/${task}-ind/truths.pckl --top_k 1 --raw | tee exps/${task}-ind-${seed}/top1.txt
    python src/main.py --datadir=datasets/${task}-perm-ind --exps_dir=exps/ --exp_name=${task}-perm-ind-${seed} --seed ${seed} --num_step 4 --top_k 50 --query_embed_size 32 --rnn_state_size 32 --vocab_embed_size 32 --max_epoch ${epoch} --no_link_percent 2.0 --gpu "" --resume "exps/${task}-trans-${seed}"
    python eval/evaluate.py --preds=exps/${task}-perm-ind-${seed}/test_predictions.txt --truths=datasets/${task}-perm-ind/truths.pckl --top_k 10 --raw | tee exps/${task}-perm-ind-${seed}/top10.txt
    python eval/evaluate.py --preds=exps/${task}-perm-ind-${seed}/test_predictions.txt --truths=datasets/${task}-perm-ind/truths.pckl --top_k 5 --raw | tee exps/${task}-perm-ind-${seed}/top5.txt
    python eval/evaluate.py --preds=exps/${task}-perm-ind-${seed}/test_predictions.txt --truths=datasets/${task}-perm-ind/truths.pckl --top_k 3 --raw | tee exps/${task}-perm-ind-${seed}/top3.txt
    python eval/evaluate.py --preds=exps/${task}-perm-ind-${seed}/test_predictions.txt --truths=datasets/${task}-perm-ind/truths.pckl --top_k 1 --raw | tee exps/${task}-perm-ind-${seed}/top1.txt
done
