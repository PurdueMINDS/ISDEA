#
set -e

#
task=FD1
epochs=40

#
rm -rf experiments/${task}-trans-*
rm -rf experiments/${task}-ind-*
for seed in 42; do
    #
    python train.py -d ${task}-trans -e ${task}-trans-${seed} --num_epochs ${epochs} --hop 3 --num_gcn_layers 3 --eval_every 1 --num_workers 8 --disable_cuda
    python test_ranking.py -d ${task}-ind -e ${task}-ind-${seed} --hop 3 --resume "experiments/${task}-trans-${seed}"
    python test_ranking.py -d ${task}-perm-ind -e ${task}-perm-ind-${seed} --hop 3 --resume "experiments/${task}-trans-${seed}"
done
