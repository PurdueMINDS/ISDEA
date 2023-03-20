#
set -e

#
task=NELL9951
rm -rf datasets/${task}Trans/processed/*
rm -rf datasets/${task}Ind/processed/*
rm -rf experiments/NBFNet/Ind${task}Trans/*
rm -rf experiments/NBFNet/Ind${task}Ind/*
rm -rf experiments/NBFNet/Ind${task}PermInd/*
for seed in 42; do
    #
    python script/run.py -c config/${task}-trans.yaml --gpus [2] --myid ${seed} --resume null
    python script/run.py -c config/${task}-ind.yaml --gpus [0] --myid ${seed} --resume "$(pwd)/experiments/NBFNet/Ind${task}Trans/${seed}"
    python script/run.py -c config/${task}-perm-ind.yaml --gpus [0] --myid ${seed} --resume "$(pwd)/experiments/NBFNet/Ind${task}Trans/${seed}"
done
