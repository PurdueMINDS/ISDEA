#
set -e

#
export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

#
task=${1}
model=dssgnn
aggr=mean
ablate=both
seed=${2}
epoch=${3}
lrexp=3


rm -rf logs/fit/${task}~dx2~${model}-${aggr}-${ablate}:_:e${epoch}-ss${seed}~l${lrexp}-sm${seed}
/usr/bin/time -f "======  ========\n  Time: %e sec\nMemory: %M KB\n======  ========" \
    python -u fit.py \
    --data data --cache cache --task ${task} --sample heuristics --bidirect \
    --num-hops 3 --num-processes 4 --unit-process 30.0 --num-epochs ${epoch} \
    --batch-size-node 128 --batch-size-edge-train 256 --batch-size-edge-valid 16 --batch-size-edge-test 16 \
    --negative-rate-train 2 --negative-rate-eval 24 --num-neg-rels-train 2 --num-neg-rels-eval 26 \
    --seed-schedule ${seed} \
    --hidden 32 --activate relu --dropout 0.0 --num-bases 4 --clip-grad-norm 1.0 --weight-decay 5e-4 \
    --device cuda --model ${model} --dss-aggr ${aggr} --ablate ${ablate} --lr 1e-${lrexp} \
    --seed-model ${seed} \
    --ks "1,3,5,10" --margin 10.0 --early-stop 15
rm -rf logs/transform/${task}~dx2~0~${model}-${aggr}-${ablate}:_:e${epoch}-ss${seed}~l${lrexp}-sm${seed}
/usr/bin/time -f "======  ========\n  Time: %e sec\nMemory: %M KB\n======  ========" \
    python -u transform.py \
    --resume logs/fit/${task}~dx2~${model}-${aggr}-${ablate}:_:e${epoch}-ss${seed}~l${lrexp}-sm${seed}
rm -rf cache/${task}-trans~dx2/
rm -rf cache/${task}-ind~dx2/
