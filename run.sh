#
set -e

#
export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

#
task=FD2
model=dssgnn
aggr="mean"
ablate="both"
seed=58
lrexp=3
epoch=50

#
rm -rf logs/cache/${task}*

#
rm -rf logs/schedule/${task}~dx2:_:e${epoch}-s${seed}
/usr/bin/time -f "======  ========\n  Time: %e sec\nMemory: %M KB\n======  ========" \
    python -u schedule.py \
    --data data --cache cache --task ${task} --sample heuristics --bidirect \
    --num-hops 2 --num-processes 4 --unit-process 60.0 --num-epochs ${epoch} \
    --batch-size-node 128 --batch-size-edge-train 256 --batch-size-edge-valid 256 --batch-size-edge-test 256 \
    --negative-rate-train 2 --negative-rate-eval 50 \
    --seed ${seed}

#
rm -rf logs/fit/${task}~dx2~${model}-${aggr}-${ablate}:_:e${epoch}-ss${seed}~l${lrexp}-sm${seed}
/usr/bin/time -f "======  ========\n  Time: %e sec\nMemory: %M KB\n======  ========" \
    python -u fit.py \
    --data data --cache cache --task ${task} --sample heuristics --bidirect \
    --num-hops 2 --num-processes 4 --unit-process 30.0 --num-epochs ${epoch} \
    --batch-size-node 128 --batch-size-edge-train 256 --batch-size-edge-valid 256 --batch-size-edge-test 256 \
    --negative-rate-train 2 --negative-rate-eval 50 \
    --seed-schedule ${seed} \
    --hidden 32 --activate relu --dropout 0.0 --num-bases 4 --clip-grad-norm 1.0 --weight-decay 5e-4 \
    --device cpu --model ${model} --dss-aggr ${aggr} --ablate ${ablate} --lr 1e-${lrexp} \
    --seed-model ${seed} \
    --ks "1,2,4" --margin 10.0 --early-stop 15
rm -rf logs/transform/${task}~dx2~0~${model}-${aggr}-${ablate}:_:e${epoch}-ss${seed}~l${lrexp}-sm${seed}
/usr/bin/time -f "======  ========\n  Time: %e sec\nMemory: %M KB\n======  ========" \
    python -u transform.py \
    --resume logs/fit/${task}~dx2~${model}-${aggr}-${ablate}:_:e${epoch}-ss${seed}~l${lrexp}-sm${seed}
