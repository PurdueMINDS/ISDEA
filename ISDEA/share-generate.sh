#
set -e

#
export OMP_NUM_THREADS=1
export MPI_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

#
task=${1}
seed=${2}
epoch=${3}
cpus=${4}

#
rm -rf cache/${task}-trans~dx2/
rm -rf cache/${task}-ind~dx2/
rm -rf logs/schedule/${task}~dx2:_:e${epoch}-s${seed}
/usr/bin/time -f "======  ========\n  Time: %e sec\nMemory: %M KB\n======  ========" \
    python -u schedule.py \
    --data data --cache cache --task ${task} --sample heuristics --bidirect \
    --num-hops 3 --num-processes ${cpus} --unit-process 60.0 --num-epochs ${epoch} \
    --batch-size-node 128 --batch-size-edge-train 256 --batch-size-edge-valid 16 --batch-size-edge-test 16 \
    --negative-rate-train 2 --negative-rate-eval 24 --num-neg-rels-train 2 --num-neg-rels-eval 26 \
    --seed ${seed}