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

#
rm -rf logs/transform/${task}~dx2~0~${model}-${aggr}-${ablate}:_:e${epoch}-ss${seed}~l${lrexp}-sm${seed}
/usr/bin/time -f "======  ========\n  Time: %e sec\nMemory: %M KB\n======  ========" \
    python -u transform.py \
    --resume logs/fit/${task}~dx2~${model}-${aggr}-${ablate}:_:e${epoch}-ss${seed}~l${lrexp}-sm${seed}
rm -rf cache/${task}-trans~dx2/
rm -rf cache/${task}-ind~dx2/
