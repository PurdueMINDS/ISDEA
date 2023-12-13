#
set -e

#
project=$(basename $(pwd))
code=$(basename $(dirname $(pwd)))
if [[ ${CONDA_DEFAULT_ENV} != ${code}-${project} ]]; then
    #
    target=${code}-${project}
    output=${CONDA_DEFAULT_ENV}
    echo -e "Conda environment must be \"\x1b[92m${target}\x1b[0m\", but get \"\x1b[91m${output}\x1b[0m\"."
    exit 1
fi

#
pip install -e .