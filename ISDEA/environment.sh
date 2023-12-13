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
declare -A installed
declare -A latests

# Using package installer for Python to maintain.
install() {
    #
    local name
    local extra
    local version

    #
    name=${1}
    extra=${2}
    version=${3}
    shift 3

    #
    if [[ ${#extra} -eq 0 ]]; then
        #
        pip install --no-cache-dir --upgrade ${name}==${version} ${*}
    else
        #
        pip install --no-cache-dir --upgrade ${name}[${extra}]==${version} ${*}
    fi
    installed[${name}]=${version}
}

#
vercu=cu117
verth=1.13.0

#
install black "" 23.1.0
install mypy "" 1.1.1
install pytest "" 7.2.2
install pytest-cov "" 4.0.0
install numpy "" 1.24.2
install more_itertools "" 8.14.0
install scikit-learn "" 1.2.2
install seaborn "" 0.12.2
install networkx "" 3.0
install pydot "" 1.4.2
install ninja "" 1.11.1
install pyyaml "" 6.0
install easydict "" 1.10
install torch "" ${verth} --extra-index-url https://download.pytorch.org/whl/${vercu}
install torch-scatter "" 2.1.0 -f https://data.pyg.org/whl/torch-${verth}+${vercu}.html
install torch-sparse "" 0.6.16 -f https://data.pyg.org/whl/torch-${verth}+${vercu}.html
install torch-cluster "" 1.6.0 -f https://data.pyg.org/whl/torch-${verth}+${vercu}.html
install torch-spline-conv "" 1.2.1 -f https://data.pyg.org/whl/torch-${verth}+${vercu}.html
install torch-geometric "" 2.2.0 -f https://data.pyg.org/whl/torch-${verth}+${vercu}.html

#
outdate() {
    #
    local nlns
    local name
    local latest

    #
    latests=()
    nlns=0
    while IFS= read -r line; do
        #
        nlns=$((nlns + 1))
        [[ ${nlns} -gt 2 ]] || continue

        #
        name=$(echo ${line} | awk "{print \$1}")
        latest=$(echo ${line} | awk "{print \$3}")
        latests[${name}]=${latest}
    done <<<$(pip list --outdated)
}

#
outdate
for package in ${!installed[@]}; do
    #
    if [[ -n ${latests[${package}]} ]]; then
        #
        msg1="\x1b[1;93m${package}\x1b[0m"
        msg2="\x1b[94m${installed[${package}]}\x1b[0m"
        msg3="${msg1} (${msg2}) is \x1b[4;93moutdated\x1b[0m"
        msg4="latest version is \x1b[94m${latests[${package}]}\x1b[0m"
        echo -e "${msg3} (${msg4})."
    fi
done
