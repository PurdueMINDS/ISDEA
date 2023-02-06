## Supplementary Code

We provide our proposal implementation and baseline implementations here.
- Our implementation is under `src`.
- Our implementation is under `clone/*`.
Baselines are cloned from repo given in original paper with essential modifications to run with our data. All modifications are marked by comment `# MODIFY` and `# \\:`.

To execute our code, a virtual environment created by CONDA is required. Then execute `environment.sh` to install all dependencies.
Execute `pip install -e .` to install our implementation as importable Python module `etexood`.
You can execute `build.sh` to run a full test over our codes.

To execute baseline codes, please follow the README in each baseline repo to create independent virtual environments.
Our environment supports to execute GraIL and NBFNet (PyTorch-implementation), but for DRUM and Neural LP, corresponding environments are required.

## Execute

All example executions will run the least amount epoch (mostly 1). To reproduce results, please follow configuration in each paper. We recommend to use 50 here.

We provide an execution example in `run.sh` for FD-1.
To run other real-world datasets, change the name to WN18RR1 or NELL9951, and use following arguments: `--num-hops 3 --batch-size-edge-train 256 --batch-size-edge-valid 16 --batch-size-edge-test 16`.

To execute any baseline, go to corresponding directory under `clone`.
In the directory, run `link.sh` first to link datasets into baseline directory, then run example execution `run.sh`.
We put our reproduction setting as default in example, if you want to strictly reproduce original paper, please follow their reported hyperparameters.
