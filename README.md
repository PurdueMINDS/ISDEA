## IS-DEA Code Repository

We provide our proposal implementation and baseline implementations here.

-   Our implementation is under `src`.
-   Our baseline reproduction is under `clone/*`.
    Baselines are cloned from repos given in original paper with essential modifications to run with our data. All modifications are marked by comment `# MODIFY` and `# \\:`.

To execute our code, a virtual environment created by CONDA is required. Then execute `environment.sh` to install all dependencies.
Execute `pip install -e .` to install our implementation as importable Python module `etexood` under debugging mode.

To execute baseline codes, please follow the README in each baseline repos to create independent virtual environments.

## Execute

All example executions (`run.sh`) will run the least amount epoch (mostly 1). To reproduce results, please follow configuration in baseline papers and our paper.
We use 50 epochs for all experiments.

In `run.sh`, we provide the simplest example for synthetic FD-1 and FD-2 tasks.
To run other real-world inductive knowledge graph completion datasets, please refer to `share-generate.sh`, `share-fit.sh` and `share-transform.sh`.
For example, to run an experiment with ISDEA using mean DSS aggregation on FB237 v1, please set following variables in those three scripts as

```bash
task=FB2371
model=dssgnn
aggr=mean
ablate=both
```

To execute any baseline, go to corresponding directory under `clone`.
In the directory, run `link.sh` first to link datasets into baseline directory, then run example execution `run.sh`.
If you want to run on a new datasets, please change `task` variable in the example scripts of baselines.
