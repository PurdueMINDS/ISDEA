# Double Equivariant InGram (DEq-InGram)

Modified from the original InGram implementation, which can be found here: https://github.com/bdi-lab/InGram

## TL;DR - Modified example train and test command

To train a model on the PediaTypes `DB2WD` dataset with logging to terminal, run

```shell
Python train.py --data_path data/PediaTypes/ --data_name DB2WD-15K-V2 --exp PediaTypes --seed 42
```

To train a model on the same dataset but log to Weights & Biases, run

```shell
Python train.py --data_path data/PediaTypes/ --data_name DB2WD-15K-V2 --exp PediaTypes --seed 42 --wandb --wandb-project <your wandb project name> --wandb-entity <your account/team name> --wandb-job-type <wandb job type>
```

To test **DEq-InGram with 10 Monte Carlo Samples** using the best checkpoint saved from the above training run, which should be located in `ckpt/PediaTypes/DB2WD-15K-V2/<run_hash>`, run

```shell
test.py --best --run_hash <run_hash> --data_name DB2WD-15K-V2 --exp PediaTypes --mc 10
```

where <run_hash> is the hash of the hyperparameters of the previous training run. The hash is printed to the terminal when the training run is started.

Or alternatively, you can specify data_name and run_hash together using the `--data_name_run_hash` argument:

```shell
test.py --best --data_name_run_hash DB2WD-15K-V2/<run_hash> --exp PediaTypes --mc 10
```

## Summary of hyperparameters

We conducted hyperparameter search on the following hyperparameters:
```
margin: [1.0, 2.0]
learning_rate: [0.0005, 0.001]
num_layer_ent: [2, 3, 4]
num_layer_rel: [1, 2, 3]
```
Values of the other hyperparameters are those used in the original paper and repo.

For final baseline comparisons, we trained the models with the following hyperparameters:

On PediaTypes all scenarios:
```
--margin 2.0 --learning_rate 0.001 --num_layer_ent 4 --num_layer_rel 3 --dimension_entity 32 --dimension_relation 32 --hidden_dimension_ratio_entity 8 --hidden_dimension_ratio_relation 4 --num_bin 10 --num_epoch 10000 --validation_epoch 200 --num_head 8 --num_neg 10
```

On WikiTopics all scenarios:
```
--margin 2.0 --learning_rate 0.001 --num_layer_ent 4 --num_layer_rel 3 --dimension_entity 32 --dimension_relation 32 --hidden_dimension_ratio_entity 8 --hidden_dimension_ratio_relation 4 --num_bin 10 --num_epoch 10000 --validation_epoch 200 --num_head 8 --num_neg 10
```

# Change Log

## Evaluation metrics

The experiments in the original paper evaluate the model performance against all negative head/tail entities in the graph. In our setting, we (1) additionally evaluate against negative relations and (2) evaluate only against 50 random negative samples. Specifically, we

- for each positive triplet, randomly sample 50 negative head, 50 negative tails, and 50 negative relations,
- aggregate the results for negative heads and negative tails as entity rankings, and
- report the MR, MRR, Hits@1, Hits@3, and Hits@10 for both entity rankings and relation rankings.

We also report a mixed metric, where each positive triplet is ranked against 12 negative heads, 12 negative tails, and 26 negative relations. These negative heads, tails, and relations are taken from thosed sampled above.

In addition, we consider the worst-case scenario when breaking ties. That is, when there are multiple negative triplets with the same score as the positive one, then our positive triplet is **ranked after all of those negative ones**. This is in contrast to the original paper where the average rank of the positive triplet against the negative ones with the same score is taken.

See `evaluation.py` and `utils.py` for modification details.

## Monte Carlo sampling of triplet scores

In addition to the original implementation that performs a single forward pass to compute the triplet scores, we also implement Monte Carlo sampling of triplet scores. Specifically, we first draw `mc` samples of the initial entity and relation embeddings, then perform `mc` forward passes to compute the triplet scores. The final triplet score is the average of the `mc` scores. This process is theoretically more sound since it is in line with the theory of Double Equivariance (See: https://arxiv.org/abs/2302.01313), where the output triplet scores of the InGram model given a random initial entity and relation embeddings should be considered as a sample from the structural representation of the triplet that is double equivariant. In practice, we also observe that this procedure gives significantly better results with much less variance. 

See `my_parser.py` for the additional argument `mc` and `evaluation.py` and `text.py` for the implementation of the Monte Carlo procedure. 

## Data split and data loader

In the original paper, the validation triplets are taken from the test-time knowledge graph. We adapt it to our setting where the validation triplets are those sampled from the training graph. Hence, the data file `train.txt`, `valid.txt`, `msg.txt`, and `test.txt` are used in the following way:

- During training, the model is trained solely on `train.txt` by making a self-supervised mask on these triplets.
- During validation, `train.txt` is taken as the message/observation/input to the model, and `valid.txt` is taken as the supervision/target triplets to predict.
- During test, `msg.txt` is taken as the message/observation/input to the model, and `test.txt` is taken as the supervision/target triplets to predict.

Additionally, when readining triplets from file, separately the line by any whitespate via `.split()` instead of by tab via `.split('\t')`.

See `dataset.py` for modification details.

## Re-initialize random entity and relation embeddings during test

Since in our setting, the validation set is sampled from the training graph instead of test graph (as in the original paper), we re-initialize the entity and relation embeddings before running the model on the test set. We use the same `initialize()` method from `initialize.py`.

See `test.py` for modification details.

## Early stop and save best model checkpoints during training

During trainin, we monitor the `mrr_ent` metric (entity MRR) on the validation set and __only__ save the model checkpoint with the best `mrr_ent` metric. The best model checkpoint is named `best.ckpt` and is saved in the same folder as the other episodic model checkpoints.

We also stop training if the `mrr_ent` metric does not improve for 10 epochs.

See `train.py` for modification details.

## Enable loading the best model checkpoint for testing

We enable loading the `best.ckpt` model checkpoint of any given experiments for testing. To do so, we modify the command line arguments so now you should specify `--best` together with `--run-hash`, `--data_name`, and `--exp` to load the best model checkpoint of the given experiment.

See `test.py` for modification details.

## Specify random seed

Allow specify random seed of the experiment in the command line with the argument `--seed`. 

See `my_parser.py` for modification details.

## Use Weights & Biases for logging

Use Weights & Biases for logging.

See `train.py` and `test.py` for modification details and `utils.py` for additional utility functions.

## Modify checkpoint saving directory

Now a hash is created for each run using the hyperparameters specified in the command line arguments, and the model checkpoints are saved in the following directory:

```
./ckpt/[experiment_name]/[dataset_name]/[run_hash]/
```

where `[experiment_name]` is the experiment name specified by `--exp` argument, `[dataset_name]` is the name of the dataset specified by `--data_name` argument, and `[hash]` is the hash of the hyperparameters. The hash is created using the `hashlib` library and is the first 8 characters of the SHA-256 hash of the hyperparameters.

See `train.py` for modification details.

## Fix divide-by-zero error which may cause NaN values

Fix a divide-by-zero error in `InGramEntityLayer.forward()` which may cause NaN values in the outputs. Specifically, the `ent_freq` tensor may contain 0s which is then used as a divisor to compute `self_rel`. We fix this by replacing 0s in `ent_freq` with 1s.

See `model.py` for modification details.

## Fix numpy int type conversion error when computing metrics

Replace `np.int` with built-in `int` to avoid type conversion error when computing metrics in `get_metrics()`.

See `utils.py` for modification details.


# InGram: Inductive Knowledge Graph Embedding via Relation Graphs
This code is the official implementation of the following [paper](https://proceedings.mlr.press/v202/lee23c.html):

> Jaejun Lee, Chanyoung Chung, and Joyce Jiyoung Whang, InGram: Inductive Knowledge Graph Embedding via Relation Graphs, The 40th International Conference on Machine Learning (ICML), 2023.

All codes are written by Jaejun Lee (jjlee98@kaist.ac.kr). When you use this code or data, please cite our paper.

```bibtex
@inproceedings{ingram,
	author={Jaejun Lee and Chanyoung Chung and Joyce Jiyoung Whang},
	title={{I}n{G}ram: Inductive Knowledge Graph Embedding via Relation Graphs},
	booktitle={Proceedings of the 40th International Conference on Machine Learning},
	year={2023},
	pages={18796--18809}
}
```

## Requirements

We used Python 3.8 and PyTorch 1.12.1 with cudatoolkit 11.3.

You can install all requirements with:

```shell
pip install -r requirements.txt
```

## Reproducing the Reported Results

We used NVIDIA RTX A6000, NVIDIA GeForce RTX 2080 Ti, and NVIDIA GeForce RTX 3090 for all our experiments. We provide the checkpoints we used to produce the inductive link prediction results on 14 datasets. If you want to use the checkpoints, place the unzipped ckpt folder in the same directory with the codes.

You can download the checkpoints from https://drive.google.com/file/d/1aZrx2dYNPT7j4TGVBOGqHMdHRwFUBqx5/view?usp=sharing.

The command to reproduce the results in our paper:

```python
python3 test.py --best --data_name [dataset_name]
```

## Training from Scratch

To train InGram from scratch, run `train.py` with arguments. Please refer to `my_parser.py` for the examples of the arguments. Please tune the hyperparameters of our model using the range provided in Appendix C of the paper because the best hyperparameters may be different due to randomness.

The list of arguments of `train.py`:
- `--data_name`: name of the dataset
- `--exp`: experiment name
- `-m, --margin`: $\gamma$
- `-lr, --learning_rate`: learning rate
- `-nle, --num_layer_ent`: $\widehat{L}$
- `-nlr, --num_layer_rel`: $L$
- `-d_e, --dimension_entity`: $\widehat{d}$
- `-d_r, --dimension_relation`: $d$
- `-hdr_e, --hidden_dimension_ratio_entity`: $\widehat{d'}/\widehat{d}$
- `-hdr_r, --hidden_dimension_ratio_relation`: $d'/d$
- `-b, --num_bin`: $B$
- `-e, --num_epoch`: number of epochs to run
- `--target_epoch`: the epoch to run test (only used for test.py)
- `-v, --validation_epoch`: duration for the validation
- `--num_head`: $\widehat{K}=K$
- `--num_neg`: number of negative triplets per triplet
- `--best`: use the provided checkpoints (only used for test.py)
- `--no_write`: don't save the checkpoints (only used for train.py)
