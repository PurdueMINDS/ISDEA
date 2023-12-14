# Double Equivariance for Inductive Link Prediction for Both New Nodes and New Relation Types

This repository is the official implementation of the following [paper](https://arxiv.org/abs/2302.01313):

> Gao, Jianfei et al. “Double Equivariance for Inductive Link Prediction for Both New Nodes and New Relation Types.” (2023).

It contains three (3) models introduced in the paper, which are located in their respective directory as follows:

- `ISDEA/`: The legacy implementation of the Inductive Structural Double Equivariant Architecture (ISDEA).
- `ISDEA_PLUS/`: The new and improved Inductive Structural Double Equivariant Architecture **Plus** (ISDEA+), which achieves 20x - 120x speedup and attains superior performance compared to the legacy ISDEA. We thank Yucheng Zhang for his contribution to this implementation, and we hereby include his repository as a git submodule in this repository.
- `DEq_InGram/`: The Double Equivariant version of InGram, which achieves better performance than the original InGram on the doubly inductive link prediction task on knowledge graphs considered in the paper.

To clone the repository, run:
```
git clone --recurse-submodules https://github.com/PurdueMINDS/ISDEA.git
```
This ensures that all files, including those from ISDEA+ that are added as git submodule to this repository, are cloned correctly.

Please refer to the `README.md` file in each directory for more details on how to run the code.


## Citation

When you use this code or data, we kindly ask you to cite *BOTH* our paper and the ISDEA+ github repository:
```
@article{gao2023double,
  title={Double Equivariance for Inductive Link Prediction for Both New Nodes and New Relation Types},
  author={Gao, Jianfei and Zhou, Yangze and Zhou, Jincheng and Ribeiro, Bruno},
  journal={arXiv preprint arXiv:2302.01313},
  year={2023}
}
```

```
@software{Zhang_ISDEA,
author = {Zhang, Yucheng},
title = {{ISDEA+}},
url = {https://github.com/yuchengz99/ISDEA_PLUS},
month = {12},
year = {2023},
version = {1.0.0}
}
```

Thank you!

