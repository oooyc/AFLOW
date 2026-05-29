# CAFG

**Cluster-Awareness and Fine-Grained Feedback Driven Optimization Method for LLM-based Multi-Agent Systems**

Chao Ouyang, et al. *Computer Engineering and Applications (计算机工程与应用), 2025.*

CAFG is the official implementation of the paper above. It builds on the [AFlow](https://github.com/FoundationAgents/AFlow) automatic agentic-workflow optimizer and improves its search with two components:

- **Cluster-aware merging of redundant workflows.** Candidate workflows are represented by structural + semantic features; functionally redundant search branches are detected by Louvain community detection over a workflow-similarity graph and merged, cutting wasted exploration (design redundancy reduced from **63% to 54%** in our experiments). See `scripts/optimizer_utils/diversity_utils.py` and `select_merge.py`.
- **Node-level fine-grained feedback.** Sparse end-to-end rewards are decomposed into local, per-node signals (input-quality scoring + self-reflection), giving the optimizer per-agent-interaction credit instead of a single trajectory-level score.

Across multiple reasoning and code-generation benchmarks, these additions raise the average score to **84.2** while improving search convergence and robustness.

> Built on **AFlow** (Zhang et al., ICLR 2025). This repository keeps AFlow's workflow representation, operators, and MCTS-style optimizer, and adds the cluster-aware and fine-grained-feedback components above.

## Components added over AFlow

| Component | Where |
|---|---|
| Workflow structural + semantic similarity, Louvain clustering, redundant-branch merging | `scripts/optimizer_utils/diversity_utils.py`, `select_merge.py` |
| Node-level fine-grained feedback (input-quality scoring + self-reflection) | `scripts/optimizer.py`, `scripts/evaluator.py`, `workspace/*/workflows/template/` |

AFlow's original abstractions — Node, Operator, Workflow, Optimizer, Evaluator — are retained; see the AFlow repo for their design.

## Installation

```bash
conda create -n cafg python=3.9 -y
conda activate cafg
pip install -r requirements.txt
```

Cluster-aware merging additionally uses `networkx`, `python-louvain` (imported as `community`), and `scikit-learn`, with optional `sentence-transformers` for semantic features (a TF-IDF fallback is used when it is unavailable).

## Quick Start

1. Configure LLM parameters in the config file (see `config/config2.example.yaml` for reference).
2. Run optimization on a benchmark:
   ```bash
   python run.py --dataset MATH
   # or with custom parameters:
   python run.py --dataset MATH --sample 4 --optimized_path optimized --max_rounds 20
   ```
   Supported datasets: HumanEval, MBPP, GSM8K, MATH, HotpotQA, DROP.

## Datasets

Six benchmarks (HumanEval, MBPP, GSM8K, MATH, HotpotQA, DROP) with evaluation code under `benchmarks/`. For a custom task, inherit `BaseBenchmark` and register it in `evaluator.py` / `optimizer.py` (same interface as AFlow).

## Citation

If you use this code, please cite:

```bibtex
@article{ouyang2025cafg,
  title   = {Cluster-Awareness and Fine-Grained Feedback Driven Optimization Method for LLM-based Multi-Agent Systems},
  author  = {Ouyang, Chao and others},
  journal = {Computer Engineering and Applications},
  year    = {2025}
}
```

This work builds on AFlow; please also cite:

```bibtex
@inproceedings{zhang2025aflow,
  title     = {AFlow: Automating Agentic Workflow Generation},
  author    = {Zhang, Jiayi and Xiang, Jinyu and Yu, Zhaoyang and Teng, Fengwei and Chen, Xiong-Hui and Chen, Jiaqi and Zhuge, Mingchen and Cheng, Xin and Hong, Sirui and Wang, Jinlin and Zheng, Bingnan and Liu, Bang and Luo, Yuyu and Wu, Chenglin},
  booktitle = {The Thirteenth International Conference on Learning Representations (ICLR)},
  year      = {2025}
}
```

## Acknowledgement

This project is built on [AFlow](https://github.com/FoundationAgents/AFlow) and MetaGPT. We thank the authors for releasing their code. Licensed under MIT (see [LICENSE](LICENSE)).
