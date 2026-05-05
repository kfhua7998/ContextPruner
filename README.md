# ContextPruner

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A training-free context pruning method that removes redundant sentences before feeding them to LLMs.**

## One-line Summary

Remove 50% of the context, get 76% accuracy — random removal drops to 52%.

## Quick Start

```python
from src.pruner import ContextPruner

pruner = ContextPruner()
pruned = pruner.prune(
    question="What is the largest planet?",
    context="Jupiter is the largest planet. Earth is third. Saturn has rings.",
    keep_ratio=0.5
)
```

## Results on SQuAD (100 samples)

Evaluated with Qwen2-1.5B-Instruct on 100 SQuAD validation samples.

| Keep Ratio | Accuracy | Compression | vs Random |
|------------|----------|-------------|-----------|
| 100% | 80.0% | 1.00x | - |
| 70% | 77.0% | 1.85x | - |
| 50% | 76.0% | 2.18x | +24% |
| 30% | 71.0% | 3.69x | - |

**Key findings**:
- Semantic pruning (76%) significantly outperforms random pruning (52%) at 50% keep ratio
- Shuffling sentence order does not hurt accuracy (82% vs 80%)
- Compression ratio up to 2.18x with only 4% accuracy loss

## How It Works

1. Split context into sentences
2. Compute semantic similarity between each sentence and the question using BGE embedding
3. Keep only the top-K most similar sentences
4. Feed pruned context to any LLM

## Installation

```bash
pip install -r requirements.txt
```

## Run Experiment

```bash
python experiments/run_squad.py
```

## File Structure

```
ContextPruner/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   └── pruner.py
├── experiments/
│   └── run_squad.py
└── examples/
    └── simple_demo.py
```

## Citation

```bibtex
@misc{contextpruner2025,
  author = {Your Name},
  title = {ContextPruner: Training-Free Context Pruning for LLMs},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/ContextPruner}
}
```

## License

MIT
