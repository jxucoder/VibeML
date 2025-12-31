# 🧠 VibeML

**Tiny ML models from scratch. Inspired by [Karpathy's micrograd](https://github.com/karpathy/micrograd).**

No frameworks. No abstractions. Just the raw math.

## Goals

🎯 **Vibe Coding** — Build ML models through intuition, not boilerplate. Let the code flow.

📚 **Learning** — Understand what's *really* happening inside neural networks, trees, and beyond.

*Vibe coded with Claude Opus 4.5*

## Models

| Category | Model | Run |
|----------|-------|-----|
| **Neural Networks** | | |
| └ Feedforward | MLP (XOR) | `python3 models/neural_networks/feedforward/mlp/micro/neural_network.py` |
| └ Convolutional | CNN | `python3 models/neural_networks/convolutional/cnn/micro/cnn.py` |
| └ Convolutional | Capsule Network | `python3 models/neural_networks/convolutional/capsule/micro/capsule.py` |
| └ Recurrent | RNN | `python3 models/neural_networks/recurrent/rnn/micro/rnn.py` |
| └ Graph | GNN | `python3 models/neural_networks/graph/gnn/micro/gnn.py` |
| └ Generative | VAE | `python3 models/neural_networks/generative/vae/micro/vae.py` |
| **Self-Supervised** | JEPA | `python3 models/self_supervised/jepa/micro/jepa.py` |
| **Tree-Based** | Decision Tree | `python3 models/tree_based/decision_tree/micro/decision_tree.py` |
| **Tree-Based** | Gradient Boosting | `python3 models/tree_based/gradient_boosting/micro/gbdt.py` |
| **Probabilistic** | CRF | `python3 models/probabilistic/crf/micro/crf.py` |
| **Factorization** | FM | `python3 models/factorization/fm/micro/fm.py` |

## Structure

```
models/
├── neural_networks/
│   ├── feedforward/
│   │   └── mlp/
│   │       ├── micro/              # From-scratch Python
│   │       └── other_languages/    # COBOL, Pascal
│   ├── convolutional/
│   │   ├── cnn/micro/
│   │   └── capsule/micro/
│   ├── recurrent/
│   │   └── rnn/micro/
│   ├── graph/
│   │   └── gnn/micro/
│   └── generative/
│       └── vae/micro/
│
├── self_supervised/
│   └── jepa/micro/
│
├── tree_based/
│   ├── decision_tree/micro/
│   └── gradient_boosting/micro/
│
├── probabilistic/
│   └── crf/micro/
│
├── factorization/
│   └── fm/micro/
│
├── meta_learning/              # Coming soon: PFN, MAML
├── reinforcement_learning/     # Coming soon: DQN, PPO
└── techniques/                 # Coming soon: LoRA, Fine-tuning
```

### Folder Convention

- `micro/` — From-scratch implementations (NumPy only)
- `torch/` — PyTorch implementations (coming later)
- `other_languages/` — Non-Python implementations

## Roadmap

**Models:** MLP ✓ | Decision Tree ✓ | GBDT ✓ | RNN ✓ | CNN ✓ | Capsule ✓ | GNN ✓ | VAE ✓ | JEPA ✓ | CRF ✓ | FM ✓ | LSTM | Transformer | Word2Vec | Boltzmann | RL

**Categories:** Neural Networks ✓ | Tree-Based ✓ | Probabilistic ✓ | Self-Supervised ✓ | Meta-Learning | Reinforcement Learning

## Contributing

1. Pick a model + category
2. Keep it minimal (<200 lines for micro/)
3. No ML libraries (NumPy only)
4. Add a README explaining the math
5. PR it

## License

Apache 2.0
