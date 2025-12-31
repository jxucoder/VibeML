# 🧠 VibeML

**Tiny ML models from scratch. Inspired by [Karpathy's micrograd](https://github.com/karpathy/micrograd).**

No frameworks. No abstractions. Just the raw math.

## Goals

🎯 **Vibe Coding** — Build ML models through intuition, not boilerplate. Let the code flow.

📚 **Learning** — Understand what's *really* happening inside neural networks, trees, and beyond.

*Vibe coded with Claude Opus 4.5*

## Models

### Neural Networks
- **MLP** — Feedforward network that learns XOR
- **CNN** — Convolutional layers for image patterns
- **Capsule** — Dynamic routing between capsules
- **RNN** — Character-level language model with BPTT
- **GNN** — Message passing on graph structures
- **VAE** — Variational autoencoder for generation

### Self-Supervised
- **JEPA** — Joint Embedding Predictive Architecture

### Tree-Based
- **Decision Tree** — Recursive splitting with Gini impurity
- **Gradient Boosting** — Fit trees to residuals (XGBoost-style)

### Probabilistic
- **CRF** — Conditional Random Fields for sequences

### Factorization
- **FM** — Factorization Machines for sparse features

## Structure

```
models/
├── neural_networks/
│   ├── feedforward/mlp/
│   ├── convolutional/{cnn, capsule}/
│   ├── recurrent/rnn/
│   ├── graph/gnn/
│   └── generative/vae/
├── self_supervised/jepa/
├── tree_based/{decision_tree, gradient_boosting}/
├── probabilistic/crf/
├── factorization/fm/
├── meta_learning/           # Soon: PFN, MAML
├── reinforcement_learning/  # Soon: DQN, PPO
└── techniques/              # Soon: LoRA, Fine-tuning
```

Each model has a `micro/` folder for from-scratch implementations.

## Quick Start

```bash
# Run any model
python3 models/neural_networks/feedforward/mlp/micro/neural_network.py
python3 models/self_supervised/jepa/micro/jepa.py
python3 models/tree_based/gradient_boosting/micro/gbdt.py
```

## Roadmap

**Models:** MLP ✓ | Decision Tree ✓ | GBDT ✓ | RNN ✓ | CNN ✓ | Capsule ✓ | GNN ✓ | VAE ✓ | JEPA ✓ | CRF ✓ | FM ✓ | LSTM | Transformer | Word2Vec

**Coming:** Meta-Learning | Reinforcement Learning | Fine-tuning Techniques

## Contributing

1. Pick a model + category
2. Keep it minimal (<200 lines)
3. NumPy only — no ML libraries
4. Add a README with the math
5. PR it

## License

Apache 2.0
