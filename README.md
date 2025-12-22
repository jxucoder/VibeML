# 🧠 VibeML

**Tiny ML models in every language. Inspired by [Karpathy's micrograd](https://github.com/karpathy/micrograd).**

No frameworks. No abstractions. Just the raw math.

*Vibe coded with Claude Opus 4.5 for learning & education.*

## Models

### Simple Neural Network
A 2→4→1 network that learns XOR.

- **Python** (~35 lines): `python3 neural_network.py`
- **Pascal** (~45 lines): `fpc neural_network.pas && ./neural_network`
- **COBOL** (~70 lines): `cobc -x -free neural_network.cob && ./a.out`

### Vanilla Decision Tree
Recursive splitting with Gini impurity.

- **Python** (~60 lines): `python3 decision_tree.py`

### Gradient Boosting Decision Tree
Fit trees to residuals, sum predictions. Powers XGBoost/LightGBM.

- **Python** (~75 lines): `python3 gbdt.py`

## Structure

```
models/
├── simple_neural_network/
│   ├── python/
│   ├── pascal/
│   └── cobol/
├── vanilla_decision_tree/
│   └── python/
└── gradient_boosting_decision_tree/
    └── python/
```

## Roadmap

**Models:** Simple NN ✓ | Vanilla Decision Tree ✓ | Gradient Boosting ✓ | Random Forest | KNN | Linear Regression | CNN | RNN | Transformer

**Languages:** Python ✓ | Pascal ✓ | COBOL ✓ | C | Rust | Go | JavaScript | Haskell

## Contributing

1. Pick a model + language
2. Keep it minimal (<100 lines)
3. No ML libraries
4. PR it

## License

Apache 2.0
