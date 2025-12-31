# microgbm - Histogram GBDT from Scratch

A minimal Gradient Boosting Machine implementation in pure Python. No dependencies.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Histogram Binning** | Discretize features into bins for fast split finding |
| **Second-Order Gradients** | Use both gradient (G) and hessian (H) |
| **Optimal Leaf Weights** | w* = -G / (H + λ) |
| **Split Gain** | 0.5 × [G²ₗ/(Hₗ+λ) + G²ᵣ/(Hᵣ+λ) - G²/(H+λ)] - γ |
| **L2 Regularization** | λ controls leaf weight shrinkage |

## Run

```bash
python3 gbdt.py
```

## Usage

```python
from gbdt import GBMClassifier

model = GBMClassifier(
    n_estimators=10,
    max_depth=3,
    learning_rate=0.3,
    reg_lambda=1.0,
    num_bins=256
)
model.fit(X, y)
probs = model.predict_proba(X_test)
```
