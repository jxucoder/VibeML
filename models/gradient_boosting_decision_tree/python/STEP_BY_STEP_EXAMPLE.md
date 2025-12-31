# microgbm: Step-by-Step Example

This guide walks through the Gradient Boosting Decision Tree algorithm using a concrete example with actual calculations.

---

## 1. The Problem: Binary Classification

We have a simple dataset with 2 features and binary labels:

| Sample | Feature 0 (x₀) | Feature 1 (x₁) | Label (y) |
|--------|----------------|----------------|-----------|
| 0      | 0.5            | 0.2            | 0         |
| 1      | 1.0            | 0.4            | 0         |
| 2      | 1.5            | 0.6            | 0         |
| 3      | 2.0            | 0.8            | 0         |
| 4      | 2.5            | 1.0            | 0         |
| 5      | 3.0            | 1.2            | 0         |
| 6      | 3.5            | 1.4            | 1         |
| 7      | 4.0            | 1.6            | 1         |
| 8      | 4.5            | 1.8            | 1         |
| 9      | 5.0            | 2.0            | 1         |
| 10     | 5.5            | 2.2            | 1         |
| 11     | 6.0            | 2.4            | 1         |

**Goal:** Learn to predict the probability of class 1.

---

## 2. Step 1: Histogram Binning

Instead of considering every unique value for splits, we discretize features into bins. This dramatically speeds up split finding.

### How it works:

```
For Feature 0: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
                 ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
Bin indices:     0    1    2    3    4    5    6    7    8    9   10   11
```

With `num_bins=16`, since we only have 12 unique values, each value gets its own bin.

### Binned Data:

| Sample | Bin[0] | Bin[1] | Label |
|--------|--------|--------|-------|
| 0      | 0      | 0      | 0     |
| 1      | 1      | 1      | 0     |
| 2      | 2      | 2      | 0     |
| 3      | 3      | 3      | 0     |
| 4      | 4      | 4      | 0     |
| 5      | 5      | 5      | 0     |
| 6      | 6      | 6      | 1     |
| 7      | 7      | 7      | 1     |
| 8      | 8      | 8      | 1     |
| 9      | 9      | 9      | 1     |
| 10     | 10     | 10     | 1     |
| 11     | 11     | 11     | 1     |

---

## 3. Step 2: Initialize Base Score

The initial prediction is based on the class distribution:

```
mean(y) = 6/12 = 0.5

base_score = log(mean_y / (1 - mean_y))
           = log(0.5 / 0.5)
           = log(1)
           = 0.0
```

**Initial prediction for all samples:** `ŷ = [0.0, 0.0, 0.0, ..., 0.0]`

**Initial probability:** `p = sigmoid(0) = 0.5` for all samples

---

## 4. Step 3: Compute Gradients and Hessians

For binary classification with log loss:

| Formula | Description |
|---------|-------------|
| **Gradient:** `g = p - y` | First derivative of loss |
| **Hessian:** `h = p × (1 - p)` | Second derivative of loss |

### Calculations for Round 1:

Since `p = 0.5` for all samples initially:

| Sample | y | p   | Gradient (g = p - y) | Hessian (h = p(1-p)) |
|--------|---|-----|----------------------|----------------------|
| 0      | 0 | 0.5 | 0.5 - 0 = **0.5**    | 0.5 × 0.5 = **0.25** |
| 1      | 0 | 0.5 | **0.5**              | **0.25**             |
| 2      | 0 | 0.5 | **0.5**              | **0.25**             |
| 3      | 0 | 0.5 | **0.5**              | **0.25**             |
| 4      | 0 | 0.5 | **0.5**              | **0.25**             |
| 5      | 0 | 0.5 | **0.5**              | **0.25**             |
| 6      | 1 | 0.5 | 0.5 - 1 = **-0.5**   | **0.25**             |
| 7      | 1 | 0.5 | **-0.5**             | **0.25**             |
| 8      | 1 | 0.5 | **-0.5**             | **0.25**             |
| 9      | 1 | 0.5 | **-0.5**             | **0.25**             |
| 10     | 1 | 0.5 | **-0.5**             | **0.25**             |
| 11     | 1 | 0.5 | **-0.5**             | **0.25**             |

**Totals:**
- `G = Σg = 6 × 0.5 + 6 × (-0.5) = 0`
- `H = Σh = 12 × 0.25 = 3.0`

---

## 5. Step 4: Build a Tree (Finding the Best Split)

### 5.1 Build Histogram for Each Feature

For **Feature 0**, we accumulate gradients and hessians into bins:

| Bin | Gradient | Hessian | Sample |
|-----|----------|---------|--------|
| 0   | 0.5      | 0.25    | 0      |
| 1   | 0.5      | 0.25    | 1      |
| 2   | 0.5      | 0.25    | 2      |
| 3   | 0.5      | 0.25    | 3      |
| 4   | 0.5      | 0.25    | 4      |
| 5   | 0.5      | 0.25    | 5      |
| 6   | -0.5     | 0.25    | 6      |
| 7   | -0.5     | 0.25    | 7      |
| 8   | -0.5     | 0.25    | 8      |
| 9   | -0.5     | 0.25    | 9      |
| 10  | -0.5     | 0.25    | 10     |
| 11  | -0.5     | 0.25    | 11     |

### 5.2 Evaluate Split Candidates

For each potential split point, calculate the **gain**:

```
Gain = 0.5 × [G²_left/(H_left + λ) + G²_right/(H_right + λ) - G²/(H + λ)] - γ
```

With `λ = 1.0` (regularization) and `γ = 0` (min gain):

**Current score (no split):**
```
current_score = G² / (H + λ) = 0² / (3.0 + 1.0) = 0
```

**Evaluating split at bin 5 (x₀ ≤ 3.0 vs x₀ > 3.0):**

```
Left (bins 0-5):   G_left = 6 × 0.5 = 3.0,    H_left = 6 × 0.25 = 1.5
Right (bins 6-11): G_right = 6 × (-0.5) = -3.0, H_right = 6 × 0.25 = 1.5

left_score  = G²_left / (H_left + λ)   = 9.0 / 2.5 = 3.6
right_score = G²_right / (H_right + λ) = 9.0 / 2.5 = 3.6

Gain = 0.5 × (3.6 + 3.6 - 0) - 0 = 3.6
```

This is a perfect split! ✓

### 5.3 Calculate Leaf Weights

After splitting at Feature 0, bin 5:

**Left leaf (samples 0-5, all y=0):**
```
weight_left = -G_left / (H_left + λ)
            = -3.0 / (1.5 + 1.0)
            = -3.0 / 2.5
            = -1.2
```

**Right leaf (samples 6-11, all y=1):**
```
weight_right = -G_right / (H_right + λ)
             = -(-3.0) / (1.5 + 1.0)
             = 3.0 / 2.5
             = 1.2
```

### Tree 1 Structure:

```
           [Feature 0 ≤ bin 5?]
                  /    \
                 /      \
         Yes (≤3.0)   No (>3.0)
              ↓           ↓
          w = -1.2    w = +1.2
```

---

## 6. Step 5: Update Predictions

Update predictions using learning rate `η = 0.3`:

```
ŷ_new = ŷ_old + η × tree_prediction
```

| Sample | Old ŷ | Tree Output | New ŷ = 0 + 0.3 × output |
|--------|-------|-------------|--------------------------|
| 0-5    | 0.0   | -1.2        | 0.0 + 0.3 × (-1.2) = **-0.36** |
| 6-11   | 0.0   | +1.2        | 0.0 + 0.3 × (+1.2) = **+0.36** |

**New probabilities:**
```
For samples 0-5:  p = sigmoid(-0.36) = 1/(1+e^0.36) ≈ 0.411
For samples 6-11: p = sigmoid(+0.36) = 1/(1+e^-0.36) ≈ 0.589
```

---

## 7. Step 6: Repeat (Build More Trees)

### Round 2: New Gradients and Hessians

| Sample | y | p     | Gradient (p - y) | Hessian (p(1-p)) |
|--------|---|-------|------------------|------------------|
| 0-5    | 0 | 0.411 | 0.411            | 0.242            |
| 6-11   | 1 | 0.589 | -0.411           | 0.242            |

The algorithm continues building trees to reduce the residual errors...

---

## 8. Final Predictions

After all trees are built, the final prediction combines all tree outputs:

```
raw_score = base_score + η × Σ tree_i(x)
probability = sigmoid(raw_score)
class = 1 if probability > 0.5 else 0
```

### Example Output:

```
x=[0.5, 0.2] -> prob=0.0067, pred=0, true=0 ✓
x=[1.0, 0.4] -> prob=0.0067, pred=0, true=0 ✓
x=[1.5, 0.6] -> prob=0.0067, pred=0, true=0 ✓
x=[2.0, 0.8] -> prob=0.0067, pred=0, true=0 ✓
x=[2.5, 1.0] -> prob=0.0067, pred=0, true=0 ✓
x=[3.0, 1.2] -> prob=0.0067, pred=0, true=0 ✓
x=[3.5, 1.4] -> prob=0.9933, pred=1, true=1 ✓
x=[4.0, 1.6] -> prob=0.9933, pred=1, true=1 ✓
x=[4.5, 1.8] -> prob=0.9933, pred=1, true=1 ✓
x=[5.0, 2.0] -> prob=0.9933, pred=1, true=1 ✓
x=[5.5, 2.2] -> prob=0.9933, pred=1, true=1 ✓
x=[6.0, 2.4] -> prob=0.9933, pred=1, true=1 ✓

Accuracy: 100.0%
```

---

## 9. Key Formulas Summary

| Component | Formula | Purpose |
|-----------|---------|---------|
| **Sigmoid** | σ(x) = 1/(1 + e⁻ˣ) | Convert raw score to probability |
| **Gradient** | g = p - y | Direction of steepest descent |
| **Hessian** | h = p(1-p) | Curvature (how fast gradient changes) |
| **Leaf Weight** | w* = -G/(H + λ) | Optimal prediction for a leaf |
| **Split Gain** | 0.5[G²ₗ/(Hₗ+λ) + G²ᵣ/(Hᵣ+λ) - G²/(H+λ)] - γ | Improvement from splitting |
| **Prediction** | ŷ = base + η × Σtree(x) | Ensemble output |

---

## 10. Hyperparameters Explained

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `n_estimators` | More trees = more capacity (can overfit) | 10-1000 |
| `max_depth` | Deeper trees = more complex patterns | 3-10 |
| `learning_rate` (η) | Lower = more trees needed, but more stable | 0.01-0.3 |
| `reg_lambda` (λ) | Higher = more regularization, simpler leaves | 0-10 |
| `min_child_weight` | Minimum H required to make a leaf | 1-10 |
| `gamma` (γ) | Minimum gain required to make a split | 0-5 |
| `num_bins` | More bins = finer splits, slower | 32-256 |

---

## 11. Code Walkthrough

### Initialize and Train:

```python
from gbdt import GBMClassifier

model = GBMClassifier(
    n_estimators=10,    # Build 10 trees
    max_depth=3,        # Each tree has max 3 levels
    learning_rate=0.3,  # Step size
    reg_lambda=1.0,     # L2 regularization
    num_bins=16         # Histogram bins
)

model.fit(X, y)
```

### Make Predictions:

```python
# Get probabilities
probs = model.predict_proba(X_test)

# Get class labels (0 or 1)
predictions = model.predict(X_test)
```

---

## 12. Why Histograms?

Traditional GBDT examines every unique value as a potential split point:
- **O(n × features × unique_values)** per tree

Histogram-based GBDT:
1. Pre-bin all features (once)
2. Build gradient/hessian histograms
3. Only consider bin boundaries as splits
- **O(n × features + features × bins)** per tree

For large datasets, this is **dramatically faster**! This is the core idea behind LightGBM and XGBoost's `tree_method='hist'`.

---

## Run the Demo

```bash
python3 gbdt.py
```

Output:
```
==================================================
microgbm - Histogram GBDT from scratch
==================================================

microgbm training
params: n_estimators=10, max_depth=3, lr=0.3, lambda=1.0
--------------------------------------------------
[  1] log_loss: 0.525990
[  2] log_loss: 0.402471
[  3] log_loss: 0.308782
...
[ 10] log_loss: 0.037838
--------------------------------------------------

Predictions:
--------------------------------------------------
x=[0.5, 0.2] -> prob=0.0067, pred=0, true=0 ✓
...
```

