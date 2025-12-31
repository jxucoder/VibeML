# Micro Capsule Network

A capsule network that learns to classify image patterns using dynamic routing. Unlike traditional CNNs, capsules output **vectors** instead of scalars—encoding not just *what* is detected, but *how* it appears.

## Python
```bash
python3 capsule.py
```

## What Makes Capsules Special?

Traditional neurons output a scalar (one number). Capsules output a **vector** where:
- **Length** = probability that an entity exists (0 to 1)
- **Orientation** = properties of the entity (pose, deformation, etc.)

```
Traditional CNN: neuron → 0.85 (just probability)
Capsule Network: capsule → [0.2, 0.7, -0.1, 0.3] (probability + pose!)
```

## Architecture

```
Input (6×6) → Conv (3×3) → Primary Capsules → Dynamic Routing → Digit Capsules
                                    ↓
                          [vectors encoding features]
                                    ↓
                          Routing by Agreement
                                    ↓
                          [class vectors with pose info]
```

## Key Components

### Squash Function
Non-linear activation that keeps vector length between 0 and 1:
```
squash(s) = ||s||² / (1 + ||s||²) × s / ||s||
```

### Dynamic Routing
Capsules "vote" on parent capsules through iterative agreement:
1. Each primary capsule predicts each digit capsule: `û_ij = W_ij × u_i`
2. Coupling coefficients computed via softmax: `c_ij = softmax(b_ij)`
3. Weighted sum + squash: `v_j = squash(Σ c_ij × û_ij)`
4. Update agreement: `b_ij += û_ij · v_j`
5. Repeat for `r` iterations

### Margin Loss
Encourages correct capsule to have length > 0.9, others < 0.1:
```
L_k = T_k × max(0, 0.9 - ||v_k||)² + 0.5 × (1-T_k) × max(0, ||v_k|| - 0.1)²
```

## Why Capsules Matter

**Equivariance**: When input transforms (rotates, shifts), capsule vectors transform correspondingly rather than losing information through pooling.

**Part-Whole Relationships**: Dynamic routing captures hierarchical relationships—parts "agree" on the whole they belong to.

**Viewpoint Invariance**: Same object from different angles produces capsules with same length but different orientation.

## Example Output

```
Vertical line at column 1:
  Capsule vector: [0.234, 0.812, -0.102, 0.445]
  Probability: 0.95

Vertical line at column 4:
  Capsule vector: [0.234, 0.398, -0.102, 0.821]  ← Different pose!
  Probability: 0.95

Same class, same confidence, but the vector encodes POSITION.
```

## References

- Sabour, Frosst, Hinton (2017): "Dynamic Routing Between Capsules"

