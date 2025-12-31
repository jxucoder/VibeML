# Micro JEPA

A tiny implementation of **Joint Embedding Predictive Architecture (JEPA)** in pure Python. No frameworks, no dependencies—just the math.

## What is JEPA?

JEPA is a self-supervised learning approach that learns representations by **predicting in embedding space** rather than pixel space. Unlike autoencoders that reconstruct inputs, JEPA predicts the *embeddings* of masked patches from visible patches.

Key insight: By predicting abstract representations instead of low-level pixels, the model learns more semantic and useful features.

## Architecture

```
4x4 Image → 2x2 Patches → Embeddings → Predictions

┌─────────────────┐
│  Context Encoder │←── Learns via gradient descent
└────────┬────────┘
         ↓
    Context Embs ──→ ┌───────────┐
         +           │ Predictor │──→ Predicted Target Emb
    Position Enc ──→ └───────────┘            ↓
                                         MSE Loss
                                              ↑
┌─────────────────┐                  Actual Target Emb
│  Target Encoder  │←── EMA of Context Encoder (no gradients!)
└─────────────────┘
```

## Key Components

1. **Context Encoder**: Encodes visible patches into embeddings (trained)
2. **Target Encoder**: EMA copy of context encoder (provides stable targets)
3. **Predictor**: Predicts target embeddings from context embeddings
4. **Masking**: Random patch masking (≈75% context, ≈25% targets)

## Why EMA for Target Encoder?

The exponential moving average prevents **representation collapse**—where the model learns trivial constant outputs. By making targets a slowly-changing average, JEPA maintains useful signal.

## Run

```bash
python3 jepa.py
```

## Output

```
VibeML Micro JEPA - Learning Representations via Prediction
============================================================

Generated 200 training images (4x4)
Architecture: 2x2 patches -> 6-dim embeddings

Training for 500 epochs...
Learning rate: 0.02, EMA momentum: 0.99
------------------------------------------------------------
Epoch   50 | Loss: 0.4150 | Pred: 0.4128 | Var: 0.0021
Epoch  100 | Loss: 0.8301 | Pred: 0.8274 | Var: 0.0027
...

Representation Quality Test
============================================================
Embeddings for different patterns:
h_gradient  : [ 0.527,  0.058,  1.827,  0.873, -0.869,  1.492]
v_gradient  : [ 0.084, -0.721, -0.348,  1.334, -2.093,  0.589]
checker     : [ 4.333, -1.872, -1.072,  1.934,  0.099,  2.651]
uniform     : [-0.069, -1.001, -0.650,  0.334, -2.366, -0.428]

Cosine similarities between patterns:
            h_gradient  v_gradient     checker     uniform
h_gradient       1.000       0.445       0.364       0.058
v_gradient       0.445       1.000       0.385       0.841
checker          0.364       0.385       1.000       0.098

Linear Probe Test: 100% accuracy on downstream classification!
```

## Code Structure

| Function | Purpose |
|----------|---------|
| `context_encode()` | Encode patch with trainable encoder |
| `target_encode()` | Encode patch with EMA encoder |
| `predict()` | Predict target embedding from context |
| `jepa_forward()` | Full forward pass with masking |
| `ema_update()` | Update target encoder with EMA |

## Differences from I-JEPA (Meta's Full Version)

| Aspect | Micro JEPA | I-JEPA |
|--------|------------|--------|
| Encoder | 2-layer MLP | Vision Transformer |
| Image size | 4×4 | 224×224 |
| Patches | 4 (2×2) | 196 (14×14) |
| Embedding dim | 4 | 768+ |
| Masking | Random | Multi-block |

## References

- [I-JEPA Paper (Meta AI, 2023)](https://arxiv.org/abs/2301.08243)
- [Yann LeCun's JEPA Talk](https://www.youtube.com/watch?v=DokLw1tILlw)
- [Self-Supervised Learning Overview](https://ai.meta.com/blog/self-supervised-learning/)

## ~50 Lines Core Logic

The essential JEPA forward pass:

```python
def jepa_forward(img, context_mask, target_mask):
    patches = extract_patches(img)
    
    # Encode context with trainable encoder
    context_embs = [context_encode(patches[i]) for i in context_mask]
    avg_context = average(context_embs)
    
    # Encode targets with EMA encoder (no gradients)
    target_embs = [target_encode(patches[i]) for i in target_mask]
    
    # Predict target embeddings
    predictions = [predict(avg_context, pos) for pos in target_mask]
    
    # Loss is MSE in embedding space
    loss = mse(predictions, target_embs)
    return loss
```

