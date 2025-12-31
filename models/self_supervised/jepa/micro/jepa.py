"""
VibeML - A tiny JEPA (Joint Embedding Predictive Architecture) in Python
Learn representations by predicting embeddings, not pixels. Just the math.
"""
import math
import random

random.seed(42)

# =============================================================================
# JEPA Architecture Overview:
# - Context Encoder: Encodes visible patches into embeddings
# - Target Encoder:  EMA copy of context encoder (momentum updated)
# - Predictor:       Predicts target embeddings from context embeddings
# - Loss:            MSE in embedding space (not pixel space!)
# =============================================================================

# Architecture: 4x4 image -> 2x2 patches -> 6-dim embeddings
PATCH_SIZE = 2
IMG_SIZE = 4
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 4 patches
EMBED_DIM = 6
HIDDEN_DIM = 12

# =============================================================================
# Weight initialization
# =============================================================================

def init_weights(rows, cols):
    """Xavier initialization"""
    scale = math.sqrt(2.0 / (rows + cols))
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

def init_bias(size):
    return [0.0 for _ in range(size)]

def copy_weights(w):
    """Deep copy weights"""
    if isinstance(w[0], list):
        return [[v for v in row] for row in w]
    return [v for v in w]

# Context Encoder: patch (4 pixels) -> embedding (4-dim)
ctx_enc_w1 = init_weights(PATCH_SIZE * PATCH_SIZE, HIDDEN_DIM)  # 4 -> 8
ctx_enc_b1 = init_bias(HIDDEN_DIM)
ctx_enc_w2 = init_weights(HIDDEN_DIM, EMBED_DIM)  # 8 -> 4
ctx_enc_b2 = init_bias(EMBED_DIM)

# Target Encoder: EMA copy of context encoder (initialized same)
tgt_enc_w1 = copy_weights(ctx_enc_w1)
tgt_enc_b1 = copy_weights(ctx_enc_b1)
tgt_enc_w2 = copy_weights(ctx_enc_w2)
tgt_enc_b2 = copy_weights(ctx_enc_b2)

# Predictor: context embeddings + positional info -> target embedding prediction
# Input: context embedding (4) + target position encoding (4) = 8
# Output: predicted target embedding (4)
pred_w1 = init_weights(EMBED_DIM + NUM_PATCHES, HIDDEN_DIM)  # 8 -> 8
pred_b1 = init_bias(HIDDEN_DIM)
pred_w2 = init_weights(HIDDEN_DIM, EMBED_DIM)  # 8 -> 4
pred_b2 = init_bias(EMBED_DIM)

# =============================================================================
# Positional Encoding (simple one-hot for each patch position)
# =============================================================================

def get_pos_encoding(pos):
    """One-hot positional encoding for patch position"""
    enc = [0.0] * NUM_PATCHES
    enc[pos] = 1.0
    return enc

# =============================================================================
# Activation functions
# =============================================================================

def relu(x):
    return max(0.0, x)

def relu_deriv(x):
    return 1.0 if x > 0 else 0.0

# =============================================================================
# Patch extraction
# =============================================================================

def extract_patches(img):
    """Extract 2x2 patches from 4x4 image -> list of 4 patches (each 4 pixels)"""
    patches = []
    for pi in range(IMG_SIZE // PATCH_SIZE):
        for pj in range(IMG_SIZE // PATCH_SIZE):
            patch = []
            for i in range(PATCH_SIZE):
                for j in range(PATCH_SIZE):
                    patch.append(img[pi * PATCH_SIZE + i][pj * PATCH_SIZE + j])
            patches.append(patch)
    return patches

# =============================================================================
# Encoder forward pass
# =============================================================================

def encode(patch, w1, b1, w2, b2):
    """Encode a patch into an embedding"""
    # Hidden layer
    h_pre = [sum(patch[i] * w1[i][j] for i in range(len(patch))) + b1[j] 
             for j in range(HIDDEN_DIM)]
    h = [relu(h_pre[j]) for j in range(HIDDEN_DIM)]
    
    # Output embedding (linear - no normalization to allow diverse representations)
    emb = [sum(h[j] * w2[j][k] for j in range(HIDDEN_DIM)) + b2[k] 
           for k in range(EMBED_DIM)]
    
    return emb, h, h_pre

def context_encode(patch):
    """Encode with context encoder"""
    return encode(patch, ctx_enc_w1, ctx_enc_b1, ctx_enc_w2, ctx_enc_b2)

def target_encode(patch):
    """Encode with target encoder (no gradients - just for targets)"""
    return encode(patch, tgt_enc_w1, tgt_enc_b1, tgt_enc_w2, tgt_enc_b2)

# =============================================================================
# Predictor forward pass
# =============================================================================

def predict(context_emb, target_pos):
    """Predict target embedding from context embedding and target position"""
    pos_enc = get_pos_encoding(target_pos)
    inp = context_emb + pos_enc  # Concatenate: 4 + 4 = 8
    
    # Hidden layer
    h_pre = [sum(inp[i] * pred_w1[i][j] for i in range(len(inp))) + pred_b1[j]
             for j in range(HIDDEN_DIM)]
    h = [relu(h_pre[j]) for j in range(HIDDEN_DIM)]
    
    # Output prediction
    pred = [sum(h[j] * pred_w2[j][k] for j in range(HIDDEN_DIM)) + pred_b2[k]
            for k in range(EMBED_DIM)]
    
    return pred, h, h_pre, inp

# =============================================================================
# JEPA Forward pass
# =============================================================================

def jepa_forward(img, context_mask, target_mask):
    """
    Full JEPA forward pass
    context_mask: list of patch indices to use as context
    target_mask: list of patch indices to predict
    """
    patches = extract_patches(img)
    
    # Encode context patches
    context_embs = []
    context_cache = []
    for idx in context_mask:
        emb, h, h_pre = context_encode(patches[idx])
        context_embs.append(emb)
        context_cache.append((h, h_pre, patches[idx]))
    
    # Average context embeddings (simple aggregation)
    avg_context = [0.0] * EMBED_DIM
    for emb in context_embs:
        for k in range(EMBED_DIM):
            avg_context[k] += emb[k]
    avg_context = [v / len(context_embs) for v in avg_context]
    
    # Encode target patches with target encoder (no gradients)
    target_embs = []
    for idx in target_mask:
        emb, _, _ = target_encode(patches[idx])
        target_embs.append(emb)
    
    # Predict target embeddings
    predictions = []
    pred_cache = []
    for i, idx in enumerate(target_mask):
        pred, h, h_pre, inp = predict(avg_context, idx)
        predictions.append(pred)
        pred_cache.append((h, h_pre, inp))
    
    return predictions, target_embs, avg_context, context_cache, pred_cache, context_embs

# =============================================================================
# Loss function
# =============================================================================

def embedding_mse(pred, target):
    """MSE loss between predicted and target embeddings"""
    return sum((pred[k] - target[k]) ** 2 for k in range(EMBED_DIM)) / EMBED_DIM

def variance_loss(embeddings, min_std=0.1):
    """Variance regularization to prevent collapse (VICReg-style)
    Penalizes when embedding variance falls below threshold"""
    if len(embeddings) < 2:
        return 0.0
    
    loss = 0.0
    for k in range(EMBED_DIM):
        values = [emb[k] for emb in embeddings]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(var + 1e-8)
        # Hinge loss: penalize if std < min_std
        loss += max(0, min_std - std)
    return loss / EMBED_DIM

def jepa_loss(predictions, targets, context_embs, var_weight=1.0):
    """Total JEPA loss = prediction MSE + variance regularization"""
    # Prediction loss
    pred_loss = 0.0
    for pred, tgt in zip(predictions, targets):
        pred_loss += embedding_mse(pred, tgt)
    pred_loss /= len(predictions)
    
    # Variance loss on context embeddings (prevent collapse)
    var_loss = variance_loss(context_embs)
    
    return pred_loss + var_weight * var_loss, pred_loss, var_loss

# =============================================================================
# Backward pass
# =============================================================================

def backward(predictions, targets, avg_context, context_cache, pred_cache, 
             context_embs, context_mask, patches, var_weight=0.5, min_std=0.1):
    """Compute gradients for context encoder and predictor"""
    
    # Initialize gradient accumulators
    d_pred_w1 = [[0.0] * HIDDEN_DIM for _ in range(EMBED_DIM + NUM_PATCHES)]
    d_pred_b1 = [0.0] * HIDDEN_DIM
    d_pred_w2 = [[0.0] * EMBED_DIM for _ in range(HIDDEN_DIM)]
    d_pred_b2 = [0.0] * EMBED_DIM
    
    d_ctx_enc_w1 = [[0.0] * HIDDEN_DIM for _ in range(PATCH_SIZE * PATCH_SIZE)]
    d_ctx_enc_b1 = [0.0] * HIDDEN_DIM
    d_ctx_enc_w2 = [[0.0] * EMBED_DIM for _ in range(HIDDEN_DIM)]
    d_ctx_enc_b2 = [0.0] * EMBED_DIM
    
    # Gradient of context embedding (accumulated from all predictions)
    d_avg_context = [0.0] * EMBED_DIM
    
    # Backprop through each prediction
    for pred, tgt, (h, h_pre, inp) in zip(predictions, targets, pred_cache):
        # d(MSE)/d(pred)
        d_pred = [2 * (pred[k] - tgt[k]) / EMBED_DIM / len(predictions) 
                  for k in range(EMBED_DIM)]
        
        # Predictor output layer gradients
        for j in range(HIDDEN_DIM):
            for k in range(EMBED_DIM):
                d_pred_w2[j][k] += h[j] * d_pred[k]
        for k in range(EMBED_DIM):
            d_pred_b2[k] += d_pred[k]
        
        # Backprop to hidden
        d_h = [sum(pred_w2[j][k] * d_pred[k] for k in range(EMBED_DIM)) 
               for j in range(HIDDEN_DIM)]
        d_h_pre = [d_h[j] * relu_deriv(h_pre[j]) for j in range(HIDDEN_DIM)]
        
        # Predictor input layer gradients
        for i in range(len(inp)):
            for j in range(HIDDEN_DIM):
                d_pred_w1[i][j] += inp[i] * d_h_pre[j]
        for j in range(HIDDEN_DIM):
            d_pred_b1[j] += d_h_pre[j]
        
        # Gradient w.r.t. input (context embedding part only)
        d_inp = [sum(pred_w1[i][j] * d_h_pre[j] for j in range(HIDDEN_DIM))
                 for i in range(len(inp))]
        
        # Accumulate gradient for context embedding
        for k in range(EMBED_DIM):
            d_avg_context[k] += d_inp[k]
    
    # Initialize per-embedding gradients from prediction loss
    d_context_embs = [[d_avg_context[k] / len(context_embs) for k in range(EMBED_DIM)]
                      for _ in range(len(context_embs))]
    
    # Add variance loss gradients (VICReg-style)
    # d(var_loss)/d(emb) encourages spread when std < min_std
    if len(context_embs) >= 2:
        for k in range(EMBED_DIM):
            values = [emb[k] for emb in context_embs]
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(var + 1e-8)
            
            if std < min_std:
                # Gradient to increase variance: push values away from mean
                for c_idx in range(len(context_embs)):
                    # d(hinge_loss)/d(std) * d(std)/d(emb[k])
                    # = -1 * (emb[k] - mean) / (len * std)
                    d_var = -var_weight * (context_embs[c_idx][k] - mean) / (len(context_embs) * std + 1e-8)
                    d_context_embs[c_idx][k] += d_var
    
    # Backprop through each context encoder
    for c_idx, (h, h_pre, patch) in enumerate(context_cache):
        d_emb = d_context_embs[c_idx]
        
        # Output layer gradients
        for j in range(HIDDEN_DIM):
            for k in range(EMBED_DIM):
                d_ctx_enc_w2[j][k] += h[j] * d_emb[k]
        for k in range(EMBED_DIM):
            d_ctx_enc_b2[k] += d_emb[k]
        
        # Backprop to hidden
        d_h = [sum(ctx_enc_w2[j][k] * d_emb[k] for k in range(EMBED_DIM))
               for j in range(HIDDEN_DIM)]
        d_h_pre = [d_h[j] * relu_deriv(h_pre[j]) for j in range(HIDDEN_DIM)]
        
        # Input layer gradients
        for i in range(PATCH_SIZE * PATCH_SIZE):
            for j in range(HIDDEN_DIM):
                d_ctx_enc_w1[i][j] += patch[i] * d_h_pre[j]
        for j in range(HIDDEN_DIM):
            d_ctx_enc_b1[j] += d_h_pre[j]
    
    return {
        'pred_w1': d_pred_w1, 'pred_b1': d_pred_b1,
        'pred_w2': d_pred_w2, 'pred_b2': d_pred_b2,
        'ctx_enc_w1': d_ctx_enc_w1, 'ctx_enc_b1': d_ctx_enc_b1,
        'ctx_enc_w2': d_ctx_enc_w2, 'ctx_enc_b2': d_ctx_enc_b2,
    }

# =============================================================================
# Parameter updates
# =============================================================================

def update_params(grads, lr):
    """Update context encoder and predictor with gradients"""
    global ctx_enc_w1, ctx_enc_b1, ctx_enc_w2, ctx_enc_b2
    global pred_w1, pred_b1, pred_w2, pred_b2
    
    for i in range(PATCH_SIZE * PATCH_SIZE):
        for j in range(HIDDEN_DIM):
            ctx_enc_w1[i][j] -= lr * grads['ctx_enc_w1'][i][j]
    for j in range(HIDDEN_DIM):
        ctx_enc_b1[j] -= lr * grads['ctx_enc_b1'][j]
    
    for j in range(HIDDEN_DIM):
        for k in range(EMBED_DIM):
            ctx_enc_w2[j][k] -= lr * grads['ctx_enc_w2'][j][k]
    for k in range(EMBED_DIM):
        ctx_enc_b2[k] -= lr * grads['ctx_enc_b2'][k]
    
    for i in range(EMBED_DIM + NUM_PATCHES):
        for j in range(HIDDEN_DIM):
            pred_w1[i][j] -= lr * grads['pred_w1'][i][j]
    for j in range(HIDDEN_DIM):
        pred_b1[j] -= lr * grads['pred_b1'][j]
    
    for j in range(HIDDEN_DIM):
        for k in range(EMBED_DIM):
            pred_w2[j][k] -= lr * grads['pred_w2'][j][k]
    for k in range(EMBED_DIM):
        pred_b2[k] -= lr * grads['pred_b2'][k]

def ema_update(momentum=0.996):
    """Update target encoder with EMA of context encoder"""
    global tgt_enc_w1, tgt_enc_b1, tgt_enc_w2, tgt_enc_b2
    
    for i in range(PATCH_SIZE * PATCH_SIZE):
        for j in range(HIDDEN_DIM):
            tgt_enc_w1[i][j] = momentum * tgt_enc_w1[i][j] + (1 - momentum) * ctx_enc_w1[i][j]
    for j in range(HIDDEN_DIM):
        tgt_enc_b1[j] = momentum * tgt_enc_b1[j] + (1 - momentum) * ctx_enc_b1[j]
    
    for j in range(HIDDEN_DIM):
        for k in range(EMBED_DIM):
            tgt_enc_w2[j][k] = momentum * tgt_enc_w2[j][k] + (1 - momentum) * ctx_enc_w2[j][k]
    for k in range(EMBED_DIM):
        tgt_enc_b2[k] = momentum * tgt_enc_b2[k] + (1 - momentum) * ctx_enc_b2[k]

# =============================================================================
# Generate training data: simple patterns
# =============================================================================

def generate_pattern_data(n_samples):
    """Generate 4x4 images with simple patterns (gradients, edges, etc.)"""
    data = []
    for _ in range(n_samples):
        img = [[0.0] * IMG_SIZE for _ in range(IMG_SIZE)]
        pattern = random.choice(['h_gradient', 'v_gradient', 'diagonal', 'checker', 'block'])
        
        if pattern == 'h_gradient':
            for i in range(IMG_SIZE):
                for j in range(IMG_SIZE):
                    img[i][j] = j / (IMG_SIZE - 1)
        elif pattern == 'v_gradient':
            for i in range(IMG_SIZE):
                for j in range(IMG_SIZE):
                    img[i][j] = i / (IMG_SIZE - 1)
        elif pattern == 'diagonal':
            for i in range(IMG_SIZE):
                for j in range(IMG_SIZE):
                    img[i][j] = 1.0 if i == j else 0.0
        elif pattern == 'checker':
            for i in range(IMG_SIZE):
                for j in range(IMG_SIZE):
                    img[i][j] = 1.0 if (i + j) % 2 == 0 else 0.0
        elif pattern == 'block':
            bi, bj = random.randint(0, 2), random.randint(0, 2)
            for i in range(bi, min(bi + 2, IMG_SIZE)):
                for j in range(bj, min(bj + 2, IMG_SIZE)):
                    img[i][j] = 1.0
        
        # Add noise
        for i in range(IMG_SIZE):
            for j in range(IMG_SIZE):
                img[i][j] += random.gauss(0, 0.1)
                img[i][j] = max(0, min(1, img[i][j]))
        
        data.append(img)
    return data

def generate_mask():
    """Generate random context/target masks (JEPA masking strategy)"""
    all_patches = list(range(NUM_PATCHES))
    random.shuffle(all_patches)
    
    # Use ~75% as context, ~25% as targets (typical JEPA ratio)
    num_context = max(1, NUM_PATCHES - 1)  # At least 1 context, 1 target
    context_mask = sorted(all_patches[:num_context])
    target_mask = sorted(all_patches[num_context:])
    
    if len(target_mask) == 0:
        # Ensure at least one target
        target_mask = [context_mask.pop()]
    
    return context_mask, target_mask

# =============================================================================
# Training
# =============================================================================

print("=" * 60)
print("VibeML Micro JEPA - Learning Representations via Prediction")
print("=" * 60)

train_data = generate_pattern_data(200)
print(f"\nGenerated {len(train_data)} training images ({IMG_SIZE}x{IMG_SIZE})")
print(f"Architecture: {PATCH_SIZE}x{PATCH_SIZE} patches -> {EMBED_DIM}-dim embeddings")

lr = 0.02
momentum = 0.99  # Lower momentum for faster adaptation in tiny model
epochs = 500

print(f"\nTraining for {epochs} epochs...")
print(f"Learning rate: {lr}, EMA momentum: {momentum}")
print("-" * 60)

for epoch in range(epochs):
    total_loss = 0
    total_pred_loss = 0
    total_var_loss = 0
    random.shuffle(train_data)
    
    for img in train_data:
        # Generate masks for this sample
        context_mask, target_mask = generate_mask()
        
        # Forward pass
        predictions, targets, avg_ctx, ctx_cache, pred_cache, ctx_embs = \
            jepa_forward(img, context_mask, target_mask)
        
        # Compute loss (prediction + variance regularization)
        loss, pred_loss, var_loss = jepa_loss(predictions, targets, ctx_embs)
        total_loss += loss
        total_pred_loss += pred_loss
        total_var_loss += var_loss
        
        # Backward pass (includes variance regularization)
        patches = extract_patches(img)
        grads = backward(predictions, targets, avg_ctx, ctx_cache, pred_cache,
                        ctx_embs, context_mask, patches, var_weight=1.0)
        
        # Update context encoder and predictor
        update_params(grads, lr)
        
        # EMA update target encoder
        ema_update(momentum)
    
    if (epoch + 1) % 50 == 0:
        n = len(train_data)
        print(f"Epoch {epoch+1:4d} | Loss: {total_loss/n:.4f} | Pred: {total_pred_loss/n:.4f} | Var: {total_var_loss/n:.4f}")

# =============================================================================
# Test: Embedding quality
# =============================================================================

print("\n" + "=" * 60)
print("Representation Quality Test")
print("=" * 60)

# Test that similar patches get similar embeddings
test_patterns = {
    'h_gradient': [[j / 3 for j in range(4)] for i in range(4)],
    'v_gradient': [[i / 3 for j in range(4)] for i in range(4)],
    'checker': [[1.0 if (i+j) % 2 == 0 else 0.0 for j in range(4)] for i in range(4)],
    'uniform': [[0.5 for j in range(4)] for i in range(4)],
}

print("\nEmbeddings for different patterns:")
print("-" * 50)

embeddings = {}
for name, img in test_patterns.items():
    patches = extract_patches(img)
    # Encode all patches and average
    embs = [context_encode(p)[0] for p in patches]
    avg_emb = [sum(e[k] for e in embs) / len(embs) for k in range(EMBED_DIM)]
    embeddings[name] = avg_emb
    emb_str = ", ".join(f"{v:6.3f}" for v in avg_emb)
    print(f"{name:12s}: [{emb_str}]")

# Compute cosine similarities
print("\nCosine similarities between pattern embeddings:")
print("-" * 50)

def cosine_sim(a, b):
    dot = sum(a[i] * b[i] for i in range(len(a)))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    return dot / (norm_a * norm_b + 1e-8)

patterns = list(embeddings.keys())
print("          ", end="")
for p in patterns:
    print(f"{p:>12s}", end="")
print()

for p1 in patterns:
    print(f"{p1:10s}", end="")
    for p2 in patterns:
        sim = cosine_sim(embeddings[p1], embeddings[p2])
        print(f"{sim:12.3f}", end="")
    print()

# =============================================================================
# Test: Prediction quality
# =============================================================================

print("\n" + "=" * 60)
print("Prediction Quality Test")
print("=" * 60)

print("\nPredicting masked patch embeddings:")
print("-" * 50)

test_img = test_patterns['h_gradient']
patches = extract_patches(test_img)

# Test different context/target splits
test_cases = [
    ([0, 1, 2], [3], "Predict bottom-right from top+bottom-left"),
    ([0, 1, 3], [2], "Predict bottom-left from top+bottom-right"),
    ([0, 2, 3], [1], "Predict top-right from rest"),
    ([1, 2, 3], [0], "Predict top-left from rest"),
]

for ctx_mask, tgt_mask, desc in test_cases:
    predictions, targets, *_ = jepa_forward(test_img, ctx_mask, tgt_mask)
    pred = predictions[0]
    tgt = targets[0]
    
    mse = embedding_mse(pred, tgt)
    sim = cosine_sim(pred, tgt)
    
    print(f"\n{desc}:")
    pred_str = ", ".join(f"{v:6.3f}" for v in pred)
    tgt_str = ", ".join(f"{v:6.3f}" for v in tgt)
    print(f"  Predicted: [{pred_str}]")
    print(f"  Target:    [{tgt_str}]")
    print(f"  MSE: {mse:.4f}, Cosine Sim: {sim:.4f}")

# =============================================================================
# Test: Invariance check
# =============================================================================

print("\n" + "=" * 60)
print("Noise Invariance Test")
print("=" * 60)

base_img = test_patterns['checker']
base_patches = extract_patches(base_img)
base_embs = [context_encode(p)[0] for p in base_patches]
base_avg = [sum(e[k] for e in base_embs) / len(base_embs) for k in range(EMBED_DIM)]

print("\nComparing clean vs noisy versions of same pattern:")
print("-" * 50)

for noise_level in [0.05, 0.1, 0.2]:
    noisy_img = [[v + random.gauss(0, noise_level) for v in row] for row in base_img]
    noisy_patches = extract_patches(noisy_img)
    noisy_embs = [context_encode(p)[0] for p in noisy_patches]
    noisy_avg = [sum(e[k] for e in noisy_embs) / len(noisy_embs) for k in range(EMBED_DIM)]
    
    sim = cosine_sim(base_avg, noisy_avg)
    print(f"Noise σ={noise_level:.2f}: Cosine similarity = {sim:.4f}")

# =============================================================================
# Test: Linear probe classification (downstream task)
# =============================================================================

print("\n" + "=" * 60)
print("Linear Probe Test (Downstream Classification)")
print("=" * 60)

print("\nTraining linear classifier on frozen JEPA embeddings:")
print("-" * 50)

# Generate labeled data: distinguish patterns
labeled_data = []
for _ in range(50):
    # Horizontal gradient (class 0)
    img = [[j / 3 + random.gauss(0, 0.1) for j in range(4)] for i in range(4)]
    labeled_data.append((img, 0))
    
    # Vertical gradient (class 1) 
    img = [[i / 3 + random.gauss(0, 0.1) for j in range(4)] for i in range(4)]
    labeled_data.append((img, 1))
    
    # Checker pattern (class 2)
    img = [[1.0 if (i+j) % 2 == 0 else 0.0 for j in range(4)] for i in range(4)]
    img = [[v + random.gauss(0, 0.1) for v in row] for row in img]
    labeled_data.append((img, 2))

# Get JEPA embeddings for all images (frozen encoder)
def get_embedding(img):
    patches = extract_patches(img)
    embs = [context_encode(p)[0] for p in patches]
    return [sum(e[k] for e in embs) / len(embs) for k in range(EMBED_DIM)]

# Train simple linear classifier on embeddings
probe_w = [[random.gauss(0, 0.1) for _ in range(3)] for _ in range(EMBED_DIM)]
probe_b = [0.0, 0.0, 0.0]

def probe_forward(emb):
    logits = [sum(emb[i] * probe_w[i][j] for i in range(EMBED_DIM)) + probe_b[j] 
              for j in range(3)]
    exp_logits = [math.exp(l - max(logits)) for l in logits]
    s = sum(exp_logits)
    return [e / s for e in exp_logits]

# Train linear probe
probe_lr = 0.5
for ep in range(100):
    random.shuffle(labeled_data)
    correct = 0
    for img, label in labeled_data:
        emb = get_embedding(img)
        probs = probe_forward(emb)
        
        # Cross entropy gradient
        d_logits = probs[:]
        d_logits[label] -= 1.0
        
        # Update probe weights
        for i in range(EMBED_DIM):
            for j in range(3):
                probe_w[i][j] -= probe_lr * emb[i] * d_logits[j]
        for j in range(3):
            probe_b[j] -= probe_lr * d_logits[j]
        
        if probs.index(max(probs)) == label:
            correct += 1
    
    if (ep + 1) % 25 == 0:
        print(f"Probe Epoch {ep+1:3d} | Accuracy: {correct}/{len(labeled_data)} ({100*correct/len(labeled_data):.1f}%)")

# Final test
test_data = []
for _ in range(20):
    test_data.append(([[j/3 for j in range(4)] for i in range(4)], 0, "H-grad"))
    test_data.append(([[i/3 for j in range(4)] for i in range(4)], 1, "V-grad"))
    test_data.append(([[1.0 if (i+j)%2==0 else 0.0 for j in range(4)] for i in range(4)], 2, "Checker"))

correct = 0
for img, label, name in test_data:
    emb = get_embedding(img)
    probs = probe_forward(emb)
    pred = probs.index(max(probs))
    correct += (pred == label)

print(f"\nFinal Test Accuracy: {correct}/{len(test_data)} ({100*correct/len(test_data):.1f}%)")

print("\n" + "=" * 60)
print("JEPA Training Complete!")
print("=" * 60)
print("\nKey JEPA concepts demonstrated:")
print("• Prediction in embedding space (not pixel space)")
print("• Context encoder learns via gradient descent")
print("• Target encoder uses EMA (no gradients - prevents collapse)")
print("• Variance regularization helps maintain representation quality")
print("• Learned embeddings useful for downstream tasks (linear probe)")

