"""
VibeML - A tiny Capsule Network in Python
Implements dynamic routing between capsules. Just the math, nothing else.

Capsule Networks encode features as vectors where:
- Vector LENGTH = probability that entity exists
- Vector ORIENTATION = entity properties (pose, deformation, etc.)
"""
import math
import random

random.seed(42)

# ============================================================================
# Data: Classify patterns - vertical lines (0), horizontal lines (1), diagonal (2)
# ============================================================================
def make_data(n=30):
    """Generate toy 6x6 images with different line patterns"""
    data = []
    for _ in range(n):
        img = [[0.0]*6 for _ in range(6)]
        pattern = random.randint(0, 2)
        
        if pattern == 0:  # Vertical line
            col = random.randint(1, 4)
            for row in range(6): 
                img[row][col] = 1.0
        elif pattern == 1:  # Horizontal line
            row = random.randint(1, 4)
            for col in range(6): 
                img[row][col] = 1.0
        else:  # Diagonal line
            offset = random.randint(0, 1)
            for i in range(6):
                if 0 <= i + offset < 6:
                    img[i][i + offset] = 1.0
        
        data.append((img, pattern))
    return data

# ============================================================================
# Hyperparameters
# ============================================================================
lr = 0.02
num_classes = 3
capsule_dim = 4          # Dimension of each capsule vector
num_primary_caps = 8     # Number of primary capsules
routing_iterations = 3   # Dynamic routing iterations

# ============================================================================
# Weight Initialization
# ============================================================================
# Conv layer: extract features (single 3x3 filter for simplicity)
conv_filter = [[random.gauss(0, 0.3) for _ in range(3)] for _ in range(3)]
conv_bias = 0.0

# Primary capsule weights: map conv features to capsule vectors
# Shape: (feature_size, num_primary_caps, capsule_dim)
feature_size = 16  # 4x4 conv output flattened
W_primary = [[[random.gauss(0, 0.3) for _ in range(capsule_dim)] 
              for _ in range(num_primary_caps)] 
             for _ in range(feature_size)]

# Routing weights: transform primary capsules to digit capsules
# Shape: (num_primary_caps, num_classes, capsule_dim, capsule_dim)
W_route = [[[[random.gauss(0, 0.3) for _ in range(capsule_dim)]
             for _ in range(capsule_dim)]
            for _ in range(num_classes)]
           for _ in range(num_primary_caps)]

# ============================================================================
# Core Functions
# ============================================================================
def conv2d(img, filt, bias):
    """Convolve 6x6 image with 3x3 filter -> 4x4 output"""
    out = [[0.0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            val = bias
            for ki in range(3):
                for kj in range(3):
                    val += img[i+ki][j+kj] * filt[ki][kj]
            out[i][j] = max(0, val)  # ReLU
    return out

def flatten(grid):
    """Flatten 2D grid to 1D list"""
    return [val for row in grid for val in row]

def squash(vec):
    """
    Squash function: non-linear activation for capsules
    Preserves direction, scales length to (0, 1)
    
    squash(s) = ||s||² / (1 + ||s||²) * s / ||s||
    """
    squared_norm = sum(v*v for v in vec)
    norm = math.sqrt(squared_norm + 1e-8)
    scale = squared_norm / (1 + squared_norm) / norm
    return [v * scale for v in vec]

def vec_length(vec):
    """Vector magnitude (L2 norm)"""
    return math.sqrt(sum(v*v for v in vec) + 1e-8)

def dot_product(v1, v2):
    """Dot product of two vectors"""
    return sum(a*b for a, b in zip(v1, v2))

def mat_vec_mul(mat, vec):
    """Matrix-vector multiplication"""
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]

def softmax(x):
    """Softmax over list of values"""
    exp_x = [math.exp(xi - max(x)) for xi in x]
    s = sum(exp_x)
    return [e / s for e in exp_x]

# ============================================================================
# Forward Pass
# ============================================================================
def forward(img):
    """
    Forward pass through capsule network:
    1. Conv layer extracts spatial features
    2. Primary capsules: reshape features into capsule vectors
    3. Dynamic routing: agreement between capsule layers
    4. Output: digit capsule vectors (length = class probability)
    """
    # Step 1: Convolution
    conv_out = conv2d(img, conv_filter, conv_bias)
    features = flatten(conv_out)
    
    # Step 2: Primary Capsules
    # Each primary capsule is a weighted sum of features
    primary_caps = []
    for cap_idx in range(num_primary_caps):
        cap_vec = [0.0] * capsule_dim
        for feat_idx in range(feature_size):
            for d in range(capsule_dim):
                cap_vec[d] += features[feat_idx] * W_primary[feat_idx][cap_idx][d]
        primary_caps.append(squash(cap_vec))
    
    # Step 3: Compute prediction vectors (u_hat)
    # Each primary capsule predicts each digit capsule
    u_hat = [[None for _ in range(num_classes)] for _ in range(num_primary_caps)]
    for i in range(num_primary_caps):
        for j in range(num_classes):
            u_hat[i][j] = mat_vec_mul(W_route[i][j], primary_caps[i])
    
    # Step 4: Dynamic Routing
    # Initialize coupling coefficients (log probabilities)
    b = [[0.0 for _ in range(num_classes)] for _ in range(num_primary_caps)]
    
    digit_caps = None
    for iteration in range(routing_iterations):
        # Coupling coefficients via softmax
        c = [softmax(b[i]) for i in range(num_primary_caps)]
        
        # Weighted sum of predictions -> digit capsule inputs
        s = [[0.0] * capsule_dim for _ in range(num_classes)]
        for j in range(num_classes):
            for i in range(num_primary_caps):
                for d in range(capsule_dim):
                    s[j][d] += c[i][j] * u_hat[i][j][d]
        
        # Squash to get digit capsule outputs
        digit_caps = [squash(s[j]) for j in range(num_classes)]
        
        # Update routing logits (agreement)
        if iteration < routing_iterations - 1:
            for i in range(num_primary_caps):
                for j in range(num_classes):
                    b[i][j] += dot_product(u_hat[i][j], digit_caps[j])
    
    # Output probabilities = capsule lengths
    probs = [vec_length(digit_caps[j]) for j in range(num_classes)]
    
    return conv_out, features, primary_caps, u_hat, digit_caps, probs

# ============================================================================
# Backward Pass (Simplified gradient computation)
# ============================================================================
def backward(img, label, conv_out, features, primary_caps, u_hat, digit_caps, probs):
    """
    Backpropagation through capsule network using margin loss
    
    Margin Loss: L_k = T_k * max(0, m+ - ||v_k||)² + λ * (1-T_k) * max(0, ||v_k|| - m-)²
    
    where T_k = 1 if class k present, m+ = 0.9, m- = 0.1, λ = 0.5
    """
    m_plus = 0.9
    m_minus = 0.1
    lambda_val = 0.5
    
    # Gradient w.r.t. digit capsule lengths
    d_lengths = [0.0] * num_classes
    for k in range(num_classes):
        length = probs[k]
        if k == label:  # T_k = 1
            if length < m_plus:
                d_lengths[k] = -2 * (m_plus - length)
        else:  # T_k = 0
            if length > m_minus:
                d_lengths[k] = 2 * lambda_val * (length - m_minus)
    
    # Gradient w.r.t. digit capsule vectors
    d_digit_caps = []
    for k in range(num_classes):
        length = probs[k]
        # d_length/d_v = v / ||v||
        d_v = [digit_caps[k][d] / length * d_lengths[k] for d in range(capsule_dim)]
        d_digit_caps.append(d_v)
    
    # Gradient through squash (approximate)
    # d_squash/d_s ≈ identity for small gradients
    d_s = d_digit_caps
    
    # Recompute coupling coefficients for gradient
    b = [[0.0 for _ in range(num_classes)] for _ in range(num_primary_caps)]
    for _ in range(routing_iterations - 1):
        c = [softmax(b[i]) for i in range(num_primary_caps)]
        s = [[0.0] * capsule_dim for _ in range(num_classes)]
        for j in range(num_classes):
            for i in range(num_primary_caps):
                for d in range(capsule_dim):
                    s[j][d] += c[i][j] * u_hat[i][j][d]
        v = [squash(s[j]) for j in range(num_classes)]
        for i in range(num_primary_caps):
            for j in range(num_classes):
                b[i][j] += dot_product(u_hat[i][j], v[j])
    c = [softmax(b[i]) for i in range(num_primary_caps)]
    
    # Gradient w.r.t. routing weights
    d_W_route = [[[[0.0 for _ in range(capsule_dim)]
                   for _ in range(capsule_dim)]
                  for _ in range(num_classes)]
                 for _ in range(num_primary_caps)]
    
    for i in range(num_primary_caps):
        for j in range(num_classes):
            for d_out in range(capsule_dim):
                for d_in in range(capsule_dim):
                    d_W_route[i][j][d_out][d_in] = (
                        c[i][j] * d_s[j][d_out] * primary_caps[i][d_in]
                    )
    
    # Gradient w.r.t. primary capsule weights (simplified)
    d_W_primary = [[[0.0 for _ in range(capsule_dim)]
                    for _ in range(num_primary_caps)]
                   for _ in range(feature_size)]
    
    # Backprop through primary capsules
    d_primary = [[0.0] * capsule_dim for _ in range(num_primary_caps)]
    for i in range(num_primary_caps):
        for j in range(num_classes):
            for d_out in range(capsule_dim):
                for d_in in range(capsule_dim):
                    d_primary[i][d_in] += c[i][j] * d_s[j][d_out] * W_route[i][j][d_out][d_in]
    
    for feat_idx in range(feature_size):
        for cap_idx in range(num_primary_caps):
            for d in range(capsule_dim):
                d_W_primary[feat_idx][cap_idx][d] = features[feat_idx] * d_primary[cap_idx][d]
    
    # Gradient w.r.t. conv filter (simplified)
    d_conv_filter = [[0.0 for _ in range(3)] for _ in range(3)]
    d_conv_bias = 0.0
    
    # Backprop through features to conv
    d_features = [0.0] * feature_size
    for feat_idx in range(feature_size):
        for cap_idx in range(num_primary_caps):
            for d in range(capsule_dim):
                d_features[feat_idx] += d_primary[cap_idx][d] * W_primary[feat_idx][cap_idx][d]
    
    # Reshape to conv output shape and backprop through ReLU
    d_conv = [[d_features[i*4 + j] if conv_out[i][j] > 0 else 0.0 
               for j in range(4)] for i in range(4)]
    
    # Conv gradient
    for ki in range(3):
        for kj in range(3):
            for i in range(4):
                for j in range(4):
                    d_conv_filter[ki][kj] += img[i+ki][j+kj] * d_conv[i][j]
    d_conv_bias = sum(sum(row) for row in d_conv)
    
    return d_W_route, d_W_primary, d_conv_filter, d_conv_bias

# ============================================================================
# Training
# ============================================================================
train_data = make_data(60)
test_data = make_data(20)

print("=" * 60)
print("Training Capsule Network")
print("Pattern Classification: Vertical | Horizontal | Diagonal")
print("=" * 60)
print(f"Architecture: Conv(3x3) -> {num_primary_caps} Primary Caps({capsule_dim}D)")
print(f"              -> Dynamic Routing ({routing_iterations} iters) -> {num_classes} Digit Caps")
print("=" * 60)

for epoch in range(150):
    total_loss = 0
    correct = 0
    
    for img, label in train_data:
        # Forward
        conv_out, features, primary_caps, u_hat, digit_caps, probs = forward(img)
        
        # Margin loss
        m_plus, m_minus, lambda_val = 0.9, 0.1, 0.5
        loss = 0
        for k in range(num_classes):
            if k == label:
                loss += max(0, m_plus - probs[k]) ** 2
            else:
                loss += lambda_val * max(0, probs[k] - m_minus) ** 2
        total_loss += loss
        
        # Accuracy
        pred = probs.index(max(probs))
        correct += (pred == label)
        
        # Backward
        d_W_route, d_W_primary, d_conv_filter, d_conv_bias = backward(
            img, label, conv_out, features, primary_caps, u_hat, digit_caps, probs)
        
        # Update routing weights
        for i in range(num_primary_caps):
            for j in range(num_classes):
                for d_out in range(capsule_dim):
                    for d_in in range(capsule_dim):
                        W_route[i][j][d_out][d_in] -= lr * d_W_route[i][j][d_out][d_in]
        
        # Update primary capsule weights
        for feat_idx in range(feature_size):
            for cap_idx in range(num_primary_caps):
                for d in range(capsule_dim):
                    W_primary[feat_idx][cap_idx][d] -= lr * d_W_primary[feat_idx][cap_idx][d]
        
        # Update conv weights
        for ki in range(3):
            for kj in range(3):
                conv_filter[ki][kj] -= lr * 0.1 * d_conv_filter[ki][kj]
        conv_bias -= lr * 0.1 * d_conv_bias
    
    if epoch % 25 == 0:
        acc = 100 * correct / len(train_data)
        print(f"Epoch {epoch:3d} | Loss: {total_loss/len(train_data):.4f} | Acc: {correct}/{len(train_data)} ({acc:.0f}%)")

# ============================================================================
# Testing
# ============================================================================
print("=" * 60)
print("Testing...")
correct = 0
confusion = [[0]*num_classes for _ in range(num_classes)]

for img, label in test_data:
    _, _, _, _, digit_caps, probs = forward(img)
    pred = probs.index(max(probs))
    correct += (pred == label)
    confusion[label][pred] += 1

print(f"Test Accuracy: {correct}/{len(test_data)} ({100*correct/len(test_data):.0f}%)")
print("\nConfusion Matrix:")
labels = ["Vert", "Horiz", "Diag"]
print("         " + "  ".join(f"{l:>5}" for l in labels))
for i, row in enumerate(confusion):
    print(f"{labels[i]:>5} -> " + "  ".join(f"{v:>5}" for v in row))

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "=" * 60)
print("Capsule Analysis: Vector properties encode MORE than just class")
print("=" * 60)

class_names = ["Vertical", "Horizontal", "Diagonal"]

for pattern_idx, pattern_name in enumerate(class_names):
    print(f"\n{pattern_name} Line:")
    
    # Create example
    img = [[0.0]*6 for _ in range(6)]
    if pattern_idx == 0:
        for row in range(6): img[row][3] = 1.0
    elif pattern_idx == 1:
        for col in range(6): img[3][col] = 1.0
    else:
        for i in range(6): img[i][i] = 1.0
    
    # Forward pass
    _, _, _, _, digit_caps, probs = forward(img)
    
    # Show input
    print("  Input:")
    for row in img:
        print("    " + "".join("█" if v else "·" for v in row))
    
    # Show capsule outputs
    print(f"  Capsule lengths (probabilities):")
    for k in range(num_classes):
        bar = "█" * int(probs[k] * 20)
        print(f"    {class_names[k]:>10}: {probs[k]:.3f} {bar}")
    
    # Show winning capsule vector (encodes pose info)
    winner = probs.index(max(probs))
    print(f"  Winning capsule vector (encodes pose):")
    print(f"    {digit_caps[winner]}")
    print(f"  Prediction: {class_names[winner]}")

# ============================================================================
# Key Insight Demo
# ============================================================================
print("\n" + "=" * 60)
print("KEY INSIGHT: Capsule orientation captures POSE variations")
print("=" * 60)

print("\nTwo vertical lines at different positions:")
positions = [1, 4]
for pos in positions:
    img = [[0.0]*6 for _ in range(6)]
    for row in range(6): 
        img[row][pos] = 1.0
    
    _, _, _, _, digit_caps, probs = forward(img)
    print(f"\n  Vertical line at column {pos}:")
    for row in img:
        print("    " + "".join("█" if v else "·" for v in row))
    print(f"  Capsule vector: [{', '.join(f'{v:.3f}' for v in digit_caps[0])}]")
    print(f"  Probability: {probs[0]:.3f}")

print("\nNotice: Same class probability, but DIFFERENT vector orientations!")
print("This is the power of capsules: they encode WHERE/HOW, not just WHAT.")
print("=" * 60)

