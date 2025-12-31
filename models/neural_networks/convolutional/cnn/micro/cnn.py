"""
VibeML - A tiny CNN in Python
Learns to classify simple patterns. Just the math, nothing else.
"""
import math
import random

random.seed(42)

# Generate toy data: vertical lines (class 0) vs horizontal lines (class 1)
def make_data(n=20):
    data = []
    for _ in range(n):
        img = [[0.0]*5 for _ in range(5)]
        if random.random() < 0.5:
            # Vertical line
            col = random.randint(0, 4)
            for row in range(5): img[row][col] = 1.0
            data.append((img, 0))
        else:
            # Horizontal line
            row = random.randint(0, 4)
            for col in range(5): img[row][col] = 1.0
            data.append((img, 1))
    return data

# Hyperparams
lr = 0.01
num_filters = 2
kernel_size = 3

# Initialize weights
# Conv filters: num_filters x kernel_size x kernel_size
filters = [[[random.gauss(0, 0.5) for _ in range(kernel_size)] 
            for _ in range(kernel_size)] for _ in range(num_filters)]
filter_bias = [0.0] * num_filters

# After conv (3x3 output) -> pool (1x1 output per filter) -> dense
# Dense: num_filters -> 2 classes
W_dense = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(num_filters)]
b_dense = [0.0, 0.0]

def conv2d(img, filt, bias):
    """Convolve 5x5 image with 3x3 filter -> 3x3 output"""
    out_size = len(img) - kernel_size + 1  # 3
    out = [[0.0]*out_size for _ in range(out_size)]
    for i in range(out_size):
        for j in range(out_size):
            val = bias
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    val += img[i+ki][j+kj] * filt[ki][kj]
            out[i][j] = val
    return out

def relu(x):
    """ReLU activation on 2D array"""
    return [[max(0, val) for val in row] for row in x]

def relu_deriv(x):
    """ReLU derivative"""
    return [[1.0 if val > 0 else 0.0 for val in row] for row in x]

def max_pool(x):
    """Global max pooling -> single value"""
    return max(max(row) for row in x)

def max_pool_mask(x, max_val):
    """Mask showing where max value was"""
    mask = [[0.0]*len(x[0]) for _ in range(len(x))]
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j] == max_val:
                mask[i][j] = 1.0
                return mask  # Only one max
    return mask

def softmax(x):
    exp_x = [math.exp(xi - max(x)) for xi in x]
    s = sum(exp_x)
    return [e / s for e in exp_x]

def forward(img):
    """Forward pass: conv -> relu -> pool -> dense -> softmax"""
    # Convolution + ReLU for each filter
    conv_outs = []
    relu_outs = []
    for f in range(num_filters):
        conv = conv2d(img, filters[f], filter_bias[f])
        conv_outs.append(conv)
        relu_outs.append(relu(conv))
    
    # Global max pooling
    pool_outs = [max_pool(relu_outs[f]) for f in range(num_filters)]
    
    # Dense layer
    logits = [0.0, 0.0]
    for j in range(2):
        for i in range(num_filters):
            logits[j] += pool_outs[i] * W_dense[i][j]
        logits[j] += b_dense[j]
    
    probs = softmax(logits)
    return conv_outs, relu_outs, pool_outs, probs

def backward(img, label, conv_outs, relu_outs, pool_outs, probs):
    """Backprop through entire network"""
    # Output gradient (cross-entropy + softmax)
    d_logits = probs[:]
    d_logits[label] -= 1  # [p0-y0, p1-y1]
    
    # Dense gradients
    d_W_dense = [[0.0]*2 for _ in range(num_filters)]
    d_b_dense = d_logits[:]
    d_pool = [0.0] * num_filters
    
    for i in range(num_filters):
        for j in range(2):
            d_W_dense[i][j] = pool_outs[i] * d_logits[j]
            d_pool[i] += W_dense[i][j] * d_logits[j]
    
    # Backprop through pooling and conv
    d_filters = [[[0.0]*kernel_size for _ in range(kernel_size)] for _ in range(num_filters)]
    d_filter_bias = [0.0] * num_filters
    
    for f in range(num_filters):
        # Pool gradient -> distribute to max location
        max_val = pool_outs[f]
        d_relu = [[0.0]*3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                if relu_outs[f][i][j] == max_val:
                    d_relu[i][j] = d_pool[f]
                    break
            else:
                continue
            break
        
        # ReLU gradient
        relu_mask = relu_deriv(conv_outs[f])
        d_conv = [[d_relu[i][j] * relu_mask[i][j] for j in range(3)] for i in range(3)]
        
        # Conv gradients
        d_filter_bias[f] = sum(sum(row) for row in d_conv)
        for ki in range(kernel_size):
            for kj in range(kernel_size):
                for i in range(3):
                    for j in range(3):
                        d_filters[f][ki][kj] += img[i+ki][j+kj] * d_conv[i][j]
    
    return d_filters, d_filter_bias, d_W_dense, d_b_dense

# Training
train_data = make_data(40)
test_data = make_data(10)

print("Training CNN on vertical vs horizontal lines...")
print("=" * 50)

for epoch in range(200):
    total_loss = 0
    correct = 0
    
    for img, label in train_data:
        conv_outs, relu_outs, pool_outs, probs = forward(img)
        
        # Cross-entropy loss
        total_loss += -math.log(probs[label] + 1e-8)
        if probs.index(max(probs)) == label:
            correct += 1
        
        # Backward
        d_filters, d_filter_bias, d_W_dense, d_b_dense = backward(
            img, label, conv_outs, relu_outs, pool_outs, probs)
        
        # Update weights
        for f in range(num_filters):
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    filters[f][ki][kj] -= lr * d_filters[f][ki][kj]
            filter_bias[f] -= lr * d_filter_bias[f]
        
        for i in range(num_filters):
            for j in range(2):
                W_dense[i][j] -= lr * d_W_dense[i][j]
        for j in range(2):
            b_dense[j] -= lr * d_b_dense[j]
    
    if epoch % 40 == 0:
        print(f"Epoch {epoch:3d} | Loss: {total_loss/len(train_data):.4f} | Acc: {correct}/{len(train_data)}")

# Test
print("=" * 50)
print("Testing...")
correct = 0
for img, label in test_data:
    _, _, _, probs = forward(img)
    pred = probs.index(max(probs))
    correct += (pred == label)

print(f"Test Accuracy: {correct}/{len(test_data)} ({100*correct/len(test_data):.0f}%)")

# Visualize learned filters
print("\n" + "=" * 50)
print("Learned filters (detecting edges):")
for f in range(num_filters):
    print(f"\nFilter {f}:")
    for row in filters[f]:
        print("  " + " ".join(f"{v:+.2f}" for v in row))

# Demo prediction
print("\n" + "=" * 50)
print("Demo: Classifying a vertical line")
demo = [[0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0]]
_, _, _, probs = forward(demo)
print("Input:")
for row in demo:
    print("  " + "".join("█" if v else "·" for v in row))
print(f"Prediction: {'Vertical' if probs[0] > probs[1] else 'Horizontal'} (conf: {max(probs)*100:.1f}%)")

