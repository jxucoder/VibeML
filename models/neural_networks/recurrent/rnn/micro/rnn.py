"""
VibeML - A tiny RNN in Python
Character-level language model. Just the math, nothing else.
"""
import math
import random

random.seed(42)

# Data: learn to predict next char in "hello"
data = "hello"
chars = list(set(data))
vocab_size = len(chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# Hyperparams
hidden_size = 10
lr = 0.1

# Weights: Wxh (input->hidden), Whh (hidden->hidden), Why (hidden->output)
Wxh = [[random.gauss(0, 0.1) for _ in range(hidden_size)] for _ in range(vocab_size)]
Whh = [[random.gauss(0, 0.1) for _ in range(hidden_size)] for _ in range(hidden_size)]
Why = [[random.gauss(0, 0.1) for _ in range(vocab_size)] for _ in range(hidden_size)]
bh = [0.0] * hidden_size
by = [0.0] * vocab_size

def tanh(x): 
    return math.tanh(x)

def softmax(x):
    exp_x = [math.exp(xi - max(x)) for xi in x]
    s = sum(exp_x)
    return [e / s for e in exp_x]

def forward(inputs, h_prev):
    """Forward pass through time. Returns hidden states, outputs, probs."""
    xs, hs, ps = {}, {-1: h_prev[:]}, {}
    
    for t, ch in enumerate(inputs):
        # One-hot encode input
        xs[t] = [0] * vocab_size
        xs[t][char_to_idx[ch]] = 1
        
        # Hidden state: h = tanh(Wxh @ x + Whh @ h_prev + bh)
        hs[t] = [0.0] * hidden_size
        for j in range(hidden_size):
            for i in range(vocab_size):
                hs[t][j] += Wxh[i][j] * xs[t][i]
            for i in range(hidden_size):
                hs[t][j] += Whh[i][j] * hs[t-1][i]
            hs[t][j] = tanh(hs[t][j] + bh[j])
        
        # Output: y = Why @ h + by
        y = [0.0] * vocab_size
        for i in range(vocab_size):
            for j in range(hidden_size):
                y[i] += Why[j][i] * hs[t][j]
            y[i] += by[i]
        ps[t] = softmax(y)
    
    return xs, hs, ps

def backward(inputs, targets, xs, hs, ps):
    """Backward pass (BPTT). Returns gradients."""
    dWxh = [[0.0]*hidden_size for _ in range(vocab_size)]
    dWhh = [[0.0]*hidden_size for _ in range(hidden_size)]
    dWhy = [[0.0]*vocab_size for _ in range(hidden_size)]
    dbh, dby = [0.0]*hidden_size, [0.0]*vocab_size
    dh_next = [0.0] * hidden_size
    
    for t in reversed(range(len(inputs))):
        # Output gradient: dy = p - target (cross-entropy + softmax)
        dy = ps[t][:]
        dy[char_to_idx[targets[t]]] -= 1
        
        # Why gradient
        for j in range(hidden_size):
            for i in range(vocab_size):
                dWhy[j][i] += hs[t][j] * dy[i]
        for i in range(vocab_size):
            dby[i] += dy[i]
        
        # Hidden gradient: dh = Why.T @ dy + dh_next
        dh = [0.0] * hidden_size
        for j in range(hidden_size):
            for i in range(vocab_size):
                dh[j] += Why[j][i] * dy[i]
            dh[j] += dh_next[j]
        
        # Backprop through tanh: dh_raw = dh * (1 - h^2)
        dh_raw = [dh[j] * (1 - hs[t][j]**2) for j in range(hidden_size)]
        
        # Wxh, Whh, bh gradients
        for j in range(hidden_size):
            for i in range(vocab_size):
                dWxh[i][j] += xs[t][i] * dh_raw[j]
            for i in range(hidden_size):
                dWhh[i][j] += hs[t-1][i] * dh_raw[j]
            dbh[j] += dh_raw[j]
        
        # Pass gradient to previous timestep
        dh_next = [0.0] * hidden_size
        for j in range(hidden_size):
            for i in range(hidden_size):
                dh_next[i] += Whh[i][j] * dh_raw[j]
    
    return dWxh, dWhh, dWhy, dbh, dby

# Training
inputs = list(data[:-1])   # "hell"
targets = list(data[1:])   # "ello"
h = [0.0] * hidden_size

print("Training RNN on 'hello'...")
for epoch in range(1000):
    xs, hs, ps = forward(inputs, h)
    
    # Loss: cross-entropy
    loss = sum(-math.log(ps[t][char_to_idx[targets[t]]] + 1e-8) for t in range(len(inputs)))
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    # Backward
    dWxh, dWhh, dWhy, dbh, dby = backward(inputs, targets, xs, hs, ps)
    
    # Gradient clipping & update
    for param, dparam in [(Wxh, dWxh), (Whh, dWhh), (Why, dWhy)]:
        for i in range(len(param)):
            for j in range(len(param[0])):
                dparam[i][j] = max(-5, min(5, dparam[i][j]))  # clip
                param[i][j] -= lr * dparam[i][j]
    for i in range(len(bh)):
        dbh[i] = max(-5, min(5, dbh[i]))
        bh[i] -= lr * dbh[i]
    for i in range(len(by)):
        dby[i] = max(-5, min(5, dby[i]))
        by[i] -= lr * dby[i]
    
    h = hs[len(inputs)-1]  # carry hidden state

# Generate text starting from 'h'
print("\nGenerating from 'h':")
h = [0.0] * hidden_size
ch = 'h'
output = ch
for _ in range(10):
    x = [0] * vocab_size
    x[char_to_idx[ch]] = 1
    
    # Forward one step
    h_new = [0.0] * hidden_size
    for j in range(hidden_size):
        for i in range(vocab_size):
            h_new[j] += Wxh[i][j] * x[i]
        for i in range(hidden_size):
            h_new[j] += Whh[i][j] * h[i]
        h_new[j] = tanh(h_new[j] + bh[j])
    h = h_new
    
    y = [0.0] * vocab_size
    for i in range(vocab_size):
        for j in range(hidden_size):
            y[i] += Why[j][i] * h[j]
        y[i] += by[i]
    p = softmax(y)
    
    # Sample from distribution
    r = random.random()
    cumsum = 0
    for i, prob in enumerate(p):
        cumsum += prob
        if r < cumsum:
            ch = idx_to_char[i]
            break
    output += ch

print(f"Generated: {output}")

