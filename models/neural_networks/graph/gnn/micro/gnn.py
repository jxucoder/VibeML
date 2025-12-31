"""
VibeML - A tiny Graph Neural Network in Python
Node classification via message passing. Just the math, nothing else.
"""
import math
import random

random.seed(42)

# =============================================================================
# GRAPH DATA: Zachary's Karate Club (simplified)
# A social network where nodes are members, edges are friendships.
# Task: Predict which faction each member belongs to (binary classification)
# =============================================================================

# Adjacency list: node -> list of neighbor nodes
# This is a simplified version of the famous karate club graph
edges = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3),
    (2, 3), (2, 7),
    (3, 7),
    (4, 5), (4, 6),
    (5, 6),
    (6, 7),
    (7, 8), (7, 9),
    (8, 9), (8, 10),
    (9, 10), (9, 11),
    (10, 11),
]

num_nodes = 12
# Build adjacency list (undirected graph)
neighbors = [[] for _ in range(num_nodes)]
for i, j in edges:
    neighbors[i].append(j)
    neighbors[j].append(i)

# Node features: random 4-dimensional features
input_dim = 4
node_features = [[random.gauss(0, 1) for _ in range(input_dim)] for _ in range(num_nodes)]

# Labels: two factions (0 = left group, 1 = right group)
labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# =============================================================================
# GNN ARCHITECTURE
# 2 Graph Convolutional Layers + 1 Output Layer
# Message Passing: h_v = σ(W · AGGREGATE({h_u : u ∈ N(v) ∪ {v}}))
# =============================================================================

hidden_dim = 8
output_dim = 2  # binary classification

# Layer 1: input_dim -> hidden_dim
W1 = [[random.gauss(0, 0.5) for _ in range(hidden_dim)] for _ in range(input_dim)]
b1 = [0.0] * hidden_dim

# Layer 2: hidden_dim -> hidden_dim
W2 = [[random.gauss(0, 0.5) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
b2 = [0.0] * hidden_dim

# Output layer: hidden_dim -> output_dim
W_out = [[random.gauss(0, 0.5) for _ in range(output_dim)] for _ in range(hidden_dim)]
b_out = [0.0] * output_dim

lr = 0.01


def relu(x):
    """ReLU activation"""
    return max(0, x)


def relu_deriv(x):
    """ReLU derivative"""
    return 1.0 if x > 0 else 0.0


def softmax(x):
    """Softmax for classification"""
    exp_x = [math.exp(xi - max(x)) for xi in x]
    s = sum(exp_x)
    return [e / s for e in exp_x]


def aggregate(node, features):
    """
    Aggregate features from neighbors + self (mean aggregation)
    This is the core of GNN: information flows along edges
    """
    neighbor_list = neighbors[node] + [node]  # include self
    agg = [0.0] * len(features[0])
    for neighbor in neighbor_list:
        for d in range(len(features[0])):
            agg[d] += features[neighbor][d]
    # Mean pooling
    n = len(neighbor_list)
    return [a / n for a in agg]


def graph_conv(features, W, b):
    """
    Graph Convolutional Layer:
    1. Aggregate neighbor features for each node
    2. Transform with linear layer + ReLU
    """
    new_features = []
    for node in range(num_nodes):
        # Aggregate
        agg = aggregate(node, features)
        
        # Linear transform: h = W @ agg + b
        h = [0.0] * len(W[0])
        for j in range(len(W[0])):
            for i in range(len(W)):
                h[j] += W[i][j] * agg[i]
            h[j] += b[j]
        
        # ReLU activation
        h = [relu(x) for x in h]
        new_features.append(h)
    
    return new_features


def forward(features):
    """
    Forward pass through the GNN:
    Input -> GCN Layer 1 -> GCN Layer 2 -> Output Layer -> Softmax
    """
    # Store intermediate values for backprop
    cache = {'input': features}
    
    # GCN Layer 1
    h1_pre = []  # pre-activation
    h1 = []      # post-activation
    for node in range(num_nodes):
        agg = aggregate(node, features)
        pre = [sum(W1[i][j] * agg[i] for i in range(input_dim)) + b1[j] 
               for j in range(hidden_dim)]
        h1_pre.append(pre)
        h1.append([relu(x) for x in pre])
    cache['agg1'] = [aggregate(node, features) for node in range(num_nodes)]
    cache['h1_pre'] = h1_pre
    cache['h1'] = h1
    
    # GCN Layer 2
    h2_pre = []
    h2 = []
    for node in range(num_nodes):
        agg = aggregate(node, h1)
        pre = [sum(W2[i][j] * agg[i] for i in range(hidden_dim)) + b2[j] 
               for j in range(hidden_dim)]
        h2_pre.append(pre)
        h2.append([relu(x) for x in pre])
    cache['agg2'] = [aggregate(node, h1) for node in range(num_nodes)]
    cache['h2_pre'] = h2_pre
    cache['h2'] = h2
    
    # Output layer (no aggregation, per-node classification)
    logits = []
    probs = []
    for node in range(num_nodes):
        out = [sum(W_out[i][j] * h2[node][i] for i in range(hidden_dim)) + b_out[j] 
               for j in range(output_dim)]
        logits.append(out)
        probs.append(softmax(out))
    cache['logits'] = logits
    cache['probs'] = probs
    
    return probs, cache


def backward(cache, labels):
    """
    Backward pass through the GNN
    Computes gradients for all parameters
    """
    probs = cache['probs']
    h2 = cache['h2']
    h2_pre = cache['h2_pre']
    h1 = cache['h1']
    h1_pre = cache['h1_pre']
    agg1 = cache['agg1']
    agg2 = cache['agg2']
    
    # Initialize gradients
    dW_out = [[0.0] * output_dim for _ in range(hidden_dim)]
    db_out = [0.0] * output_dim
    dW2 = [[0.0] * hidden_dim for _ in range(hidden_dim)]
    db2 = [0.0] * hidden_dim
    dW1 = [[0.0] * hidden_dim for _ in range(input_dim)]
    db1 = [0.0] * hidden_dim
    
    # Gradient at h2 level (accumulated from output layer)
    dh2 = [[0.0] * hidden_dim for _ in range(num_nodes)]
    
    # Output layer gradients
    for node in range(num_nodes):
        # Cross-entropy gradient: dL/d_logits = probs - one_hot(label)
        d_logits = probs[node][:]
        d_logits[labels[node]] -= 1
        
        # W_out gradient
        for i in range(hidden_dim):
            for j in range(output_dim):
                dW_out[i][j] += h2[node][i] * d_logits[j]
        for j in range(output_dim):
            db_out[j] += d_logits[j]
        
        # Gradient to h2
        for i in range(hidden_dim):
            for j in range(output_dim):
                dh2[node][i] += W_out[i][j] * d_logits[j]
    
    # GCN Layer 2 backward
    # Gradient needs to flow back through aggregation
    dh1 = [[0.0] * hidden_dim for _ in range(num_nodes)]
    
    for node in range(num_nodes):
        # Backprop through ReLU
        dh2_pre = [dh2[node][j] * relu_deriv(h2_pre[node][j]) for j in range(hidden_dim)]
        
        # W2, b2 gradient
        for i in range(hidden_dim):
            for j in range(hidden_dim):
                dW2[i][j] += agg2[node][i] * dh2_pre[j]
        for j in range(hidden_dim):
            db2[j] += dh2_pre[j]
        
        # Gradient to aggregated h1 -> distribute to neighbors
        d_agg = [sum(W2[i][j] * dh2_pre[j] for j in range(hidden_dim)) 
                 for i in range(hidden_dim)]
        
        # Distribute gradient to neighbors (reverse of mean aggregation)
        neighbor_list = neighbors[node] + [node]
        n = len(neighbor_list)
        for neighbor in neighbor_list:
            for d in range(hidden_dim):
                dh1[neighbor][d] += d_agg[d] / n
    
    # GCN Layer 1 backward
    for node in range(num_nodes):
        # Backprop through ReLU
        dh1_pre = [dh1[node][j] * relu_deriv(h1_pre[node][j]) for j in range(hidden_dim)]
        
        # W1, b1 gradient
        for i in range(input_dim):
            for j in range(hidden_dim):
                dW1[i][j] += agg1[node][i] * dh1_pre[j]
        for j in range(hidden_dim):
            db1[j] += dh1_pre[j]
    
    return dW1, db1, dW2, db2, dW_out, db_out


def compute_loss(probs, labels):
    """Cross-entropy loss"""
    loss = 0
    for node in range(num_nodes):
        loss -= math.log(probs[node][labels[node]] + 1e-8)
    return loss / num_nodes


def accuracy(probs, labels):
    """Classification accuracy"""
    correct = sum(1 for node in range(num_nodes) 
                  if probs[node].index(max(probs[node])) == labels[node])
    return correct / num_nodes


# =============================================================================
# TRAINING
# =============================================================================

print("=" * 50)
print("Graph Neural Network - Node Classification")
print("=" * 50)
print(f"Nodes: {num_nodes}, Edges: {len(edges)}")
print(f"Task: Classify nodes into 2 factions")
print("=" * 50)

for epoch in range(500):
    # Forward pass
    probs, cache = forward(node_features)
    
    # Compute loss
    loss = compute_loss(probs, labels)
    acc = accuracy(probs, labels)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")
    
    # Backward pass
    dW1, db1, dW2, db2, dW_out, db_out = backward(cache, labels)
    
    # Gradient descent update
    for i in range(input_dim):
        for j in range(hidden_dim):
            W1[i][j] -= lr * dW1[i][j]
    for j in range(hidden_dim):
        b1[j] -= lr * db1[j]
    
    for i in range(hidden_dim):
        for j in range(hidden_dim):
            W2[i][j] -= lr * dW2[i][j]
    for j in range(hidden_dim):
        b2[j] -= lr * db2[j]
    
    for i in range(hidden_dim):
        for j in range(output_dim):
            W_out[i][j] -= lr * dW_out[i][j]
    for j in range(output_dim):
        b_out[j] -= lr * db_out[j]

# =============================================================================
# FINAL EVALUATION
# =============================================================================

print("\n" + "=" * 50)
print("Final Predictions")
print("=" * 50)

probs, _ = forward(node_features)
print("\nNode | Predicted | Actual | Confidence")
print("-" * 42)
for node in range(num_nodes):
    pred = probs[node].index(max(probs[node]))
    conf = max(probs[node])
    status = "✓" if pred == labels[node] else "✗"
    print(f"  {node:2d} |     {pred}     |   {labels[node]}    |   {conf:.2%}   {status}")

final_acc = accuracy(probs, labels)
print("-" * 42)
print(f"Final Accuracy: {final_acc:.2%}")

# Visualize the graph structure
print("\n" + "=" * 50)
print("Graph Structure (Adjacency)")
print("=" * 50)
print("\n  Faction 0: Nodes 0-5")
print("  Faction 1: Nodes 6-11")
print("\n  Edges show connections between nodes:")
for node in range(num_nodes):
    faction = "L" if labels[node] == 0 else "R"
    neighbors_str = ", ".join(str(n) for n in neighbors[node])
    print(f"  Node {node:2d} [{faction}] -> {neighbors_str}")

