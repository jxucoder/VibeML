"""
VibeML - A tiny neural network in Python
Inspired by Karpathy's micrograd: just the math, nothing else.
"""
import math
import random

# Network: 2 inputs -> 4 hidden -> 1 output
# Weights initialized randomly
w1 = [[random.uniform(-1,1) for _ in range(4)] for _ in range(2)]  # 2x4
w2 = [random.uniform(-1,1) for _ in range(4)]                       # 4
b1 = [random.uniform(-1,1) for _ in range(4)]                       # 4
b2 = random.uniform(-1,1)                                           # 1

# Training data: XOR
X = [[0,0], [0,1], [1,0], [1,1]]
Y = [0, 1, 1, 0]

def sigmoid(x): 
    return 1 / (1 + math.exp(-x))

# Train
for epoch in range(10000):
    for x, y in zip(X, Y):
        # Forward
        h = [sigmoid(sum(x[i]*w1[i][j] for i in range(2)) + b1[j]) for j in range(4)]
        o = sigmoid(sum(h[j]*w2[j] for j in range(4)) + b2)
        
        # Backward
        do = (o - y) * o * (1 - o)
        dh = [do * w2[j] * h[j] * (1 - h[j]) for j in range(4)]
        
        # Update
        for j in range(4):
            w2[j] -= 0.5 * do * h[j]
            b1[j] -= 0.5 * dh[j]
            for i in range(2):
                w1[i][j] -= 0.5 * dh[j] * x[i]
        b2 -= 0.5 * do

# Test
print("XOR Neural Network")
for x, y in zip(X, Y):
    h = [sigmoid(sum(x[i]*w1[i][j] for i in range(2)) + b1[j]) for j in range(4)]
    o = sigmoid(sum(h[j]*w2[j] for j in range(4)) + b2)
    print(f"{x[0]} XOR {x[1]} = {o:.4f} (expected {y})")

