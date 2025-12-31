"""
VibeML - A tiny Factorization Machine in Python
Learns feature interactions for sparse data. Just the math, nothing else.
"""
import math
import random

random.seed(42)

# Hyperparams
k = 4          # latent factor dimension
lr = 0.01      # learning rate
reg = 0.01     # regularization
epochs = 1000

# Demo: Movie rating prediction with sparse features
# Features: [user_0, user_1, user_2, movie_0, movie_1, movie_2, genre_action, genre_comedy]
# Each sample is one-hot encoded user + movie + genre
n_features = 8

# Training data: (sparse_features, rating)
# Format: list of (feature_indices, target)
# user 0 rates movie 0 (action): 5.0
# user 0 rates movie 1 (comedy): 3.0
# user 1 rates movie 0 (action): 4.0
# user 1 rates movie 2 (comedy): 5.0
# user 2 rates movie 1 (comedy): 4.0
# user 2 rates movie 0 (action): 2.0
data = [
    ([0, 3, 6], 5.0),  # user0, movie0, action -> 5
    ([0, 4, 7], 3.0),  # user0, movie1, comedy -> 3
    ([1, 3, 6], 4.0),  # user1, movie0, action -> 4
    ([1, 5, 7], 5.0),  # user1, movie2, comedy -> 5
    ([2, 4, 7], 4.0),  # user2, movie1, comedy -> 4
    ([2, 3, 6], 2.0),  # user2, movie0, action -> 2
]

# FM Parameters
w0 = 0.0                                                      # global bias
w = [0.0] * n_features                                        # linear weights
V = [[random.gauss(0, 0.1) for _ in range(k)] for _ in range(n_features)]  # latent factors


def predict(features):
    """
    FM prediction: y = w0 + Σ wi*xi + Σ Σ <vi, vj> * xi * xj
    
    For sparse binary features (xi ∈ {0,1}), this simplifies to:
    y = w0 + Σ(i∈features) wi + pairwise_interactions
    
    Pairwise term computed efficiently in O(k*n) using:
    0.5 * Σf [(Σi vi,f * xi)² - Σi (vi,f * xi)²]
    """
    # Linear part: w0 + sum of weights for active features
    linear = w0 + sum(w[i] for i in features)
    
    # Pairwise interactions via latent factors
    # For each latent dimension f: (sum of v_if)^2 - sum of v_if^2
    pairwise = 0.0
    for f in range(k):
        sum_vf = sum(V[i][f] for i in features)
        sum_vf_sq = sum(V[i][f] ** 2 for i in features)
        pairwise += sum_vf ** 2 - sum_vf_sq
    pairwise *= 0.5
    
    return linear + pairwise


def train_step(features, target):
    """SGD update for one sample."""
    global w0
    
    pred = predict(features)
    error = pred - target
    
    # Precompute sum of V for each latent factor (needed for gradient)
    sum_v = [sum(V[i][f] for i in features) for f in range(k)]
    
    # Update w0
    w0 -= lr * (error + reg * w0)
    
    # Update w and V for active features only (sparse update!)
    for i in features:
        # Linear weight gradient
        w[i] -= lr * (error + reg * w[i])
        
        # Latent factor gradient: error * (sum_v[f] - V[i][f])
        for f in range(k):
            grad = error * (sum_v[f] - V[i][f])
            V[i][f] -= lr * (grad + reg * V[i][f])
    
    return error ** 2


def compute_loss(dataset):
    """Mean squared error."""
    return sum((predict(feat) - tgt) ** 2 for feat, tgt in dataset) / len(dataset)


# Training
print("Training Factorization Machine on movie ratings...")
print(f"Features: 3 users, 3 movies, 2 genres (one-hot) -> {n_features} features")
print(f"Latent factors: k={k}\n")

for epoch in range(epochs):
    random.shuffle(data)
    total_loss = 0
    for features, target in data:
        total_loss += train_step(features, target)
    
    if epoch % 200 == 0:
        mse = compute_loss(data)
        print(f"Epoch {epoch:4d}, MSE: {mse:.4f}")

print(f"Epoch {epochs:4d}, MSE: {compute_loss(data):.4f}")

# Test predictions
print("\n--- Predictions ---")
feature_names = ["user0", "user1", "user2", "movie0", "movie1", "movie2", "action", "comedy"]

for features, target in data:
    pred = predict(features)
    names = [feature_names[i] for i in features]
    print(f"{names} -> pred: {pred:.2f}, actual: {target:.1f}")

# Predict unseen combination: user2 + movie2 + comedy (not in training)
print("\n--- Unseen Combination ---")
unseen = [2, 5, 7]  # user2, movie2, comedy
names = [feature_names[i] for i in unseen]
pred = predict(unseen)
print(f"{names} -> pred: {pred:.2f} (never seen in training!)")

# Show learned latent factors
print("\n--- Learned Latent Factors (V) ---")
print("(Similar items should have similar vectors)\n")
for i, name in enumerate(feature_names):
    vec = [f"{V[i][f]:+.2f}" for f in range(k)]
    print(f"{name:8s}: [{', '.join(vec)}]")

# Compute similarity between items via latent factors
print("\n--- Feature Similarities (dot product of latent vectors) ---")

def dot(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

# Movie similarities
print("Movies:")
for i in [3, 4, 5]:
    for j in [3, 4, 5]:
        if i < j:
            sim = dot(V[i], V[j])
            print(f"  {feature_names[i]} <-> {feature_names[j]}: {sim:+.3f}")

print("\nGenres:")
sim = dot(V[6], V[7])
print(f"  action <-> comedy: {sim:+.3f}")

