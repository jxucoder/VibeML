"""
VibeML - A tiny Gradient Boosted Decision Tree in Python
Just the algorithm, nothing else.
"""

# Regression data: predict y from x
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
Y = [1.2, 1.8, 3.1, 4.0, 5.2, 5.8, 7.1, 8.0]

def mse(Y): 
    """Mean squared error"""
    if not Y: return 0
    m = sum(Y) / len(Y)
    return sum((y - m)**2 for y in Y) / len(Y)

def split(X, Y, feat, thresh):
    """Split data on feature < threshold"""
    lX, lY, rX, rY = [], [], [], []
    for x, y in zip(X, Y):
        if x[feat] < thresh:
            lX.append(x); lY.append(y)
        else:
            rX.append(x); rY.append(y)
    return lX, lY, rX, rY

def best_split(X, Y):
    """Find best split to minimize MSE"""
    best_gain, best_feat, best_thresh = -1, None, None
    parent_mse = mse(Y)
    n = len(Y)
    
    for feat in range(len(X[0])):
        for thresh in sorted(set(x[feat] for x in X)):
            _, lY, _, rY = split(X, Y, feat, thresh)
            if not lY or not rY: continue
            gain = parent_mse - (len(lY)/n * mse(lY) + len(rY)/n * mse(rY))
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat, thresh
    return best_feat, best_thresh

def build_tree(X, Y, depth=0, max_depth=2):
    """Build regression tree (returns mean at leaves)"""
    if len(Y) <= 1 or depth >= max_depth:
        return sum(Y) / len(Y) if Y else 0
    
    feat, thresh = best_split(X, Y)
    if feat is None:
        return sum(Y) / len(Y)
    
    lX, lY, rX, rY = split(X, Y, feat, thresh)
    return {
        'feat': feat, 'thresh': thresh,
        'left': build_tree(lX, lY, depth+1, max_depth),
        'right': build_tree(rX, rY, depth+1, max_depth)
    }

def predict_tree(tree, x):
    """Predict with single tree"""
    if not isinstance(tree, dict):
        return tree
    if x[tree['feat']] < tree['thresh']:
        return predict_tree(tree['left'], x)
    return predict_tree(tree['right'], x)

# GBDT: Gradient Boosting
n_trees = 5
lr = 0.5
trees = []

# Initial prediction: mean
pred = [sum(Y) / len(Y)] * len(Y)

print("Gradient Boosting Decision Tree")
print(f"Trees: {n_trees}, Learning rate: {lr}")
print()

for i in range(n_trees):
    # Compute residuals (negative gradient of MSE)
    residuals = [y - p for y, p in zip(Y, pred)]
    
    # Fit tree to residuals
    tree = build_tree(X, residuals, max_depth=2)
    trees.append(tree)
    
    # Update predictions
    pred = [p + lr * predict_tree(tree, x) for p, x in zip(pred, X)]
    
    # Print loss
    loss = sum((y - p)**2 for y, p in zip(Y, pred)) / len(Y)
    print(f"Tree {i+1}: MSE = {loss:.4f}")

print()
print("Predictions:")
for x, y in zip(X, Y):
    p = sum(Y) / len(Y)  # Start with mean
    for tree in trees:
        p += lr * predict_tree(tree, x)
    print(f"x={x[0]} -> {p:.2f} (actual {y})")

