"""
VibeML - A tiny decision tree in Python
Just the algorithm, nothing else.
"""

# Training data: [feature1, feature2] -> label
# Simple 2D dataset that's not linearly separable
X = [[2,3], [1,1], [2,1], [3,2], [4,4], [5,3], [4,2], [5,5]]
Y = [0, 0, 0, 0, 1, 1, 1, 1]

def gini(labels):
    """Gini impurity: 1 - sum(p^2)"""
    n = len(labels)
    if n == 0: return 0
    counts = {}
    for l in labels: counts[l] = counts.get(l, 0) + 1
    return 1 - sum((c/n)**2 for c in counts.values())

def split(X, Y, feat, thresh):
    """Split data on feature < threshold"""
    left_X, left_Y, right_X, right_Y = [], [], [], []
    for x, y in zip(X, Y):
        if x[feat] < thresh:
            left_X.append(x); left_Y.append(y)
        else:
            right_X.append(x); right_Y.append(y)
    return left_X, left_Y, right_X, right_Y

def best_split(X, Y):
    """Find best feature and threshold to split on"""
    best_gain, best_feat, best_thresh = -1, None, None
    parent_gini = gini(Y)
    n = len(Y)
    
    for feat in range(len(X[0])):
        thresholds = sorted(set(x[feat] for x in X))
        for thresh in thresholds:
            _, left_Y, _, right_Y = split(X, Y, feat, thresh)
            if not left_Y or not right_Y: continue
            
            # Information gain
            gain = parent_gini - (len(left_Y)/n * gini(left_Y) + len(right_Y)/n * gini(right_Y))
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat, thresh
    
    return best_feat, best_thresh

def build_tree(X, Y, depth=0, max_depth=3):
    """Recursively build tree"""
    # Stop: pure node or max depth
    if len(set(Y)) == 1 or depth >= max_depth:
        return max(set(Y), key=Y.count)  # Return majority class
    
    feat, thresh = best_split(X, Y)
    if feat is None:
        return max(set(Y), key=Y.count)
    
    left_X, left_Y, right_X, right_Y = split(X, Y, feat, thresh)
    
    return {
        'feat': feat,
        'thresh': thresh,
        'left': build_tree(left_X, left_Y, depth+1, max_depth),
        'right': build_tree(right_X, right_Y, depth+1, max_depth)
    }

def predict(tree, x):
    """Traverse tree to predict"""
    if not isinstance(tree, dict):
        return tree
    if x[tree['feat']] < tree['thresh']:
        return predict(tree['left'], x)
    return predict(tree['right'], x)

# Build and test
tree = build_tree(X, Y)
print("Vanilla Decision Tree")
print(f"Tree: {tree}")
print()
for x, y in zip(X, Y):
    pred = predict(tree, x)
    print(f"{x} -> {pred} (expected {y})")

