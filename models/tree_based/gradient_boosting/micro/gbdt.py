"""
microgbm - A tiny histogram-based Gradient Boosting Machine from scratch
Pure Python, no dependencies. Educational implementation of GBDT with histogram binning.

Key concepts implemented:
- Histogram binning for fast split finding
- Second-order gradients (gradient + hessian)
- Optimal leaf weights: -G / (H + lambda)
- Split gain: G²/(H + lambda)
- L2 regularization
"""

import math


# ============================================================================
# HISTOGRAM BINNING
# ============================================================================

def build_histograms(X, num_bins=256):
    """Build histogram bin edges for each feature."""
    num_features = len(X[0])
    bin_edges = []
    
    for feature in range(num_features):
        values = sorted(set(x[feature] for x in X))
        if len(values) <= num_bins:
            edges = values
        else:
            step = len(values) // num_bins
            edges = [values[i * step] for i in range(num_bins)]
            edges.append(values[-1])
        bin_edges.append(edges)
    
    return bin_edges


def bin_data(X, bin_edges):
    """Convert continuous features to bin indices."""
    binned = []
    for x in X:
        binned_row = []
        for feature, val in enumerate(x):
            edges = bin_edges[feature]
            bin_idx = 0
            for i, edge in enumerate(edges):
                if val >= edge:
                    bin_idx = i
            binned_row.append(bin_idx)
        binned.append(binned_row)
    return binned


# ============================================================================
# GRADIENT & HESSIAN (for binary classification with log loss)
# ============================================================================

def sigmoid(x):
    """Sigmoid function with overflow protection."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def compute_gradients_hessians(y_true, y_pred):
    """
    Compute gradients and hessians for log loss (binary classification).
    
    Gradient: dL/d(y_pred) = p - y
    Hessian:  d²L/d(y_pred)² = p * (1 - p)
    """
    gradients = []
    hessians = []
    
    for y, pred in zip(y_true, y_pred):
        p = sigmoid(pred)
        g = p - y
        h = p * (1.0 - p) + 1e-8
        gradients.append(g)
        hessians.append(h)
    
    return gradients, hessians


# ============================================================================
# GBDT TREE
# ============================================================================

class GBDTTree:
    """A single regression tree for gradient boosting."""
    
    def __init__(self, value=None, feature=None, threshold=None, left=None, right=None):
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
    
    def is_leaf(self):
        return self.value is not None
    
    def predict(self, x_binned):
        """Predict using binned features."""
        if self.is_leaf():
            return self.value
        if x_binned[self.feature] <= self.threshold:
            return self.left.predict(x_binned)
        return self.right.predict(x_binned)
    
    @staticmethod
    def build(X_binned, gradients, hessians, bin_edges, 
              depth=0, max_depth=3, reg_lambda=1.0, min_child_weight=1.0, gamma=0.0):
        """Build tree recursively using histogram-based split finding."""
        n = len(gradients)
        G = sum(gradients)
        H = sum(hessians)
        
        leaf_weight = -G / (H + reg_lambda)
        
        if depth >= max_depth or n <= 1 or H < min_child_weight:
            return GBDTTree(value=leaf_weight)
        
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        current_score = (G * G) / (H + reg_lambda)
        num_features = len(X_binned[0])
        
        for feature in range(num_features):
            num_bins = len(bin_edges[feature])
            grad_hist = [0.0] * num_bins
            hess_hist = [0.0] * num_bins
            
            for i in range(n):
                bin_idx = X_binned[i][feature]
                grad_hist[bin_idx] += gradients[i]
                hess_hist[bin_idx] += hessians[i]
            
            G_left, H_left = 0.0, 0.0
            
            for bin_idx in range(num_bins - 1):
                G_left += grad_hist[bin_idx]
                H_left += hess_hist[bin_idx]
                G_right = G - G_left
                H_right = H - H_left
                
                if H_left < min_child_weight or H_right < min_child_weight:
                    continue
                
                left_score = (G_left * G_left) / (H_left + reg_lambda)
                right_score = (G_right * G_right) / (H_right + reg_lambda)
                gain = 0.5 * (left_score + right_score - current_score) - gamma
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = bin_idx
        
        if best_feature is None:
            return GBDTTree(value=leaf_weight)
        
        left_idx = [i for i in range(n) if X_binned[i][best_feature] <= best_threshold]
        right_idx = [i for i in range(n) if X_binned[i][best_feature] > best_threshold]
        
        left_X = [X_binned[i] for i in left_idx]
        left_g = [gradients[i] for i in left_idx]
        left_h = [hessians[i] for i in left_idx]
        
        right_X = [X_binned[i] for i in right_idx]
        right_g = [gradients[i] for i in right_idx]
        right_h = [hessians[i] for i in right_idx]
        
        return GBDTTree(
            feature=best_feature,
            threshold=best_threshold,
            left=GBDTTree.build(left_X, left_g, left_h, bin_edges, 
                               depth + 1, max_depth, reg_lambda, min_child_weight, gamma),
            right=GBDTTree.build(right_X, right_g, right_h, bin_edges,
                                depth + 1, max_depth, reg_lambda, min_child_weight, gamma)
        )


# ============================================================================
# GBM CLASSIFIER
# ============================================================================

class GBMClassifier:
    """Gradient Boosting Machine for binary classification."""
    
    def __init__(self, n_estimators=10, max_depth=3, learning_rate=0.3,
                 reg_lambda=1.0, min_child_weight=1.0, gamma=0.0, num_bins=256):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.num_bins = num_bins
        self.trees = []
        self.bin_edges = None
        self.base_score = 0.0
    
    def fit(self, X, y, verbose=True):
        """Train the model."""
        self.bin_edges = build_histograms(X, self.num_bins)
        X_binned = bin_data(X, self.bin_edges)
        
        mean_y = sum(y) / len(y)
        self.base_score = math.log(mean_y / (1 - mean_y + 1e-8) + 1e-8)
        y_pred = [self.base_score] * len(y)
        
        if verbose:
            print("microgbm training")
            print(f"params: n_estimators={self.n_estimators}, max_depth={self.max_depth}, "
                  f"lr={self.learning_rate}, lambda={self.reg_lambda}")
            print("-" * 50)
        
        for i in range(self.n_estimators):
            gradients, hessians = compute_gradients_hessians(y, y_pred)
            
            tree = GBDTTree.build(
                X_binned, gradients, hessians, self.bin_edges,
                max_depth=self.max_depth,
                reg_lambda=self.reg_lambda,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma
            )
            self.trees.append(tree)
            
            for j in range(len(y)):
                y_pred[j] += self.learning_rate * tree.predict(X_binned[j])
            
            if verbose:
                probs = [sigmoid(p) for p in y_pred]
                log_loss = -sum(
                    y[j] * math.log(probs[j] + 1e-10) + (1 - y[j]) * math.log(1 - probs[j] + 1e-10)
                    for j in range(len(y))
                ) / len(y)
                print(f"[{i+1:3d}] log_loss: {log_loss:.6f}")
        
        if verbose:
            print("-" * 50)
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities."""
        X_binned = bin_data(X, self.bin_edges)
        probs = []
        for x in X_binned:
            raw_score = self.base_score
            for tree in self.trees:
                raw_score += self.learning_rate * tree.predict(x)
            probs.append(sigmoid(raw_score))
        return probs
    
    def predict(self, X):
        """Predict class labels."""
        probs = self.predict_proba(X)
        return [1 if p > 0.5 else 0 for p in probs]


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    X = [
        [0.5, 0.2], [1.0, 0.4], [1.5, 0.6], [2.0, 0.8],
        [2.5, 1.0], [3.0, 1.2], [3.5, 1.4], [4.0, 1.6],
        [4.5, 1.8], [5.0, 2.0], [5.5, 2.2], [6.0, 2.4]
    ]
    y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    
    print("=" * 50)
    print("microgbm - Histogram GBDT from scratch")
    print("=" * 50)
    print()
    
    model = GBMClassifier(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.3,
        reg_lambda=1.0,
        num_bins=16
    )
    model.fit(X, y)
    
    print("\nPredictions:")
    print("-" * 50)
    probs = model.predict_proba(X)
    preds = model.predict(X)
    
    for i, (x, true_y) in enumerate(zip(X, y)):
        status = "✓" if preds[i] == true_y else "✗"
        print(f"x={x} -> prob={probs[i]:.4f}, pred={preds[i]}, true={true_y} {status}")
    
    accuracy = sum(p == t for p, t in zip(preds, y)) / len(y)
    print(f"\nAccuracy: {accuracy * 100:.1f}%")
