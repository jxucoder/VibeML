"""
VibeML - A tiny Conditional Random Field in Python
Linear-chain CRF for sequence labeling. Just the math, nothing else.
"""
import math
import random

random.seed(42)

# =============================================================================
# DATA: Simple POS-tagging-like task
# Words -> Tags (DET, NOUN, VERB, ADJ)
# =============================================================================

# Vocabulary and tag set
words = ["the", "a", "cat", "dog", "runs", "jumps", "big", "small", "quickly"]
tags = ["DET", "NOUN", "VERB", "ADJ"]

word_to_idx = {w: i for i, w in enumerate(words)}
tag_to_idx = {t: i for i, t in enumerate(tags)}
idx_to_tag = {i: t for t, i in tag_to_idx.items()}

num_words = len(words)
num_tags = len(tags)

# Training sequences: (words, gold_tags)
train_data = [
    (["the", "cat", "runs"], ["DET", "NOUN", "VERB"]),
    (["a", "dog", "jumps"], ["DET", "NOUN", "VERB"]),
    (["the", "big", "cat"], ["DET", "ADJ", "NOUN"]),
    (["a", "small", "dog"], ["DET", "ADJ", "NOUN"]),
    (["the", "cat", "runs", "quickly"], ["DET", "NOUN", "VERB", "ADJ"]),
    (["the", "big", "dog", "jumps"], ["DET", "ADJ", "NOUN", "VERB"]),
]

# =============================================================================
# CRF PARAMETERS
# Emission weights: W_emit[word][tag] = weight for word emitting tag
# Transition weights: W_trans[tag_prev][tag_curr] = weight for tag transition
# =============================================================================

# Initialize weights randomly
W_emit = [[random.gauss(0, 0.1) for _ in range(num_tags)] for _ in range(num_words)]
W_trans = [[random.gauss(0, 0.1) for _ in range(num_tags)] for _ in range(num_tags)]

# Start/end transition weights
W_start = [random.gauss(0, 0.1) for _ in range(num_tags)]
W_end = [random.gauss(0, 0.1) for _ in range(num_tags)]

lr = 0.1


def compute_scores(sentence):
    """
    Compute emission and transition scores for a sentence.
    
    Returns:
        emit_scores: [T][num_tags] emission score at each position
        trans_scores: W_trans (same for all positions)
    """
    T = len(sentence)
    emit_scores = []
    for t in range(T):
        word_idx = word_to_idx.get(sentence[t], 0)
        emit_scores.append(W_emit[word_idx][:])
    return emit_scores


def forward_algorithm(emit_scores):
    """
    Forward algorithm: compute log partition function Z(x).
    
    alpha[t][j] = log sum over all paths ending at tag j at position t
    
    Uses log-sum-exp trick for numerical stability:
    log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
    """
    T = len(emit_scores)
    
    # Initialize: alpha[0][j] = W_start[j] + emit[0][j]
    alpha = [[0.0] * num_tags for _ in range(T)]
    for j in range(num_tags):
        alpha[0][j] = W_start[j] + emit_scores[0][j]
    
    # Recursion: alpha[t][j] = emit[t][j] + logsumexp(alpha[t-1] + trans[:,j])
    for t in range(1, T):
        for j in range(num_tags):
            # Collect alpha[t-1][i] + W_trans[i][j] for all i
            scores = [alpha[t-1][i] + W_trans[i][j] for i in range(num_tags)]
            # Log-sum-exp
            max_score = max(scores)
            alpha[t][j] = emit_scores[t][j] + max_score + math.log(
                sum(math.exp(s - max_score) for s in scores)
            )
    
    # Termination: Z = logsumexp(alpha[T-1] + W_end)
    final_scores = [alpha[T-1][j] + W_end[j] for j in range(num_tags)]
    max_score = max(final_scores)
    log_Z = max_score + math.log(sum(math.exp(s - max_score) for s in final_scores))
    
    return alpha, log_Z


def backward_algorithm(emit_scores):
    """
    Backward algorithm: compute backward probabilities.
    
    beta[t][i] = log sum over all paths starting from tag i at position t
    """
    T = len(emit_scores)
    
    # Initialize: beta[T-1][i] = W_end[i]
    beta = [[0.0] * num_tags for _ in range(T)]
    for i in range(num_tags):
        beta[T-1][i] = W_end[i]
    
    # Recursion: beta[t][i] = logsumexp(W_trans[i,:] + emit[t+1] + beta[t+1])
    for t in range(T-2, -1, -1):
        for i in range(num_tags):
            scores = [W_trans[i][j] + emit_scores[t+1][j] + beta[t+1][j] 
                      for j in range(num_tags)]
            max_score = max(scores)
            beta[t][i] = max_score + math.log(
                sum(math.exp(s - max_score) for s in scores)
            )
    
    return beta


def compute_marginals(emit_scores, alpha, beta, log_Z):
    """
    Compute marginal probabilities using forward-backward.
    
    P(y_t = j | x) = exp(alpha[t][j] + beta[t][j] - log_Z)
    P(y_{t-1}=i, y_t=j | x) = exp(alpha[t-1][i] + trans[i][j] + emit[t][j] + beta[t][j] - log_Z)
    """
    T = len(emit_scores)
    
    # Node marginals: P(y_t = j | x)
    node_marginals = [[0.0] * num_tags for _ in range(T)]
    for t in range(T):
        for j in range(num_tags):
            node_marginals[t][j] = math.exp(alpha[t][j] + beta[t][j] - log_Z)
    
    # Edge marginals: P(y_{t-1}=i, y_t=j | x)
    edge_marginals = []
    for t in range(1, T):
        edge_t = [[0.0] * num_tags for _ in range(num_tags)]
        for i in range(num_tags):
            for j in range(num_tags):
                score = (alpha[t-1][i] + W_trans[i][j] + 
                         emit_scores[t][j] + beta[t][j] - log_Z)
                edge_t[i][j] = math.exp(score)
        edge_marginals.append(edge_t)
    
    # Start marginals: P(y_0 = j | x)
    start_marginals = [math.exp(W_start[j] + emit_scores[0][j] + beta[0][j] - log_Z) 
                       for j in range(num_tags)]
    
    # End marginals: P(y_{T-1} = j | x)
    end_marginals = [math.exp(alpha[T-1][j] + W_end[j] - log_Z) 
                     for j in range(num_tags)]
    
    return node_marginals, edge_marginals, start_marginals, end_marginals


def viterbi(emit_scores):
    """
    Viterbi algorithm: find the best tag sequence.
    
    Returns the most likely tag sequence given the emission scores.
    """
    T = len(emit_scores)
    
    # Viterbi scores and backpointers
    viterbi_scores = [[0.0] * num_tags for _ in range(T)]
    backpointers = [[0] * num_tags for _ in range(T)]
    
    # Initialize
    for j in range(num_tags):
        viterbi_scores[0][j] = W_start[j] + emit_scores[0][j]
    
    # Recursion
    for t in range(1, T):
        for j in range(num_tags):
            best_score = float('-inf')
            best_prev = 0
            for i in range(num_tags):
                score = viterbi_scores[t-1][i] + W_trans[i][j]
                if score > best_score:
                    best_score = score
                    best_prev = i
            viterbi_scores[t][j] = best_score + emit_scores[t][j]
            backpointers[t][j] = best_prev
    
    # Termination: find best final tag
    best_final_score = float('-inf')
    best_final_tag = 0
    for j in range(num_tags):
        score = viterbi_scores[T-1][j] + W_end[j]
        if score > best_final_score:
            best_final_score = score
            best_final_tag = j
    
    # Backtrack
    best_path = [0] * T
    best_path[T-1] = best_final_tag
    for t in range(T-2, -1, -1):
        best_path[t] = backpointers[t+1][best_path[t+1]]
    
    return best_path, best_final_score


def compute_score(sentence, tag_sequence):
    """
    Compute the score of a specific tag sequence for a sentence.
    score = sum of emission scores + transition scores
    """
    T = len(sentence)
    score = 0.0
    
    for t in range(T):
        word_idx = word_to_idx.get(sentence[t], 0)
        tag_idx = tag_to_idx[tag_sequence[t]]
        score += W_emit[word_idx][tag_idx]
        
        if t == 0:
            score += W_start[tag_idx]
        else:
            prev_tag_idx = tag_to_idx[tag_sequence[t-1]]
            score += W_trans[prev_tag_idx][tag_idx]
        
        if t == T - 1:
            score += W_end[tag_idx]
    
    return score


def compute_gradients(sentence, gold_tags):
    """
    Compute gradients for CRF parameters.
    
    Gradient = empirical feature counts - expected feature counts
    
    For emission weights: dL/dW_emit[w][t] = count(w,t in gold) - E[count(w,t)]
    For transition weights: dL/dW_trans[i][j] = count(i->j in gold) - E[count(i->j)]
    """
    T = len(sentence)
    emit_scores = compute_scores(sentence)
    alpha, log_Z = forward_algorithm(emit_scores)
    beta = backward_algorithm(emit_scores)
    node_marg, edge_marg, start_marg, end_marg = compute_marginals(
        emit_scores, alpha, beta, log_Z
    )
    
    # Initialize gradients
    dW_emit = [[0.0] * num_tags for _ in range(num_words)]
    dW_trans = [[0.0] * num_tags for _ in range(num_tags)]
    dW_start = [0.0] * num_tags
    dW_end = [0.0] * num_tags
    
    # Emission gradients
    for t in range(T):
        word_idx = word_to_idx.get(sentence[t], 0)
        gold_tag_idx = tag_to_idx[gold_tags[t]]
        
        # Empirical count (1 for gold tag)
        dW_emit[word_idx][gold_tag_idx] += 1.0
        
        # Expected count (subtract)
        for j in range(num_tags):
            dW_emit[word_idx][j] -= node_marg[t][j]
    
    # Transition gradients
    for t in range(1, T):
        prev_tag_idx = tag_to_idx[gold_tags[t-1]]
        curr_tag_idx = tag_to_idx[gold_tags[t]]
        
        # Empirical count
        dW_trans[prev_tag_idx][curr_tag_idx] += 1.0
        
        # Expected count (subtract)
        for i in range(num_tags):
            for j in range(num_tags):
                dW_trans[i][j] -= edge_marg[t-1][i][j]
    
    # Start transition gradients
    gold_start_tag = tag_to_idx[gold_tags[0]]
    dW_start[gold_start_tag] += 1.0
    for j in range(num_tags):
        dW_start[j] -= start_marg[j]
    
    # End transition gradients
    gold_end_tag = tag_to_idx[gold_tags[T-1]]
    dW_end[gold_end_tag] += 1.0
    for j in range(num_tags):
        dW_end[j] -= end_marg[j]
    
    return dW_emit, dW_trans, dW_start, dW_end, log_Z


def train_step(sentence, gold_tags):
    """
    Perform one training step: compute loss and update weights.
    
    Loss = -log P(y|x) = -score(x,y) + log Z(x)
    """
    # Compute gradients
    dW_emit, dW_trans, dW_start, dW_end, log_Z = compute_gradients(sentence, gold_tags)
    
    # Compute loss
    gold_score = compute_score(sentence, gold_tags)
    loss = -gold_score + log_Z
    
    # Update weights (gradient ascent on log-likelihood = gradient descent on loss)
    for w in range(num_words):
        for t in range(num_tags):
            W_emit[w][t] += lr * dW_emit[w][t]
    
    for i in range(num_tags):
        for j in range(num_tags):
            W_trans[i][j] += lr * dW_trans[i][j]
    
    for j in range(num_tags):
        W_start[j] += lr * dW_start[j]
        W_end[j] += lr * dW_end[j]
    
    return loss


def predict(sentence):
    """Predict the best tag sequence for a sentence."""
    emit_scores = compute_scores(sentence)
    best_path, _ = viterbi(emit_scores)
    return [idx_to_tag[idx] for idx in best_path]


def evaluate(data):
    """Evaluate accuracy on a dataset."""
    correct = 0
    total = 0
    for sentence, gold_tags in data:
        pred_tags = predict(sentence)
        for pred, gold in zip(pred_tags, gold_tags):
            if pred == gold:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


# =============================================================================
# TRAINING
# =============================================================================

print("=" * 55)
print("Conditional Random Field - Sequence Labeling")
print("=" * 55)
print(f"Vocabulary: {num_words} words, Tags: {tags}")
print(f"Training sequences: {len(train_data)}")
print("=" * 55)

for epoch in range(200):
    total_loss = 0
    for sentence, gold_tags in train_data:
        loss = train_step(sentence, gold_tags)
        total_loss += loss
    
    if epoch % 40 == 0:
        acc = evaluate(train_data)
        print(f"Epoch {epoch:3d} | Loss: {total_loss:.4f} | Accuracy: {acc:.2%}")

# =============================================================================
# EVALUATION
# =============================================================================

print("\n" + "=" * 55)
print("Predictions on Training Data")
print("=" * 55)

for sentence, gold_tags in train_data:
    pred_tags = predict(sentence)
    match = "✓" if pred_tags == gold_tags else "✗"
    print(f"\nSentence: {' '.join(sentence)}")
    print(f"  Gold: {' '.join(gold_tags)}")
    print(f"  Pred: {' '.join(pred_tags)} {match}")

# Test on new sentences
print("\n" + "=" * 55)
print("Predictions on New Sequences")
print("=" * 55)

test_sentences = [
    ["the", "small", "cat", "runs"],
    ["a", "big", "dog"],
    ["the", "dog", "jumps", "quickly"],
]

for sentence in test_sentences:
    pred_tags = predict(sentence)
    print(f"\nSentence: {' '.join(sentence)}")
    print(f"  Pred: {' '.join(pred_tags)}")

# Show learned transition weights
print("\n" + "=" * 55)
print("Learned Transition Weights")
print("=" * 55)
print("\n     " + "  ".join(f"{t:>6}" for t in tags))
for i, tag_i in enumerate(tags):
    row = "  ".join(f"{W_trans[i][j]:6.2f}" for j in range(num_tags))
    print(f"{tag_i:4} {row}")

print("\nStart weights:", " ".join(f"{tags[j]}:{W_start[j]:.2f}" for j in range(num_tags)))
print("End weights:  ", " ".join(f"{tags[j]}:{W_end[j]:.2f}" for j in range(num_tags)))

final_acc = evaluate(train_data)
print(f"\nFinal Training Accuracy: {final_acc:.2%}")

