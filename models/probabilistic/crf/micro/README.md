# Micro CRF

A minimal Conditional Random Field (CRF) for sequence labeling tasks like POS tagging.

## Python
```bash
python3 crf.py
```

## How It Works

CRFs model the conditional probability of a label sequence **y** given an input sequence **x**:

```
P(y|x) = exp(score(x, y)) / Z(x)
```

Where:
- **score(x, y)** = sum of emission scores + transition scores
- **Z(x)** = partition function (normalizing constant over all possible label sequences)

### Scoring Function

```
score(x, y) = Σ W_emit[x_t][y_t] + Σ W_trans[y_{t-1}][y_t] + W_start[y_0] + W_end[y_T]
```

- **Emission weights**: How likely word x_t emits tag y_t
- **Transition weights**: How likely tag y_{t-1} transitions to y_t
- **Start/End weights**: Probability of starting/ending with each tag

### Key Algorithms

**Forward Algorithm** - Computes log Z(x) efficiently:
```
α[t][j] = logsumexp(α[t-1] + W_trans[:,j]) + emit[t][j]
Z = logsumexp(α[T-1] + W_end)
```

**Backward Algorithm** - Computes backward probabilities for gradients:
```
β[t][i] = logsumexp(W_trans[i,:] + emit[t+1] + β[t+1])
```

**Viterbi Algorithm** - Finds the most likely tag sequence:
```
v[t][j] = max_i(v[t-1][i] + W_trans[i][j]) + emit[t][j]
```

### Training

Maximize log-likelihood using gradient ascent:
```
∇L = empirical_counts - expected_counts
```

- **Empirical counts**: Feature counts from gold labels
- **Expected counts**: Feature counts under model distribution (computed via forward-backward)

## Why CRFs?

Unlike independent classifiers, CRFs model **dependencies between labels**:

| Model | Considers |
|-------|-----------|
| Naive Bayes / Logistic Regression | P(y_t \| x_t) independently |
| HMM | P(y_t \| y_{t-1}) generatively |
| **CRF** | P(y \| x) with full sequence context |

This allows CRFs to learn that:
- DET → NOUN is common
- VERB → DET is rare  
- Sequences don't start with VERB

## Task: POS Tagging

Simple part-of-speech tagging with 4 tags:
- **DET**: Determiners (the, a)
- **NOUN**: Nouns (cat, dog)
- **VERB**: Verbs (runs, jumps)
- **ADJ**: Adjectives (big, small, quickly)

The CRF learns both word-tag associations and valid tag transition patterns.

