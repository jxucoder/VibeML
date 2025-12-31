# Micro Graph Neural Network

A minimal Graph Neural Network (GNN) for node classification using message passing.

## Python
```bash
python3 gnn.py
```

## How It Works

GNNs learn node representations by **aggregating information from neighbors** in a graph. This is called **message passing**:

```
For each node v:
    1. AGGREGATE: Collect features from neighbors N(v)
       agg_v = MEAN({h_u : u ∈ N(v) ∪ {v}})
    
    2. UPDATE: Transform aggregated features
       h_v = ReLU(W · agg_v + b)
```

### Architecture

```
Input Features → GCN Layer 1 → GCN Layer 2 → Output Layer → Softmax
     (4)            (8)            (8)           (2)
```

Each **Graph Convolutional (GCN) Layer**:
- Aggregates neighbor features (mean pooling)
- Applies a linear transformation
- ReLU activation

### The Key Insight

Unlike regular neural networks that treat inputs independently, GNNs **share information along edges**. After 2 layers, each node's representation contains information from nodes up to 2 hops away.

```
Layer 0: Node knows only itself
Layer 1: Node knows 1-hop neighbors  
Layer 2: Node knows 2-hop neighbors
```

This allows the network to learn graph structure and make predictions based on local neighborhoods.

## Task: Node Classification

We use a simplified social network (inspired by Zachary's Karate Club):
- 12 nodes (people)
- 21 edges (friendships)
- 2 factions (ground truth labels)

The GNN learns to classify which faction each person belongs to, using only random initial features and the graph structure.

