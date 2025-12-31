# Micro FM (Factorization Machine)

A factorization machine that learns feature interactions from sparse data—perfect for recommendation systems.

## Python
```bash
python3 fm.py
```

## How It Works

Factorization Machines model both linear effects and pairwise feature interactions:

```
y(x) = w₀ + Σᵢ wᵢxᵢ + Σᵢ Σⱼ₍ᵢ₎ <vᵢ, vⱼ> xᵢxⱼ
```

Where `<vᵢ, vⱼ>` is the dot product of latent vectors for features i and j.

### The Magic Trick

Computing all pairs naively is O(n²). FMs use a reformulation:

```
Σᵢ Σⱼ <vᵢ, vⱼ> xᵢxⱼ = ½ Σf [(Σᵢ vᵢf xᵢ)² - Σᵢ (vᵢf xᵢ)²]
```

This reduces complexity to **O(kn)** where k is the latent dimension!

### Why FMs Work

- **Sparse data**: Each feature pair may rarely co-occur, but latent factors generalize
- **Cold start**: New items get reasonable predictions from their feature vectors
- **Efficiency**: Linear complexity in features, works with millions of sparse features

## Parameters

- **w₀**: global bias
- **w**: linear feature weights (n)
- **V**: latent factor matrix (n × k)

