# Micro CNN

A convolutional neural network that learns to classify simple image patterns (vertical vs horizontal lines).

## Python
```bash
python3 cnn.py
```

## How It Works

The CNN processes images through layers that learn spatial features:

```
Input (5×5) → Conv (3×3 kernel) → ReLU → MaxPool → Dense → Softmax
```

### Convolution
Slides a learned filter across the image, detecting local patterns:
```
output[i,j] = Σ input[i+ki, j+kj] × filter[ki, kj] + bias
```

### ReLU Activation
```
ReLU(x) = max(0, x)
```

### Max Pooling
Takes the maximum value from each feature map, providing translation invariance.

### Backpropagation
Gradients flow backwards through:
1. **Dense layer**: Standard gradient computation
2. **Max pool**: Gradient only flows to the max location
3. **ReLU**: Gradient passes through where input > 0
4. **Conv**: Filter gradients computed via correlation with input

## Architecture
- **Input**: 5×5 grayscale image
- **Conv layer**: 2 filters of 3×3 → 3×3×2 feature maps
- **Pooling**: Global max pool → 2 values
- **Dense**: 2 → 2 (num_classes)
- **Output**: Softmax probabilities

## What It Learns
The filters learn to detect edges—one typically learns vertical patterns, the other horizontal.

