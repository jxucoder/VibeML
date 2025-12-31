# Micro RNN

A character-level recurrent neural network that learns to predict the next character in a sequence.

## Python
```bash
python3 rnn.py
```

## How It Works

The RNN maintains a **hidden state** that gets updated at each time step, allowing it to "remember" previous characters:

```
h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)
y_t = Why @ h_t + by
p_t = softmax(y_t)
```

- **Wxh**: input → hidden weights
- **Whh**: hidden → hidden weights (this is what makes it recurrent!)
- **Why**: hidden → output weights

Training uses **Backpropagation Through Time (BPTT)** - gradients flow backwards through the sequence.

