# Variational Autoencoder (VAE)

A minimal VAE implementation from scratch. No frameworks, just pure Python math.

## What is a VAE?

A Variational Autoencoder learns to:
1. **Encode** data into a compressed latent space
2. **Decode** from latent space back to data
3. Keep the latent space **smooth and continuous** (via KL divergence)

## Architecture

```
Input (2D) → Encoder → μ, σ → Sample z → Decoder → Output (2D)
                         ↑
              Reparameterization trick
              z = μ + σ * ε, ε ~ N(0,1)
```

## Loss Function

```
Loss = Reconstruction Loss + β × KL Divergence
     = MSE(x, x̂) + β × KL(q(z|x) || N(0,1))
```

## Run

```bash
python3 vae.py
```

## Key Concepts Implemented

- **Reparameterization Trick**: Makes sampling differentiable
- **KL Divergence**: Regularizes latent space to be Gaussian
- **Beta-VAE**: Adjustable KL weight for disentanglement
- **Manual Backprop**: All gradients computed by hand

