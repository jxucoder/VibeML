"""
VibeML - A tiny Variational Autoencoder in Python
Just the math, nothing else. No frameworks, no dependencies.
"""
import math
import random

random.seed(42)

# =============================================================================
# VAE Architecture: 2D input -> 4 hidden -> 2 latent (mu, logvar) -> 4 hidden -> 2D output
# =============================================================================

def init_weights(rows, cols):
    """Xavier initialization"""
    scale = math.sqrt(2.0 / (rows + cols))
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

def init_bias(size):
    return [0.0 for _ in range(size)]

# Encoder: input(2) -> hidden(4) -> mu(2), logvar(2)
enc_w1 = init_weights(2, 4)   # input to hidden
enc_b1 = init_bias(4)
enc_w_mu = init_weights(4, 2)  # hidden to latent mean
enc_b_mu = init_bias(2)
enc_w_logvar = init_weights(4, 2)  # hidden to latent log-variance
enc_b_logvar = init_bias(2)

# Decoder: latent(2) -> hidden(4) -> output(2)
dec_w1 = init_weights(2, 4)   # latent to hidden
dec_b1 = init_bias(4)
dec_w2 = init_weights(4, 2)   # hidden to output
dec_b2 = init_bias(2)

# =============================================================================
# Activation functions
# =============================================================================

def relu(x):
    return max(0, x)

def relu_deriv(x):
    return 1.0 if x > 0 else 0.0

def tanh(x):
    return math.tanh(x)

def tanh_deriv(x):
    t = math.tanh(x)
    return 1 - t * t

# =============================================================================
# Forward pass
# =============================================================================

def encode(x):
    """Encoder: x -> (mu, logvar)"""
    # Hidden layer with ReLU
    h_pre = [sum(x[i] * enc_w1[i][j] for i in range(2)) + enc_b1[j] for j in range(4)]
    h = [relu(h_pre[j]) for j in range(4)]
    
    # Latent parameters (no activation - linear output)
    mu = [sum(h[j] * enc_w_mu[j][k] for j in range(4)) + enc_b_mu[k] for k in range(2)]
    logvar = [sum(h[j] * enc_w_logvar[j][k] for j in range(4)) + enc_b_logvar[k] for k in range(2)]
    
    return mu, logvar, h, h_pre

def reparameterize(mu, logvar):
    """Reparameterization trick: z = mu + std * epsilon"""
    std = [math.exp(0.5 * lv) for lv in logvar]
    eps = [random.gauss(0, 1) for _ in range(2)]
    z = [mu[k] + std[k] * eps[k] for k in range(2)]
    return z, eps, std

def decode(z):
    """Decoder: z -> x_reconstructed"""
    # Hidden layer with ReLU
    h_pre = [sum(z[k] * dec_w1[k][j] for k in range(2)) + dec_b1[j] for j in range(4)]
    h = [relu(h_pre[j]) for j in range(4)]
    
    # Output layer with tanh (outputs in [-1, 1] range)
    out_pre = [sum(h[j] * dec_w2[j][k] for j in range(4)) + dec_b2[k] for k in range(2)]
    out = [tanh(out_pre[k]) for k in range(2)]
    
    return out, h, h_pre, out_pre

def forward(x):
    """Full forward pass"""
    mu, logvar, enc_h, enc_h_pre = encode(x)
    z, eps, std = reparameterize(mu, logvar)
    x_recon, dec_h, dec_h_pre, dec_out_pre = decode(z)
    return x_recon, mu, logvar, z, eps, std, enc_h, enc_h_pre, dec_h, dec_h_pre, dec_out_pre

# =============================================================================
# Loss functions
# =============================================================================

def reconstruction_loss(x, x_recon):
    """Mean Squared Error"""
    return sum((x[i] - x_recon[i]) ** 2 for i in range(2)) / 2

def kl_divergence(mu, logvar):
    """KL(q(z|x) || p(z)) where p(z) = N(0,1)
    = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    return -0.5 * sum(1 + logvar[k] - mu[k]**2 - math.exp(logvar[k]) for k in range(2))

def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    """Total VAE loss = Reconstruction + beta * KL"""
    recon = reconstruction_loss(x, x_recon)
    kl = kl_divergence(mu, logvar)
    return recon + beta * kl, recon, kl

# =============================================================================
# Backward pass (manual gradients)
# =============================================================================

def backward(x, x_recon, mu, logvar, z, eps, std, enc_h, enc_h_pre, dec_h, dec_h_pre, dec_out_pre, beta=1.0):
    """Compute gradients for all parameters"""
    
    # Gradients for decoder output (reconstruction loss)
    d_out = [(x_recon[k] - x[k]) for k in range(2)]  # d(MSE)/d(x_recon)
    d_out_pre = [d_out[k] * tanh_deriv(dec_out_pre[k]) for k in range(2)]
    
    # Gradients for decoder weights (dec_w2, dec_b2)
    d_dec_w2 = [[dec_h[j] * d_out_pre[k] for k in range(2)] for j in range(4)]
    d_dec_b2 = d_out_pre[:]
    
    # Backprop through decoder hidden
    d_dec_h = [sum(dec_w2[j][k] * d_out_pre[k] for k in range(2)) for j in range(4)]
    d_dec_h_pre = [d_dec_h[j] * relu_deriv(dec_h_pre[j]) for j in range(4)]
    
    # Gradients for decoder weights (dec_w1, dec_b1)
    d_dec_w1 = [[z[k] * d_dec_h_pre[j] for j in range(4)] for k in range(2)]
    d_dec_b1 = d_dec_h_pre[:]
    
    # Gradients w.r.t. z
    d_z = [sum(dec_w1[k][j] * d_dec_h_pre[j] for j in range(4)) for k in range(2)]
    
    # Gradients through reparameterization: z = mu + std * eps
    # d_mu from reconstruction
    d_mu_recon = d_z[:]
    # d_logvar from reconstruction: dz/d_logvar = dz/d_std * d_std/d_logvar = eps * 0.5 * std
    d_logvar_recon = [d_z[k] * eps[k] * 0.5 * std[k] for k in range(2)]
    
    # KL divergence gradients
    # d(KL)/d(mu) = mu
    d_mu_kl = [beta * mu[k] for k in range(2)]
    # d(KL)/d(logvar) = 0.5 * (exp(logvar) - 1)
    d_logvar_kl = [beta * 0.5 * (math.exp(logvar[k]) - 1) for k in range(2)]
    
    # Total gradients for mu and logvar
    d_mu = [d_mu_recon[k] + d_mu_kl[k] for k in range(2)]
    d_logvar = [d_logvar_recon[k] + d_logvar_kl[k] for k in range(2)]
    
    # Gradients for encoder mu layer
    d_enc_w_mu = [[enc_h[j] * d_mu[k] for k in range(2)] for j in range(4)]
    d_enc_b_mu = d_mu[:]
    
    # Gradients for encoder logvar layer
    d_enc_w_logvar = [[enc_h[j] * d_logvar[k] for k in range(2)] for j in range(4)]
    d_enc_b_logvar = d_logvar[:]
    
    # Backprop through encoder hidden
    d_enc_h = [
        sum(enc_w_mu[j][k] * d_mu[k] for k in range(2)) +
        sum(enc_w_logvar[j][k] * d_logvar[k] for k in range(2))
        for j in range(4)
    ]
    d_enc_h_pre = [d_enc_h[j] * relu_deriv(enc_h_pre[j]) for j in range(4)]
    
    # Gradients for encoder input layer
    d_enc_w1 = [[x[i] * d_enc_h_pre[j] for j in range(4)] for i in range(2)]
    d_enc_b1 = d_enc_h_pre[:]
    
    return {
        'enc_w1': d_enc_w1, 'enc_b1': d_enc_b1,
        'enc_w_mu': d_enc_w_mu, 'enc_b_mu': d_enc_b_mu,
        'enc_w_logvar': d_enc_w_logvar, 'enc_b_logvar': d_enc_b_logvar,
        'dec_w1': d_dec_w1, 'dec_b1': d_dec_b1,
        'dec_w2': d_dec_w2, 'dec_b2': d_dec_b2,
    }

def update_params(grads, lr):
    """Update all parameters with gradients"""
    global enc_w1, enc_b1, enc_w_mu, enc_b_mu, enc_w_logvar, enc_b_logvar
    global dec_w1, dec_b1, dec_w2, dec_b2
    
    for i in range(2):
        for j in range(4):
            enc_w1[i][j] -= lr * grads['enc_w1'][i][j]
            dec_w1[i][j] -= lr * grads['dec_w1'][i][j]
    
    for j in range(4):
        enc_b1[j] -= lr * grads['enc_b1'][j]
        dec_b1[j] -= lr * grads['dec_b1'][j]
        for k in range(2):
            enc_w_mu[j][k] -= lr * grads['enc_w_mu'][j][k]
            enc_w_logvar[j][k] -= lr * grads['enc_w_logvar'][j][k]
            dec_w2[j][k] -= lr * grads['dec_w2'][j][k]
    
    for k in range(2):
        enc_b_mu[k] -= lr * grads['enc_b_mu'][k]
        enc_b_logvar[k] -= lr * grads['enc_b_logvar'][k]
        dec_b2[k] -= lr * grads['dec_b2'][k]

# =============================================================================
# Generate training data: points on a circle (simple 2D manifold)
# =============================================================================

def generate_circle_data(n_samples, noise=0.05):
    """Generate points on a unit circle with some noise"""
    data = []
    for _ in range(n_samples):
        theta = random.uniform(0, 2 * math.pi)
        x = math.cos(theta) + random.gauss(0, noise)
        y = math.sin(theta) + random.gauss(0, noise)
        data.append([x * 0.8, y * 0.8])  # Scale to fit in tanh range
    return data

# =============================================================================
# Training
# =============================================================================

print("=" * 60)
print("VibeML Micro VAE - Learning a Circle Manifold")
print("=" * 60)

# Generate training data
train_data = generate_circle_data(200)
print(f"\nGenerated {len(train_data)} training points on a circle")

# Training hyperparameters
lr = 0.01
beta = 0.1  # KL weight (start small for stable training)
epochs = 500

print(f"\nTraining VAE for {epochs} epochs...")
print(f"Learning rate: {lr}, Beta (KL weight): {beta}")
print("-" * 60)

for epoch in range(epochs):
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    random.shuffle(train_data)
    
    for x in train_data:
        # Forward pass
        x_recon, mu, logvar, z, eps, std, enc_h, enc_h_pre, dec_h, dec_h_pre, dec_out_pre = forward(x)
        
        # Compute loss
        loss, recon, kl = vae_loss(x, x_recon, mu, logvar, beta)
        total_loss += loss
        total_recon += recon
        total_kl += kl
        
        # Backward pass
        grads = backward(x, x_recon, mu, logvar, z, eps, std, enc_h, enc_h_pre, dec_h, dec_h_pre, dec_out_pre, beta)
        
        # Update parameters
        update_params(grads, lr)
    
    if (epoch + 1) % 100 == 0:
        avg_loss = total_loss / len(train_data)
        avg_recon = total_recon / len(train_data)
        avg_kl = total_kl / len(train_data)
        print(f"Epoch {epoch+1:4d} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

# =============================================================================
# Test: Reconstruction quality
# =============================================================================

print("\n" + "=" * 60)
print("Reconstruction Test")
print("=" * 60)

test_points = [
    [0.8, 0.0],   # Right
    [0.0, 0.8],   # Top
    [-0.8, 0.0],  # Left
    [0.0, -0.8],  # Bottom
    [0.566, 0.566],  # Top-right diagonal
]

print("\nInput Point      -> Reconstructed    | Error")
print("-" * 50)
for x in test_points:
    x_recon, mu, logvar, *_ = forward(x)
    error = math.sqrt(sum((x[i] - x_recon[i])**2 for i in range(2)))
    print(f"({x[0]:6.3f}, {x[1]:6.3f}) -> ({x_recon[0]:6.3f}, {x_recon[1]:6.3f}) | {error:.4f}")

# =============================================================================
# Test: Generation from latent space
# =============================================================================

print("\n" + "=" * 60)
print("Generation Test (Sampling from Latent Space)")
print("=" * 60)

print("\nSampling z from N(0,1) and decoding:")
print("-" * 40)
for i in range(5):
    z_sample = [random.gauss(0, 1) for _ in range(2)]
    generated, *_ = decode(z_sample)
    # Check if generated point is roughly on the circle
    radius = math.sqrt(generated[0]**2 + generated[1]**2)
    print(f"z=({z_sample[0]:6.3f}, {z_sample[1]:6.3f}) -> ({generated[0]:6.3f}, {generated[1]:6.3f}) | radius={radius:.3f}")

# =============================================================================
# Test: Latent space structure
# =============================================================================

print("\n" + "=" * 60)
print("Latent Space Analysis")
print("=" * 60)

print("\nEncoding test points to latent space:")
print("Input Point      -> Latent mu        | Latent std")
print("-" * 55)
for x in test_points:
    mu, logvar, *_ = encode(x)
    std = [math.exp(0.5 * lv) for lv in logvar]
    print(f"({x[0]:6.3f}, {x[1]:6.3f}) -> ({mu[0]:6.3f}, {mu[1]:6.3f}) | ({std[0]:.3f}, {std[1]:.3f})")

print("\n" + "=" * 60)
print("VAE Training Complete!")
print("=" * 60)

