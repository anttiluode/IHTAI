#
# iht_full_analysis.py
#
# This is the complete, combined script for analyzing the Inverse Holographic Theory (IHT-AI) model.
# It contains three parts that can be run sequentially:
#
# PART 1: Re-runs the 'exploratory.py' sweep (Tensor vs. Learned) and adds robust statistical analysis.
# PART 2: Analyzes the internal structure of the random 'W_learn' matrix (SVD, Participation Ratio).
# PART 3: A full, runnable PyTorch script to *train* an optimal 'W' matrix for attractor survival.
#
# Requirements: numpy, matplotlib, scipy, torch
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, iqr
import time

# --- PART 1 & 2: NumPy/SciPy Analysis ---

print("--- [IHT-AI Analysis] ---")
print("Starting Part 1: Comparative Sweep (Tensor vs. Learned)...")

# -- Helpers (NumPy-based) --
def gaussian_1d(x, mu=0.0, sigma=0.08):
    return np.exp(-0.5 * ((x - mu)/sigma)**2)

def make_kernel(ksize=4, freq=3.5, phase_offset=0.0):
    coords = np.linspace(-1,1,ksize)
    amp = np.exp(-coords**2 * 4.0)
    phase = np.exp(1j * (freq * coords + phase_offset))
    return amp * phase

def expand_dimension_tensor(psi, kernel):
    new_psi = np.tensordot(psi, kernel, axes=0)
    norm = np.sqrt(np.sum(np.abs(new_psi)**2))
    return new_psi / (norm + 1e-16)

def expand_dimension_learned(psi_vec, shape_out, W_learn):
    out = W_learn @ psi_vec
    out = out.reshape(shape_out)
    out = out / np.sqrt(np.sum(np.abs(out)**2) + 1e-16)
    return out

def local_phase_rotation(psi, strength=0.18):
    shape = psi.shape
    grids = np.meshgrid(*[np.linspace(-1,1,s) for s in shape], indexing='ij')
    phi = np.zeros_like(psi, dtype=np.float64)
    for g in grids:
        phi += g * 2.0
    return psi * np.exp(1j * strength * phi)

def marginal_project_to_1d(psi):
    if psi.ndim == 1:
        proj = psi
    else:
        axes = tuple(range(1, psi.ndim))
        proj = np.sum(psi, axis=axes)
    density = np.abs(proj)**2
    max_dens = np.max(density)
    if max_dens < 1e-16:
        density = density / (max_dens + 1e-16)
    else:
        density = density / max_dens
    phase = np.angle(proj)
    return density, np.abs(np.mean(np.exp(1j * phase))) 

def run_simulation(psi_init, kernels, W_learn, gamma, expansion_type, t_evolve, ticks_per_expansion, strength, seed):
    np.random.seed(seed + 1000)
    psi = psi_init.copy()
    shape_out = (len(psi_init), kernels[0].size, kernels[1].size, kernels[2].size)

    # 1. Expand
    if expansion_type == 'tensor':
        for kern in kernels:
            psi = expand_dimension_tensor(psi, kern)
            for _ in range(ticks_per_expansion):
                psi = local_phase_rotation(psi, strength=strength + 0.01*np.random.randn())
                psi = psi * (1.0 - gamma)
    elif expansion_type == 'learned':
        psi_vec = psi_init.reshape(-1)
        psi = expand_dimension_learned(psi_vec, shape_out, W_learn) 
        for _ in range(len(kernels) * ticks_per_expansion):
            psi = local_phase_rotation(psi, strength=strength + 0.01*np.random.randn())
            psi = psi * (1.0 - gamma)
    
    density0, _ = marginal_project_to_1d(psi)
    peak0 = np.max(density0)
    
    # 2. Evolve
    peaks = np.zeros(t_evolve)
    coherence_ts = np.zeros(t_evolve)
    
    for t in range(t_evolve):
        psi = local_phase_rotation(psi, strength=strength + 0.01*np.random.randn())
        psi = psi * (1.0 - gamma)
        density, coherence = marginal_project_to_1d(psi)
        peaks[t] = np.max(density)
        coherence_ts[t] = coherence
        
    below50 = np.where(peaks <= 0.5 * peak0)[0]
    half50 = below50[0] if below50.size > 0 else t_evolve 
    final_coh = coherence_ts[-1]
    
    return half50, final_coh

# -- Main Simulation Parameters --
L = 192 
x = np.linspace(-1,1,L)
base_psi = gaussian_1d(x, mu=0.0, sigma=0.08)
base_psi = base_psi * np.exp(1j * 0.12 * np.sin(5*x))
base_psi = base_psi / np.sqrt(np.sum(np.abs(base_psi)**2))

kernel_size = 4
kern1 = make_kernel(ksize=kernel_size, freq=3.5, phase_offset=0.0)
kern2 = make_kernel(ksize=kernel_size, freq=4.5, phase_offset=0.6)
kern3 = make_kernel(ksize=kernel_size, freq=2.8, phase_offset=-0.4)
kernels = (kern1, kern2, kern3)

ticks_per_expansion = 5
strength = 0.18
t_evolve = 600
gamma_vals = np.concatenate((np.linspace(0.0, 0.015, 6), np.linspace(0.02, 0.05, 5), np.linspace(0.06, 0.1, 4)))
seeds = list(range(8)) # Use 8 seeds for better statistics

target_flat_size = L * (kernel_size**3)
W_rand = np.random.randn(target_flat_size, L) + 1j * np.random.randn(target_flat_size, L)
W_rand = W_rand * 0.01 
U, S, Vh = np.linalg.svd(W_rand, full_matrices=False)
W_learn = U @ Vh # Orthonormal-ish complex matrix

# Storage
results_tensor = {'half50': np.zeros((len(gamma_vals), len(seeds))), 'final_coh': np.zeros((len(gamma_vals), len(seeds)))}
results_learned = {'half50': np.zeros((len(gamma_vals), len(seeds))), 'final_coh': np.zeros((len(gamma_vals), len(seeds)))}

# Run Sweep
for gi, gamma in enumerate(gamma_vals):
    for si, seed in enumerate(seeds):
        h50_t, fc_t = run_simulation(base_psi, kernels, W_learn, gamma, 'tensor', t_evolve, ticks_per_expansion, strength, seed)
        results_tensor['half50'][gi, si] = h50_t
        results_tensor['final_coh'][gi, si] = fc_t
        
        h50_l, fc_l = run_simulation(base_psi, kernels, W_learn, gamma, 'learned', t_evolve, ticks_per_expansion, strength, seed)
        results_learned['half50'][gi, si] = h50_l
        results_learned['final_coh'][gi, si] = fc_l
    print(f"-> Part 1: Completed gamma={gamma:.4f}")

# -- Part 1b: Statistical Analysis & Plotting --
print("\nStarting Part 1b: Statistical Analysis...")

# Aggregate results (median and IQR)
median_h50_t = np.median(results_tensor['half50'], axis=1)
iqr_h50_t = iqr(results_tensor['half50'], axis=1)
median_h50_l = np.median(results_learned['half50'], axis=1)
iqr_h50_l = iqr(results_learned['half50'], axis=1)

median_coh_t = np.median(results_tensor['final_coh'], axis=1)
iqr_coh_t = iqr(results_tensor['final_coh'], axis=1)
median_coh_l = np.median(results_learned['final_coh'], axis=1)
iqr_coh_l = iqr(results_learned['final_coh'], axis=1)

# Plot with error bars (Median +/- IQR/2)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Part 1: Statistical Comparison (Median and IQR over 8 seeds)")

# Half-life plot
ax[0].plot(gamma_vals, median_h50_t, '-o', label='Tensor Unfolding (Baseline)', color='C0')
ax[0].fill_between(gamma_vals, median_h50_t - iqr_h50_t/2, median_h50_t + iqr_h50_t/2, color='C0', alpha=0.2)
ax[0].plot(gamma_vals, median_h50_l, '--x', label='Learned Mapping (Exploratory)', color='C1')
ax[0].fill_between(gamma_vals, median_h50_l - iqr_h50_l/2, median_h50_l + iqr_h50_l/2, color='C1', alpha=0.2)
ax[0].set_title('Attractor Half-Life (50%) vs Dilution (Gamma)')
ax[0].set_xlabel('Dilution $\\gamma$')
ax[0].set_ylabel('Median Half-Life (Timesteps)')
ax[0].axhline(t_evolve, color='gray', linestyle=':', label='Max Run Time')
ax[0].legend()
ax[0].grid(True)

# Coherence plot
ax[1].plot(gamma_vals, median_coh_t, '-o', label='Tensor Unfolding (Baseline)', color='C0')
ax[1].fill_between(gamma_vals, median_coh_t - iqr_coh_t/2, median_coh_t + iqr_coh_t/2, color='C0', alpha=0.2)
ax[1].plot(gamma_vals, median_coh_l, '--x', label='Learned Mapping (Exploratory)', color='C1')
ax[1].fill_between(gamma_vals, median_coh_l - iqr_coh_l/2, median_coh_l + iqr_coh_l/2, color='C1', alpha=0.2)
ax[1].set_title('Final Projected Phase Coherence vs Dilution (Gamma)')
ax[1].set_xlabel('Dilution $\\gamma$')
ax[1].set_ylabel('Median Final Coherence $|\\langle e^{i\\theta} \\rangle|$')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('iht_stats_comparison.png')
print("Saved statistical plot to iht_stats_comparison.png")
plt.show()

# Run Wilcoxon signed-rank test at a critical gamma (e.g., the last one)
gamma_test_idx = -1 # Test at the highest gamma
h50_t_vals = results_tensor['half50'][gamma_test_idx, :]
h50_l_vals = results_learned['half50'][gamma_test_idx, :]
coh_t_vals = results_tensor['final_coh'][gamma_test_idx, :]
coh_l_vals = results_learned['final_coh'][gamma_test_idx, :]

# Check for non-zero variance before running test
if np.any(h50_t_vals != h50_l_vals):
    stat_h50, p_h50 = wilcoxon(h50_t_vals, h50_l_vals)
    print(f"\nWilcoxon Test (Half-Life) at gamma={gamma_vals[gamma_test_idx]:.4f}: p-value = {p_h50:.4f}")
else:
    print(f"\nWilcoxon Test (Half-Life) at gamma={gamma_vals[gamma_test_idx]:.4f}: No difference.")

if np.any(coh_t_vals != coh_l_vals):
    stat_coh, p_coh = wilcoxon(coh_t_vals, coh_l_vals)
    print(f"Wilcoxon Test (Coherence) at gamma={gamma_vals[gamma_test_idx]:.4f}: p-value = {p_coh:.4f}")
else:
    print(f"Wilcoxon Test (Coherence) at gamma={gamma_vals[gamma_test_idx]:.4f}: No difference.")


# --- PART 2: W_learn Internal Analysis ---
print("\nStarting Part 2: Internal Analysis of W_learn...")

# 1. Singular Value Spectrum
s = np.linalg.svd(W_learn, compute_uv=False)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.semilogy(s)
plt.title('Singular Values of $W_{learn}$')
plt.xlabel('Mode Index')
plt.ylabel('Singular Value (log)')
plt.grid(True)

# 2. Participation Ratio (PR)
psi_vec = base_psi.reshape(-1)
out_vec = W_learn @ psi_vec
energy = np.abs(out_vec)**2
p = energy / (energy.sum() + 1e-16)
PR = 1.0 / np.sum(p**2)
print(f"Participation Ratio (PR) of mapped vector: {PR:.2f}")
print(f"(Max possible PR for {target_flat_size} modes is {target_flat_size})")

# 3. Energy spread in hidden dims
out_reshaped = out_vec.reshape(L, kernel_size, kernel_size, kernel_size)
energy_reshaped = np.abs(out_reshaped)**2

# Sum over base dimension (L) to see energy in hidden subspace
hidden_energy = np.sum(energy_reshaped, axis=0)

plt.subplot(1, 2, 2)
plt.plot(np.sum(hidden_energy, axis=(1,2))) # Project hidden 3D to 1D
plt.title('Energy in 1st Hidden Dim')
plt.xlabel('Index')
plt.ylabel('Projected Energy')
plt.tight_layout()
plt.savefig('iht_W_analysis.png')
print("Saved W_learn analysis plot to iht_W_analysis.png")
plt.show()


# --- PART 3: Advanced IHT-AI Training (PyTorch) ---
print("\nStarting Part 3: Trainable W_learn (PyTorch)...")
print("This part is computationally heavy and requires PyTorch.")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # --- PyTorch Helper Functions (Differentiable) ---
    
    def pt_local_phase_rotation(psi_complex, strength, grids):
        """ Differentiable phase rotation """
        # psi_complex shape: (batch, L, k, k, k)
        # grids: precomputed meshgrid
        
        # phi needs to be (1, L, k, k, k)
        phi = torch.zeros_like(psi_complex.real) # (batch, L, k, k, k)
        for g in grids:
            phi += g * 2.0
        
        # Create complex rotation
        rot_real = torch.cos(strength * phi)
        rot_imag = torch.sin(strength * phi)
        
        # Manual complex multiplication
        out_real = psi_complex.real * rot_real - psi_complex.imag * rot_imag
        out_imag = psi_complex.real * rot_imag + psi_complex.imag * rot_real
        
        return torch.complex(out_real, out_imag)

    def pt_marginal_project_to_1d(psi_complex):
        """ Differentiable projection and coherence calculation """
        # psi_complex shape: (batch, L, k, k, k)
        
        proj = torch.sum(psi_complex, dim=(2,3,4)) # Sum over hidden dims -> (batch, L)
        
        # Coherence
        phase = torch.angle(proj)
        # mean(exp(i*phase))
        mean_exp_real = torch.mean(torch.cos(phase), dim=1)
        mean_exp_imag = torch.mean(torch.sin(phase), dim=1)
        
        # abs(mean(exp(i*phase))) = sqrt(real^2 + imag^2)
        coherence = torch.sqrt(mean_exp_real**2 + mean_exp_imag**2)
        
        return coherence # (batch,)
        
    class IHT_Model(nn.Module):
        def __init__(self, L, k, strength, t_evolve, gamma, device):
            super().__init__()
            self.L = L
            self.k = k
            self.target_flat_size = L * (k**3)
            self.strength = strength
            self.t_evolve = t_evolve
            self.gamma = gamma
            self.device = device

            # The trainable mapping: maps 1D complex (2*L) to 4D complex (2*flat)
            # We map (real, imag) -> (real_out, imag_out)
            self.W_mapping = nn.Linear(2 * L, 2 * self.target_flat_size, bias=False)
            
            # Precompute meshgrid for phase rotation (move to device)
            shape = (L, k, k, k)
            grid_coords = [torch.linspace(-1, 1, s, device=device) for s in shape]
            self.grids = torch.meshgrid(*grid_coords, indexing='ij')
            # Add batch dim: (1, L, k, k, k)
            self.grids = [g.unsqueeze(0) for g in self.grids]

        def forward(self, psi_1d_realimag):
            # psi_1d_realimag shape: (batch, 2*L)
            
            # 1. Apply Learned Mapping
            psi_4d_realimag = self.W_mapping(psi_1d_realimag) # (batch, 2*flat)
            
            # Reshape to (batch, L, k, k, k) complex
            real_part = psi_4d_realimag[:, :self.target_flat_size].view(-1, self.L, self.k, self.k, self.k)
            imag_part = psi_4d_realimag[:, self.target_flat_size:].view(-1, self.L, self.k, self.k, self.k)
            psi_complex = torch.complex(real_part, imag_part)
            
            # Normalize
            norm = torch.sqrt(torch.sum(torch.abs(psi_complex)**2, dim=(1,2,3,4), keepdim=True))
            psi_complex = psi_complex / (norm + 1e-16)
            
            # 2. Evolve
            for t in range(self.t_evolve):
                # Differentiable phase rotation + damping
                psi_complex = pt_local_phase_rotation(psi_complex, self.strength, self.grids)
                psi_complex = psi_complex * (1.0 - self.gamma)
                
                # Re-normalize every few steps to prevent numerical vanishing
                if t % 20 == 0:
                    norm = torch.sqrt(torch.sum(torch.abs(psi_complex)**2, dim=(1,2,3,4), keepdim=True))
                    psi_complex = psi_complex / (norm + 1e-16)

            # 3. Compute metric (Final Coherence)
            final_coherence = pt_marginal_project_to_1d(psi_complex) # (batch,)
            
            return final_coherence

    # -- Training Setup --
    
    # Use settings from Part 1
    L_pt = 128 # Use L=128 for training speed
    k_pt = 4
    t_evolve_pt = 50 # Use 50 steps for speed
    gamma_train = 0.02 # Train at the critical threshold
    strength_pt = 0.18
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nPyTorch using device: {device}")
    
    # Create base_psi (numpy) for L=128
    x_pt = np.linspace(-1,1,L_pt)
    base_psi_pt = gaussian_1d(x_pt, mu=0.0, sigma=0.08) * np.exp(1j * 0.12 * np.sin(5*x_pt))
    base_psi_pt = base_psi_pt / np.sqrt(np.sum(np.abs(base_psi_pt)**2))
    
    # Convert to real/imag vector
    base_psi_realimag = np.concatenate([base_psi_pt.real, base_psi_pt.imag]).astype(np.float32)
    base_psi_tensor = torch.tensor(base_psi_realimag, device=device)

    # Init model and optimizer
    model = IHT_Model(L_pt, k_pt, strength_pt, t_evolve_pt, gamma_train, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Starting PyTorch training... (L={L_pt}, t_evolve={t_evolve_pt}, gamma={gamma_train})")
    
    n_epochs = 100
    batch_size = 4
    loss_history = []

    start_time = time.time()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Create a batch by repeating the base_psi tensor
        batch_in = base_psi_tensor.unsqueeze(0).repeat(batch_size, 1)
        
        # Add small noise to each item in batch
        noise = 0.01 * torch.randn_like(batch_in)
        batch_in = batch_in + noise
        
        # Forward pass
        final_coherence_batch = model(batch_in)
        
        # Loss = -coherence (we want to MAXIMIZE coherence)
        loss = -torch.mean(final_coherence_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss (Neg Coherence): {loss.item():.4f}")

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history)
    plt.title('Part 3: Training Loss (Negative Coherence)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('iht_training_loss.png')
    print("Saved training loss plot to iht_training_loss.png")
    plt.show()

    # We could now save the trained model.state_dict()
    # torch.save(model.state_dict(), 'iht_trained_W.pth')
    # print("Saved trained model weights to iht_trained_W.pth")

except ImportError:
    print("\nPyTorch not found. Skipping Part 3 (Advanced Training).")
    print("To run Part 3, please install PyTorch: pip install torch")

print("\n--- [IHT-AI Analysis Complete] ---")