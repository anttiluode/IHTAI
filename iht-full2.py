#
# iht_optimized_analysis.py
#
# This script first trains an optimized mapping (W) and saves it,
# then runs a full comparative sweep to test its survival curve
# against the 'Tensor' and 'Random Learned' baselines.
#
# Requirements: numpy, matplotlib, scipy, torch
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
import time
import torch
import torch.nn as nn
import torch.optim as optim

print("--- [IHT-AI Optimized Analysis] ---")

# --- PART 1: TRAIN AND SAVE THE OPTIMIZED 'W' MATRIX ---

print("Starting Part 1: Training Optimized 'W' Matrix...")

# -- PyTorch Helper Functions --
def pt_local_phase_rotation(psi_complex, strength, grids):
    """ Differentiable phase rotation """
    phi = torch.zeros_like(psi_complex.real)
    for g in grids:
        phi += g * 2.0
    rot_real = torch.cos(strength * phi)
    rot_imag = torch.sin(strength * phi)
    out_real = psi_complex.real * rot_real - psi_complex.imag * rot_imag
    out_imag = psi_complex.real * rot_imag + psi_complex.imag * rot_real
    return torch.complex(out_real, out_imag)

def pt_marginal_project_to_1d(psi_complex):
    """ Differentiable projection and coherence calculation """
    proj = torch.sum(psi_complex, dim=(2,3,4)) # -> (batch, L)

    # Calculate Density (This was the missing part)
    density = torch.abs(proj)**2
    max_dens, _ = torch.max(density, dim=1, keepdim=True)
    density_norm = density / (max_dens + 1e-16) # (batch, L)
    
    # Calculate Coherence
    phase = torch.angle(proj)
    mean_exp_real = torch.mean(torch.cos(phase), dim=1)
    mean_exp_imag = torch.mean(torch.sin(phase), dim=1)
    coherence = torch.sqrt(mean_exp_real**2 + mean_exp_imag**2) # (batch,)
    
    # Return both
    return density_norm, coherence

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
        
        self.W_mapping = nn.Linear(2 * L, 2 * self.target_flat_size, bias=False)
        
        shape = (L, k, k, k)
        grid_coords = [torch.linspace(-1, 1, s, device=device) for s in shape]
        self.grids = torch.meshgrid(*grid_coords, indexing='ij')
        self.grids = [g.unsqueeze(0) for g in self.grids]

    def forward(self, psi_1d_realimag):
        psi_4d_realimag = self.W_mapping(psi_1d_realimag)
        
        real_part = psi_4d_realimag[:, :self.target_flat_size].view(-1, self.L, self.k, self.k, self.k)
        imag_part = psi_4d_realimag[:, self.target_flat_size:].view(-1, self.L, self.k, self.k, self.k)
        psi_complex = torch.complex(real_part, imag_part)
        
        norm = torch.sqrt(torch.sum(torch.abs(psi_complex)**2, dim=(1,2,3,4), keepdim=True))
        psi_complex = psi_complex / (norm + 1e-16)
        
        for t in range(self.t_evolve):
            psi_complex = pt_local_phase_rotation(psi_complex, self.strength, self.grids)
            psi_complex = psi_complex * (1.0 - self.gamma)
            
            if t % 20 == 0:
                norm = torch.sqrt(torch.sum(torch.abs(psi_complex)**2, dim=(1,2,3,4), keepdim=True))
                psi_complex = psi_complex / (norm + 1e-16)

        final_coherence = pt_marginal_project_to_1d(psi_complex)
        return final_coherence

# -- NumPy gaussian_1d for base state --
def gaussian_1d(x, mu=0.0, sigma=0.08):
    return np.exp(-0.5 * ((x - mu)/sigma)**2)

# -- Training Setup --
L_pt = 128
k_pt = 4
t_evolve_pt = 50
gamma_train = 0.02
strength_pt = 0.18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch using device: {device}")

x_pt = np.linspace(-1,1,L_pt)
base_psi_pt = gaussian_1d(x_pt, mu=0.0, sigma=0.08) * np.exp(1j * 0.12 * np.sin(5*x_pt))
base_psi_pt = base_psi_pt / np.sqrt(np.sum(np.abs(base_psi_pt)**2))
base_psi_realimag = np.concatenate([base_psi_pt.real, base_psi_pt.imag]).astype(np.float32)
base_psi_tensor = torch.tensor(base_psi_realimag, device=device)

model = IHT_Model(L_pt, k_pt, strength_pt, t_evolve_pt, gamma_train, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 100
batch_size = 4
start_time = time.time()

print(f"Starting PyTorch training... (L={L_pt}, t_evolve={t_evolve_pt}, gamma={gamma_train})")
for epoch in range(n_epochs):
    optimizer.zero_grad()
    batch_in = base_psi_tensor.unsqueeze(0).repeat(batch_size, 1)
    noise = 0.01 * torch.randn_like(batch_in)
    batch_in = batch_in + noise
    
    final_coherence_batch = model(batch_in)
    loss = -torch.mean(final_coherence_batch)
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss (Neg Coherence): {loss.item():.4f}")

print(f"Training finished in {time.time() - start_time:.2f} seconds.")

# --- CRITICAL STEP: Save the trained model ---
torch.save(model.state_dict(), 'iht_trained_W.pth')
print("Saved optimized model weights to 'iht_trained_W.pth'")


# --- PART 2: RUN FULL SWEEP WITH OPTIMIZED MODEL ---

print("\nStarting Part 2: Comparative Sweep (Tensor vs. Random vs. Optimized)...")

# -- NumPy Helpers for Baselines --
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

def np_local_phase_rotation(psi, strength=0.18):
    shape = psi.shape
    grids = np.meshgrid(*[np.linspace(-1,1,s) for s in shape], indexing='ij')
    phi = np.zeros_like(psi, dtype=np.float64)
    for g in grids:
        phi += g * 2.0
    return psi * np.exp(1j * strength * phi)

def np_marginal_project_to_1d(psi):
    if psi.ndim == 1: proj = psi
    else: proj = np.sum(psi, axis=tuple(range(1, psi.ndim)))
    density = np.abs(proj)**2
    max_dens = np.max(density)
    density = density / (max_dens + 1e-16)
    phase = np.angle(proj)
    return density, np.abs(np.mean(np.exp(1j * phase)))

def run_simulation_numpy(psi_init, kernels, W_learn, gamma, expansion_type, t_evolve, ticks_per_expansion, strength, seed):
    np.random.seed(seed + 2000)
    psi = psi_init.copy()
    shape_out = (len(psi_init), kernels[0].size, kernels[1].size, kernels[2].size)
    
    if expansion_type == 'tensor':
        for kern in kernels:
            psi = expand_dimension_tensor(psi, kern)
            for _ in range(ticks_per_expansion):
                psi = np_local_phase_rotation(psi, strength=strength + 0.01*np.random.randn())
                psi = psi * (1.0 - gamma)
    elif expansion_type == 'learned':
        psi_vec = psi_init.reshape(-1)
        psi = expand_dimension_learned(psi_vec, shape_out, W_learn) 
        for _ in range(len(kernels) * ticks_per_expansion):
            psi = np_local_phase_rotation(psi, strength=strength + 0.01*np.random.randn())
            psi = psi * (1.0 - gamma)
            
    density0, _ = np_marginal_project_to_1d(psi)
    peak0 = np.max(density0)
    if peak0 < 1e-9: peak0 = 1.0 # Handle immediate collapse
    
    peaks = np.zeros(t_evolve)
    coherence_ts = np.zeros(t_evolve)
    for t in range(t_evolve):
        psi = np_local_phase_rotation(psi, strength=strength + 0.01*np.random.randn())
        psi = psi * (1.0 - gamma)
        density, coherence = np_marginal_project_to_1d(psi)
        peaks[t] = np.max(density)
        coherence_ts[t] = coherence
        
    below50 = np.where(peaks <= 0.5 * peak0)[0]
    half50 = below50[0] if below50.size > 0 else t_evolve
    final_coh = coherence_ts[-1]
    return half50, final_coh

# -- PyTorch Helper for Optimized Sweep --
def run_simulation_optimized(psi_init_realimag_tensor, model_state_dict, gamma, t_evolve, seed):
    # This simulation runs entirely on the PyTorch device
    torch.manual_seed(seed + 3000)
    
    # Load the trained model architecture and weights
    # Note: We use the *training* parameters (L=128, k=4) for the loaded model
    L_m, k_m, s_m, t_m, g_m = 128, 4, 0.18, 50, 0.02 
    model = IHT_Model(L_m, k_m, s_m, t_m, g_m, device).to(device)
    model.load_state_dict(model_state_dict)
    model.eval() # Set to evaluation mode
    
    # We must re-define the forward pass for inference with *variable gamma* and *t_evolve*
    with torch.no_grad(): # No gradients needed
        psi_4d_realimag = model.W_mapping(psi_init_realimag_tensor)
        real_part = psi_4d_realimag[:, :model.target_flat_size].view(-1, L_m, k_m, k_m, k_m)
        imag_part = psi_4d_realimag[:, model.target_flat_size:].view(-1, L_m, k_m, k_m, k_m)
        psi_complex = torch.complex(real_part, imag_part)
        norm = torch.sqrt(torch.sum(torch.abs(psi_complex)**2, dim=(1,2,3,4), keepdim=True))
        psi_complex = psi_complex / (norm + 1e-16)

        # Get initial peak
        density0_all, _ = pt_marginal_project_to_1d(psi_complex)
        peak0 = torch.max(density0_all)
        if peak0 < 1e-9: peak0 = 1.0

        peaks = torch.zeros(t_evolve, device=device)
        coherence_ts = torch.zeros(t_evolve, device=device)

        for t in range(t_evolve):
            # Evolve with the *new* gamma
            psi_complex = pt_local_phase_rotation(psi_complex, model.strength, model.grids)
            psi_complex = psi_complex * (1.0 - gamma) 
            
            density, coherence = pt_marginal_project_to_1d(psi_complex)
            peaks[t] = torch.max(density)
            coherence_ts[t] = coherence

        # Move to CPU/Numpy for metric calculation
        peaks_np = peaks.cpu().numpy()
        coherence_ts_np = coherence_ts.cpu().numpy()
        
        below50 = np.where(peaks_np <= 0.5 * peak0.cpu().numpy())[0]
        half50 = below50[0] if below50.size > 0 else t_evolve
        final_coh = coherence_ts_np[-1]
        
    return half50, final_coh


# -- Sweep Parameters --
L_sweep = 128 # Use L=128 for all sweeps to match trained model
k_sweep = 4
t_evolve_sweep = 300 # Shorter evolution for the sweep
strength_sweep = 0.18
ticks_sweep = 5

# Match the gamma grid from your previous run for a fair comparison
gamma_vals_sweep = np.concatenate((np.linspace(0.0, 0.015, 6), np.linspace(0.02, 0.05, 5), np.linspace(0.06, 0.1, 4), np.linspace(0.12, 0.2, 3)))
seeds_sweep = list(range(6)) # 6 seeds

# Create base state (L=128)
x_sweep = np.linspace(-1,1,L_sweep)
base_psi_sweep = gaussian_1d(x_sweep, mu=0.0, sigma=0.08) * np.exp(1j * 0.12 * np.sin(5*x_sweep))
base_psi_sweep = base_psi_sweep / np.sqrt(np.sum(np.abs(base_psi_sweep)**2))
# Create PyTorch version for optimized model
base_psi_sweep_realimag = np.concatenate([base_psi_sweep.real, base_psi_sweep.imag]).astype(np.float32)
base_psi_sweep_tensor = torch.tensor(base_psi_sweep_realimag, device=device).unsqueeze(0) # Add batch dim

# Create Kernels
kernels_sweep = (
    make_kernel(ksize=k_sweep, freq=3.5, phase_offset=0.0),
    make_kernel(ksize=k_sweep, freq=4.5, phase_offset=0.6),
    make_kernel(ksize=k_sweep, freq=2.8, phase_offset=-0.4)
)
# Create Random W_learn (NumPy, complex)
target_flat_size_sweep = L_sweep * (k_sweep**3)
W_rand_sweep = np.random.randn(target_flat_size_sweep, L_sweep) + 1j * np.random.randn(target_flat_size_sweep, L_sweep)
W_rand_sweep = W_rand_sweep * 0.01 
U_s, S_s, Vh_s = np.linalg.svd(W_rand_sweep, full_matrices=False)
W_learn_random_sweep = U_s @ Vh_s

# Load the Optimized Model State
optimized_model_state = torch.load('iht_trained_W.pth', map_location=device)

# Storage
results = {
    'tensor': {'half50': np.zeros((len(gamma_vals_sweep), len(seeds_sweep))), 'final_coh': np.zeros((len(gamma_vals_sweep), len(seeds_sweep)))},
    'random': {'half50': np.zeros((len(gamma_vals_sweep), len(seeds_sweep))), 'final_coh': np.zeros((len(gamma_vals_sweep), len(seeds_sweep)))},
    'trained': {'half50': np.zeros((len(gamma_vals_sweep), len(seeds_sweep))), 'final_coh': np.zeros((len(gamma_vals_sweep), len(seeds_sweep)))}
}

# --- Run the Final Sweep ---
print(f"Running final sweep (L={L_sweep}, t_evolve={t_evolve_sweep})...")
for gi, gamma in enumerate(gamma_vals_sweep):
    for si, seed in enumerate(seeds_sweep):
        # 1. Tensor (Baseline)
        h50_t, fc_t = run_simulation_numpy(base_psi_sweep, kernels_sweep, None, gamma, 'tensor', t_evolve_sweep, ticks_sweep, strength_sweep, seed)
        results['tensor']['half50'][gi, si] = h50_t
        results['tensor']['final_coh'][gi, si] = fc_t

        # 2. Random Learned (Baseline)
        h50_r, fc_r = run_simulation_numpy(base_psi_sweep, kernels_sweep, W_learn_random_sweep, gamma, 'learned', t_evolve_sweep, ticks_sweep, strength_sweep, seed)
        results['random']['half50'][gi, si] = h50_r
        results['random']['final_coh'][gi, si] = fc_r
        
        # 3. Optimized (Trained)
        h50_o, fc_o = run_simulation_optimized(base_psi_sweep_tensor, optimized_model_state, gamma, t_evolve_sweep, seed)
        results['trained']['half50'][gi, si] = h50_o
        results['trained']['final_coh'][gi, si] = fc_o
        
    print(f"-> Part 2: Completed gamma={gamma:.4f}")

# --- Aggregate & Plot Final Results ---
print("\nAggregating final results and plotting...")

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("IHT-AI Final Comparison: Optimized vs. Baselines (Median & IQR over 6 seeds)", fontsize=16)

colors = {'tensor': 'C0', 'random': 'C1', 'trained': 'C2'}
labels = {'tensor': 'Tensor (Baseline)', 'random': 'Random Learned', 'trained': 'Optimized (Trained)'}
styles = {'tensor': '-o', 'random': '--x', 'trained': '-s'}

for model_type in ['tensor', 'random', 'trained']:
    # Half-life
    median_h50 = np.median(results[model_type]['half50'], axis=1)
    iqr_h50 = iqr(results[model_type]['half50'], axis=1)
    ax[0].plot(gamma_vals_sweep, median_h50, styles[model_type], label=labels[model_type], color=colors[model_type])
    ax[0].fill_between(gamma_vals_sweep, median_h50 - iqr_h50/2, median_h50 + iqr_h50/2, color=colors[model_type], alpha=0.2)
    
    # Coherence
    median_coh = np.median(results[model_type]['final_coh'], axis=1)
    iqr_coh = iqr(results[model_type]['final_coh'], axis=1)
    ax[1].plot(gamma_vals_sweep, median_coh, styles[model_type], label=labels[model_type], color=colors[model_type])
    ax[1].fill_between(gamma_vals_sweep, median_coh - iqr_coh/2, median_coh + iqr_coh/2, color=colors[model_type], alpha=0.2)

ax[0].set_title('Attractor Half-Life (50%) vs Dilution (Gamma)')
ax[0].set_xlabel('Dilution $\\gamma$')
ax[0].set_ylabel('Median Half-Life (Timesteps)')
ax[0].axhline(t_evolve_sweep, color='gray', linestyle=':', label='Max Run Time')
ax[0].legend()
ax[0].grid(True)
ax[0].set_ylim(bottom=0)

ax[1].set_title('Final Projected Phase Coherence vs Dilution (Gamma)')
ax[1].set_xlabel('Dilution $\\gamma$')
ax[1].set_ylabel('Median Final Coherence $|\\langle e^{i\\theta} \\rangle|$')
ax[1].legend()
ax[1].grid(True)
ax[1].set_ylim(0, 1.05)

plt.tight_layout(rect=[0, 0.03, 1, 0.94])
plt.savefig('iht_FINAL_comparison.png')
print("Saved final comparison plot to 'iht_FINAL_comparison.png'")
plt.show()

print("\n--- [IHT-AI Optimized Analysis Complete] ---")