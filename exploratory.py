# Inverse Holographic Theory (IHT-AI) Experiment: Learned vs Tensor Unfolding
# This script compares how robust the "latent attractor" (the localized peak) is
# under two different expansion methods (Tensor-Product vs Learned Linear Mapping)
# across a range of decoherence (Dilution / Gamma) values.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# -- Helpers --
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
    """Applies the linear mapping W_learn to the 1D psi_vec to create the high-dim state."""
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
    """Projects to 1D, returns normalized density and phase coherence."""
    if psi.ndim == 1:
        proj = psi
    else:
        axes = tuple(range(1, psi.ndim))
        proj = np.sum(psi, axis=axes)
    density = np.abs(proj)**2
    density = density / (np.max(density) + 1e-16)
    phase = np.angle(proj)
    return density, np.abs(np.mean(np.exp(1j * phase))) 

def run_simulation(psi_init, kernels, W_learn, gamma, expansion_type, t_evolve, ticks_per_expansion, strength, seed):
    np.random.seed(seed + 1000)
    psi = psi_init.copy()
    
    # 1. Expand to 4D state (either via tensor products or single learned map)
    if expansion_type == 'tensor':
        for kern in kernels:
            psi = expand_dimension_tensor(psi, kern)
            for _ in range(ticks_per_expansion):
                # Apply initial rotation + dilution during expansion ticks
                psi = local_phase_rotation(psi, strength=strength + 0.01*np.random.randn())
                psi = psi * (1.0 - gamma)
    elif expansion_type == 'learned':
        psi_vec = psi_init.reshape(-1)
        shape_out = (len(psi_init), kernels[0].size, kernels[1].size, kernels[2].size)
        psi = expand_dimension_learned(psi_vec, shape_out, W_learn) 
        # Apply phase rotation and decay ticks over the same period as tensor's expansion
        for _ in range(len(kernels) * ticks_per_expansion):
            psi = local_phase_rotation(psi, strength=strength + 0.01*np.random.randn())
            psi = psi * (1.0 - gamma)
    
    # Get the initial peak after expansion (t=0 for the evolution phase)
    density0, _ = marginal_project_to_1d(psi)
    peak0 = np.max(density0)
    
    # 2. Evolve (Decay Curve)
    peaks = np.zeros(t_evolve)
    coherence_ts = np.zeros(t_evolve)
    
    for t in range(t_evolve):
        psi = local_phase_rotation(psi, strength=strength + 0.01*np.random.randn())
        psi = psi * (1.0 - gamma)
        density, coherence = marginal_project_to_1d(psi)
        peaks[t] = np.max(density)
        coherence_ts[t] = coherence
        
    # Metrics calculation
    below50 = np.where(peaks <= 0.5 * peak0)[0]
    half50 = below50[0] if below50.size > 0 else t_evolve 
    final_coh = coherence_ts[-1]
    
    return half50, final_coh

# -- Main Simulation Parameters --
L = 192 # Base grid size
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
t_evolve = 600 # Evolution steps to measure half-life

# Gamma grid and seeds
gamma_vals = np.concatenate((np.linspace(0.0, 0.015, 6), np.linspace(0.02, 0.05, 5), np.linspace(0.06, 0.1, 4)))
seeds = [0,1,2,3]

# Initialize random linear mapping matrix W_learn (Orthonormal-ish)
target_flat_size = L * (kernel_size**3)
W_rand = np.random.randn(target_flat_size, L) * 0.01 
U, S, Vh = np.linalg.svd(W_rand, full_matrices=False)
W_learn = U @ Vh 

# Storage for results
results_tensor = {'half50': np.zeros((len(gamma_vals), len(seeds))), 'final_coh': np.zeros((len(gamma_vals), len(seeds)))}
results_learned = {'half50': np.zeros((len(gamma_vals), len(seeds))), 'final_coh': np.zeros((len(gamma_vals), len(seeds)))}

# Run the sweep
print("Starting comparative sweep (Tensor vs Learned)...")
for gi, gamma in enumerate(gamma_vals):
    for si, seed in enumerate(seeds):
        # Tensor Expansion Run (Baseline)
        h50_t, fc_t = run_simulation(base_psi, kernels, W_learn, gamma, 'tensor', t_evolve, ticks_per_expansion, strength, seed)
        results_tensor['half50'][gi, si] = h50_t
        results_tensor['final_coh'][gi, si] = fc_t
        
        # Learned Expansion Run (Exploratory)
        h50_l, fc_l = run_simulation(base_psi, kernels, W_learn, gamma, 'learned', t_evolve, ticks_per_expansion, strength, seed)
        results_learned['half50'][gi, si] = h50_l
        results_learned['final_coh'][gi, si] = fc_l
    print(f"-> Completed gamma={gamma:.4f}")

# Aggregate results (median across seeds)
median_h50_t = np.median(results_tensor['half50'], axis=1)
median_coh_t = np.median(results_tensor['final_coh'], axis=1)
median_h50_l = np.median(results_learned['half50'], axis=1)
median_coh_l = np.median(results_learned['final_coh'], axis=1)

# Plot comparison
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Half-life comparison
ax[0].plot(gamma_vals, median_h50_t, '-o', label='Tensor Unfolding (Baseline)', color='C0')
ax[0].plot(gamma_vals, median_h50_l, '--x', label='Learned Mapping (Exploratory)', color='C1')
ax[0].set_title('Attractor Half-Life (50%) vs Dilution (Gamma)')
ax[0].set_xlabel('Dilution $\\gamma$')
ax[0].set_ylabel('Median Half-Life (Timesteps)')
ax[0].axhline(t_evolve, color='gray', linestyle=':', label='Max Run Time')
ax[0].legend()
ax[0].grid(True)

# Coherence comparison
ax[1].plot(gamma_vals, median_coh_t, '-o', label='Tensor Unfolding (Baseline)', color='C0')
ax[1].plot(gamma_vals, median_coh_l, '--x', label='Learned Mapping (Exploratory)', color='C1')
ax[1].set_title('Final Projected Phase Coherence vs Dilution (Gamma)')
ax[1].set_xlabel('Dilution $\\gamma$')
ax[1].set_ylabel('Median Final Coherence $|\\langle e^{i\\theta} \\rangle|$')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()

# Print key comparison summary
print("\n--- Comparative Summary ---")
print("Critical Gamma (Tensor Unfolding): approx 0.015 (from previous analysis)")
print("Median Half-Life (Last Gamma Tested, 0.1000):")
print(f"  Tensor (Baseline): {median_h50_t[-1]:.2f} steps")
print(f"  Learned (Exploratory): {median_h50_l[-1]:.2f} steps")
print("Median Final Coherence (Last Gamma Tested, 0.1000):")
print(f"  Tensor (Baseline): {median_coh_t[-1]:.4f}")
print(f"  Learned (Exploratory): {median_coh_l[-1]:.4f}")