# iht_run_trained_sweep.py
import numpy as np
import matplotlib.pyplot as plt
import torch
import csv
import os
from scipy.stats import iqr

# === Parameters (tweakable) ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
L_sweep = 128           # must match model's L used in training
k = 4
t_evolve = 400          # a bit longer for robustness
ticks = 5
strength = 0.18
seeds = list(range(8))  # ensemble seeds
# gamma grid: fine near small values, extend to more extremes
gamma_vals = np.concatenate((
    np.linspace(0.0, 0.02, 9),
    np.linspace(0.025, 0.06, 8),
    np.linspace(0.07, 0.12, 6),
    np.linspace(0.15, 0.30, 4)
))

# paths
trained_path = 'iht_trained_W.pth'
out_csv = 'iht_trained_sweep_results.csv'
out_plot = 'iht_trained_sweep_plot.png'

# === helpers (numpy) ===
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

def expand_dimension_learned_np(psi_vec, shape_out, Wnp):
    out = Wnp @ psi_vec
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
    if psi.ndim == 1:
        proj = psi
    else:
        proj = np.sum(psi, axis=tuple(range(1, psi.ndim)))
    density = np.abs(proj)**2
    maxd = np.max(density)
    density = density / (maxd + 1e-16)
    phase = np.angle(proj)
    return density, np.abs(np.mean(np.exp(1j * phase)))

# === load or create bases ===
x = np.linspace(-1,1,L_sweep)
base_psi = gaussian_1d(x, mu=0.0, sigma=0.08) * np.exp(1j * 0.12 * np.sin(5*x))
base_psi = base_psi / np.sqrt(np.sum(np.abs(base_psi)**2))

kernels = (
    make_kernel(ksize=k, freq=3.5, phase_offset=0.0),
    make_kernel(ksize=k, freq=4.5, phase_offset=0.6),
    make_kernel(ksize=k, freq=2.8, phase_offset=-0.4)
)
shape_out = (L_sweep, k, k, k)
target_flat_size = L_sweep * (k**3)

# === load trained model ===
if not os.path.exists(trained_path):
    raise FileNotFoundError(f"Trained model not found at {trained_path}")

# We only need the linear weights W mapping (2*L -> 2*flat). We'll extract W and convert to complex mapping
trained = torch.load(trained_path, map_location='cpu')

# find key for linear weight (assumes default IHT_Model.W_mapping weight name)
# weight shape is (2*target_flat, 2*L)
weight_key = None
for kname in trained.keys():
    if 'W_mapping.weight' in kname:
        weight_key = kname
        break
if weight_key is None:
    # common fallback
    weight_key = 'W_mapping.weight'
W_torch = trained[weight_key].cpu().numpy()
# build complex W that maps complex psi (L) -> flat complex vector
# The saved mapping is real-valued mapping of concatenated [real, imag] -> [real_out, imag_out]
# Convert: for complex psi z = a + i b, then W_complex = (W_real + i W_imag) arranged properly.
half_out = W_torch.shape[0] // 2
half_in = W_torch.shape[1] // 2
W_rr = W_torch[:half_out, :half_in]
W_ri = W_torch[:half_out, half_in:]
W_ir = W_torch[half_out:, :half_in]
W_ii = W_torch[half_out:, half_in:]
# complex mapping: out = (W_rr + i W_ir) * real + (W_ri + i W_ii) * imag
# equivalently create W_complex (out_flat x in_L) = (W_rr + i W_ir) + (W_ri + i W_ii) * ?? easier build by combining columns
W_complex = (W_rr + 1j * W_ir) + 1j * (W_ri + 1j * W_ii)  # careful but works when applied to psi_vec
# A safer construction: apply W via real/imag piecewise in numpy function below

def apply_W_complex_numpy(psi_vec_complex, W_torch_mat):
    # psi_vec_complex shape (L,), W_torch_mat is (2*out, 2*L) from saved weights
    a = psi_vec_complex.real
    b = psi_vec_complex.imag
    W = W_torch_mat
    top = W[:W.shape[0]//2, :W.shape[1]//2] @ a + W[:W.shape[0]//2, W.shape[1]//2:] @ b
    bot = W[W.shape[0]//2:, :W.shape[1]//2] @ a + W[W.shape[0]//2:, W.shape[1]//2:] @ b
    out_complex = top + 1j * bot
    return out_complex

# === Sweep function for each model type ===
def run_numpy_tensor(base_psi, kernels, gamma, seed, t_evolve_local):
    np.random.seed(seed + 100)
    psi = base_psi.copy()
    for kern in kernels:
        psi = expand_dimension_tensor(psi, kern)
        for _ in range(ticks):
            psi = np_local_phase_rotation(psi, strength=strength + 0.01*np.random.randn())
            psi = psi * (1.0 - gamma)
    density0, _ = np_marginal_project_to_1d(psi)
    peak0 = np.max(density0)
    peaks = np.zeros(t_evolve_local)
    for t in range(t_evolve_local):
        psi = np_local_phase_rotation(psi, strength=strength + 0.01*np.random.randn())
        psi = psi * (1.0 - gamma)
        density, coh = np_marginal_project_to_1d(psi)
        peaks[t] = np.max(density)
    below = np.where(peaks <= 0.5*peak0)[0]
    half = below[0] if below.size>0 else t_evolve_local
    final_coh = coh
    return half, final_coh

def run_numpy_randomlearned(base_psi, Wrand, gamma, seed, t_evolve_local):
    np.random.seed(seed + 200)
    psi = expand_dimension_learned_np(base_psi.reshape(-1), shape_out, Wrand)
    for _ in range(len(kernels)*ticks):
        psi = np_local_phase_rotation(psi, strength=strength + 0.01*np.random.randn())
        psi = psi * (1.0 - gamma)
    density0, _ = np_marginal_project_to_1d(psi)
    peak0 = np.max(density0) or 1.0
    peaks = np.zeros(t_evolve_local)
    for t in range(t_evolve_local):
        psi = np_local_phase_rotation(psi, strength=strength + 0.01*np.random.randn())
        psi = psi * (1.0 - gamma)
        density, coh = np_marginal_project_to_1d(psi)
        peaks[t] = np.max(density)
    below = np.where(peaks <= 0.5*peak0)[0]
    half = below[0] if below.size>0 else t_evolve_local
    final_coh = coh
    return half, final_coh

def run_numpy_trained(base_psi_realimag_tensor, gamma, seed, t_evolve_local):
    # call PyTorch code path to apply trained W and evolve using pt_local_phase_rotation
    # We'll create minimal model and load weights already done, then reuse run_simulation_optimized idea
    from importlib import import_module
    # The code to build the model class is long; but we can reuse the IHT_Model defined in your training script if available.
    # For portability, assume your training script saved both weights and a small helper function 'run_simulation_optimized'
    # To keep this script self-contained, we will directly apply the saved W weights here using apply_W_complex_numpy
    np.random.seed(seed + 300)
    # Apply W via apply_W_complex_numpy
    psi_complex = apply_W_complex_numpy(base_psi, W_torch)
    # reshape into (L,k,k,k)
    psi_complex = psi_complex.reshape(shape_out)
    # apply a sequence of rotation & damping steps
    for _ in range(len(kernels)*ticks):
        psi_complex = np_local_phase_rotation(psi_complex, strength=strength + 0.01*np.random.randn())
        psi_complex = psi_complex * (1.0 - gamma)
    density0, _ = np_marginal_project_to_1d(psi_complex)
    peak0 = np.max(density0) or 1.0
    peaks = np.zeros(t_evolve_local)
    for t in range(t_evolve_local):
        psi_complex = np_local_phase_rotation(psi_complex, strength=strength + 0.01*np.random.randn())
        psi_complex = psi_complex * (1.0 - gamma)
        density, coh = np_marginal_project_to_1d(psi_complex)
        peaks[t] = np.max(density)
    below = np.where(peaks <= 0.5*peak0)[0]
    half = below[0] if below.size>0 else t_evolve_local
    final_coh = coh
    return half, final_coh

# === prepare random W baseline ===
W_rand = np.random.randn(target_flat_size, L_sweep) + 1j * np.random.randn(target_flat_size, L_sweep)
W_rand *= 0.01
U, S, Vh = np.linalg.svd(W_rand, full_matrices=False)
W_rand_orth = U @ Vh

# === Run sweep ===
rows = []
results = {'tensor':[], 'random':[], 'trained':[]}

print("Starting final sweep over gamma grid:", gamma_vals)
for gi, gamma in enumerate(gamma_vals):
    half_tensor = np.zeros(len(seeds))
    coh_tensor = np.zeros(len(seeds))
    half_rand = np.zeros(len(seeds))
    coh_rand = np.zeros(len(seeds))
    half_tr = np.zeros(len(seeds))
    coh_tr = np.zeros(len(seeds))
    for si, sd in enumerate(seeds):
        h_t, c_t = run_numpy_tensor(base_psi, kernels, gamma, sd, t_evolve)
        h_r, c_r = run_numpy_randomlearned(base_psi, W_rand_orth, gamma, sd, t_evolve)
        h_o, c_o = run_numpy_trained(base_psi, gamma, sd, t_evolve)
        half_tensor[si], coh_tensor[si] = h_t, c_t
        half_rand[si], coh_rand[si] = h_r, c_r
        half_tr[si], coh_tr[si] = h_o, c_o
        rows.append([gamma, sd, h_t, c_t, h_r, c_r, h_o, c_o])
    results['tensor'].append((half_tensor, coh_tensor))
    results['random'].append((half_rand, coh_rand))
    results['trained'].append((half_tr, coh_tr))
    print(f"Completed gamma {gamma:.4f}")

# === Save CSV ===
with open(out_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['gamma','seed','half_tensor','coh_tensor','half_random','coh_random','half_trained','coh_trained'])
    writer.writerows(rows)
print("Saved per-run CSV to", out_csv)

# === Aggregate & plot ===
def aggregate_medians(results_list):
    med_half = np.median(np.vstack([r[0] for r in results_list]), axis=1)
    med_coh = np.median(np.vstack([r[1] for r in results_list]), axis=1)
    return med_half, med_coh

med_half_t, med_coh_t = aggregate_medians([(a,b) for (a,b) in results['tensor']])
med_half_r, med_coh_r = aggregate_medians([(a,b) for (a,b) in results['random']])
med_half_o, med_coh_o = aggregate_medians([(a,b) for (a,b) in results['trained']])

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(gamma_vals, med_half_t, '-o', label='Tensor')
plt.plot(gamma_vals, med_half_r, '--x', label='Random-Learned')
plt.plot(gamma_vals, med_half_o, '-s', label='Trained (Optimized)')
plt.xlabel('gamma'); plt.ylabel('median half-life'); plt.legend(); plt.grid(True)
plt.subplot(1,2,2)
plt.plot(gamma_vals, med_coh_t, '-o', label='Tensor')
plt.plot(gamma_vals, med_coh_r, '--x', label='Random-Learned')
plt.plot(gamma_vals, med_coh_o, '-s', label='Trained (Optimized)')
plt.xlabel('gamma'); plt.ylabel('median final coherence'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(out_plot)
print("Saved final comparison plot to", out_plot)
plt.show()
