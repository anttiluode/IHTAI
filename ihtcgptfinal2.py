# iht_analyze_trained_W.py
# Analyze trained IHT mapping (iht_trained_W.pth): SVD, PR, energy spread, mode inspection.
# Requirements: numpy, matplotlib, torch, scipy (optional)
# Run in the same folder as iht_trained_W.pth

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.linalg import svd

# --------------------- Config ---------------------
trained_path = 'iht_trained_W.pth'   # change if needed
output_prefix = 'iht_W_trained_analysis'
# Dimensions used during training (must match training)
L = 128        # input 1D length used in training
k = 4          # hidden kernel size per axis (k^3)
# Derived
M = L * (k**3)  # complex output flat length
# ------------------- End Config -------------------

if not os.path.exists(trained_path):
    raise FileNotFoundError(f"Trained file {trained_path} not found in cwd {os.getcwd()}")

# Load PyTorch state dict
state = torch.load(trained_path, map_location='cpu')
# attempt to find mapping weight key
weight_key = None
for key in state.keys():
    if key.endswith('W_mapping.weight') or 'W_mapping.weight' in key:
        weight_key = key
        break
if weight_key is None:
    # fallback guesses
    for key in state.keys():
        if 'weight' in key and state[key].ndim == 2 and state[key].shape[1] % 2 == 0:
            # heuristically pick the 2*target x 2*L matrix
            if state[key].shape[0] >= 2*M and state[key].shape[1] >= 2*L:
                weight_key = key
                break

if weight_key is None:
    raise RuntimeError("Could not locate W_mapping weight in state dict. Keys found:\n" + "\n".join(state.keys()))

W_torch = state[weight_key].cpu().numpy()  # shape: (2*M, 2*L)
print("Found weight key:", weight_key, "shape:", W_torch.shape)

# Validate shapes
if W_torch.shape != (2*M, 2*L):
    print("Warning: expected shape (2*M, 2*L) = ({},{}) but got {}. Attempting coercion...".format(2*M, 2*L, W_torch.shape))

# ---------------- Construct complex W matrix ----------------
# The mapping saved is real: maps [Re(x); Im(x)] -> [Re(out); Im(out)]
# We want W_complex (M x L) such that out_complex = W_complex @ x_complex
# Let x = a + i b. The saved mapping does:
# top = W_rr a + W_ri b
# bot = W_ir a + W_ii b
# out_complex = top + i * bot
# So W_complex = (W_rr + i W_ir) + i*(W_ri + i W_ii) ??? Simpler to form by action.
# We'll build W_complex by applying the mapping to basis vectors.

W = W_torch
rows, cols = W.shape
half_out = rows // 2
half_in = cols // 2

W_rr = W[:half_out, :half_in]
W_ri = W[:half_out, half_in:]
W_ir = W[half_out:, :half_in]
W_ii = W[half_out:, half_in:]

# Construct function to apply saved real mapping to complex vector:
def apply_saved_W_real_to_complex(psi_complex):
    """psi_complex: shape (L,), complex numpy"""
    a = psi_complex.real
    b = psi_complex.imag
    top = W_rr @ a + W_ri @ b
    bot = W_ir @ a + W_ii @ b
    out = top + 1j * bot
    return out  # shape (2*M?) Actually top has length half_out = M

# Build explicit W_complex by applying mapping to complex basis vectors e_j
print("Constructing explicit W_complex by applying mapping to complex basis (this may take a moment)...")
Wc = np.zeros((half_out, half_in), dtype=np.complex128)  # (M, L)
for j in range(L):
    e = np.zeros(L, dtype=np.complex128)
    e[j] = 1.0 + 0j
    Wc[:, j] = apply_saved_W_real_to_complex(e)

print("W_complex shape:", Wc.shape)  # should be (M, L)

# ----------------- SVD and singular spectrum -----------------
print("Computing SVD of W_complex (this can be expensive)...")
U,sigma,Vh = svd(Wc, full_matrices=False)  # sigma is 1D array length = min(M,L) = L (since L < M)
print("Singular values shape:", sigma.shape)

# Plot singular values (semilogy)
plt.figure(figsize=(8,4))
plt.semilogy(sigma, '.', markersize=4)
plt.title('Singular values of trained W (log scale)')
plt.xlabel('mode index')
plt.ylabel('singular value (log)')
plt.grid(True)
plt.tight_layout()
plt.savefig(output_prefix + '_singular_values.png', dpi=200)
print("Saved singular values plot:", output_prefix + '_singular_values.png')

# Cumulative energy
cum = np.cumsum(sigma**2) / np.sum(sigma**2)
plt.figure(figsize=(8,4))
plt.plot(cum, '-o', markersize=4)
plt.title('Cumulative energy of singular values')
plt.xlabel('k')
plt.ylabel('fraction energy explained')
plt.grid(True)
plt.tight_layout()
plt.savefig(output_prefix + '_cumulative_energy.png', dpi=200)
print("Saved cumulative energy plot:", output_prefix + '_cumulative_energy.png')

# Effective rank metrics
def participation_ratio(vec):
    p = np.abs(vec)**2
    p = p / (p.sum() + 1e-16)
    return 1.0 / np.sum(p**2)

# ----------------- Participation Ratio for inputs -----------------
# Build base input (should match training)
x = np.linspace(-1,1,L)
base_psi = np.exp(-0.5 * ((x - 0.0)/0.08)**2) * np.exp(1j * 0.12 * np.sin(5*x))
base_psi = base_psi / np.sqrt(np.sum(np.abs(base_psi)**2))

# Mapped output
out_base = Wc @ base_psi  # shape M
PR_base = participation_ratio(out_base)
print(f"Participation Ratio (base psi): {PR_base:.2f} / max {M}")

# Also compute PR for several random seeds
prs = []
for seed in range(10):
    rng = np.random.RandomState(seed+7)
    noise = 0.01 * (rng.randn(L) + 1j*rng.randn(L))
    psi_noisy = base_psi * np.exp(1j*0.02*rng.randn(L)) + noise
    psi_noisy = psi_noisy / np.sqrt(np.sum(np.abs(psi_noisy)**2))
    outn = Wc @ psi_noisy
    prs.append(participation_ratio(outn))
print("PRs for 10 noisy variants (approx):", np.round(prs,2))

# Save PR summary
with open(output_prefix + '_PR_summary.txt','w') as f:
    f.write(f"PR_base: {PR_base}\n")
    f.write("PRs noisy:\n")
    f.write("\n".join(map(str, prs)))
print("Saved PR summary to", output_prefix + '_PR_summary.txt')

# -------------- Visualize energy in hidden tensor --------------
out_resh = out_base.reshape(L, k, k, k)  # shape (L,k,k,k)
energy = np.abs(out_resh)**2  # shape (L,k,k,k)

# Sum energy over base dimension L -> energy per (k,k,k)
hidden_energy = np.sum(energy, axis=0)  # shape (k,k,k)
# Visualize marginal sums along axes
marg0 = np.sum(hidden_energy, axis=(1,2))  # length k (axis along first hidden dim)
marg1 = np.sum(hidden_energy, axis=(0,2))
marg2 = np.sum(hidden_energy, axis=(0,1))

plt.figure(figsize=(10,3))
plt.subplot(1,3,1); plt.bar(np.arange(k), marg0); plt.title('Hidden dim 0 marginal'); plt.grid(True)
plt.subplot(1,3,2); plt.bar(np.arange(k), marg1); plt.title('Hidden dim 1 marginal'); plt.grid(True)
plt.subplot(1,3,3); plt.bar(np.arange(k), marg2); plt.title('Hidden dim 2 marginal'); plt.grid(True)
plt.tight_layout()
plt.savefig(output_prefix + '_hidden_marginals.png', dpi=200)
print("Saved hidden marginals:", output_prefix + '_hidden_marginals.png')

# Show a heatmap of flattened hidden-energy projection (collapse k,k to a grid)
flat_hidden = hidden_energy.reshape(k, k*k)  # shape (k, k^2)
plt.figure(figsize=(6,4))
plt.imshow(flat_hidden, aspect='auto')
plt.colorbar()
plt.title('Hidden energy (collapsed slice)') 
plt.xlabel('flattened indices'); plt.ylabel('hidden dim 0 index')
plt.tight_layout()
plt.savefig(output_prefix + '_hidden_heatmap.png', dpi=200)
print("Saved hidden heatmap:", output_prefix + '_hidden_heatmap.png')

# -------------- Inspect top singular vectors --------------
# U has shape (M, min(M,L)) and Vh has shape (min(M,L), L)
# Visualize first few left singular vectors reshaped to (L,k,k,k)
nplot = min(6, U.shape[1])
plt.figure(figsize=(12, 2*nplot))
for i in range(nplot):
    vec = U[:, i]  # size M
    try:
        vec_resh = vec.reshape(L, k, k, k)
        # show sum over L (energy per hidden position)
        img = np.sum(np.abs(vec_resh)**2, axis=0)  # shape (k,k,k)
        # flatten to 2D for display
        img2 = img.reshape(k, k*k)
        ax = plt.subplot(nplot, 1, i+1)
        ax.imshow(img2, aspect='auto')
        ax.set_title(f'Left singular vec {i} energy (flattened hidden dims)')
        ax.set_ylabel('mode {}'.format(i))
    except Exception as e:
        print("Couldn't reshape singular vec", i, "->", e)
plt.tight_layout()
plt.savefig(output_prefix + '_singvecs_hidden.png', dpi=200)
print("Saved singular vector hidden visualizations:", output_prefix + '_singvecs_hidden.png')

# -------------- Compare with random baseline --------------
# Build a random orthonormal-like complex W for comparison with same norm
rng = np.random.RandomState(42)
W_rand = rng.randn(M, L) + 1j * rng.randn(M, L)
# normalize singular values scale similar to trained W: scale random W to match Frobenius norm
scale = np.linalg.norm(Wc) / np.linalg.norm(W_rand)
W_rand *= scale

# PR for the same base psi
out_rand = W_rand @ base_psi
PR_rand = participation_ratio(out_rand)
print(f"Random baseline PR: {PR_rand:.2f}")

# singular values of random
_, s_rand, _ = svd(W_rand, full_matrices=False)

plt.figure(figsize=(8,4))
plt.semilogy(sigma, label='trained')
plt.semilogy(s_rand, label='random baseline', alpha=0.7)
plt.title('Singular values: trained vs random baseline')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_prefix + '_svd_trained_vs_random.png', dpi=200)
print("Saved SVD comparison:", output_prefix + '_svd_trained_vs_random.png')

# ------------- Summary printout -------------
print("\n--- SUMMARY ---")
print("W_complex shape:", Wc.shape)
print("Singular values (first 10):", np.round(sigma[:10], 6))
print("Cumulative energy top-10:", np.round(np.sum(sigma[:10]**2)/np.sum(sigma**2), 4))
print("Participation ratio (base psi):", PR_base)
print("Participation ratio (random baseline):", PR_rand)

# Save some numeric results to CSV/text
with open(output_prefix + '_numeric_summary.txt','w') as f:
    f.write("Singular values (first 50):\n")
    f.write(",".join(map(str, sigma[:50].tolist())) + "\n\n")
    f.write(f"PR_base: {PR_base}\n")
    f.write(f"PR_random: {PR_rand}\n")
    f.write("PRs noisy:\n")
    f.write(",".join(map(str, prs)) + "\n")
print("Saved numeric summary:", output_prefix + '_numeric_summary.txt')

print("\nAll done. Generated files (prefix):", output_prefix, "\nLook for PNGs and numeric summary in current directory.")
