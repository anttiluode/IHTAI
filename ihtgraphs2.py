# Mixed run: parameter sweeps + decoherence + learned mixing expansion
# - Sweeps over kernel frequency and phase rotation strength
# - Adds decoherence (multiplicative damping) per tick
# - Compares tensor-product expansion vs learned linear mixing expansion
# - Computes metrics: peak density, entropy of projected density, coherence (|<e^{iθ}>|)
# - Plots heatmaps and example projected density/phase traces
#
# Run in notebook. Uses matplotlib. No external internet.


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy

np.random.seed(1)

# base 1D grid
L = 256
x = np.linspace(-1,1,L)

def gaussian_1d(x, mu=0.0, sigma=0.08):
    return np.exp(-0.5 * ((x - mu)/sigma)**2)

def make_kernel(ksize=6, freq=3.0, phase_offset=0.0):
    coords = np.linspace(-1,1,ksize)
    amp = np.exp(-coords**2 * 4.0)
    phase = np.exp(1j * (freq * coords + phase_offset))
    return amp * phase

def expand_dimension_tensor(psi, kernel):
    new_psi = np.tensordot(psi, kernel, axes=0)
    norm = np.sqrt(np.sum(np.abs(new_psi)**2))
    return new_psi / (norm + 1e-16)

def expand_dimension_learned(psi, shape_out, W):
    # psi shape (..., n); we flatten all but last into vector and apply linear mixing to create extra dims
    # Here, simple approach: reshape to vector, multiply by W, reshape to desired shape_out
    vec = psi.reshape(-1)
    out = W @ vec
    out = out.reshape(shape_out)
    out = out / np.sqrt(np.sum(np.abs(out)**2) + 1e-16)
    return out

def local_phase_rotation(psi, strength=0.5):
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
    density = density / (np.max(density) + 1e-16)
    phase = np.angle(proj)
    return proj, density, phase

def shannon_entropy_normalized(p):
    p = np.array(p, dtype=float)
    p = p / (p.sum() + 1e-16)
    # small epsilon to avoid log(0)
    return scipy_entropy(p + 1e-16, base=2) / np.log2(len(p))

# initial 1D psi
psi1 = gaussian_1d(x, mu=0.0, sigma=0.08)
psi1 = psi1 * np.exp(1j * 0.2 * np.sin(5*x) + 1j*0.05*np.random.randn(L))
psi1 = psi1 / np.sqrt(np.sum(np.abs(psi1)**2))

# parameter ranges
freqs = np.array([2.0, 3.5, 5.0, 7.0])   # kernel frequencies
strengths = np.array([0.05, 0.1, 0.2, 0.4])  # phase rotation strengths
gamma_vals = np.array([0.0, 0.002, 0.01])  # decoherence (multiplicative damping per tick)

# results storage: for tensor and learned
results_tensor = np.zeros((len(gamma_vals), len(freqs), len(strengths), 3))  # metrics: peak, entropy, coherence
results_learned = np.zeros_like(results_tensor)

# we will run 3 expansion steps (1D->2D->3D->4D) as before, with small internal ticks each step
ticks_per_expansion = 6
kernel_size = 6

# prepare a random orthonormal mixing matrix for learned expansion scenario
# target shape: (L, kernel_size, kernel_size, kernel_size) flattened size = L * k^3
target_shape = (L, kernel_size, kernel_size, kernel_size)
flat_size = np.prod(target_shape)
# for computational feasibility, create a smaller learned mapping: map initial L vector -> L * k^3 vector
# using a random Gaussian matrix then orthonormalize via QR on a smaller dimension to keep memory low
randA = np.random.randn(flat_size, L) * 0.02
# Orthonormalize rows approximately using QR on randA @ randA.T (keeps size manageable)
Q, R = np.linalg.qr(randA)
W_learn = Q  # shape (flat_size, L)

# run sweeps
for gi, gamma in enumerate(gamma_vals):
    for fi, freq in enumerate(freqs):
        kern1 = make_kernel(ksize=kernel_size, freq=freq, phase_offset=0.0)
        kern2 = make_kernel(ksize=kernel_size, freq=freq*1.3, phase_offset=0.7)
        kern3 = make_kernel(ksize=kernel_size, freq=freq*0.7, phase_offset=-0.4)
        for si, strength in enumerate(strengths):
            # --- tensor expansion run ---
            psi = psi1.copy()
            for kern in (kern1, kern2, kern3):
                psi = expand_dimension_tensor(psi, kern)
                for _ in range(ticks_per_expansion):
                    psi = local_phase_rotation(psi, strength=strength)
                    # apply decoherence (multiplicative damping)
                    if gamma > 0:
                        psi = psi * (1.0 - gamma)
            proj, density, phase = marginal_project_to_1d(psi)
            peak = np.max(density)
            ent = shannon_entropy_normalized(density)
            coh = np.abs(np.mean(np.exp(1j*phase)))
            results_tensor[gi,fi,si,:] = [peak, ent, coh]

            # --- learned expansion run ---
            # apply learned linear mapping to create same target shape as tensor run
            psi_vec = psi1.reshape(-1)  # always start from initial for learned mapping
            psi_learn = expand_dimension_learned(psi_vec, target_shape, W_learn)
            for _ in range(3 * ticks_per_expansion):
                psi_learn = local_phase_rotation(psi_learn, strength=strength*0.9)
                if gamma > 0:
                    psi_learn = psi_learn * (1.0 - gamma)
            proj2, density2, phase2 = marginal_project_to_1d(psi_learn)
            peak2 = np.max(density2)
            ent2 = shannon_entropy_normalized(density2)
            coh2 = np.abs(np.mean(np.exp(1j*phase2)))
            results_learned[gi,fi,si,:] = [peak2, ent2, coh2]

# plot results: for each gamma, show heatmaps of peak/entropy/coherence across freq x strength
metrics = ['peak', 'entropy', 'coherence']
for mi in range(3):
    fig, axes = plt.subplots(2, len(gamma_vals), figsize=(4*len(gamma_vals), 6))
    for gi, gamma in enumerate(gamma_vals):
        # tensor heatmap
        data_t = results_tensor[gi,:,:,mi]
        im1 = axes[0,gi].imshow(data_t, origin='lower', aspect='auto', 
                                 extent=[strengths[0], strengths[-1], freqs[0], freqs[-1]])
        axes[0,gi].set_title(f"Tensor {metrics[mi]} (gamma={gamma})")
        axes[0,gi].set_xlabel('phase strength')
        axes[0,gi].set_ylabel('kernel freq')
        fig.colorbar(im1, ax=axes[0,gi])
        # learned heatmap
        data_l = results_learned[gi,:,:,mi]
        im2 = axes[1,gi].imshow(data_l, origin='lower', aspect='auto',
                                 extent=[strengths[0], strengths[-1], freqs[0], freqs[-1]])
        axes[1,gi].set_title(f"Learned {metrics[mi]} (gamma={gamma})")
        axes[1,gi].set_xlabel('phase strength')
        axes[1,gi].set_ylabel('kernel freq')
        fig.colorbar(im2, ax=axes[1,gi])
    plt.tight_layout()
    plt.show()

# Also show example projected density & phase for one interesting cell (middle gamma, middle freq, middle strength)
gi = len(gamma_vals)//2
fi = len(freqs)//2
si = len(strengths)//2

# recompute example states for visualization
freq = freqs[fi]; strength = strengths[si]; gamma = gamma_vals[gi]
kern1 = make_kernel(ksize=kernel_size, freq=freq, phase_offset=0.0)
kern2 = make_kernel(ksize=kernel_size, freq=freq*1.3, phase_offset=0.7)
kern3 = make_kernel(ksize=kernel_size, freq=freq*0.7, phase_offset=-0.4)

# tensor run visuals
psi = psi1.copy()
states_vis = []
states_vis.append(('1D initial', psi.copy()))
for kern in (kern1, kern2, kern3):
    psi = expand_dimension_tensor(psi, kern)
    for _ in range(ticks_per_expansion):
        psi = local_phase_rotation(psi, strength=strength)
        if gamma>0:
            psi = psi * (1.0-gamma)
    states_vis.append((f'{psi.ndim}D', psi.copy()))

# learned run visuals
psi_learn = expand_dimension_learned(psi1.reshape(-1), target_shape, W_learn)
for _ in range(3 * ticks_per_expansion):
    psi_learn = local_phase_rotation(psi_learn, strength=strength*0.9)
    if gamma>0:
        psi_learn = psi_learn * (1.0-gamma)
states_vis.append(('learned 4D', psi_learn.copy()))

# plot projected densities + phases for these states
for title, state in states_vis:
    proj, density, phase = marginal_project_to_1d(state)
    fig, ax = plt.subplots(1,2,figsize=(10,3))
    ax[0].plot(x, density); ax[0].set_title(f'Projected density - {title}'); ax[0].set_xlim(-1,1)
    ax[1].plot(x, phase); ax[1].set_title(f'Projected phase - {title}'); ax[1].set_xlim(-1,1)
    plt.show()

# Print a compact numerical summary for selected cell
print("Selected example metrics (gamma={}, freq={}, strength={}):".format(gamma, freq, strength))
print(" Tensor -> peak, entropy, coherence:", results_tensor[gi,fi,si,:])
print(" Learned-> peak, entropy, coherence:", results_learned[gi,fi,si,:])
