# Bundle run: robust gamma scan (half-life), coherence vs gamma, and animation near threshold.
# - Uses tensor-product unfolding like before.
# - Measures half-life (time until projected peak falls below 50% of initial peak) and area-under-peak.
# - Computes coherence |<e^{i theta}>| over time and records median final coherence across seeds.
# - Picks gamma around transition and makes an animation showing density+phase evolution.
# Compact settings chosen so it runs reasonably fast but produces clear trends.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

np.random.seed(0)

# Helpers (similar to earlier)
def gaussian_1d(x, mu=0.0, sigma=0.08):
    return np.exp(-0.5 * ((x - mu)/sigma)**2)

def make_kernel(ksize=4, freq=4.0, phase_offset=0.0):
    coords = np.linspace(-1,1,ksize)
    amp = np.exp(-coords**2 * 4.0)
    phase = np.exp(1j * (freq * coords + phase_offset))
    return amp * phase

def expand_dimension_tensor(psi, kernel):
    new_psi = np.tensordot(psi, kernel, axes=0)
    norm = np.sqrt(np.sum(np.abs(new_psi)**2))
    return new_psi / (norm + 1e-16)

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
    density = density / (np.max(density) + 1e-16)
    phase = np.angle(proj)
    return proj, density, phase

# Simulation params
L = 128
x = np.linspace(-1,1,L)
base_psi = gaussian_1d(x, mu=0.0, sigma=0.08)
base_psi = base_psi * np.exp(1j * 0.15 * np.sin(5*x))
base_psi = base_psi / np.sqrt(np.sum(np.abs(base_psi)**2))

kernel_size = 4
kern1 = make_kernel(ksize=kernel_size, freq=3.5, phase_offset=0.0)
kern2 = make_kernel(ksize=kernel_size, freq=4.5, phase_offset=0.6)
kern3 = make_kernel(ksize=kernel_size, freq=2.8, phase_offset=-0.4)
kernels = (kern1, kern2, kern3)

ticks_per_expansion = 6
strength = 0.18

# gamma grid focused around transition
gamma_vals = np.concatenate((np.linspace(0.0, 0.02, 6), np.linspace(0.025, 0.06, 8)))
seeds = [0,1,2,3,4]  # ensemble seeds

t_evolve = 240  # evolution steps to measure half-life and coherence
time = np.arange(t_evolve)

# storage
half_lives = np.full((len(gamma_vals), len(seeds)), np.nan)
areas = np.full_like(half_lives, np.nan)
final_coherence = np.full_like(half_lives, np.nan)
coherence_time_series = np.zeros((len(gamma_vals), len(seeds), t_evolve))

for gi, gamma in enumerate(gamma_vals):
    for si, sd in enumerate(seeds):
        np.random.seed(sd + 100)  # vary seed bank
        psi = base_psi.copy()
        # expand (with small jitter per expansion to break symmetry)
        for kern in kernels:
            psi = expand_dimension_tensor(psi, kern)
            for _ in range(ticks_per_expansion):
                psi = local_phase_rotation(psi, strength=strength + 0.01*np.random.randn())
                psi = psi * (1.0 - gamma)
        # record initial peak
        proj0, density0, phase0 = marginal_project_to_1d(psi)
        peak0 = np.max(density0)
        # evolve and record metrics
        peaks = np.zeros(t_evolve)
        coherence_ts = np.zeros(t_evolve)
        for t in range(t_evolve):
            psi = local_phase_rotation(psi, strength=strength + 0.01*np.random.randn())
            psi = psi * (1.0 - gamma)
            proj, density, phase = marginal_project_to_1d(psi)
            peaks[t] = np.max(density)
            coherence_ts[t] = np.abs(np.mean(np.exp(1j * phase)))
        # half-life: first t where peak <= 0.5 * peak0 (if never, set to inf-like)
        below_idx = np.where(peaks <= 0.5 * peak0)[0]
        hl = below_idx[0] if below_idx.size>0 else np.inf
        half_lives[gi,si] = hl
        areas[gi,si] = np.trapz(peaks, dx=1.0)
        final_coherence[gi,si] = coherence_ts[-1]
        coherence_time_series[gi,si,:] = coherence_ts

# Aggregate metrics across seeds: median half-life, median area, median final coherence
median_half = np.nanmedian(half_lives, axis=1)
median_area = np.nanmedian(areas, axis=1)
median_final_coh = np.nanmedian(final_coherence, axis=1)

# Find gamma where median_half drops below threshold (e.g., 20 timesteps)
threshold = 20
crit_idx = np.where(median_half < threshold)[0]
crit_gamma = gamma_vals[crit_idx[0]] if crit_idx.size>0 else None

# Plot results: half-life, area, final coherence vs gamma
fig, ax = plt.subplots(1,3,figsize=(15,4))
ax[0].plot(gamma_vals, median_half, '-o'); ax[0].set_xlabel('gamma'); ax[0].set_ylabel('median half-life (timesteps)'); ax[0].set_title('Half-life vs gamma'); ax[0].grid(True)
ax[1].plot(gamma_vals, median_area, '-o'); ax[1].set_xlabel('gamma'); ax[1].set_ylabel('median area under peak'); ax[1].set_title('Area vs gamma'); ax[1].grid(True)
ax[2].plot(gamma_vals, median_final_coh, '-o'); ax[2].set_xlabel('gamma'); ax[2].set_ylabel('median final coherence'); ax[2].set_title('Final coherence vs gamma'); ax[2].grid(True)
plt.tight_layout()
plt.show()

print("Estimated critical gamma (median half-life < {}): {}".format(threshold, crit_gamma))

# Choose animation gamma: pick the gamma just below the critical gamma (if exists) or mid-grid
if crit_gamma is not None:
    # choose gamma just below critical if possible
    cind = max(0, crit_idx[0]-1)
    anim_gamma = gamma_vals[cind]
else:
    anim_gamma = gamma_vals[len(gamma_vals)//2]
print("Animation gamma chosen:", anim_gamma)

# Build animation frames at anim_gamma for a representative seed
np.random.seed(42)
psi = base_psi.copy()
frames = []
titles = []

# initial
proj, density, phase = marginal_project_to_1d(psi)
frames.append((density.copy(), phase.copy())); titles.append('1D initial')

# expansion + ticks
for ei, kern in enumerate(kernels):
    psi = expand_dimension_tensor(psi, kern)
    for tt in range(ticks_per_expansion):
        psi = local_phase_rotation(psi, strength=strength)
        psi = psi * (1.0 - anim_gamma)
        proj, density, phase = marginal_project_to_1d(psi)
        frames.append((density.copy(), phase.copy())); titles.append(f'exp {ei+1} tick {tt+1}')

# evolution frames
for tt in range(120):
    psi = local_phase_rotation(psi, strength=strength)
    psi = psi * (1.0 - anim_gamma)
    proj, density, phase = marginal_project_to_1d(psi)
    frames.append((density.copy(), phase.copy())); titles.append(f'evol {tt+1}')

# Create animation (density + phase)
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))
line1, = ax1.plot(x, frames[0][0], color='C1'); ax1.set_ylim(0,1.05); ax1.set_xlim(-1,1); ax1.grid(True)
line2, = ax2.plot(x, frames[0][1], color='C2'); ax2.set_ylim(-4,4); ax2.set_xlim(-1,1); ax2.grid(True)
ax1.set_title(titles[0]); ax2.set_title('phase')

def animate_func(i):
    density, phase = frames[i]
    line1.set_ydata(density)
    line2.set_ydata(phase)
    ax1.set_title(titles[i])
    return line1, line2

anim = animation.FuncAnimation(fig, animate_func, frames=len(frames), interval=80, blit=True)
plt.close(fig)
html_anim = HTML(anim.to_jshtml())
display(html_anim)

# show a few static snapshots
for idx in [0, 3, 6, 9, 15, 30, 80, len(frames)-1]:
    d, p = frames[idx]
    fig, ax = plt.subplots(1,2,figsize=(10,3))
    ax[0].plot(x, d); ax[0].set_title(f'{titles[idx]} density'); ax[0].set_ylim(0,1.05); ax[0].grid(True)
    ax[1].plot(x, p); ax[1].set_title(f'{titles[idx]} phase'); ax[1].set_ylim(-4,4); ax[1].grid(True)
    plt.show()

# print brief diagnostics
print("Gamma grid:", gamma_vals)
print("Median half-lives:", median_half)
print("Median final coherence:", median_final_coh)
