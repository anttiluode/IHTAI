# Full experiment A (wide gamma sweep + longer runs + animations near critical gamma)
# - Tensor unfolding (1D->4D)
# - Wider gamma grid (0.0 to 0.2, denser near 0)
# - Larger grid L=256, longer evolution t_evolve=1000
# - Ensemble seeds = 6
# - Metrics: half-life (50% and 10%), area under peak, final coherence, survival fraction
# - Find approximate critical gamma where survival fraction drops, produce animations at gamma just below/above that
#
# NOTE: This is a heavier run; it may take a bit of time. I've tuned parameters to be reasonably heavy but runnable.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
# from IPython.display import HTML # REMOVED: display() is not available in standard Python

np.random.seed(0)

# helpers
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

# simulation params (heavier)
L = 256
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

# gamma grid: finer near 0 and extend to 0.2
gamma_vals = np.concatenate((np.linspace(0.0, 0.02, 11), np.linspace(0.025, 0.05, 10), np.linspace(0.06, 0.2, 8)))
seeds = list(range(6))

t_evolve = 1000
time = np.arange(t_evolve)

# storage arrays
half50 = np.full((len(gamma_vals), len(seeds)), np.inf)
half10 = np.full_like(half50, np.inf)
areas = np.zeros_like(half50, dtype=float)
final_coh = np.zeros_like(areas)
survived = np.zeros((len(gamma_vals), len(seeds)), dtype=bool)

# run ensemble
for gi, gamma in enumerate(gamma_vals):
    for si, sd in enumerate(seeds):
        np.random.seed(sd + 1000)
        psi = base_psi.copy()
        # expand
        for kern in kernels:
            psi = expand_dimension_tensor(psi, kern)
            for _ in range(ticks_per_expansion):
                psi = local_phase_rotation(psi, strength=strength + 0.01*np.random.randn())
                psi = psi * (1.0 - gamma)
        proj0, density0, phase0 = marginal_project_to_1d(psi)
        peak0 = np.max(density0)
        peaks = np.zeros(t_evolve)
        coherence_ts = np.zeros(t_evolve)
        for t in range(t_evolve):
            psi = local_phase_rotation(psi, strength=strength + 0.01*np.random.randn())
            psi = psi * (1.0 - gamma)
            proj, density, phase = marginal_project_to_1d(psi)
            peaks[t] = np.max(density)
            coherence_ts[t] = np.abs(np.mean(np.exp(1j * phase)))
        # compute metrics
        below50 = np.where(peaks <= 0.5 * peak0)[0]
        half50[gi,si] = below50[0] if below50.size>0 else np.inf
        below10 = np.where(peaks <= 0.1 * peak0)[0]
        half10[gi,si] = below10[0] if below10.size>0 else np.inf
        # FIX: Replaced deprecated np.trapz with np.trapezoid
        areas[gi,si] = np.trapezoid(peaks, dx=1.0)
        final_coh[gi,si] = coherence_ts[-1]
        survived[gi,si] = np.isinf(half50[gi,si])

# aggregate
median_half50 = np.nanmedian(np.where(np.isfinite(half50), half50, np.nan), axis=1)
median_half10 = np.nanmedian(np.where(np.isfinite(half10), half10, np.nan), axis=1)
median_area = np.median(areas, axis=1)
median_final_coh = np.median(final_coh, axis=1)
survival_fraction = np.mean(survived, axis=1)

# find approximate critical gamma where survival fraction drops below 0.5
crit_idx = np.where(survival_fraction < 0.5)[0]
crit_gamma = gamma_vals[crit_idx[0]] if crit_idx.size>0 else None

# Plot metrics (will display a static plot)
fig, ax = plt.subplots(2,2, figsize=(12,8))
ax[0,0].plot(gamma_vals, median_half50, '-o'); ax[0,0].set_title('median half-life (50%)'); ax[0,0].set_xlabel('gamma'); ax[0,0].grid(True)
ax[0,1].plot(gamma_vals, median_half10, '-o'); ax[0,1].set_title('median half-life (10%)'); ax[0,1].set_xlabel('gamma'); ax[0,1].grid(True)
ax[1,0].plot(gamma_vals, median_area, '-o'); ax[1,0].set_title('median area under peak'); ax[1,0].set_xlabel('gamma'); ax[1,0].grid(True)
ax[1,1].plot(gamma_vals, survival_fraction, '-o'); ax[1,1].set_title('survival fraction (half50 infinite)'); ax[1,1].set_xlabel('gamma'); ax[1,1].grid(True)
plt.tight_layout()

# Save the plot since display may not work or is undesired in a script environment
plt.savefig('ihtgraphs_metrics_plot.png')
plt.show() # Still call show, as some environments (like Spyder) can still show it.

print("Approx critical gamma (survival fraction < 0.5):", crit_gamma)

# choose gamma below and above critical for animations
if crit_gamma is None:
    gamma_below = gamma_vals[len(gamma_vals)//3]
    gamma_above = gamma_vals[2*len(gamma_vals)//3]
else:
    # Find the index of crit_gamma
    try:
        idx = np.where(np.isclose(gamma_vals, crit_gamma))[0][0]
    except IndexError:
        # Fallback if crit_gamma isn't perfectly in gamma_vals
        idx = np.searchsorted(gamma_vals, crit_gamma)
    
    gamma_below = gamma_vals[max(0, idx-1)]
    gamma_above = gamma_vals[min(len(gamma_vals)-1, idx)] if survival_fraction[idx] < 0.5 else gamma_vals[min(len(gamma_vals)-1, idx+1)]


print("Animation gammas:", gamma_below, gamma_above)

# MODIFICATION: Reduced capture_evol from 300 to 100 to prevent animation size limit warning
CAPTURE_EVOL = 100 

# function to build frames for a given gamma and seed
def build_frames_for_gamma(gamma, seed, capture_evol=CAPTURE_EVOL):
    np.random.seed(seed + 2000)
    psi = base_psi.copy()
    frames = []
    # initial
    proj, density, phase = marginal_project_to_1d(psi)
    frames.append((density.copy(), phase.copy(), '1D initial'))
    # expansion ticks
    for ei, kern in enumerate(kernels):
        psi = expand_dimension_tensor(psi, kern)
        for tt in range(ticks_per_expansion):
            psi = local_phase_rotation(psi, strength=strength)
            psi = psi * (1.0 - gamma)
            proj, density, phase = marginal_project_to_1d(psi)
            frames.append((density.copy(), phase.copy(), f'exp{ei+1} tick{tt+1}'))
    # evolution capture
    for tt in range(capture_evol):
        psi = local_phase_rotation(psi, strength=strength)
        psi = psi * (1.0 - gamma)
        proj, density, phase = marginal_project_to_1d(psi)
        frames.append((density.copy(), phase.copy(), f'evol{tt+1}'))
    return frames

# Build animations for gamma_below and gamma_above with representative seed 0
frames_below = build_frames_for_gamma(gamma_below, seed=0, capture_evol=CAPTURE_EVOL)
frames_above = build_frames_for_gamma(gamma_above, seed=0, capture_evol=CAPTURE_EVOL)

# Render animation for gamma_below (density + phase)
def render_animation(frames, title_prefix):
    # This function now returns the animation object directly.
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))
    line1, = ax1.plot(x, frames[0][0], color='C1'); ax1.set_ylim(0,1.05); ax1.set_xlim(-1,1); ax1.grid(True)
    line2, = ax2.plot(x, frames[0][1], color='C2'); ax2.set_ylim(-4,4); ax2.set_xlim(-1,1); ax2.grid(True)
    ax1.set_title(frames[0][2]); ax2.set_title('phase')

    def animate_func(i):
        d, p, ttl = frames[i]
        line1.set_ydata(d)
        line2.set_ydata(p)
        ax1.set_title(f"{title_prefix}: {ttl}")
        return line1, line2

    anim = animation.FuncAnimation(fig, animate_func, frames=len(frames), interval=60, blit=True)
    plt.close(fig) # Close figure to avoid plotting many empty frames
    return anim

print("Rendering animation for gamma_below:", gamma_below)
anim_below = render_animation(frames_below, f"gamma={gamma_below:.4f}")
# FIX: Use .save() to write the animation to a file instead of using display()
# Note: You may need to install the 'pillow' library (`pip install pillow`) to save as GIF.
try:
    anim_below.save(f"animation_gamma_{gamma_below:.4f}.gif", writer='pillow', fps=15)
    print(f"-> Saved animation for gamma_below to animation_gamma_{gamma_below:.4f}.gif")
except Exception as e:
    print(f"-> WARNING: Could not save GIF for gamma_below. Ensure 'pillow' is installed. Error: {e}")

print("Rendering animation for gamma_above:", gamma_above)
anim_above = render_animation(frames_above, f"gamma={gamma_above:.4f}")
try:
    anim_above.save(f"animation_gamma_{gamma_above:.4f}.gif", writer='pillow', fps=15)
    print(f"-> Saved animation for gamma_above to animation_gamma_{gamma_above:.4f}.gif")
except Exception as e:
    print(f"-> WARNING: Could not save GIF for gamma_above. Ensure 'pillow' is installed. Error: {e}")


# Print summary diagnostics
print("\n--- Summary ---")
print("Gamma grid length:", len(gamma_vals))
print("Median half50 (first 10):", median_half50[:10])
print("Survival fraction (first 10):", survival_fraction[:10])
print("Animations saved as GIF files in the current directory.")