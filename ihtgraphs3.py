# Critical experiments: (1) Gamma (dilution) sweep to find threshold; (2) Animation of expansion showing phase+density evolution.
# We'll run compact but informative simulations with multiple seeds, measure peak decay time constant (tau) for each gamma,
# and produce plots. Then create an animation for one representative gamma showing expansion and phase evolution.
#
# Uses numpy + matplotlib. No external downloads. Keeps sizes small for speed.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import animation, rc
from IPython.display import HTML

np.random.seed(0)

# -- helpers --
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

def local_phase_rotation(psi, strength=0.2):
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

def fit_exponential_decay(t, y):
    # fit y = A * exp(-t/tau) + C ; require positive tau. We'll fit on log for initial estimate.
    try:
        # shift so min ~0 for stability
        ymin = np.min(y)
        y0 = y - ymin + 1e-12
        # fit log-linear
        logy = np.log(y0)
        p = np.polyfit(t, logy, 1)
        tau_init = -1.0 / p[0]
        A_init = np.exp(p[1])
        C_init = ymin
        popt, _ = curve_fit(lambda tt, A, tau, C: A * np.exp(-tt / tau) + C, t, y, p0=[A_init, tau_init, C_init], maxfev=2000)
        return popt  # A, tau, C
    except Exception as e:
        return None

# -- simulation params --
L = 128
x = np.linspace(-1,1,L)
base_psi = gaussian_1d(x, mu=0.0, sigma=0.08)
base_psi = base_psi * np.exp(1j * 0.2 * np.sin(5*x))
base_psi = base_psi / np.sqrt(np.sum(np.abs(base_psi)**2))

# kernels
kernel_size = 4
kern1 = make_kernel(ksize=kernel_size, freq=3.5, phase_offset=0.0)
kern2 = make_kernel(ksize=kernel_size, freq=4.5, phase_offset=0.6)
kern3 = make_kernel(ksize=kernel_size, freq=2.8, phase_offset=-0.4)
kernels = (kern1, kern2, kern3)

# evolution settings
ticks_per_expansion = 6  # small per expansion
total_steps = 3 * ticks_per_expansion  # total internal ticks after full expansion
strength = 0.18

# gamma sweep and seeds
gamma_vals = np.concatenate((np.linspace(0.0, 0.02, 9), np.linspace(0.03, 0.1, 8)))  # finer near 0
seeds = [0,1,2,3]  # ensemble seeds for robustness

# storage
taus = np.zeros((len(gamma_vals), len(seeds))) * np.nan
final_peaks = np.zeros_like(taus)

# run sweep: for each gamma and seed, expand then evolve for many steps, record peak vs time
t_evolve = 120  # time steps to record decay curve (longer to catch slow tau)
time_array = np.arange(t_evolve)

for gi, gamma in enumerate(gamma_vals):
    for si, sd in enumerate(seeds):
        np.random.seed(sd)
        psi = base_psi.copy()
        # expand dims tensorwise
        for kern in kernels:
            psi = expand_dimension_tensor(psi, kern)
            # small random phase jitter per seed for variability
            for _ in range(ticks_per_expansion):
                psi = local_phase_rotation(psi, strength=strength + 0.02*np.random.randn())
                # apply decoherence each internal tick in expansion too
                psi = psi * (1.0 - gamma)
        # now evolve further for t_evolve steps under phase rotation + decoherence and record peak
        peaks = np.zeros(t_evolve)
        for t in range(t_evolve):
            psi = local_phase_rotation(psi, strength=strength + 0.02*np.random.randn())
            psi = psi * (1.0 - gamma)
            proj, density, phase = marginal_project_to_1d(psi)
            peaks[t] = np.max(density)
        # fit exponential decay to peaks over time window (skip initial transient a bit)
        fit = fit_exponential_decay(time_array[10:], peaks[10:])
        if fit is not None:
            A, tau, C = fit
            taus[gi,si] = tau
        final_peaks[gi,si] = peaks[-1]

# summarize tau vs gamma (median across seeds)
median_tau = np.nanmedian(taus, axis=1)
mean_final_peak = np.mean(final_peaks, axis=1)

# Plot tau vs gamma and final peak vs gamma
fig, ax = plt.subplots(1,2,figsize=(12,4))
ax[0].plot(gamma_vals, median_tau, '-o')
ax[0].set_xlabel('gamma (dilution)')
ax[0].set_ylabel('median tau (fit)')
ax[0].set_title('Median decay time constant vs gamma (ensemble seeds)')
ax[0].grid(True)

ax[1].plot(gamma_vals, mean_final_peak, '-o')
ax[1].set_xlabel('gamma (dilution)')
ax[1].set_ylabel('mean final projected peak')
ax[1].set_title('Final projected peak after evolution vs gamma')
ax[1].grid(True)

plt.show()

# Estimate critical gamma where tau drops dramatically (heuristic): where median_tau < 10
crit_indices = np.where(median_tau < 10)[0]
crit_gamma = gamma_vals[crit_indices[0]] if crit_indices.size>0 else None

print("Estimated critical gamma (tau<10):", crit_gamma)

# -----------------
# Animation for a representative gamma (near threshold if found, otherwise mid)
rep_gamma = crit_gamma if crit_gamma is not None else 0.01
print("Representative gamma for animation:", rep_gamma)

# Build animation states: start 1D then expand and run evolution, capturing frames
np.random.seed(7)
psi = base_psi.copy()
frames = []
titles = []

# store initial 1D
proj, density, phase = marginal_project_to_1d(psi)
frames.append((density.copy(), phase.copy())); titles.append('1D initial')

# expansion + ticks, capture after each internal tick
for ei, kern in enumerate(kernels):
    psi = expand_dimension_tensor(psi, kern)
    for tt in range(ticks_per_expansion):
        psi = local_phase_rotation(psi, strength=strength)
        psi = psi * (1.0 - rep_gamma)
        proj, density, phase = marginal_project_to_1d(psi)
        frames.append((density.copy(), phase.copy()))
        titles.append(f'exp {ei+1} tick {tt+1}')

# continue evolving for extra steps to show decay
for tt in range(40):
    psi = local_phase_rotation(psi, strength=strength)
    psi = psi * (1.0 - rep_gamma)
    proj, density, phase = marginal_project_to_1d(psi)
    frames.append((density.copy(), phase.copy()))
    titles.append(f'evol {tt+1}')

# Create animation: two subplots density and phase
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))
line1, = ax1.plot(x, frames[0][0], color='C1')
ax1.set_title(titles[0]); ax1.set_ylim(0,1.05); ax1.set_xlim(-1,1); ax1.grid(True)
line2, = ax2.plot(x, frames[0][1], color='C1')
ax2.set_title('phase'); ax2.set_ylim(-4,4); ax2.set_xlim(-1,1); ax2.grid(True)

def animate_func(i):
    density, phase = frames[i]
    line1.set_ydata(density)
    line2.set_ydata(phase)
    ax1.set_title(titles[i])
    return line1, line2

anim = animation.FuncAnimation(fig, animate_func, frames=len(frames), interval=120, blit=True)
plt.close(fig)  # prevent duplicate static display

# Display the animation inline
html_anim = HTML(anim.to_jshtml())
display(html_anim)

# Also show a few snapshots as static plots for reference
for idx in [0, 3, 6, 9, 15, 30, 50, len(frames)-1]:
    density, phase = frames[idx]
    fig, ax = plt.subplots(1,2,figsize=(10,3))
    ax[0].plot(x, density); ax[0].set_title(f'{titles[idx]} density'); ax[0].set_ylim(0,1.05); ax[0].grid(True)
    ax[1].plot(x, phase); ax[1].set_title(f'{titles[idx]} phase'); ax[1].set_ylim(-4,4); ax[1].grid(True)
    plt.show()

# Print short diagnostics
print("Median tau vs gamma (first 10):")
for g, t in zip(gamma_vals[:10], median_tau[:10]):
    print(f" gamma={g:.4f} -> median tau={t:.2f}")
print("...")
print("Representative gamma used for animation:", rep_gamma)
