# Simulation of a "Quantum Walker" unfolding through holographic dimensions
# - Implements a complex wavefunction psi initially 1D (Gaussian)
# - At each "Planck tick" we expand into a higher dimension by tensoring with a small complex kernel
# - We apply a simple local phase rotation to simulate phase evolution in the high-dim substrate
# - Project observable back to 1D by summing over extra dims (marginalization) to produce the "perceived" density and phase
# - Displays the projected |psi|^2 and phase for dimensions 1..4 at a few ticks
#
# Run in notebook. Uses matplotlib (no seaborn), one plot per figure. 
# No external data or internet is required.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def gaussian_1d(x, mu=0.0, sigma=0.05):
    return np.exp(-0.5 * ((x - mu)/sigma)**2)

def make_kernel(ksize=8, freq=3.0, phase_offset=0.0):
    """Small complex oscillatory kernel used to expand dimensionality.
       It has both amplitude modulation and a phase factor to create phase structure."""
    coords = np.linspace(-1,1,ksize)
    amp = np.exp(-coords**2 * 4.0)  # localized amplitude
    phase = np.exp(1j * (freq * coords + phase_offset))
    return amp * phase

def expand_dimension(psi, kernel):
    """Tensor product expansion of psi with kernel to add one dimension.
       Normalizes the new wavefunction."""
    new_psi = np.tensordot(psi, kernel, axes=0)  # outer product -> adds an axis at end
    # normalize
    norm = np.sqrt(np.sum(np.abs(new_psi)**2))
    return new_psi / (norm + 1e-16)

def local_phase_rotation(psi, strength=0.5):
    """Apply a local, position-dependent phase rotation across all axes,
       simulating a Hermitian generator acting across the multi-dim substrate."""
    # build an axis grid for the tensor
    shape = psi.shape
    grids = np.meshgrid(*[np.linspace(-1,1,s) for s in shape], indexing='ij')
    # simple generator: sum of coordinates -> phase field
    phi = np.zeros_like(psi, dtype=np.float64)
    for g in grids:
        phi += g * 2.0  # weight of coordinate contributions
    # apply phase exponential; strength controls rotation per tick
    return psi * np.exp(1j * strength * phi)

def marginal_project_to_1d(psi):
    """Project high-dim psi to a 1D complex field by summing (coherent projection)
       over all but axis 0. Returns the projected complex amplitude and density."""
    if psi.ndim == 1:
        proj = psi
    else:
        # sum coherently across axes 1..N to keep phase relationships
        axes = tuple(range(1, psi.ndim))
        proj = np.sum(psi, axis=axes)
    density = np.abs(proj)**2
    # normalize density for plotting convenience
    density = density / (np.max(density) + 1e-16)
    return proj, density

# simulation parameters
L = 256  # resolution of base 1D axis
x = np.linspace(-1,1,L)
psi1 = gaussian_1d(x, mu=0.0, sigma=0.08)
# add a small random phase to start
psi1 = psi1 * np.exp(1j * 0.3 * np.sin(5*x) + 1j*0.1*np.random.randn(L))
# normalize
psi1 = psi1 / np.sqrt(np.sum(np.abs(psi1)**2))

# prepare kernels (different small kernels for each expansion)
kernels = [make_kernel(ksize=6, freq=3.0, phase_offset=0.0),
           make_kernel(ksize=6, freq=4.5, phase_offset=0.7),
           make_kernel(ksize=6, freq=2.1, phase_offset=-0.4)]

# store states for dims 1..4 after a few ticks
states = []
psi = psi1.copy()
states.append(('1D initial', psi.copy()))

# simulate a few planck ticks, expanding dimensionality each tick
for i, kern in enumerate(kernels):
    # expand to add one dimension (1D -> 2D -> 3D -> 4D)
    psi = expand_dimension(psi, kern)
    # apply a few local phase rotations to simulate evolution in the new substrate
    for _ in range(6):
        psi = local_phase_rotation(psi, strength=0.15 + 0.05*i)
    states.append((f'{psi.ndim}D after expansion {i+1}', psi.copy()))

# For each stored state, compute projected 1D amplitude & density and phase
projections = []
for title, state in states:
    proj_complex, density = marginal_project_to_1d(state)
    phase = np.angle(proj_complex)
    projections.append((title, density, phase, proj_complex))

# Plot results: one figure per stored state showing density and phase
for idx, (title, density, phase, proj_complex) in enumerate(projections):
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(x, density)
    ax.set_title(f'Projected density - {title}')
    ax.set_xlabel('x (projected)')
    ax.set_xlim(-1,1)
    ax.grid(True)
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(8,3))
    ax2.plot(x, phase)
    ax2.set_title(f'Projected phase - {title}')
    ax2.set_xlabel('x (projected)')
    ax2.set_xlim(-1,1)
    ax2.grid(True)
    plt.show()

# Additionally show a 2D slice (magnitude) for the 2D state to visualize internal structure
# Only if a 2D state exists in stored states
for title, state in states:
    if state.ndim == 2:
        mag = np.abs(state)**2
        fig, ax = plt.subplots(figsize=(5,5))
        im = ax.imshow(mag, origin='lower', extent=[-1,1,-1,1])
        ax.set_title('2D internal |psi|^2 structure for state: ' + title)
        plt.colorbar(im, ax=ax)
        plt.show()
        break

# Final: show effective dimensionality evolution and a tiny summary printout
print("Summary of stored states:")
for title, state in states:
    print(f" - {title}: ndim={state.ndim}, shape={state.shape}, max|psi|^2={np.max(np.abs(state)**2):.4e}")
