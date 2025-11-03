# knotiverse_interactive.py
# Interactive version with indefinite run, pause/resume, and detailed logging.
# Requires: numpy, scipy, matplotlib, json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import hilbert
import json
import time
import os
from matplotlib.widgets import Button # <<< Import Button widget

# ----------------------
# Simulation parameters
# ----------------------
L = 4096            # number of lattice sites
dt = 0.05             # nominal timestep
# steps = 10000       # REMOVED - runs indefinitely
record_window = 512   # history length
detect_threshold = 0.5  # knot threshold
coupling = 0.5        # neighbor coupling
nonlinear = 0.8       # self-interaction
damping = 0.005       # weak damping

# --- SPEED OPTIMIZATION ---
MAX_KNOTS_FOR_MDS = 500 # Cap for MDS

# Numerical stability parameters
MAX_AMPLITUDE_CLIP = 1e3 # Max amplitude
SATURATION_THRESHOLD = 2.0 # tanh divisor

# --- Logging Setup ---
LOG_PATH = "knotiverse_interactive_log.jsonl"
LOG_EVERY_N_FRAMES = 5 # Log every 5 animation frames
# Delete previous log file if it exists
if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)
    print(f"Removed previous log file: {LOG_PATH}")

# ----------------------
# Initialize 1D complex field psi
# ----------------------
rng = np.random.default_rng(1234)
x = np.arange(L)
psi = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) * 0.01
seed_positions = [L//4, L//2, 3*L//4]
for p in seed_positions:
    width = 4
    amp = 2.0 * (0.8 + 0.4 * rng.random())
    gauss = amp * np.exp(-((x - p)**2) / (2 * width**2))
    phase = np.exp(1j * 2.0 * np.pi * rng.random())
    psi += gauss * phase

history = np.zeros((L, record_window), dtype=np.complex128)
hist_ptr = 0
simulation_time = 0.0
current_frame_index = 0 # <<< Track frame index globally

# helper: discrete laplacian with periodic boundary
def laplacian_1d(arr):
    return np.roll(arr, -1) - 2*arr + np.roll(arr, 1)

# ----------------------
# Perceived embedding functions (classical MDS) - ROBUST VERSION
# (Unchanged)
# ----------------------
def pairwise_correlation_distance(series_matrix): #
    # ... (code from knotiverse4.py) ...
    X = series_matrix - np.mean(series_matrix, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std[std < 1e-9] = 1.0
    Xn = X / std
    C = (Xn @ Xn.T) / Xn.shape[1]
    C = np.nan_to_num(C, nan=0.0, posinf=1.0, neginf=-1.0)
    D = 1.0 - np.abs(C)
    return D, C

def classical_mds_from_dists(D, ndim=3, eps_reg=1e-10): #
    # ... (robust MDS code from knotiverse4.py) ...
    if not np.all(np.isfinite(D)):
        safe_max = np.nanmax(np.abs(D[np.isfinite(D)])) if np.sum(np.isfinite(D)) > 0 else 2.0
        D = np.nan_to_num(D, nan=safe_max + 1.0, posinf=safe_max + 1.0, neginf=safe_max + 1.0)
    n = D.shape[0]
    D2 = D**2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    trace = np.trace(B)
    ridge_scale = trace if trace != 0 else 1.0
    B_reg = B + eps_reg * ridge_scale * np.eye(n)
    try:
        vals, vecs = np.linalg.eigh(B_reg)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        pos = vals > 1e-10
        if np.sum(pos) == 0:
            return np.zeros((n, ndim))
        L_vals = np.sqrt(np.maximum(vals[pos][:ndim], 0.0))
        coords = vecs[:, pos][:, :ndim] * L_vals[:ndim]
        if coords.shape[1] < ndim:
            coords = np.pad(coords, ((0,0),(0, ndim-coords.shape[1])), mode='constant')
        return coords
    except np.linalg.LinAlgError:
        try:
            U, S, Vt = np.linalg.svd(B_reg)
            S_clip = np.maximum(S[:ndim], 0.0)
            coords = U[:, :ndim] * np.sqrt(S_clip)
            if coords.shape[1] < ndim:
                coords = np.pad(coords, ((0,0),(0, ndim-coords.shape[1])), mode='constant')
            return coords
        except Exception:
            return np.zeros((n, ndim))

# ----------------------
# Phase Winding Calculation Function
# (Unchanged)
# ----------------------
def compute_winding_for_site(phase_arr, site, radius=3): #
    # ... (code from knotiverse_logger.py) ...
    n = len(phase_arr)
    eff_radius = min(radius, n // 2 - 1)
    if eff_radius < 1: return 0
    idxs = [(site + i) % n for i in range(-eff_radius, eff_radius + 1)]
    idxs_loop = idxs + [idxs[0]]
    ph = phase_arr[idxs_loop]
    unwrapped_ph = np.unwrap(ph)
    total_phase_change = unwrapped_ph[-1] - unwrapped_ph[0]
    winding = int(np.round(total_phase_change / (2 * np.pi)))
    return winding

# ----------------------
# Visualization setup
# ----------------------
fig = plt.figure(figsize=(12, 7)) # Adjusted size for button
# Use GridSpec for better layout control
gs = fig.add_gridspec(2, 2, height_ratios=[10, 1])

ax_field = fig.add_subplot(gs[0, 0])
ax_embed = fig.add_subplot(gs[0, 1], projection='3d')
ax_controls = fig.add_subplot(gs[1, :]) # Span both columns for controls

ax_field.set_title("1D String amplitude |psi| (current)")
ax_embed.set_title("Perceived 3D embedding (Fast)")
line_field, = ax_field.plot([], [], lw=1)
ax_field.set_xlim(0, L-1)
ax_field.set_ylim(-0.1, 2.5)
ax_field.set_xlabel("site")
ax_field.set_ylabel("|psi| (real/implied)")
scatter_3d = None

# --- Add Pause/Play Button ---
pause_button_ax = ax_controls # Use the bottom subplot area
pause_button_pos = [0.45, 0.1, 0.1, 0.8] # [left, bottom, width, height] within ax_controls
pause_button = Button(pause_button_ax, 'Pause', color='lightgoldenrodyellow')
is_paused = False

def toggle_pause(event):
    global is_paused
    if is_paused:
        ani.resume()
        pause_button.label.set_text('Pause')
    else:
        ani.pause()
        pause_button.label.set_text('Resume')
    is_paused = not is_paused
    fig.canvas.draw_idle()

pause_button.on_clicked(toggle_pause)

# ----------------------
# Animation update (WITH LOGGING and LIFETIME TRACKING)
# ----------------------
sim_steps_per_frame = 6
# frame_count removed - runs indefinitely

# Dictionaries for tracking lifetimes
active_knots = {} # {site: birth_frame}
active_holes = {} # {site: birth_frame}

def update_frame(frame_idx_ignored): # Use global frame index instead
    global psi, history, hist_ptr, simulation_time, scatter_3d
    global active_knots, active_holes, current_frame_index

    current_frame_index += 1 # Increment global frame index

    # --- Run Simulation Steps ---
    frame_dt_accum = 0.0
    for _ in range(sim_steps_per_frame): #
        # ... (Simulation physics code - unchanged from knotiverse_logger.py) ...
        lap = laplacian_1d(psi)
        coupling_term = 1j * coupling * lap
        amp = np.abs(psi)
        sat = np.tanh(amp / SATURATION_THRESHOLD)
        nonlin_term = -1j * nonlinear * (sat**2) * psi
        damping_term = -damping * psi
        max_amp_now = np.max(amp)
        local_dt = dt
        if max_amp_now > 100.0: local_dt = dt * 0.1
        elif max_amp_now > 10.0: local_dt = dt * 0.5
        psi = psi + local_dt * (coupling_term + nonlin_term + damping_term)
        psi = np.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0)
        amp_new = np.abs(psi)
        over = amp_new > MAX_AMPLITUDE_CLIP
        if np.any(over): psi[over] = psi[over] * (MAX_AMPLITUDE_CLIP / amp_new[over])
        history[:, hist_ptr] = psi
        simulation_time += local_dt
        frame_dt_accum += local_dt
    hist_ptr = (hist_ptr + 1) % record_window #

    # --- Calculations for Plotting and Logging ---
    amp_now = np.abs(psi) #
    line_field.set_data(np.arange(L), amp_now) #

    # --- Knot detection ---
    left = np.roll(amp_now, 1) #
    right = np.roll(amp_now, -1) #
    mask_thresh = amp_now > detect_threshold #
    mask_local_max = (amp_now >= left) & (amp_now >= right) #
    all_detected_knots_idx = np.where(mask_thresh & mask_local_max)[0] #

    # --- Cap knots for MDS ---
    n_all_knots = len(all_detected_knots_idx) #
    knot_indices_for_mds = all_detected_knots_idx
    if n_all_knots > MAX_KNOTS_FOR_MDS: #
        knot_amps = amp_now[all_detected_knots_idx] #
        top_k_indices = np.argsort(knot_amps)[-MAX_KNOTS_FOR_MDS:] #
        knot_indices_for_mds = all_detected_knots_idx[top_k_indices] #
        n_knots_mds = MAX_KNOTS_FOR_MDS #
    elif n_all_knots == 0: #
        topk = 6 #
        knot_indices_for_mds = np.argsort(amp_now)[-topk:] #
        n_knots_mds = len(knot_indices_for_mds) #
    else:
        n_knots_mds = n_all_knots # Use all if below cap

    # --- Hole Detection ---
    hole_threshold = detect_threshold * 0.8 #
    min_mask = (amp_now <= left) & (amp_now <= right) & (amp_now < hole_threshold) #
    all_detected_holes_idx = np.where(min_mask)[0] #

    # --- Calculate Phase and Winding ---
    analytic_full = hilbert(psi.real) #
    phase_full = np.angle(analytic_full) #

    holes_list_details = [] # Renamed to avoid conflict
    current_active_holes = set()
    for h_idx in all_detected_holes_idx:
        w = compute_winding_for_site(phase_full, h_idx, radius=3) #
        holes_list_details.append({ #
            'site': int(h_idx),
            'amp': float(amp_now[h_idx]),
            'winding': int(w),
            'is_vortex': abs(w) > 0
        })
        current_active_holes.add(h_idx) #

    # --- Update Hole Lifetimes ---
    holes_born_this_frame = [] #
    holes_died_this_frame = [] #
    for h_idx in current_active_holes: #
        if h_idx not in active_holes: #
            active_holes[h_idx] = current_frame_index #
            holes_born_this_frame.append(int(h_idx)) #
    dead_hole_indices = set(active_holes.keys()) - current_active_holes #
    for h_idx in dead_hole_indices: #
        birth_frame = active_holes.pop(h_idx) #
        holes_died_this_frame.append({'site': int(h_idx), 'birth': int(birth_frame), 'death': int(current_frame_index), 'lifetime': int(current_frame_index - birth_frame)}) #

    # --- Update Knot Lifetimes ---
    knots_list_details = [] # Renamed
    current_active_knots = set(all_detected_knots_idx) #
    knots_born_this_frame = [] #
    knots_died_this_frame = [] #
    for k_idx in current_active_knots: #
        if k_idx not in active_knots: #
            active_knots[k_idx] = current_frame_index #
            knots_born_this_frame.append(int(k_idx)) #
        knots_list_details.append({'site': int(k_idx), 'amp': float(amp_now[k_idx])}) #
    dead_knot_indices = set(active_knots.keys()) - current_active_knots #
    for k_idx in dead_knot_indices: #
        birth_frame = active_knots.pop(k_idx) #
        knots_died_this_frame.append({'site': int(k_idx), 'birth': int(birth_frame), 'death': int(current_frame_index), 'lifetime': int(current_frame_index - birth_frame)}) #

    # --- Calculate Angular Momentum ---
    grad_psi = np.roll(psi, -1) - np.roll(psi, 1) #
    moment_density = np.imag(np.conj(psi) * grad_psi) #
    angular_momentum = float(np.sum(moment_density)) #

    # --- Build time-series matrix for MDS ---
    ordered_history = np.roll(history, -hist_ptr, axis=1) #
    knot_histories_real_mds = ordered_history[knot_indices_for_mds].real #
    analytic_series_mds = hilbert(knot_histories_real_mds, axis=1) #
    series_mds = np.abs(analytic_series_mds) #

    # Compute distances and MDS
    coords = np.zeros((n_knots_mds, 3)) # Default
    mean_corr = 0.0
    top_corr = []
    if n_knots_mds >= 2: #
        D, C = pairwise_correlation_distance(series_mds) #
        coords = classical_mds_from_dists(D, ndim=3) #
        if n_knots_mds > 1: #
            corr_values = np.abs(C[np.triu_indices(n_knots_mds, k=1)]) #
            if len(corr_values) > 0: #
                mean_corr = float(np.mean(corr_values)) #
                top_indices = np.argsort(corr_values)[-5:] #
                top_corr_vals = corr_values[top_indices] #
                top_corr = [float(v) for v in top_corr_vals] #

    # --- Update 3D scatter ---
    if scatter_3d is not None:
         scatter_3d.remove()

    ax_embed.set_title(f"Perceived 3D embedding (Frame: {current_frame_index})") # Use global index
    ax_embed.set_xlim(-1.5, 1.5); ax_embed.set_ylim(-1.5, 1.5); ax_embed.set_zlim(-1.5, 1.5) #
    ax_embed.set_xlabel("X"); ax_embed.set_ylabel("Y"); ax_embed.set_zlabel("Z") #

    scatter_3d = None
    if n_knots_mds > 0:
        xs, ys, zs = coords[:,0], coords[:,1], coords[:,2] #
        site_indices = np.array(knot_indices_for_mds) #
        scatter_3d = ax_embed.scatter(xs, ys, zs, s=20, c=site_indices, cmap='viridis', alpha=0.7) #

    # --- <<< LOGGING STEP >>> ---
    if current_frame_index % LOG_EVERY_N_FRAMES == 0:
        record = { #
            'frame': int(current_frame_index), # Use global index
            'time': float(simulation_time),
            'n_knots_detected': int(n_all_knots),
            'n_holes_detected': int(len(holes_list_details)),
            'mean_amp': float(np.mean(amp_now)),
            'max_amp': float(np.max(amp_now)),
            'total_energy': float(np.sum(amp_now**2)),
            'angular_momentum': float(angular_momentum),
            'mean_knot_correlation': float(mean_corr),
            'top_knot_correlations': top_corr,
            'knots_details': knots_list_details, # Use renamed var
            'holes_details': holes_list_details, # Use renamed var
            'knots_born': knots_born_this_frame,
            'knots_died': knots_died_this_frame,
            'holes_born': holes_born_this_frame,
            'holes_died': holes_died_this_frame,
            'mds_coords_sample': coords[:min(n_knots_mds, 10)].tolist() if n_knots_mds > 0 else []
        }
        try:
            with open(LOG_PATH, 'a', encoding='utf-8') as f: #
                f.write(json.dumps(record, separators=(',', ':')) + "\n") #
        except Exception as e: #
            print(f"Error writing log at frame {current_frame_index}: {e}") #
    # --- <<< END LOGGING STEP >>> ---

    artists = [line_field]
    if scatter_3d:
        # We need to return the PathCollection generated by scatter
        # However, blit=True is still problematic with clear(). Keep blit=False.
        pass # scatter_3d is redrawn anyway because of clear()

    return artists

# ----------------------
# Final Log Save / Cleanup
# ----------------------
def on_close(event):
    """Save log data when the plot window is closed."""
    print("\nPlot window closed.")
    # No need to explicitly save JSONL as it's written line-by-line
    final_log_message()

def final_log_message():
    print(f"\nSimulation stopped or finished at frame {current_frame_index}.")
    print(f"Log data written line-by-line to {LOG_PATH}")
    print(f"Total simulation time elapsed: {simulation_time:.2f} units")
    end_time_wall = time.time()
    print(f"Total wall clock time: {end_time_wall - start_time_wall:.2f} seconds")

fig.canvas.mpl_connect('close_event', on_close) # Connect close event handler

# ----------------------
# Run animation
# ----------------------
start_time_wall = time.time()
# Use frames=None for indefinite run, repeat=True is default but explicit is fine
ani = FuncAnimation(fig, update_frame, frames=None, interval=10, blit=False, repeat=True)
plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make space for button
try:
    plt.show() #
except Exception as e:
    print(f"Animation display error: {e}")
finally:
    # Ensure final message prints even if show() errors
    # Note: Log is saved line-by-line, no final save needed here
    # final_log_message() # Message now printed by on_close
    pass