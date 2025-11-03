import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import sys

# --- Configuration ---
LOG_FILENAME = "knotiverse_interactive_log.jsonl"
OUTPUT_DIR = "analysis_output"
SAMPLE_EVERY_N = 50  # Analyze every Nth frame to reduce memory
CHUNK_SIZE = 1000    # Process this many lines at once

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Starting analysis of {LOG_FILENAME}")
print(f"Sampling every {SAMPLE_EVERY_N} frames...")
print(f"Processing in chunks of {CHUNK_SIZE} lines...")

# --- Streaming Statistics Accumulators ---
class StreamStats:
    """Compute statistics without loading all data into memory"""
    def __init__(self):
        self.frames = []
        self.times = []
        self.n_knots = []
        self.n_holes = []
        self.n_vortex = []
        self.mean_amps = []
        self.max_amps = []
        self.energies = []
        self.angular_mom = []
        self.correlations = []
        
        # Lifetime tracking
        self.knot_lifetimes = []
        self.hole_lifetimes = []
        
        # Winding number distribution
        self.winding_counts = Counter()
        
        # Phase analysis
        self.phase_transitions = []
        self.max_frame = 0
        
    def add_frame(self, record):
        """Add a single frame's data"""
        frame = record.get('frame', 0)
        
        # Basic metrics
        self.frames.append(frame)
        self.times.append(record.get('time', 0))
        
        # Knot/hole counts
        knots_details = record.get('knots_details', [])
        holes_details = record.get('holes_details', [])
        
        self.n_knots.append(len(knots_details))
        self.n_holes.append(len(holes_details))
        
        # Count vortex holes
        n_vortex = sum(1 for h in holes_details if h.get('is_vortex', False))
        self.n_vortex.append(n_vortex)
        
        # Amplitudes and energy
        self.mean_amps.append(record.get('mean_amp', 0))
        self.max_amps.append(record.get('max_amp', 0))
        self.energies.append(record.get('total_energy', 0))
        
        # Angular momentum
        self.angular_mom.append(record.get('angular_momentum', 0))
        
        # Correlation
        self.correlations.append(record.get('mean_knot_correlation', 0))
        
        # Lifetime events
        knots_died = record.get('knots_died', [])
        holes_died = record.get('holes_died', [])
        
        for k in knots_died:
            if isinstance(k, dict) and 'lifetime' in k:
                self.knot_lifetimes.append(k['lifetime'])
        
        for h in holes_died:
            if isinstance(h, dict) and 'lifetime' in h:
                self.hole_lifetimes.append(h['lifetime'])
        
        # Winding numbers
        for hole in holes_details:
            w = hole.get('winding', 0)
            self.winding_counts[w] += 1
        
        self.max_frame = max(self.max_frame, frame)
    
    def finalize(self):
        """Convert lists to numpy arrays for efficient plotting"""
        self.frames = np.array(self.frames)
        self.times = np.array(self.times)
        self.n_knots = np.array(self.n_knots)
        self.n_holes = np.array(self.n_holes)
        self.n_vortex = np.array(self.n_vortex)
        self.mean_amps = np.array(self.mean_amps)
        self.max_amps = np.array(self.max_amps)
        self.energies = np.array(self.energies)
        self.angular_mom = np.array(self.angular_mom)
        self.correlations = np.array(self.correlations)

# --- Streaming File Reader ---
def stream_jsonl_file(filename, sample_every_n=1):
    """Yield records from JSONL file, sampling every N lines"""
    line_count = 0
    sampled_count = 0
    bytes_read = 0
    
    try:
        # Get file size for progress tracking
        file_size = os.path.getsize(filename)
        print(f"File size: {file_size / (1024**3):.2f} GB")
        
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                bytes_read += len(line.encode('utf-8'))
                
                # Sample every Nth line
                if line_count % sample_every_n != 0:
                    continue
                
                try:
                    record = json.loads(line)
                    sampled_count += 1
                    
                    # Progress update every 100 sampled frames
                    if sampled_count % 100 == 0:
                        progress = (bytes_read / file_size) * 100
                        print(f"Progress: {progress:.1f}% ({sampled_count} frames sampled, line {line_count})")
                    
                    yield record
                    
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed line {line_count}: {e}")
                    continue
        
        print(f"\nTotal lines read: {line_count}")
        print(f"Frames sampled: {sampled_count}")
        
    except FileNotFoundError:
        print(f"ERROR: File {filename} not found!")
        sys.exit(1)

# --- Main Analysis ---
print("\n=== PHASE 1: Streaming data collection ===")
stats = StreamStats()

for record in stream_jsonl_file(LOG_FILENAME, sample_every_n=SAMPLE_EVERY_N):
    stats.add_frame(record)

stats.finalize()

print(f"\n=== PHASE 2: Analysis ===")
print(f"Total frames analyzed: {len(stats.frames)}")
print(f"Frame range: {stats.frames[0]} to {stats.frames[-1]}")
print(f"Simulation time range: {stats.times[0]:.2f} to {stats.times[-1]:.2f}")

# --- PLOTTING ---
print("\n=== PHASE 3: Generating plots ===")

fig = plt.figure(figsize=(16, 20))
gs = fig.add_gridspec(6, 2, hspace=0.3, wspace=0.3)

# Plot 1: Knot and Hole Counts
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(stats.frames, stats.n_knots, label='Knots', color='blue', linewidth=0.5)
ax1.plot(stats.frames, stats.n_holes, label='Holes (all)', color='red', linewidth=0.5, alpha=0.7)
ax1.plot(stats.frames, stats.n_vortex, label='Vortex Holes (|w|>0)', color='magenta', linewidth=0.5)
ax1.set_xlabel('Frame')
ax1.set_ylabel('Count')
ax1.set_title('Population Dynamics Over 268k+ Frames')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Energy Evolution
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(stats.frames, stats.energies, color='green', linewidth=0.5)
ax2.set_xlabel('Frame')
ax2.set_ylabel('Total Energy (Σ|ψ|²)')
ax2.set_title('Energy Concentration')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# Plot 3: Max Amplitude
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(stats.frames, stats.max_amps, color='orange', linewidth=0.5)
ax3.set_xlabel('Frame')
ax3.set_ylabel('Max Amplitude')
ax3.set_title('Peak Amplitude Evolution')
ax3.grid(True, alpha=0.3)

# Plot 4: Angular Momentum
ax4 = fig.add_subplot(gs[2, :])
ax4.plot(stats.frames, stats.angular_mom, color='purple', linewidth=0.5)
ax4.set_xlabel('Frame')
ax4.set_ylabel('Angular Momentum')
ax4.set_title('Angular Momentum Evolution (Chirality Lock)')
ax4.grid(True, alpha=0.3)

# Plot 5: Correlation (Phase Coherence)
ax5 = fig.add_subplot(gs[3, :])
ax5.plot(stats.frames, stats.correlations, color='brown', linewidth=0.5)
ax5.set_xlabel('Frame')
ax5.set_ylabel('Mean Correlation')
ax5.set_title('Knot Phase Coherence')
ax5.grid(True, alpha=0.3)
ax5.set_ylim([0, 1])

# Plot 6: Knot Lifetime Distribution
ax6 = fig.add_subplot(gs[4, 0])
if stats.knot_lifetimes:
    # Convert frame lifetimes to simulation time (assuming dt=0.05, sim_steps_per_frame=6)
    dt = 0.05
    sim_steps = 6
    knot_times = np.array(stats.knot_lifetimes) * sim_steps * dt
    ax6.hist(knot_times, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Lifetime (sim time units)')
    ax6.set_ylabel('Count')
    ax6.set_title(f'Knot Lifetime Distribution (n={len(stats.knot_lifetimes)})')
    ax6.axvline(np.mean(knot_times), color='red', linestyle='--', label=f'Mean: {np.mean(knot_times):.2f}')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
else:
    ax6.text(0.5, 0.5, 'No knot lifetime data', ha='center', va='center', transform=ax6.transAxes)
    ax6.set_title('Knot Lifetimes (No Data)')

# Plot 7: Hole Lifetime Distribution
ax7 = fig.add_subplot(gs[4, 1])
if stats.hole_lifetimes:
    hole_times = np.array(stats.hole_lifetimes) * sim_steps * dt
    ax7.hist(hole_times, bins=50, color='red', alpha=0.7, edgecolor='black')
    ax7.set_xlabel('Lifetime (sim time units)')
    ax7.set_ylabel('Count')
    ax7.set_title(f'Hole Lifetime Distribution (n={len(stats.hole_lifetimes)})')
    ax7.axvline(np.mean(hole_times), color='blue', linestyle='--', label=f'Mean: {np.mean(hole_times):.2f}')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
else:
    ax7.text(0.5, 0.5, 'No hole lifetime data', ha='center', va='center', transform=ax7.transAxes)
    ax7.set_title('Hole Lifetimes (No Data)')

# Plot 8: Winding Number Distribution
ax8 = fig.add_subplot(gs[5, :])
if stats.winding_counts:
    windings = sorted(stats.winding_counts.keys())
    counts = [stats.winding_counts[w] for w in windings]
    
    colors = ['red' if w < 0 else 'green' if w > 0 else 'gray' for w in windings]
    ax8.bar(windings, counts, color=colors, alpha=0.7, edgecolor='black')
    ax8.set_xlabel('Winding Number')
    ax8.set_ylabel('Total Instances')
    ax8.set_title('Topological Charge Distribution (Vortex Types)')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Add percentage of vortex holes
    total_holes = sum(counts)
    vortex_holes = sum(c for w, c in zip(windings, counts) if w != 0)
    vortex_pct = (vortex_holes / total_holes * 100) if total_holes > 0 else 0
    ax8.text(0.02, 0.98, f'Vortex holes: {vortex_pct:.1f}%', 
             transform=ax8.transAxes, va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
else:
    ax8.text(0.5, 0.5, 'No winding data', ha='center', va='center', transform=ax8.transAxes)
    ax8.set_title('Winding Numbers (No Data)')

fig.suptitle(f'Knotiverse Analysis: {stats.max_frame} Total Frames', fontsize=16, fontweight='bold')

output_path = os.path.join(OUTPUT_DIR, 'knotiverse_full_analysis.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved comprehensive plot to {output_path}")

# --- STATISTICAL SUMMARY ---
print("\n=== PHASE 4: Statistical Summary ===")

print("\n--- Population Statistics ---")
print(f"Knots: mean={np.mean(stats.n_knots):.1f}, max={np.max(stats.n_knots)}, min={np.min(stats.n_knots)}")
print(f"Holes: mean={np.mean(stats.n_holes):.1f}, max={np.max(stats.n_holes)}, min={np.min(stats.n_holes)}")
print(f"Vortex holes: mean={np.mean(stats.n_vortex):.1f}, max={np.max(stats.n_vortex)}")

print("\n--- Energy Dynamics ---")
print(f"Energy: initial={stats.energies[0]:.2e}, final={stats.energies[-1]:.2e}")
print(f"Energy ratio (final/initial): {stats.energies[-1]/stats.energies[0]:.2f}x")
print(f"Max amplitude: peak={np.max(stats.max_amps):.2f}, final={stats.max_amps[-1]:.2f}")

print("\n--- Angular Momentum ---")
print(f"Initial: {stats.angular_mom[0]:.2f}")
print(f"Final: {stats.angular_mom[-1]:.2f}")
print(f"Change: {stats.angular_mom[-1] - stats.angular_mom[0]:.2f}")
print(f"Direction: {'Clockwise' if stats.angular_mom[-1] < 0 else 'Counterclockwise'}")

print("\n--- Phase Coherence ---")
print(f"Correlation: mean={np.mean(stats.correlations):.3f}, max={np.max(stats.correlations):.3f}")
print(f"Initial correlation: {stats.correlations[0]:.3f}")
print(f"Final correlation: {stats.correlations[-1]:.3f}")

if stats.knot_lifetimes:
    knot_sim_times = np.array(stats.knot_lifetimes) * 6 * 0.05
    print("\n--- Knot Lifetimes ---")
    print(f"Events: {len(stats.knot_lifetimes)}")
    print(f"Mean: {np.mean(knot_sim_times):.2f} sim time units")
    print(f"Median: {np.median(knot_sim_times):.2f}")
    print(f"Max: {np.max(knot_sim_times):.2f}")

if stats.hole_lifetimes:
    hole_sim_times = np.array(stats.hole_lifetimes) * 6 * 0.05
    print("\n--- Hole Lifetimes ---")
    print(f"Events: {len(stats.hole_lifetimes)}")
    print(f"Mean: {np.mean(hole_sim_times):.2f} sim time units")
    print(f"Median: {np.median(hole_sim_times):.2f}")
    print(f"Max: {np.max(hole_sim_times):.2f}")

if stats.winding_counts:
    print("\n--- Topological Charges (Winding Numbers) ---")
    total_holes_counted = sum(stats.winding_counts.values())
    vortex_holes_counted = sum(c for w, c in stats.winding_counts.items() if w != 0)
    print(f"Total hole instances: {total_holes_counted}")
    print(f"Vortex holes (|w|>0): {vortex_holes_counted} ({vortex_holes_counted/total_holes_counted*100:.1f}%)")
    print("\nWinding distribution:")
    for w in sorted(stats.winding_counts.keys()):
        count = stats.winding_counts[w]
        pct = count / total_holes_counted * 100
        print(f"  w={w:+2d}: {count:6d} instances ({pct:5.1f}%)")

print("\n=== Analysis complete! ===")
print(f"All plots saved to {OUTPUT_DIR}/")
plt.close('all')