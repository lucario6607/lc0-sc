import matplotlib.pyplot as plt
import numpy as np

sizes = []
mean_nps = []
mean_ms = []

with open("v6e_backendbench_dense_1_to_64.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("size") or line.startswith("Building") or line.startswith("Transferring") or line.startswith("Done") or line.startswith("Loading") or line.startswith("Weights") or line.startswith("Warning") or line.startswith("Devices") or line.startswith("Converting") or line.startswith("TpuDevice") or "v0.34" in line or "_" in line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                s = int(parts[0])
                nps = float(parts[1])
                ms = float(parts[2])
                sizes.append(s)
                mean_nps.append(nps)
                mean_ms.append(ms)
            except ValueError:
                pass

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, dpi=300)

# NPS Plot
ax1.plot(sizes, mean_nps, color='#1f77b4', marker='o', markersize=4, linewidth=2, label='TPU v6e-1 (BT4 BF16)')
ax1.set_ylabel('Inference NPS (Nodes/sec)', fontsize=12, fontweight='bold', color='#1f77b4')
ax1.set_title('Google Cloud TPU v6e (Trillium) - BT4 Network Scaling (Dense Batch 1 to 64)', fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='y', labelcolor='#1f77b4')

# Highlight Batch 24 and Batch 56
max_idx = np.argmax(mean_nps)
ax1.annotate(f'Peak: {sizes[max_idx]} @ {mean_nps[max_idx]:,.0f} NPS',
             xy=(sizes[max_idx], mean_nps[max_idx]),
             xytext=(sizes[max_idx] - 12, mean_nps[max_idx] + 600),
             arrowprops=dict(facecolor='#1f77b4', shrink=0.05, width=1.5, headwidth=8),
             fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#e6f2ff', edgecolor='#1f77b4', alpha=0.8))

idx24 = sizes.index(24)
ax1.annotate(f'Search Target: Batch 24\n{mean_nps[idx24]:,.0f} NPS ({mean_ms[idx24]:.2f} ms)',
             xy=(24, mean_nps[idx24]),
             xytext=(24 - 10, mean_nps[idx24] - 2500),
             arrowprops=dict(facecolor='#2ca02c', shrink=0.05, width=1.5, headwidth=8),
             fontsize=10, fontweight='bold', color='#1b661b',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', edgecolor='#2ca02c', alpha=0.8))

# Latency Plot
ax2.plot(sizes, mean_ms, color='#d62728', marker='s', markersize=4, linewidth=2, label='Batch Latency (ms)')
ax2.set_xlabel('Minibatch Size', fontsize=12, fontweight='bold')
ax2.set_ylabel('Latency per Batch (ms)', fontsize=12, fontweight='bold', color='#d62728')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.tick_params(axis='y', labelcolor='#d62728')

# Tile boundary note at batch 57
ax2.axvline(x=56.5, color='gray', linestyle=':', linewidth=1.5)
ax2.text(57.5, 4.5, 'MXU Boundary Step\n(latency jumps 4.2ms -> 5.9ms)', fontsize=9, color='#555555', fontstyle='italic')

plt.tight_layout()
plt.savefig("v6e_dense_batch_scaling.png", dpi=300)
print("Saved v6e_dense_batch_scaling.png successfully.")