# =====================================================================
# PHASE 2C — Experiment C2: LayerScale γ Heatmap Visualization
# =====================================================================
# Run these 4 cells IN ORDER after your standard Phase 2C imports.
# No GPU needed — this is pure checkpoint analysis.
# Outputs:
#   ./outputs/phase2c/gamma_heatmap.png      (main thesis figure)
#   ./outputs/phase2c/gamma_per_layer.png     (per-layer summary)
#   ./outputs/phase2c/gamma_distributions.png (histograms)
#   ./outputs/phase2c/gamma_stats.csv         (raw statistics)
# =====================================================================


# =====================================================================
# Cell C2-1 — Load Champion Checkpoint & Extract γ Parameters
# =====================================================================
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import torch

PHASE2B_DIR   = "./outputs/phase2b"
PHASE2C_DIR   = "./outputs/phase2c"
CHAMPION_CKPT = os.path.join(PHASE2B_DIR, "B4_postGN_LS_24_s42_best.pt")
os.makedirs(PHASE2C_DIR, exist_ok=True)

# ---- Load checkpoint ----
ckpt  = torch.load(CHAMPION_CKPT, map_location="cpu")
state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

# ---- Discover all gamma parameters ----
gamma_keys = sorted([k for k in state if "gamma" in k])
print(f"Found {len(gamma_keys)} gamma parameters.  First 6:")
for k in gamma_keys[:6]:
    print(f"  {k}  shape={tuple(state[k].shape)}")
if len(gamma_keys) > 6:
    print(f"  ... ({len(gamma_keys) - 6} more)")

# ---- Parse into structured containers ----
# Matches patterns like:
#   backbone.0.0.ls1.gamma          (ModuleList of ModuleList)
#   backbone.0.block0.ls1.gamma     (Sequential with named modules)
#   backbone.2.3.ls2.gamma          etc.
gamma1_data = {}          # stage_idx → [np.array per block, sorted]
gamma2_data = {}

for key in gamma_keys:
    val = state[key].detach().cpu().numpy().copy()
    m = re.match(r"backbone\.(\d+)\.\D*(\d+).*?\.ls(\d)\.gamma$", key)
    if m:
        stage, block, ls_id = int(m.group(1)), int(m.group(2)), int(m.group(3))
        target = gamma1_data if ls_id == 1 else gamma2_data
        target.setdefault(stage, []).append((block, val))
    else:
        print(f"  ⚠ Could not parse: {key}")

# Sort by block index, keep only the arrays
for d in [gamma1_data, gamma2_data]:
    for s in d:
        d[s].sort(key=lambda x: x[0])
        d[s] = [v for _, v in d[s]]

# ---- Precompute helpers used by all later cells ----
n_stages = len(gamma1_data)
stage_blocks   = [len(gamma1_data[s]) for s in range(n_stages)]
stage_channels = [gamma1_data[s][0].shape[0] for s in range(n_stages)]
total_blocks   = sum(stage_blocks)
# cumulative start index per stage
stage_offsets  = [0]
for nb in stage_blocks:
    stage_offsets.append(stage_offsets[-1] + nb)

# Global flat array (for color limits)
all_flat = np.concatenate(
    [np.concatenate(gamma1_data[s]) for s in range(n_stages)] +
    [np.concatenate(gamma2_data[s]) for s in range(n_stages)]
)

# ---- Per-block statistics → DataFrame ----
rows = []
for s in range(n_stages):
    for b in range(stage_blocks[s]):
        g1 = gamma1_data[s][b]
        g2 = gamma2_data[s][b]
        rows.append(dict(
            stage=s, block_in_stage=b,
            global_layer=stage_offsets[s] + b,
            channels=stage_channels[s],
            # γ₁ stats
            g1_mean=g1.mean(), g1_std=g1.std(),
            g1_min=g1.min(),  g1_max=g1.max(),
            g1_neg_pct=100 * (g1 < 0).mean(),
            # γ₂ stats
            g2_mean=g2.mean(), g2_std=g2.std(),
            g2_min=g2.min(),  g2_max=g2.max(),
            g2_neg_pct=100 * (g2 < 0).mean(),
        ))
df_stats = pd.DataFrame(rows)
csv_path = os.path.join(PHASE2C_DIR, "gamma_stats.csv")
df_stats.to_csv(csv_path, index=False)

# ---- Print summary table ----
print(f"\n{'='*65}")
print(f"{'Stage':>5} | {'Blks':>4} | {'Ch':>4} | {'γ₁ mean':>9} | {'γ₂ mean':>9} | {'%neg γ₁':>7} | {'%neg γ₂':>7}")
print(f"{'='*65}")
for s in range(n_stages):
    g1_all = np.concatenate(gamma1_data[s])
    g2_all = np.concatenate(gamma2_data[s])
    print(f"  {s:>3}  | {stage_blocks[s]:>4} | {stage_channels[s]:>4} "
          f"| {g1_all.mean():>+8.4f} | {g2_all.mean():>+8.4f} "
          f"| {100*(g1_all<0).mean():>5.1f}% | {100*(g2_all<0).mean():>5.1f}%")
print(f"{'─'*65}")
print(f"  ALL  | {total_blocks:>4} |      "
      f"| {all_flat.mean():>+8.4f} |           "
      f"| {100*(all_flat<0).mean():>5.1f}% |")
print(f"\n✓ Saved per-block stats → {csv_path}")


# =====================================================================
# Cell C2-2 — Main γ Heatmap  (THE key thesis figure)
# =====================================================================
#
# Layout:  4 rows (stages)  ×  2 columns (γ₁, γ₂)
# Color:   diverging RdBu_r, centred at 0
#          blue = positive (amplify),  red = negative (suppress)

fig, axes = plt.subplots(
    n_stages, 2,
    figsize=(16, 13),
    gridspec_kw=dict(
        height_ratios=stage_blocks,
        width_ratios=[1, 1],
        hspace=0.15,
        wspace=0.06,
    ),
)

# Symmetric colour limits — clip at 99.5th percentile for contrast
vabs = float(np.percentile(np.abs(all_flat), 99.5))
norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

stage_cmap = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}

for col, (gdict, col_title) in enumerate([
    (gamma1_data, "γ₁  (Graph Convolution)"),
    (gamma2_data, "γ₂  (Feed-Forward Network)"),
]):
    for s in range(n_stages):
        ax = axes[s, col]
        data = np.stack(gdict[s])          # shape (blocks, channels)
        nb, nc = data.shape

        im = ax.imshow(data, aspect="auto", cmap="RdBu_r",
                       norm=norm, interpolation="nearest")

        # ---- Column title (top row only) ----
        if s == 0:
            ax.set_title(col_title, fontsize=13, fontweight="bold", pad=10)

        # ---- Y-axis: global layer index ----
        off = stage_offsets[s]
        ax.set_yticks(range(nb))
        if col == 0:
            ax.set_ylabel(f"Stage {s}\n({nc} ch)",
                          fontsize=10, fontweight="bold")
            ax.set_yticklabels([str(off + j) for j in range(nb)], fontsize=7)
        else:
            ax.set_yticklabels([])

        # ---- X-axis: channel index (bottom row only) ----
        if s == n_stages - 1:
            step = max(1, nc // 8)
            ax.set_xticks(range(0, nc, step))
            ax.set_xticklabels(range(0, nc, step), fontsize=7)
            ax.set_xlabel("Channel Index", fontsize=10)
        else:
            ax.set_xticks([])

        # ---- Corner annotation: mean & % negative ----
        mean_v = data.mean()
        neg_p  = 100 * (data < 0).mean()
        ax.text(
            0.98, 0.92,
            f"μ={mean_v:+.3f}\n{neg_p:.0f}% neg",
            transform=ax.transAxes, fontsize=7,
            ha="right", va="top", fontfamily="monospace",
            bbox=dict(facecolor="white", alpha=0.85,
                      edgecolor="gray", linewidth=0.5,
                      boxstyle="round,pad=0.15"),
        )

        # ---- Thin border coloured by stage ----
        for spine in ax.spines.values():
            spine.set_edgecolor(stage_cmap[s])
            spine.set_linewidth(1.8)

# ---- Shared colour bar ----
cbar = fig.colorbar(im, ax=axes, location="right",
                    shrink=0.82, pad=0.015, aspect=40)
cbar.set_label("Learned  γ  value", fontsize=11)
cbar.ax.axhline(y=0, color="black", linewidth=1.0)
cbar.ax.text(1.8, 0.5, "← suppress     amplify →",
             transform=cbar.ax.transAxes, fontsize=8,
             rotation=90, va="center", ha="left", color="gray")

fig.suptitle(
    "LayerScale γ Heatmap — Champion ViG  (PostGN+LS, 24 blocks, 62.54%)",
    fontsize=15, fontweight="bold", y=1.00,
)

heatmap_path = os.path.join(PHASE2C_DIR, "gamma_heatmap.png")
plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"✓ Saved: {heatmap_path}")


# =====================================================================
# Cell C2-3 — Per-Layer Summary Plots  (mean γ, % negative, std)
# =====================================================================
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(14, 10), sharex=True,
    gridspec_kw=dict(hspace=0.08),
)

layers = df_stats["global_layer"].values

# Stage-coloured backgrounds
for ax in (ax1, ax2, ax3):
    for s in range(n_stages):
        lo = stage_offsets[s] - 0.5
        hi = stage_offsets[s + 1] - 0.5
        if s % 2 == 1:
            ax.axvspan(lo, hi, color="gray", alpha=0.06, zorder=0)
        # vertical separator
        if s > 0:
            ax.axvline(lo, color="gray", linewidth=1.2,
                       linestyle="--", alpha=0.4, zorder=1)

# ---- Panel 1: Mean γ (grouped bars) ----
w = 0.35
ax1.bar(layers - w/2, df_stats["g1_mean"], w,
        color=[stage_cmap[s] for s in df_stats["stage"]],
        alpha=0.80, edgecolor="black", linewidth=0.4, label="γ₁ (GraphConv)")
ax1.bar(layers + w/2, df_stats["g2_mean"], w,
        color=[stage_cmap[s] for s in df_stats["stage"]],
        alpha=0.40, edgecolor="black", linewidth=0.4,
        hatch="//", label="γ₂ (FFN)")
ax1.axhline(0, color="black", linewidth=0.8)
ax1.set_ylabel("Mean γ", fontsize=11)
ax1.legend(fontsize=9, loc="upper right", framealpha=0.9)
ax1.set_title("Per-Layer LayerScale Statistics — Champion Model",
              fontsize=13, fontweight="bold")
# Stage labels
for s in range(n_stages):
    mid = (stage_offsets[s] + stage_offsets[s + 1]) / 2 - 0.5
    ax1.text(mid, ax1.get_ylim()[1] * 0.88,
             f"Stage {s} ({stage_channels[s]}ch)",
             ha="center", fontsize=8, fontweight="bold",
             color=stage_cmap[s],
             bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

# ---- Panel 2: % Negative ----
ax2.plot(layers, df_stats["g1_neg_pct"], "o-",
         color="#e41a1c", markersize=5, linewidth=1.5,
         label="γ₁", zorder=3)
ax2.plot(layers, df_stats["g2_neg_pct"], "s--",
         color="#377eb8", markersize=5, linewidth=1.5,
         label="γ₂", zorder=3)
ax2.axhline(50, color="gray", linewidth=0.8, linestyle=":",
            label="50 % line")
ax2.set_ylabel("% Negative γ", fontsize=11)
ax2.set_ylim(0, 100)
ax2.legend(fontsize=9, loc="lower left", framealpha=0.9)

# ---- Panel 3: Std of γ ----
ax3.plot(layers, df_stats["g1_std"], "o-",
         color="#e41a1c", markersize=5, linewidth=1.5, label="γ₁")
ax3.plot(layers, df_stats["g2_std"], "s--",
         color="#377eb8", markersize=5, linewidth=1.5, label="γ₂")
ax3.set_ylabel("Std of γ", fontsize=11)
ax3.set_xlabel("Layer Index", fontsize=11)
ax3.set_xticks(layers)
ax3.legend(fontsize=9, loc="upper right", framealpha=0.9)

plt.tight_layout()
summary_path = os.path.join(PHASE2C_DIR, "gamma_per_layer.png")
plt.savefig(summary_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"✓ Saved: {summary_path}")


# =====================================================================
# Cell C2-4 — Distribution Histograms by Stage
# =====================================================================
fig, axes = plt.subplots(2, n_stages, figsize=(18, 7),
                         sharex=True, sharey="row")

bins = np.linspace(-vabs, vabs, 80)

for row, (gdict, row_label) in enumerate([
    (gamma1_data, "γ₁ (Graph Conv)"),
    (gamma2_data, "γ₂ (FFN)"),
]):
    for s in range(n_stages):
        ax = axes[row, s]
        vals = np.concatenate(gdict[s])

        # Histogram split by sign
        neg_vals = vals[vals < 0]
        pos_vals = vals[vals >= 0]
        ax.hist(neg_vals, bins=bins, color="#d62728", alpha=0.7,
                label=f"neg ({len(neg_vals)})")
        ax.hist(pos_vals, bins=bins, color="#1f77b4", alpha=0.7,
                label=f"pos ({len(pos_vals)})")

        ax.axvline(0, color="black", linewidth=1.0, linestyle="-")
        ax.axvline(vals.mean(), color="gold", linewidth=2.0,
                   linestyle="--", label=f"mean={vals.mean():.3f}")

        # Labels
        if row == 0:
            ax.set_title(f"Stage {s}  ({stage_channels[s]} ch, "
                         f"{stage_blocks[s]} blocks)",
                         fontsize=10, fontweight="bold")
        if s == 0:
            ax.set_ylabel(f"{row_label}\nCount", fontsize=10)
        if row == 1:
            ax.set_xlabel("γ value", fontsize=10)

        ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
        ax.tick_params(labelsize=8)

fig.suptitle("Distribution of Learned γ Values — Champion ViG",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
dist_path = os.path.join(PHASE2C_DIR, "gamma_distributions.png")
plt.savefig(dist_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"✓ Saved: {dist_path}")

# ---- Final summary ----
print("\n" + "=" * 50)
print("C2 COMPLETE — All outputs:")
print(f"  1. {os.path.join(PHASE2C_DIR, 'gamma_heatmap.png')}")
print(f"  2. {os.path.join(PHASE2C_DIR, 'gamma_per_layer.png')}")
print(f"  3. {os.path.join(PHASE2C_DIR, 'gamma_distributions.png')}")
print(f"  4. {os.path.join(PHASE2C_DIR, 'gamma_stats.csv')}")
print("=" * 50)
