# =====================================================================
# PHASE 2B — Main Experiments: 100 Epochs, 3 Seeds, Full Recipe
# =====================================================================
# Prerequisites: Run Phase 2A cells first (they define the model classes).
# This notebook uses the SAME model architecture from Phase 2A
# (PyramidViG with PostGN + LayerScale options).
#
# Changes from Phase 2A pilots:
#   - 100 epochs instead of 30
#   - 10-epoch warmup instead of 5
#   - Mixup α=0.2 + Label Smoothing 0.1
#   - SoftTargetCrossEntropy loss (for Mixup targets)
#   - 3 random seeds for statistical significance
#   - Checkpoint saving for best model
# =====================================================================


# =====================================================================
# Cell 2B-1 — Phase 2B Config
# =====================================================================
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

PHASE2B_DIR    = "./outputs/phase2b"
os.makedirs(PHASE2B_DIR, exist_ok=True)

# Main training settings
MAIN_EPOCHS    = 100
MAIN_BATCH     = 128
MAIN_LR        = 1e-3
MAIN_WARMUP    = 10
MAIN_WD        = 0.05
MAIN_GRAD_CLIP = 1.0
MAIN_SEEDS     = [42, 123, 2024]

# Mixup config (mild — appropriate for 5-10M param models on 128K images)
MIXUP_ALPHA    = 0.2
LABEL_SMOOTH   = 0.1

print(f"Phase 2B config:")
print(f"  Epochs:     {MAIN_EPOCHS}")
print(f"  Batch:      {MAIN_BATCH}")
print(f"  LR:         {MAIN_LR}")
print(f"  Warmup:     {MAIN_WARMUP}")
print(f"  Mixup α:    {MIXUP_ALPHA}")
print(f"  Label Sm.:  {LABEL_SMOOTH}")
print(f"  Seeds:      {MAIN_SEEDS}")
print(f"  Output:     {PHASE2B_DIR}")


# =====================================================================
# Cell 2B-2 — Updated training loop with Mixup + checkpointing
# =====================================================================
def train_epoch_mixup(model, loader, opt, scaler, crit, mixup_fn, device):
    """Train one epoch with optional Mixup augmentation."""
    model.train()
    total_loss = 0.0
    n = 0
    for imgs, tgts in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        tgts = tgts.to(device, non_blocking=True)

        # Apply Mixup (produces soft targets)
        if mixup_fn is not None:
            imgs, tgts = mixup_fn(imgs, tgts)

        opt.zero_grad(set_to_none=True)
        with autocast('cuda'):
            loss = crit(model(imgs), tgts)

        if torch.isnan(loss) or torch.isinf(loss):
            opt.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)

        # Sanitize gradients
        for p in model.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAIN_GRAD_CLIP)

        scaler.step(opt)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)

    return total_loss / max(n, 1)


def get_gamma_stats(model):
    """Extract LayerScale gamma statistics per stage."""
    gammas_by_stage = {}
    for name, p in model.named_parameters():
        if 'gamma' not in name:
            continue
        # Parse stage index from name like "stages.2.3.gamma1"
        parts = name.split('.')
        try:
            stage_idx = int(parts[1])
        except (IndexError, ValueError):
            stage_idx = -1
        key = f"stage_{stage_idx}"
        if key not in gammas_by_stage:
            gammas_by_stage[key] = []
        gammas_by_stage[key].append(p.data.cpu().numpy().flatten())

    stats = {}
    for stage, arrays in gammas_by_stage.items():
        all_vals = np.concatenate(arrays)
        stats[stage] = {
            'mean': float(np.mean(all_vals)),
            'std': float(np.std(all_vals)),
            'min': float(np.min(all_vals)),
            'max': float(np.max(all_vals)),
            'pct_negative': float(np.mean(all_vals < 0) * 100),
        }
    return stats


def run_main_experiment(tag, total_blocks, k=9,
                        use_post_gn=False, gn_groups=8,
                        use_layer_scale=False, layer_scale_init=1e-2,
                        seed=42, save_ckpt=True):
    """Run one full 100-epoch experiment and return results."""
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    run_tag = f"{tag}_s{seed}"
    print("\n" + "=" * 90)
    print(f"MAIN EXPERIMENT [{run_tag}]")
    print(f"  blocks={total_blocks} k={k} post_gn={use_post_gn} "
          f"layer_scale={use_layer_scale} "
          f"{'init='+str(layer_scale_init) if use_layer_scale else ''}")
    print("=" * 90)

    # Build model
    model, depths = build_model(
        total_blocks, k=k,
        use_post_gn=use_post_gn, gn_groups=gn_groups,
        use_layer_scale=use_layer_scale, layer_scale_init=layer_scale_init,
        num_classes=NUM_CLASSES,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  depths={depths} | params={n_params/1e6:.2f}M")
    model = model.to(DEVICE)

    # Loss: SoftTargetCrossEntropy for Mixup, with label smoothing
    train_crit = SoftTargetCrossEntropy()
    val_crit   = nn.CrossEntropyLoss()

    # Mixup function
    mixup_fn = Mixup(
        mixup_alpha=MIXUP_ALPHA,
        cutmix_alpha=0.0,          # no CutMix
        label_smoothing=LABEL_SMOOTH,
        num_classes=NUM_CLASSES,
    )

    # Optimizer
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if p.ndim <= 1 or 'norm' in name.lower() or 'bias' in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    opt = optim.AdamW([
        {'params': decay_params,    'weight_decay': MAIN_WD},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=MAIN_LR)

    scaler = GradScaler('cuda')
    scheduler = CosineLRScheduler(
        opt, t_initial=MAIN_EPOCHS, lr_min=1e-6,
        warmup_t=MAIN_WARMUP, warmup_lr_init=1e-6,
    )
    scheduler.step(0)

    # Training loop
    best_acc = 0.0
    best_epoch = 0
    history = []

    for ep in range(MAIN_EPOCHS):
        t0 = time.time()

        tl = train_epoch_mixup(model, train_loader, opt, scaler,
                               train_crit, mixup_fn, DEVICE)
        vl, va = validate(model, val_loader, val_crit, DEVICE)

        scheduler.step(ep + 1)
        dt = time.time() - t0

        history.append(dict(
            tag=run_tag, seed=seed, epoch=ep + 1,
            train_loss=tl, val_loss=vl, val_acc=va,
            time=dt, blocks=total_blocks, k=k,
            post_gn=use_post_gn, layer_scale=use_layer_scale,
            layer_scale_init=layer_scale_init if use_layer_scale else 0.0,
            lr=opt.param_groups[0]['lr'],
        ))

        star = ""
        if va > best_acc:
            best_acc = va
            best_epoch = ep + 1
            star = " ★"
            # Save checkpoint
            if save_ckpt:
                ckpt_path = os.path.join(PHASE2B_DIR, f"{run_tag}_best.pt")
                torch.save({
                    'epoch': ep + 1,
                    'model_state_dict': model.state_dict(),
                    'val_acc': va,
                    'config': dict(
                        total_blocks=total_blocks, k=k,
                        use_post_gn=use_post_gn, gn_groups=gn_groups,
                        use_layer_scale=use_layer_scale,
                        layer_scale_init=layer_scale_init,
                    ),
                }, ckpt_path)

        # Print every 10th epoch + first + last
        if (ep + 1) % 10 == 0 or ep == 0 or (ep + 1) == MAIN_EPOCHS:
            print(f"  Ep {ep+1:3d}/{MAIN_EPOCHS} | TrL {tl:.4f} | "
                  f"VaL {vl:.4f} | Acc {va:.4f} | "
                  f"LR {opt.param_groups[0]['lr']:.2e} | {dt:.1f}s{star}")

    # Final gamma stats
    gamma_stats = {}
    if use_layer_scale:
        gamma_stats = get_gamma_stats(model)
        print(f"\n  LayerScale γ statistics:")
        for stage, s in sorted(gamma_stats.items()):
            print(f"    {stage}: mean={s['mean']:.4f} std={s['std']:.4f} "
                  f"range=[{s['min']:.4f}, {s['max']:.4f}] "
                  f"neg={s['pct_negative']:.1f}%")

    # Measure throughput
    model.eval()
    dummy = torch.randn(MAIN_BATCH, 3, 224, 224, device=DEVICE)
    # Warmup
    for _ in range(5):
        with torch.no_grad(), autocast('cuda'):
            _ = model(dummy)
    torch.cuda.synchronize()
    t0 = time.time()
    n_iters = 20
    for _ in range(n_iters):
        with torch.no_grad(), autocast('cuda'):
            _ = model(dummy)
    torch.cuda.synchronize()
    throughput = (MAIN_BATCH * n_iters) / (time.time() - t0)

    # Peak memory
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    torch.cuda.reset_peak_memory_stats()

    print(f"\n  → Best val acc [{run_tag}]: {best_acc:.4f} (epoch {best_epoch})")
    print(f"  → Throughput: {throughput:.0f} img/s | Peak mem: {peak_mem:.0f} MB")

    result = dict(
        tag=run_tag, config=tag, seed=seed,
        blocks=total_blocks, params_M=n_params / 1e6,
        best_acc=best_acc, best_epoch=best_epoch,
        final_acc=history[-1]['val_acc'],
        throughput=throughput, peak_mem_MB=peak_mem,
        gamma_stats=json.dumps(gamma_stats) if gamma_stats else "",
    )

    del model, opt, scaler
    torch.cuda.empty_cache()

    return result, history


# =====================================================================
# Cell 2B-3 — Define experiment matrix
# =====================================================================
EXPERIMENTS = [
    # (tag, total_blocks, use_post_gn, use_layer_scale, layer_scale_init, seeds)
    ("B1_baseline_12",       12, False, False, 0.0,   MAIN_SEEDS),
    ("B2_postGN_LS_16",      16, True,  True,  1e-2,  MAIN_SEEDS),  # Champion
    ("B3_postGN_LS_20",      20, True,  True,  1e-2,  MAIN_SEEDS),  # Depth scaling
    ("B4_postGN_LS_24",      24, True,  True,  1e-2,  MAIN_SEEDS),  # Deep test
    ("B5_LS_only_16",        16, False, True,  1e-2,  MAIN_SEEDS),  # Ablation
    ("B6_LS_1e4_20",         20, False, True,  1e-4,  MAIN_SEEDS),  # Init ablation
]

print("Phase 2B Experiment Matrix:")
print(f"{'Tag':25s} {'Blocks':>6s} {'PostGN':>6s} {'LS':>4s} {'Init':>8s} {'Seeds':>5s}")
print("-" * 60)
for tag, blk, pgn, ls, lsi, seeds in EXPERIMENTS:
    print(f"{tag:25s} {blk:6d} {str(pgn):>6s} {str(ls):>4s} "
          f"{lsi:8.0e} {len(seeds):5d}")
total_runs = sum(len(s) for _, _, _, _, _, s in EXPERIMENTS)
print(f"\nTotal runs: {total_runs}")
print(f"Est. time: {total_runs * 2:.0f} hours on H200")


# =====================================================================
# Cell 2B-4 — Run ALL experiments
# =====================================================================
# WARNING: This cell takes ~36-48 hours on H200.
# You can interrupt and resume — results are saved incrementally.

ALL_RESULTS = []
ALL_HISTORY = []

# Check for existing results to enable resume
results_path = os.path.join(PHASE2B_DIR, "phase2b_results.csv")
history_path = os.path.join(PHASE2B_DIR, "phase2b_history.csv")
completed_tags = set()
if os.path.exists(results_path):
    existing = pd.read_csv(results_path)
    completed_tags = set(existing['tag'].values)
    ALL_RESULTS = existing.to_dict('records')
    print(f"Resuming: {len(completed_tags)} runs already completed")
    print(f"  Completed: {sorted(completed_tags)}")

for tag, blk, pgn, ls, lsi, seeds in EXPERIMENTS:
    for seed in seeds:
        run_tag = f"{tag}_s{seed}"
        if run_tag in completed_tags:
            print(f"\n⏭️  Skipping {run_tag} (already completed)")
            continue

        result, hist = run_main_experiment(
            tag, total_blocks=blk, k=9,
            use_post_gn=pgn, gn_groups=8,
            use_layer_scale=ls, layer_scale_init=lsi,
            seed=seed, save_ckpt=True,
        )
        ALL_RESULTS.append(result)
        ALL_HISTORY.extend(hist)

        # Save incrementally (crash-safe)
        pd.DataFrame(ALL_RESULTS).to_csv(results_path, index=False)
        pd.DataFrame(ALL_HISTORY).to_csv(history_path, index=False)
        print(f"  💾 Saved incrementally ({len(ALL_RESULTS)}/{total_runs} runs)")

print("\n" + "=" * 90)
print("✅ ALL Phase 2B experiments completed!")
print("=" * 90)


# =====================================================================
# Cell 2B-5 — Summary table with statistics
# =====================================================================
print("\n" + "=" * 90)
print("PHASE 2B — SUMMARY (Mean ± Std across seeds)")
print("=" * 90)

df_results = pd.read_csv(results_path)

# Group by config
summary = df_results.groupby('config').agg(
    blocks=('blocks', 'first'),
    params_M=('params_M', 'first'),
    acc_mean=('best_acc', 'mean'),
    acc_std=('best_acc', 'std'),
    acc_max=('best_acc', 'max'),
    epoch_mean=('best_epoch', 'mean'),
    throughput=('throughput', 'mean'),
    peak_mem=('peak_mem_MB', 'mean'),
    n_seeds=('seed', 'count'),
).reset_index().sort_values('acc_mean', ascending=False)

print(f"\n{'Config':25s} {'Blk':>4s} {'Params':>7s} "
      f"{'Acc (mean±std)':>16s} {'Best':>7s} "
      f"{'Ep':>5s} {'img/s':>7s} {'Mem':>6s}")
print("-" * 90)
for _, r in summary.iterrows():
    print(f"{r['config']:25s} {r['blocks']:4.0f} {r['params_M']:6.2f}M "
          f"{r['acc_mean']:6.4f}±{r['acc_std']:.4f} {r['acc_max']:7.4f} "
          f"{r['epoch_mean']:5.0f} {r['throughput']:7.0f} {r['peak_mem']:5.0f}M")

# Statistical significance test: B2 vs B1
from scipy import stats as sp_stats
try:
    b1_accs = df_results[df_results['config'] == 'B1_baseline_12']['best_acc'].values
    b2_accs = df_results[df_results['config'] == 'B2_postGN_LS_16']['best_acc'].values
    if len(b1_accs) >= 2 and len(b2_accs) >= 2:
        t_stat, p_val = sp_stats.ttest_ind(b2_accs, b1_accs)
        print(f"\nStatistical test (B2 vs B1): t={t_stat:.3f}, p={p_val:.4f}")
        if p_val < 0.05:
            print(f"  ✅ Significant at p<0.05: PostGN+LS at 16 blocks is "
                  f"statistically better than 12-block baseline")
        else:
            print(f"  ⚠️  Not significant at p<0.05 (need more seeds or larger gap)")
except ImportError:
    print("\n(Install scipy for significance test: pip install scipy)")


# =====================================================================
# Cell 2B-6 — Visualization
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- (0,0) Accuracy bar chart with error bars ---
ax = axes[0, 0]
configs = summary['config'].values
means = summary['acc_mean'].values
stds = summary['acc_std'].values
colors = ['tab:blue' if 'baseline' in c else
          'tab:green' if 'postGN_LS' in c else
          'tab:orange' for c in configs]
bars = ax.barh(range(len(configs)), means, xerr=stds,
               color=colors, alpha=0.8, capsize=4)
ax.set_yticks(range(len(configs)))
ax.set_yticklabels(configs, fontsize=9)
ax.set_xlabel("Val Accuracy (mean ± std)", fontsize=11)
ax.set_title("Phase 2B: All Configs (100 epochs, 3 seeds)", fontsize=12)
ax.grid(alpha=0.3, axis='x')

# --- (0,1) Learning curves for each config (seed-averaged) ---
ax = axes[0, 1]
df_hist = pd.read_csv(history_path)
for config_tag in df_results['config'].unique():
    config_hist = df_hist[df_hist['tag'].str.startswith(config_tag)]
    if config_hist.empty:
        continue
    avg = config_hist.groupby('epoch')['val_acc'].mean()
    ax.plot(avg.index, avg.values, label=config_tag, linewidth=1.5)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Val Accuracy", fontsize=11)
ax.set_title("Learning Curves (seed-averaged)", fontsize=12)
ax.legend(fontsize=7, loc='lower right')
ax.grid(alpha=0.3)

# --- (1,0) Accuracy vs Depth (PostGN+LS only) ---
ax = axes[1, 0]
pgn_ls = summary[summary['config'].str.contains('postGN_LS|baseline')]
ax.errorbar(pgn_ls['blocks'], pgn_ls['acc_mean'], yerr=pgn_ls['acc_std'],
            marker='o', capsize=5, linewidth=2, markersize=8)
for _, r in pgn_ls.iterrows():
    ax.annotate(f"{r['acc_mean']:.2%}", (r['blocks'], r['acc_mean']),
                textcoords="offset points", xytext=(10, 5), fontsize=9)
ax.set_xlabel("Total Blocks (Depth)", fontsize=11)
ax.set_ylabel("Val Accuracy (mean ± std)", fontsize=11)
ax.set_title("Depth Scaling with PostGN + LayerScale", fontsize=12)
ax.grid(alpha=0.3)

# --- (1,1) Throughput vs Accuracy scatter ---
ax = axes[1, 1]
for _, r in summary.iterrows():
    color = 'tab:blue' if 'baseline' in r['config'] else \
            'tab:green' if 'postGN_LS' in r['config'] else 'tab:orange'
    ax.scatter(r['throughput'], r['acc_mean'], s=r['params_M'] * 20,
               color=color, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.annotate(r['config'].replace('_', '\n'), (r['throughput'], r['acc_mean']),
                fontsize=7, ha='center', va='bottom')
ax.set_xlabel("Throughput (img/s)", fontsize=11)
ax.set_ylabel("Val Accuracy", fontsize=11)
ax.set_title("Accuracy vs Throughput (bubble size = params)", fontsize=12)
ax.grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(PHASE2B_DIR, "phase2b_summary.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {plot_path}")


# =====================================================================
# Cell 2B-7 — Decision for Phase 2C
# =====================================================================
print("\n" + "=" * 90)
print("PHASE 2B — DECISION")
print("=" * 90)

champion = summary.iloc[0]  # highest mean accuracy
print(f"\n🏆 Champion config: {champion['config']}")
print(f"   Accuracy: {champion['acc_mean']:.4f} ± {champion['acc_std']:.4f}")
print(f"   Blocks: {champion['blocks']:.0f} | Params: {champion['params_M']:.2f}M")
print(f"   Throughput: {champion['throughput']:.0f} img/s")
print(f"\n   → Proceed to Phase 2C validation experiments with this config.")
print(f"   → Key tests: variable resolution, over-smoothing visualization,")
print(f"     γ analysis per stage, t-SNE feature maps.")

# Save champion config for Phase 2C
champion_config = dict(
    config=champion['config'],
    blocks=int(champion['blocks']),
    params_M=float(champion['params_M']),
    acc_mean=float(champion['acc_mean']),
    acc_std=float(champion['acc_std']),
    throughput=float(champion['throughput']),
)
with open(os.path.join(PHASE2B_DIR, "champion_config.json"), "w") as f:
    json.dump(champion_config, f, indent=2)
print(f"\nSaved champion config to {PHASE2B_DIR}/champion_config.json")
