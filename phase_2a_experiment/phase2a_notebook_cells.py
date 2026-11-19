# =====================================================================
# PHASE 2A — Vision-Specific Anti-Over-Smoothing for ViG
# =====================================================================
# Run these cells AFTER your Phase 1 Cells 1–3 (imports, data, loaders)
# and Cell 5 (train_epoch, validate helpers).
#
# These cells REPLACE Cell 4 (model) with a new configurable version,
# then run experiments P6, P7, P8.
# =====================================================================


# =====================================================================
# Cell 2A-1 — Updated Model with DiffGroupNorm + LayerScale options
# =====================================================================
# Paste this AFTER your imports (Cell 2) and data loaders (Cell 3).
# It redefines the model classes with new knobs.

class LayerNorm2d(nn.Module):
    def __init__(self, ch, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ch))
        self.bias   = nn.Parameter(torch.zeros(ch))
        self.eps = eps
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        y = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * y + self.bias[:, None, None]


class MRConv2d(nn.Module):
    """Max-relative graph conv with optional DropEdge."""
    def __init__(self, in_ch, out_ch, drop_edge_p=0.0):
        super().__init__()
        self.fc = nn.Linear(in_ch * 2, out_ch)
        self.drop_edge_p = drop_edge_p

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x, edge_index, k):
        x = x.float()
        if k == 0 or edge_index.numel() == 0:
            x_zero = torch.zeros_like(x)
            return self.fc(torch.cat([x, x_zero], dim=-1))
        row, col = edge_index
        x_rel = x[col] - x[row]
        BN = x.size(0)
        x_rel = x_rel.view(BN, k, -1)
        if self.training and self.drop_edge_p > 0:
            drop_mask = (torch.rand(BN, k, 1, device=x.device) < self.drop_edge_p)
            x_rel = x_rel.masked_fill(drop_mask, float('-inf'))
        x_max = x_rel.max(dim=1)[0]
        x_max = torch.nan_to_num(x_max, nan=0.0, posinf=0.0, neginf=0.0)
        return self.fc(torch.cat([x, x_max], dim=-1))


class Grapher(nn.Module):
    """
    Pre-norm Grapher with CONFIGURABLE anti-over-smoothing:
      - use_post_gn:    GroupNorm on graph-conv output (per-node, channel-wise)
      - use_layer_scale: Per-channel learnable residual scaling (CaiT-style)
    """
    def __init__(self, ch, k=9, drop_path=0.0, drop_edge_p=0.0,
                 use_post_gn=False, gn_groups=8,
                 use_layer_scale=False, layer_scale_init=1e-4):
        super().__init__()
        self.k, self.ch = k, ch

        # Sublayer 1: graph conv
        self.norm1 = LayerNorm2d(ch)
        self.gcn   = MRConv2d(ch, ch, drop_edge_p=drop_edge_p)

        # Post-graph-conv GroupNorm (per-node, across channels in groups)
        # Unlike PairNorm (centering across nodes → destroyed spatial info),
        # this normalizes channel statistics WITHIN each patch independently.
        self.post_gn = nn.GroupNorm(gn_groups, ch) if use_post_gn else None

        # Sublayer 2: FFN
        self.norm2 = LayerNorm2d(ch)
        self.ffn = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(ch * 4, ch, 1, bias=False),
        )

        self.drop = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # LayerScale: per-channel diagonal scaling (CaiT, Touvron et al. 2021)
        # Initialized to small values so deep layers start contributing ~nothing,
        # then learn to increase their contribution during training.
        if use_layer_scale:
            self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(ch))
            self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(ch))
        else:
            self.gamma1 = None
            self.gamma2 = None

    def _knn(self, x):
        B, C, H, W = x.shape
        N = H * W
        ke = min(self.k, N - 1)
        if ke == 0:
            ei = torch.empty((2, 0), dtype=torch.long, device=x.device)
            return ei, B, N, 0
        with torch.amp.autocast('cuda', enabled=False):
            xf = x.permute(0, 2, 3, 1).reshape(B, N, C).float()
            xn = F.normalize(xf, dim=-1, eps=1e-6)
            sim = torch.bmm(xn, xn.transpose(1, 2))
            sim.diagonal(dim1=1, dim2=2).fill_(-float('inf'))
            _, idx = sim.topk(ke, dim=-1)
        base = torch.arange(B, device=x.device).view(B, 1, 1) * N
        row = (torch.arange(N, device=x.device).view(1, N, 1) + base).expand(B, N, ke)
        col = idx + base.expand(B, N, ke)
        ei = torch.stack([row.reshape(-1), col.reshape(-1)], 0)
        return ei, B, N, ke

    def forward(self, x):
        B, C, H, W = x.shape

        # --- Sublayer 1: Graph Convolution ---
        h = self.norm1(x)
        ei, Bg, N, ke = self._knn(h)
        h_flat = h.permute(0, 2, 3, 1).reshape(Bg * N, C)
        h_conv = self.gcn(h_flat, ei, ke)                    # (B*N, C)

        # Post-graph-conv GroupNorm (safe: operates on channels, not across nodes)
        if self.post_gn is not None:
            h_conv = self.post_gn(h_conv.unsqueeze(-1)).squeeze(-1)

        h_conv = h_conv.reshape(Bg, H, W, C).permute(0, 3, 1, 2)  # (B,C,H,W)

        # LayerScale on graph-conv branch
        if self.gamma1 is not None:
            h_conv = h_conv * self.gamma1.view(1, -1, 1, 1)

        x = x + self.drop(h_conv)

        # --- Sublayer 2: FFN ---
        h_ffn = self.ffn(self.norm2(x))

        # LayerScale on FFN branch
        if self.gamma2 is not None:
            h_ffn = h_ffn * self.gamma2.view(1, -1, 1, 1)

        x = x + self.drop(h_ffn)
        return x


class PyramidViG(nn.Module):
    def __init__(self, num_classes=1000, embed_dim=48, depths=(2, 2, 6, 2),
                 drop_path_rate=0.1, k=9, drop_edge_p=0.0,
                 use_post_gn=False, gn_groups=8,
                 use_layer_scale=False, layer_scale_init=1e-4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, embed_dim, 16, 16, bias=False),
            LayerNorm2d(embed_dim),
            nn.GELU(),
        )
        self.stages = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i, d in enumerate(depths):
            dim = embed_dim * (2 ** i)
            if i > 0:
                ds = nn.Sequential(
                    nn.Conv2d(dim // 2, dim, 2, 2, bias=False),
                    LayerNorm2d(dim),
                )
            else:
                ds = nn.Identity()
            stage = nn.Sequential(
                ds,
                *[Grapher(dim, k=k, drop_path=dpr[cur + j],
                          drop_edge_p=drop_edge_p,
                          use_post_gn=use_post_gn, gn_groups=gn_groups,
                          use_layer_scale=use_layer_scale,
                          layer_scale_init=layer_scale_init)
                  for j in range(d)]
            )
            self.stages.append(stage)
            cur += d
        self.norm = nn.LayerNorm(embed_dim * 8)
        self.head = nn.Linear(embed_dim * 8, num_classes)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for s in self.stages:
            x = s(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(self.norm(x))


# ---- Updated build_model with new options ----
def make_depths(total_blocks, num_stages=4, ratio=(2, 2, 6, 2)):
    r = [x / sum(ratio) for x in ratio]
    d = [max(1, round(total_blocks * x)) for x in r]
    while sum(d) < total_blocks:
        d[2] += 1
    while sum(d) > total_blocks and d[2] > 1:
        d[2] -= 1
    return tuple(d)


def build_model(total_blocks=12, k=9, drop_edge_p=0.0,
                use_post_gn=False, gn_groups=8,
                use_layer_scale=False, layer_scale_init=1e-4,
                num_classes=100):
    depths = make_depths(total_blocks)
    m = PyramidViG(
        num_classes=num_classes, embed_dim=48, depths=depths,
        drop_path_rate=0.1, k=k, drop_edge_p=drop_edge_p,
        use_post_gn=use_post_gn, gn_groups=gn_groups,
        use_layer_scale=use_layer_scale, layer_scale_init=layer_scale_init,
    )
    return m, depths


# Quick sanity check
for cfg_name, cfg in [
    ("baseline",        dict()),
    ("post_gn",         dict(use_post_gn=True)),
    ("layer_scale_1e-2", dict(use_layer_scale=True, layer_scale_init=1e-2)),
    ("combined",        dict(use_post_gn=True, use_layer_scale=True, layer_scale_init=1e-2)),
]:
    _m, _d = build_model(16, **cfg)
    n = sum(p.numel() for p in _m.parameters() if p.requires_grad)
    print(f"  {cfg_name:20s} | depths={_d} | params={n/1e6:.2f}M")
    del _m
torch.cuda.empty_cache()
print("✓ All configs build successfully")


# =====================================================================
# Cell 2A-2 — Updated run_pilot (supports new model options)
# =====================================================================
def run_pilot(tag, total_blocks=12, k=9, drop_edge_p=0.0,
              use_post_gn=False, gn_groups=8,
              use_layer_scale=False, layer_scale_init=1e-4,
              epochs=PILOT_EPOCHS, lr=PILOT_LR, warmup=PILOT_WARMUP):
    """Run one pilot config and return (best_acc, history)."""
    print("=" * 80)
    print(f"PILOT [{tag}] | blocks={total_blocks} k={k} "
          f"post_gn={use_post_gn} layer_scale={use_layer_scale}"
          f"{'(init='+str(layer_scale_init)+')' if use_layer_scale else ''}")
    print("=" * 80)

    model, depths = build_model(
        total_blocks, k=k, drop_edge_p=drop_edge_p,
        use_post_gn=use_post_gn, gn_groups=gn_groups,
        use_layer_scale=use_layer_scale, layer_scale_init=layer_scale_init,
        num_classes=NUM_CLASSES,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  depths = {depths} | params = {n_params/1e6:.2f}M")
    model = model.to(DEVICE)

    tr_c = nn.CrossEntropyLoss()
    vl_c = nn.CrossEntropyLoss()

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim <= 1 or 'norm' in name.lower() or 'bias' in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    opt = optim.AdamW([{'params': decay, 'weight_decay': PILOT_WD},
                       {'params': no_decay, 'weight_decay': 0.0}], lr=lr)
    sc  = GradScaler('cuda')
    sch = CosineLRScheduler(opt, t_initial=epochs, lr_min=1e-6,
                            warmup_t=warmup, warmup_lr_init=1e-6)
    sch.step(0)

    best, hist = 0.0, []
    for ep in range(epochs):
        t0 = time.time()
        tl = train_epoch(model, train_loader, opt, sc, tr_c, DEVICE)
        vl, va = validate(model, val_loader, vl_c, DEVICE)
        sch.step(ep + 1)
        dt = time.time() - t0
        hist.append(dict(
            tag=tag, epoch=ep + 1, train_loss=tl,
            val_loss=vl, val_acc=va, time=dt,
            blocks=total_blocks, k=k,
            post_gn=use_post_gn, layer_scale=use_layer_scale,
            layer_scale_init=layer_scale_init if use_layer_scale else 0.0,
        ))
        star = ""
        if va > best:
            best = va; star = " ★"
        # Print every 5th epoch + first + last for cleaner logs
        if (ep + 1) % 5 == 0 or ep == 0 or (ep + 1) == epochs:
            print(f"  Ep {ep+1:2d}/{epochs} | TrL {tl:.4f} | VaL {vl:.4f} | "
                  f"Acc {va:.4f} | {dt:.1f}s{star}")
    print(f"  → Best val acc [{tag}]: {best:.4f}\n")

    # If LayerScale, print final gamma values to see what the model learned
    if use_layer_scale:
        gammas = []
        for name, p in model.named_parameters():
            if 'gamma' in name:
                gammas.append(p.data.mean().item())
        if gammas:
            print(f"  LayerScale γ stats: mean={np.mean(gammas):.4f} "
                  f"min={np.min(gammas):.4f} max={np.max(gammas):.4f}")

    del model, opt, sc; torch.cuda.empty_cache()
    return best, hist


# =====================================================================
# Cell 2A-3 — P6: PostGraphGroupNorm at depth 16 and 20
# =====================================================================
# WHY THIS MIGHT WORK (unlike PairNorm):
#   PairNorm centers across NODES (patches) → destroys spatial structure
#   GroupNorm normalizes across CHANNELS within each node → safe for vision
#   This stabilizes the magnitude of graph-conv outputs in deeper layers
#   without removing spatial information.

print("\n" + "▓" * 80)
print("▓  P6: PostGraph GroupNorm at depth 16 and 20")
print("▓" * 80)

P6_RESULTS = []
P6_HISTORY = []

# Reference: Phase 1 baselines (no anti-smoothing)
P1_ACC_12 = 0.4538   # your Phase 1 result
P1_ACC_16 = 0.4486
P1_ACC_20 = 0.4402

for blocks in [16, 20]:
    best, hist = run_pilot(
        f"P6_postGN_blocks{blocks}",
        total_blocks=blocks, k=9,
        use_post_gn=True, gn_groups=8,
    )
    P6_RESULTS.append(dict(
        exp="P6", method="PostGN", blocks=blocks,
        best_acc=best,
        vs_same_depth_baseline=best - (P1_ACC_16 if blocks == 16 else P1_ACC_20),
        vs_12_baseline=best - P1_ACC_12,
    ))
    P6_HISTORY.extend(hist)

df_p6 = pd.DataFrame(P6_RESULTS)
print("\n" + df_p6.to_string(index=False))


# =====================================================================
# Cell 2A-4 — P7: LayerScale at depth 16 and 20
# =====================================================================
# WHY THIS SHOULD WORK:
#   LayerScale (from CaiT, Touvron et al. ICCV 2021) initializes a
#   per-channel scaling factor to near-zero for each sublayer output.
#   Deep layers start contributing NOTHING, then gradually learn to
#   contribute. This directly prevents over-smoothing at initialization
#   because early training is dominated by shallow layers.
#
#   This is PROVEN for deep ViTs (CaiT, DeiT-III) and should transfer
#   to deep ViGs since both use pre-norm + residual architecture.
#
#   We test two init values:
#     - 1e-2: moderate damping (layers contribute ~1% initially)
#     - 1e-4: strong damping (layers contribute ~0.01% initially)

print("\n" + "▓" * 80)
print("▓  P7: LayerScale at depth 16 and 20")
print("▓" * 80)

P7_RESULTS = []
P7_HISTORY = []

for blocks in [16, 20]:
    for ls_init in [1e-2, 1e-4]:
        tag = f"P7_LS{ls_init:.0e}_blocks{blocks}"
        best, hist = run_pilot(
            tag,
            total_blocks=blocks, k=9,
            use_layer_scale=True, layer_scale_init=ls_init,
        )
        P7_RESULTS.append(dict(
            exp="P7", method=f"LayerScale(init={ls_init:.0e})",
            blocks=blocks, best_acc=best,
            vs_same_depth_baseline=best - (P1_ACC_16 if blocks == 16 else P1_ACC_20),
            vs_12_baseline=best - P1_ACC_12,
        ))
        P7_HISTORY.extend(hist)

df_p7 = pd.DataFrame(P7_RESULTS)
print("\n" + df_p7.to_string(index=False))


# =====================================================================
# Cell 2A-5 — P8: Combined (best of P6 + P7) at depth 16 and 20
# =====================================================================
# Run PostGN + LayerScale together. Use the better init from P7.

print("\n" + "▓" * 80)
print("▓  P8: Combined PostGN + LayerScale at depth 16 and 20")
print("▓" * 80)

# Pick the better LayerScale init from P7
if len(P7_RESULTS) >= 2:
    # Compare at 16 blocks
    p7_at_16 = [r for r in P7_RESULTS if r['blocks'] == 16]
    if p7_at_16:
        best_ls_init = max(p7_at_16, key=lambda r: r['best_acc'])
        # Extract init value from method string
        best_init = 1e-2 if '1e-02' in best_ls_init['method'] else 1e-4
    else:
        best_init = 1e-2
else:
    best_init = 1e-2

print(f"  Using LayerScale init = {best_init:.0e} (best from P7)\n")

P8_RESULTS = []
P8_HISTORY = []

for blocks in [16, 20]:
    tag = f"P8_combined_blocks{blocks}"
    best, hist = run_pilot(
        tag,
        total_blocks=blocks, k=9,
        use_post_gn=True, gn_groups=8,
        use_layer_scale=True, layer_scale_init=best_init,
    )
    P8_RESULTS.append(dict(
        exp="P8", method=f"PostGN+LS(init={best_init:.0e})",
        blocks=blocks, best_acc=best,
        vs_same_depth_baseline=best - (P1_ACC_16 if blocks == 16 else P1_ACC_20),
        vs_12_baseline=best - P1_ACC_12,
    ))
    P8_HISTORY.extend(hist)

df_p8 = pd.DataFrame(P8_RESULTS)
print("\n" + df_p8.to_string(index=False))


# =====================================================================
# Cell 2A-6 — Complete Phase 2A summary table
# =====================================================================
print("\n" + "=" * 90)
print("PHASE 2A — COMPLETE RESULTS")
print("=" * 90)

# Combine with Phase 1 baselines for context
phase1_baselines = pd.DataFrame([
    dict(exp="P1(ref)", method="Baseline", blocks=12,
         best_acc=P1_ACC_12, vs_same_depth_baseline=0.0, vs_12_baseline=0.0),
    dict(exp="P1(ref)", method="Baseline", blocks=16,
         best_acc=P1_ACC_16, vs_same_depth_baseline=0.0,
         vs_12_baseline=P1_ACC_16 - P1_ACC_12),
    dict(exp="P1(ref)", method="Baseline", blocks=20,
         best_acc=P1_ACC_20, vs_same_depth_baseline=0.0,
         vs_12_baseline=P1_ACC_20 - P1_ACC_12),
])

all_2a = pd.concat([phase1_baselines, df_p6, df_p7, df_p8], ignore_index=True)
all_2a_path = os.path.join(OUTPUT_DIR, "phase2a_results.csv")
all_2a.to_csv(all_2a_path, index=False)

# Pretty print with color-coding via symbols
print(f"\n{'Exp':10s} {'Method':30s} {'Blocks':>6s} {'Acc':>8s} {'vs Baseline':>12s} {'vs 12-blk':>10s}")
print("-" * 80)
for _, row in all_2a.iterrows():
    delta_same = row.get('vs_same_depth_baseline', 0)
    delta_12   = row.get('vs_12_baseline', 0)
    # Symbol: ★ if beats 12-block baseline, ▲ if beats same-depth, ▼ if worse
    if delta_12 > 0.001:
        sym = "★"
    elif delta_same > 0.001:
        sym = "▲"
    else:
        sym = " "
    print(f"{row['exp']:10s} {row['method']:30s} {row['blocks']:6d} "
          f"{row['best_acc']:8.4f} {delta_same:+11.4f} {delta_12:+9.4f} {sym}")

print(f"\nSaved: {all_2a_path}")


# =====================================================================
# Cell 2A-7 — Visualization
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left: Accuracy vs depth, all methods ---
methods = {}
for _, row in all_2a.iterrows():
    key = row['method']
    if key not in methods:
        methods[key] = {'blocks': [], 'acc': []}
    methods[key]['blocks'].append(row['blocks'])
    methods[key]['acc'].append(row['best_acc'])

colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green',
          'tab:purple', 'tab:brown', 'tab:pink']
for i, (method, data) in enumerate(methods.items()):
    sorted_pairs = sorted(zip(data['blocks'], data['acc']))
    bks, acs = zip(*sorted_pairs)
    marker = 'o' if method == 'Baseline' else 's'
    lw = 2.5 if method == 'Baseline' else 1.5
    ls = '-' if method == 'Baseline' else '--'
    axes[0].plot(bks, acs, marker=marker, linewidth=lw, linestyle=ls,
                 color=colors[i % len(colors)], label=method, markersize=8)

axes[0].axhline(P1_ACC_12, ls=':', color='grey', alpha=0.5,
                label=f'12-block baseline ({P1_ACC_12:.4f})')
axes[0].set_xlabel("Total Blocks (Depth)", fontsize=12)
axes[0].set_ylabel("Best Val Accuracy", fontsize=12)
axes[0].set_title("Phase 2A: Anti-Over-Smoothing Methods vs Depth", fontsize=13)
axes[0].legend(fontsize=8, loc='lower left')
axes[0].grid(alpha=0.3)

# --- Right: Bar chart comparing all methods at 16 blocks ---
data_16 = all_2a[all_2a['blocks'] == 16].copy()
if not data_16.empty:
    data_16 = data_16.sort_values('best_acc', ascending=True)
    bar_colors = ['tab:red' if acc < P1_ACC_16 else 'tab:green'
                  for acc in data_16['best_acc']]
    bars = axes[1].barh(range(len(data_16)), data_16['best_acc'],
                        color=bar_colors, alpha=0.8)
    axes[1].set_yticks(range(len(data_16)))
    axes[1].set_yticklabels([f"{r['exp']}: {r['method']}"
                             for _, r in data_16.iterrows()], fontsize=9)
    axes[1].axvline(P1_ACC_12, ls=':', color='blue', alpha=0.7,
                    label=f'12-block baseline ({P1_ACC_12:.4f})')
    axes[1].axvline(P1_ACC_16, ls='--', color='grey', alpha=0.5,
                    label=f'16-block baseline ({P1_ACC_16:.4f})')
    axes[1].set_xlabel("Best Val Accuracy", fontsize=12)
    axes[1].set_title("All Methods @ 16 Blocks", fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "phase2a_summary.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {plot_path}")


# =====================================================================
# Cell 2A-8 — DECISION LOGIC
# =====================================================================
print("\n" + "=" * 80)
print("PHASE 2A — DECISION")
print("=" * 80)

# Check if ANY method at 16 or 20 blocks beats the 12-block baseline
winners = all_2a[(all_2a['exp'] != 'P1(ref)') & (all_2a['vs_12_baseline'] > 0.0)]

if not winners.empty:
    best_row = winners.loc[winners['best_acc'].idxmax()]
    print(f"\n✅ SUCCESS: {best_row['method']} at {best_row['blocks']} blocks")
    print(f"   Accuracy: {best_row['best_acc']:.4f} "
          f"(+{best_row['vs_12_baseline']:.4f} vs 12-block baseline)")
    print(f"\n   → PROCEED to Phase 2B main experiments with this method.")
    print(f"   → Train on full ImageNet-100 for 100 epochs, 3 seeds.")
    print(f"   → Also test at 24 blocks to see if depth scales further.")
else:
    # Check if any method at least improves over SAME-DEPTH baseline
    improvers = all_2a[(all_2a['exp'] != 'P1(ref)') &
                       (all_2a['vs_same_depth_baseline'] > 0.005)]
    if not improvers.empty:
        best_row = improvers.loc[improvers['best_acc'].idxmax()]
        print(f"\n⚠️  PARTIAL SUCCESS: {best_row['method']} at {best_row['blocks']} blocks")
        print(f"   Accuracy: {best_row['best_acc']:.4f} "
              f"(+{best_row['vs_same_depth_baseline']:.4f} vs same-depth baseline, "
              f"but {best_row['vs_12_baseline']:+.4f} vs 12-block)")
        print(f"\n   → Anti-smoothing HELPS at deeper depths but doesn't overcome")
        print(f"     the capacity-vs-regularization trade-off.")
        print(f"   → Consider: the optimal depth may be ~12-14 blocks for this")
        print(f"     model size (5-8M params). Try embed_dim=64 with the winning")
        print(f"     method at 16 blocks in Phase 2B.")
    else:
        print(f"\n❌ NO METHOD beats the baseline at any depth.")
        print(f"   → Conclusion: ViG's over-smoothing at 16-20 blocks is NOT")
        print(f"     caused by feature homogenization (which these methods fix).")
        print(f"     It is likely a capacity/optimization issue.")
        print(f"\n   → PIVOT thesis to efficiency study at 12-block sweet spot.")
        print(f"   → Document all negative results as a systematic study.")
        print(f"     This analysis is itself publishable.")

print(f"\n{'='*80}")

# Save all history
all_history_2a = P6_HISTORY + P7_HISTORY + P8_HISTORY
if all_history_2a:
    hist_path = os.path.join(OUTPUT_DIR, "phase2a_history.csv")
    pd.DataFrame(all_history_2a).to_csv(hist_path, index=False)
    print(f"Saved: {hist_path}")
