import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
out = root / "figures"
out.mkdir(exist_ok=True)

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

deltas = {r["condition_id"]: r for r in csv.DictReader(open(root / "paired_classification_deltas.csv")) if r["metric"] == "roc_auc"}
cis = {r["condition_id"]: r for r in csv.DictReader(open(root / "classification_cis.csv")) if r["metric"] == "roc_auc"}

rows = [("Median", ["median_L1", "median_L2", "median_L3"]),
        ("CLAHE + DWT", ["clahe_dwt_strength_L1", "clahe_dwt_strength_L2", "clahe_dwt_strength_L3"]),
        ("DAE + CLAHE", ["dae_clahe_strength_L1", "dae_clahe_strength_L2", "dae_clahe_strength_L3"]),
        ("Frequency fusion", ["dwt_L1_L4_fusion", "dft_low_mid_high_fusion"])]
short = {"dwt_L1_L4_fusion": "DWT L1–L4", "dft_low_mid_high_fusion": "DFT low/mid/high"}

# Figure 1: paired delta ROC-AUC vs baseline (forest plot)
fig, ax = plt.subplots(figsize=(7.2, 5.6))
y, labels, ys = 0, [], []
for group, ids in rows:
    ax.text(-0.071, y, group, ha="left", va="center", fontweight="bold", color=INK, fontsize=10)
    y -= 1
    for cid in ids:
        d = deltas[cid]
        m, lo, hi = float(d["mean_delta"]), float(d["ci_low"]), float(d["ci_high"])
        sig = hi < 0 or lo > 0
        color = BLUE if sig else MUTED
        ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round")
        ax.plot(m, y, "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=3)
        ax.text(0.024, y, f"{m:+.3f}  [{lo:+.3f}, {hi:+.3f}]", va="center", color=INK2, fontsize=9)
        labels.append(short.get(cid, "L" + cid[-1])); ys.append(y)
        y -= 1
    y -= 0.4
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(ys, labels)
ax.set_xlim(-0.071, 0.07)
ax.set_xticks([-0.06, -0.04, -0.02, 0, 0.02])
ax.set_xlabel("ΔROC-AUC vs baseline (paired patient bootstrap, 95% CI)")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.plot([], [], "o-", color=BLUE, lw=2, ms=7, label="CI excludes 0")
ax.plot([], [], "o-", color=MUTED, lw=2, ms=7, label="CI includes 0")
ax.set_ylim(y + 0.6, 0.8)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
ax.set_title("Change in ROC-AUC from baseline (0.668)", loc="left", color=INK, fontsize=12, pad=12)
fig.tight_layout()
fig.savefig(out / "delta_roc_auc.png", dpi=160)

# Figure 2: ROC-AUC by denoising strength
fig, ax = plt.subplots(figsize=(6.4, 4.2))
b = cis["baseline"]
bm, blo, bhi = float(b["mean"]), float(b["ci_low"]), float(b["ci_high"])
ax.axhspan(blo, bhi, color=GRID, alpha=0.7, lw=0)
ax.axhline(bm, color=INK2, lw=1, ls="--")
ax.text(0.85, bm + 0.0006, "baseline 0.668", va="bottom", color=INK2, fontsize=9)
for (group, ids), color in zip(rows[:3], (BLUE, ORANGE, AQUA)):
    vals = [float(cis[c]["mean"]) for c in ids]
    ax.plot([1, 2, 3], vals, "-o", color=color, lw=2, ms=8, mec=SURFACE, mew=2, label=group)
    ax.text(3.08, vals[-1], group, va="center", color=INK, fontsize=9)
ax.set_xticks([1, 2, 3], ["L1 (weak)", "L2", "L3 (strong)"])
ax.set_xlim(0.8, 3.75)
ax.set_ylabel("ROC-AUC (official test, n = 12,081)")
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK2)
ax.set_title("ROC-AUC by denoising strength (band = baseline 95% CI)", loc="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(out / "roc_auc_by_strength.png", dpi=160)
print("ok")
