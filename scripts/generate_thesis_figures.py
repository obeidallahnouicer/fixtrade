"""
Generate ML thesis figures from real training metrics.
Run with: python scripts/generate_thesis_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path("docs/thesis/assets/ml")
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────
C_XGB     = "#10b981"   # emerald
C_LSTM    = "#6366f1"   # indigo
C_PROPHET = "#f59e0b"   # amber
C_VOL     = "#3b82f6"   # blue
C_LIQ     = "#ec4899"   # pink
BG        = "#0f0f11"
GRID      = "#27272a"
TEXT      = "#e4e4e7"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  TEXT,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "text.color":       TEXT,
    "grid.color":       GRID,
    "grid.linestyle":   "--",
    "grid.linewidth":   0.5,
    "font.family":      "monospace",
    "axes.titlepad":    14,
})

# ─────────────────────────────────────────────────────────────────────
# 1. Model metrics comparison (BIAT, averaged across 3 CV splits)
# ─────────────────────────────────────────────────────────────────────
models  = ["XGBoost", "Prophet"]
mae     = [0.8199,    10.7541]
rmse    = [1.2935,    11.3725]
mape    = [0.84,      11.02]

x   = np.arange(len(models))
w   = 0.28
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Comparaison des métriques de prédiction — BIAT (moyenne walk-forward CV)", fontsize=12)

colors = [C_XGB, C_PROPHET]
for ax, vals, label, unit in zip(
    axes,
    [mae, rmse, mape],
    ["MAE (TND)", "RMSE (TND)", "MAPE (%)"],
    ["TND", "TND", "%"],
):
    bars = ax.bar(x, vals, color=colors, width=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.grid(axis="y", zorder=0)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(vals) * 0.02,
            f"{v:.4f}" if unit == "TND" else f"{v:.2f}%",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylim(0, max(vals) * 1.25)

fig.tight_layout()
fig.savefig(OUT / "model_metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] model_metrics_comparison.png")


# ─────────────────────────────────────────────────────────────────────
# 2. Directional accuracy comparison
# ─────────────────────────────────────────────────────────────────────
dir_models = ["XGBoost", "Prophet"]
dir_acc    = [43.83,      45.12]

fig, ax = plt.subplots(figsize=(7, 4))
ax.set_title("Exactitude directionnelle — BIAT (moyenne walk-forward CV)")
bars = ax.barh(dir_models, dir_acc, color=[C_XGB, C_PROPHET], height=0.4, zorder=3)
ax.axvline(50, color="#ef4444", linestyle="--", linewidth=1.2, label="Seuil aléatoire 50 %")
ax.set_xlabel("Exactitude directionnelle (%)")
ax.set_xlim(0, 70)
ax.grid(axis="x", zorder=0)
for bar, v in zip(bars, dir_acc):
    ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}%", va="center", fontsize=10)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "directional_accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] directional_accuracy_comparison.png")


# ─────────────────────────────────────────────────────────────────────
# 3. R² comparison
# ─────────────────────────────────────────────────────────────────────
r2_models = ["XGBoost", "Prophet"]
r2_vals   = [0.8879,    -6.1271]

fig, ax = plt.subplots(figsize=(7, 4))
ax.set_title("Coefficient de détermination R² — BIAT (moyenne walk-forward CV)")
bar_colors = [C_XGB if v >= 0 else "#ef4444" for v in r2_vals]
bars = ax.bar(r2_models, r2_vals, color=bar_colors, width=0.4, zorder=3)
ax.axhline(0, color=TEXT, linewidth=0.8, linestyle="-")
ax.axhline(1, color="#22c55e", linewidth=0.8, linestyle="--", label="R²=1 (parfait)")
ax.set_ylabel("R²")
ax.grid(axis="y", zorder=0)
for bar, v in zip(bars, r2_vals):
    ypos = v + 0.1 if v >= 0 else v - 0.3
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f"{v:.4f}", ha="center", fontsize=10)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "r2_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] r2_comparison.png")


# ─────────────────────────────────────────────────────────────────────
# 4. XGBoost per-split metrics
# ─────────────────────────────────────────────────────────────────────
splits       = ["Split 1\n(val 2023)", "Split 2\n(val 2024)", "Split 3\n(val 2025)"]
xgb_mae_s    = [0.7823, 0.9608, 0.7167]
xgb_rmse_s   = [1.1798, 1.5404, 1.1604]
xgb_r2_s     = [0.7809, 0.9304, 0.9524]
xgb_dirac_s  = [49.80,  41.60,  40.08]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("XGBoost — Métriques par découpage walk-forward (BIAT)", fontsize=12)
xs = np.arange(len(splits))

for ax, vals, title, fmt in zip(
    axes.flat,
    [xgb_mae_s, xgb_rmse_s, xgb_r2_s, xgb_dirac_s],
    ["MAE (TND)", "RMSE (TND)", "R²", "Exactitude directionnelle (%)"],
    ["{:.4f}", "{:.4f}", "{:.4f}", "{:.2f}%"],
):
    bars = ax.bar(xs, vals, color=C_XGB, width=0.5, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(splits, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", zorder=0)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(abs(x) for x in vals) * 0.02,
                fmt.format(v), ha="center", va="bottom", fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "xgboost_cv_splits.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] xgboost_cv_splits.png")


# ─────────────────────────────────────────────────────────────────────
# 5. XGBoost feature importance (BIAT)
# ─────────────────────────────────────────────────────────────────────
feat_labels = [
    "plus_haut", "plus_bas", "cloture",
    "close_lag_2", "close_lag_1", "close_lag_3",
    "ouverture", "sma_5", "bb_lower", "vwap",
]
feat_imps = [30.47, 27.70, 26.30, 5.47, 5.47, 1.33, 0.62, 0.53, 0.48, 0.38]
feat_labels_fr = [
    "Plus haut", "Plus bas", "Clôture",
    "Clôture J-2", "Clôture J-1", "Clôture J-3",
    "Ouverture", "SMA 5", "Boll. Inf.", "VWAP",
]

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_title("Importance des variables — XGBoost BIAT (top 10)")
colors_feat = [C_XGB] * 3 + [C_VOL] * 3 + [C_PROPHET] * 4
bars = ax.barh(feat_labels_fr[::-1], feat_imps[::-1], color=colors_feat[::-1], height=0.6, zorder=3)
ax.set_xlabel("Importance (%)")
ax.grid(axis="x", zorder=0)
for bar, v in zip(bars, feat_imps[::-1]):
    ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}%", va="center", fontsize=9)
ax.set_xlim(0, 38)

patches = [
    mpatches.Patch(color=C_XGB,     label="Prix intrajournalier"),
    mpatches.Patch(color=C_VOL,     label="Retards de clôture"),
    mpatches.Patch(color=C_PROPHET, label="Indicateurs techniques"),
]
ax.legend(handles=patches, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "xgboost_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] xgboost_feature_importance.png")


# ─────────────────────────────────────────────────────────────────────
# 6. Ensemble weights (global vs BIAT)
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Poids optimisés de l'ensemble — inverse du RMSE de validation")

for ax, weights, title in zip(
    axes,
    [
        {"XGBoost": 89.978, "LSTM": 9.128, "Prophet": 0.894},
        {"XGBoost": 92.19, "Prophet": 7.81, "LSTM": 0.00},
    ],
    ["Ensemble global", "Ensemble BIAT"],
):
    non_zero = {k: v for k, v in weights.items() if v > 0}
    colors_w  = [C_XGB if "XGB" in k else (C_LSTM if "LSTM" in k else C_PROPHET)
                 for k in non_zero]
    wedges, texts, autotexts = ax.pie(
        list(non_zero.values()),
        labels=list(non_zero.keys()),
        colors=colors_w,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10, "color": TEXT},
    )
    for at in autotexts:
        at.set_color(BG)
        at.set_fontsize(9)
    ax.set_title(title, fontsize=11)
    ax.set_facecolor(BG)

fig.tight_layout()
fig.savefig(OUT / "ensemble_weights.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] ensemble_weights.png")


# ─────────────────────────────────────────────────────────────────────
# 7. Liquidity classifier accuracy per split
# ─────────────────────────────────────────────────────────────────────
liq_acc  = [69.44, 62.15, 70.37]
liq_size = [252,   251,   243]

fig, ax = plt.subplots(figsize=(7, 4))
ax.set_title("LiquidityXGB — Précision par découpage walk-forward (BIAT)")
bars = ax.bar(splits, liq_acc, color=C_LIQ, width=0.4, zorder=3)
ax.axhline(67.32, color="#facc15", linewidth=1.2, linestyle="--", label=f"Moyenne = 67.32 %")
ax.set_ylabel("Précision (%)")
ax.set_ylim(50, 80)
ax.grid(axis="y", zorder=0)
for bar, acc, n in zip(bars, liq_acc, liq_size):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.2f}%\n(n={n})", ha="center", fontsize=9)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "liquidity_classifier_accuracy.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] liquidity_classifier_accuracy.png")


# ─────────────────────────────────────────────────────────────────────
# 8. Walk-forward validation schema (Gantt-style)
# ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_title("Protocole de validation walk-forward — FixTrade")

splits_wf = [
    ("Split 1", 2016, 2022, 2023, 2023, 2024, 2024),
    ("Split 2", 2016, 2023, 2023, 2024, 2024, 2025),
    ("Split 3", 2016, 2024, 2024, 2025, None, None),
]
yticks = [2, 1, 0]
labels = [s[0] for s in splits_wf]
ax.set_yticks(yticks)
ax.set_yticklabels(labels, fontsize=10)

for y, (name, ts, te, vs, ve, tsts, tste) in zip(yticks, splits_wf):
    ax.barh(y, te - ts, left=ts, height=0.4, color=C_XGB,     alpha=0.85, zorder=3, label="Entraînement" if y == 2 else "")
    ax.barh(y, ve - vs, left=vs, height=0.4, color=C_PROPHET,  alpha=0.85, zorder=3, label="Validation" if y == 2 else "")
    if tsts is not None:
        ax.barh(y, tste - tsts, left=tsts, height=0.4, color=C_LIQ, alpha=0.85, zorder=3, label="Test" if y == 2 else "")

ax.set_xlabel("Année")
ax.set_xlim(2015, 2026)
ax.grid(axis="x", zorder=0)
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "walkforward_validation_schema.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] walkforward_validation_schema.png")


# ─────────────────────────────────────────────────────────────────────
# 9. Synthetic actual vs predicted (illustrative — clearly labelled)
# ─────────────────────────────────────────────────────────────────────
rng  = np.random.default_rng(42)
n    = 60
days = np.arange(n)
# Simulate a BIAT-like price around 100 TND with realistic noise
actual = 100 + np.cumsum(rng.normal(0, 0.4, n))
# XGBoost prediction: close but not perfect; MA-like on actual
pred   = actual + rng.normal(0, 0.8, n)

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_title(
    "Cours réel vs. cours prédit — illustration sur 60 séances\n"
    "(données simulées à titre illustratif, métriques réelles issues du pipeline)"
)
ax.plot(days, actual, color=TEXT,    linewidth=1.8, label="Cours réel")
ax.plot(days, pred,   color=C_XGB,   linewidth=1.4, linestyle="--", label="Prédiction XGBoost")
ax.fill_between(days, pred - 1.30, pred + 1.30, color=C_XGB, alpha=0.12, label="Intervalle ±RMSE")
ax.axvline(40, color="#facc15", linewidth=1, linestyle=":", label="Début prédiction")
ax.set_xlabel("Séance de bourse")
ax.set_ylabel("Cours (TND)")
ax.legend(fontsize=9)
ax.grid(zorder=0)
ax.text(0.01, 0.04,
        "Note : trajectoire simulée à titre illustratif — RMSE réel XGBoost = 1.2935 TND",
        transform=ax.transAxes, fontsize=8, color="#a1a1aa", style="italic")
fig.tight_layout()
fig.savefig(OUT / "prediction_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] prediction_actual_vs_predicted.png")


# ─────────────────────────────────────────────────────────────────────
# 10. Volume prediction per split
# ─────────────────────────────────────────────────────────────────────
vol_mae  = [5813.17, 4085.32, 2101.42]
vol_rmse = [41908.04, 15275.11, 4205.09]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("VolumeXGB — MAE et RMSE par découpage walk-forward (BIAT)")
for ax, vals, label in zip(axes, [vol_mae, vol_rmse], ["MAE (titres)", "RMSE (titres)"]):
    bars = ax.bar(splits, vals, color=C_VOL, width=0.4, zorder=3)
    ax.set_ylabel(label)
    ax.grid(axis="y", zorder=0)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                f"{v:,.0f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "volume_prediction_metrics.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[OK] volume_prediction_metrics.png")


print("\nAll figures saved to", OUT.resolve())
