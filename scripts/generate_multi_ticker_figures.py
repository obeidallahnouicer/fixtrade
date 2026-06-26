"""
Generate thesis figures from multi-ticker training results.
Input:  logs/metrics_all_tickers.json
Output: docs/thesis/assets/ml/*.png

Run with: python scripts/generate_multi_ticker_figures.py
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.absolute()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

METRICS_FILE = _ROOT / "logs" / "metrics_all_tickers.json"
OUT = _ROOT / "docs" / "thesis" / "assets" / "ml"
OUT.mkdir(parents=True, exist_ok=True)

BG = "#0f0f11"; GRID = "#27272a"; TEXT = "#e4e4e7"
C_XGB = "#10b981"; C_LSTM = "#6366f1"; C_PROPHET = "#f59e0b"
C_VOL = "#3b82f6"; C_LIQ = "#ec4899"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT, "text.color": TEXT,
    "grid.color": GRID, "grid.linestyle": "--", "grid.linewidth": 0.5,
    "font.family": "monospace", "axes.titlepad": 12,
})


def short(name: str, maxlen: int = 12) -> str:
    return name[:maxlen] if len(name) > maxlen else name


def main():
    if not METRICS_FILE.exists():
        print(f"ERROR: {METRICS_FILE} not found. Run train_all_tickers.py first.")
        sys.exit(1)

    raw = json.loads(METRICS_FILE.read_text())

    # Filter to tickers that succeeded (no error key)
    data = {k: v for k, v in raw.items() if "models" in v}
    print(f"Loaded {len(data)} tickers with metrics.")

    tickers = list(data.keys())
    n = len(tickers)
    if n == 0:
        print("No data to plot.")
        return

    # ── Extract per-model metrics ──
    def get(ticker, model, metric):
        try:
            return data[ticker]["models"][model][metric]
        except (KeyError, TypeError):
            return None

    xgb_mae  = [get(t, "XGBoost", "mae")     for t in tickers]
    xgb_rmse = [get(t, "XGBoost", "rmse")    for t in tickers]
    xgb_r2   = [get(t, "XGBoost", "r2")      for t in tickers]
    xgb_da   = [get(t, "XGBoost", "dir_acc") for t in tickers]

    lstm_mae  = [get(t, "LSTM", "mae")     for t in tickers]
    lstm_r2   = [get(t, "LSTM", "r2")      for t in tickers]
    lstm_da   = [get(t, "LSTM", "dir_acc") for t in tickers]

    liq_da   = [get(t, "LiquidityXGB", "dir_acc") for t in tickers]

    labels = [short(t) for t in tickers]
    xs = np.arange(n)

    # ─────────────────────────────────────────────────────────────────
    # 1. XGBoost MAE across all tickers
    # ─────────────────────────────────────────────────────────────────
    valid_mae = [(l, v) for l, v in zip(labels, xgb_mae) if v is not None]
    if valid_mae:
        vlabels, vvals = zip(*sorted(valid_mae, key=lambda x: x[1]))
        fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 5))
        ax.set_title(f"XGBoost — MAE par titre (walk-forward CV, {n} tickers)")
        bars = ax.bar(range(len(vlabels)), vvals, color=C_XGB, width=0.6, zorder=3)
        avg = np.mean(vvals)
        ax.axhline(avg, color="#facc15", linewidth=1.2, linestyle="--",
                   label=f"Moyenne = {avg:.4f} TND")
        ax.set_xticks(range(len(vlabels)))
        ax.set_xticklabels(vlabels, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("MAE (TND)")
        ax.grid(axis="y", zorder=0)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "xgboost_mae_all_tickers.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("[OK] xgboost_mae_all_tickers.png")

    # ─────────────────────────────────────────────────────────────────
    # 2. XGBoost RMSE across all tickers
    # ─────────────────────────────────────────────────────────────────
    valid_rmse = [(l, v) for l, v in zip(labels, xgb_rmse) if v is not None]
    if valid_rmse:
        vlabels, vvals = zip(*sorted(valid_rmse, key=lambda x: x[1]))
        fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 5))
        ax.set_title(f"XGBoost — RMSE par titre (walk-forward CV, {n} tickers)")
        bars = ax.bar(range(len(vlabels)), vvals, color=C_XGB, width=0.6, zorder=3)
        avg = np.mean(vvals)
        ax.axhline(avg, color="#facc15", linewidth=1.2, linestyle="--",
                   label=f"Moyenne = {avg:.4f} TND")
        ax.set_xticks(range(len(vlabels)))
        ax.set_xticklabels(vlabels, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("RMSE (TND)")
        ax.grid(axis="y", zorder=0)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "xgboost_rmse_all_tickers.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("[OK] xgboost_rmse_all_tickers.png")

    # ─────────────────────────────────────────────────────────────────
    # 3. XGBoost R² across all tickers
    # ─────────────────────────────────────────────────────────────────
    valid_r2 = [(l, v) for l, v in zip(labels, xgb_r2) if v is not None]
    if valid_r2:
        vlabels, vvals = zip(*sorted(valid_r2, key=lambda x: x[1], reverse=True))
        colors_r2 = [C_XGB if v >= 0 else "#ef4444" for v in vvals]
        fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 5))
        ax.set_title(f"XGBoost — R² par titre (walk-forward CV, {n} tickers)")
        bars = ax.bar(range(len(vlabels)), vvals, color=colors_r2, width=0.6, zorder=3)
        ax.axhline(0, color=TEXT, linewidth=0.8, linestyle="-")
        ax.axhline(1, color="#22c55e", linewidth=0.8, linestyle="--", label="R²=1 (parfait)")
        ax.set_xticks(range(len(vlabels)))
        ax.set_xticklabels(vlabels, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("R²")
        ax.grid(axis="y", zorder=0)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "xgboost_r2_all_tickers.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("[OK] xgboost_r2_all_tickers.png")

    # ─────────────────────────────────────────────────────────────────
    # 4. Directional accuracy: XGBoost vs LSTM
    # ─────────────────────────────────────────────────────────────────
    paired_da = [
        (l, xg, ls)
        for l, xg, ls in zip(labels, xgb_da, lstm_da)
        if xg is not None and ls is not None
    ]
    if paired_da:
        plabels, pxgb, plstm = zip(*paired_da)
        xsp = np.arange(len(plabels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(max(10, len(plabels) * 0.7), 5))
        ax.set_title("Exactitude directionnelle — XGBoost vs LSTM par titre")
        ax.bar(xsp - w/2, pxgb,  width=w, color=C_XGB,  label="XGBoost", zorder=3)
        ax.bar(xsp + w/2, plstm, width=w, color=C_LSTM, label="LSTM",    zorder=3)
        ax.axhline(50, color="#ef4444", linewidth=1, linestyle="--", label="Seuil 50%")
        ax.set_xticks(xsp)
        ax.set_xticklabels(plabels, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("Exactitude directionnelle (%)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", zorder=0)
        fig.tight_layout()
        fig.savefig(OUT / "directional_accuracy_all_tickers.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("[OK] directional_accuracy_all_tickers.png")

    # ─────────────────────────────────────────────────────────────────
    # 5. Liquidity classifier accuracy per ticker
    # ─────────────────────────────────────────────────────────────────
    valid_liq = [(l, v) for l, v in zip(labels, liq_da) if v is not None and v > 0]
    if valid_liq:
        vlabels, vvals = zip(*sorted(valid_liq, key=lambda x: x[1], reverse=True))
        avg_liq = np.mean(vvals)
        fig, ax = plt.subplots(figsize=(max(10, len(vlabels) * 0.6), 5))
        ax.set_title("LiquidityXGB — Precision par titre (walk-forward CV)")
        ax.bar(range(len(vlabels)), vvals, color=C_LIQ, width=0.6, zorder=3)
        ax.axhline(avg_liq, color="#facc15", linewidth=1.2, linestyle="--",
                   label=f"Moyenne = {avg_liq:.1f}%")
        ax.axhline(33.3, color="#ef4444", linewidth=1, linestyle=":",
                   label="Seuil aléatoire (3 classes, 33.3%)")
        ax.set_xticks(range(len(vlabels)))
        ax.set_xticklabels(vlabels, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("Precision (%)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", zorder=0)
        fig.tight_layout()
        fig.savefig(OUT / "liquidity_accuracy_all_tickers.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("[OK] liquidity_accuracy_all_tickers.png")

    # ─────────────────────────────────────────────────────────────────
    # 6. Heatmap: XGBoost metrics per ticker
    # ─────────────────────────────────────────────────────────────────
    metric_names = ["MAE", "RMSE", "MAPE%", "DirAcc%", "R2"]
    matrix = []
    valid_tickers_heat = []
    for t, lbl in zip(tickers, labels):
        row = [
            get(t, "XGBoost", "mae"),
            get(t, "XGBoost", "rmse"),
            get(t, "XGBoost", "mape"),
            get(t, "XGBoost", "dir_acc"),
            get(t, "XGBoost", "r2"),
        ]
        if all(v is not None for v in row):
            matrix.append(row)
            valid_tickers_heat.append(lbl)

    if matrix:
        mat = np.array(matrix)
        # Normalize each column to [0,1] for color
        mat_norm = np.zeros_like(mat, dtype=float)
        for col in range(mat.shape[1]):
            col_min, col_max = mat[:, col].min(), mat[:, col].max()
            if col_max != col_min:
                mat_norm[:, col] = (mat[:, col] - col_min) / (col_max - col_min)
            else:
                mat_norm[:, col] = 0.5

        fig, ax = plt.subplots(figsize=(9, max(6, len(valid_tickers_heat) * 0.4)))
        im = ax.imshow(mat_norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names, fontsize=9)
        ax.set_yticks(range(len(valid_tickers_heat)))
        ax.set_yticklabels(valid_tickers_heat, fontsize=8)
        ax.set_title("XGBoost — Carte de chaleur des métriques (normalisée)")
        # Annotate cells with actual values
        for i in range(len(valid_tickers_heat)):
            for j in range(len(metric_names)):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if 0.3 < mat_norm[i, j] < 0.7 else "white")
        fig.colorbar(im, ax=ax, label="Score normalisé (vert = meilleur)")
        fig.tight_layout()
        fig.savefig(OUT / "xgboost_heatmap_all_tickers.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("[OK] xgboost_heatmap_all_tickers.png")

    # ─────────────────────────────────────────────────────────────────
    # 7. Summary stats bar chart (model comparison, averaged across all tickers)
    # ─────────────────────────────────────────────────────────────────
    def avg_metric(model, metric):
        vals = [get(t, model, metric) for t in tickers]
        vals = [v for v in vals if v is not None and not (v != v)]
        return np.mean(vals) if vals else None

    summary = {
        "XGBoost": {
            "MAE": avg_metric("XGBoost", "mae"),
            "RMSE": avg_metric("XGBoost", "rmse"),
            "DirAcc%": avg_metric("XGBoost", "dir_acc"),
            "R2": avg_metric("XGBoost", "r2"),
        },
        "LSTM": {
            "MAE": avg_metric("LSTM", "mae"),
            "RMSE": avg_metric("LSTM", "rmse"),
            "DirAcc%": avg_metric("LSTM", "dir_acc"),
            "R2": avg_metric("LSTM", "r2"),
        },
        "Prophet": {
            "MAE": avg_metric("Prophet", "mae"),
            "RMSE": avg_metric("Prophet", "rmse"),
            "DirAcc%": avg_metric("Prophet", "dir_acc"),
            "R2": avg_metric("Prophet", "r2"),
        },
    }

    print("\nSummary (averaged across all tickers):")
    print(f"{'Model':<12} {'MAE':>8} {'RMSE':>8} {'DirAcc%':>9} {'R2':>8}")
    print("-" * 50)
    for model, ms in summary.items():
        print(f"{model:<12} {ms['MAE']:>8.4f} {ms['RMSE']:>8.4f} {ms['DirAcc%']:>9.2f} {ms['R2']:>8.4f}")

    # Plot: average metrics comparison
    metric_keys = ["MAE", "RMSE", "DirAcc%"]
    model_list = ["XGBoost", "LSTM", "Prophet"]
    model_colors = [C_XGB, C_LSTM, C_PROPHET]
    xs2 = np.arange(len(metric_keys))
    w2 = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"Comparaison des modeles — moyenne sur {n} titres BVMT")
    for i, (model, color) in enumerate(zip(model_list, model_colors)):
        vals2 = [summary[model].get(mk) for mk in metric_keys]
        vals2 = [v if v is not None else 0 for v in vals2]
        bars2 = ax.bar(xs2 + (i - 1) * w2, vals2, width=w2, color=color, label=model, zorder=3)
        for bar, v in zip(bars2, vals2):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(xs2)
    ax.set_xticklabels(metric_keys, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", zorder=0)
    ax.set_ylabel("Valeur (TND pour MAE/RMSE, % pour DirAcc)")
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison_all_tickers_avg.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] model_comparison_all_tickers_avg.png")

    print(f"\nAll figures saved to {OUT.resolve()}")


if __name__ == "__main__":
    main()
