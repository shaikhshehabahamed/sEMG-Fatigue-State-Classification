"""Post-hoc statistical analysis, feature importance, and figure generation."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import confusion_matrix

# Colorblind-safe palette (Wong 2011)
_PALETTE: Dict[str, str] = {
    "Raw": "#0072B2",
    "Monotonic": "#E69F00",
    "Progress": "#009E73",
}
_CLASS_ABBREV = {0: "NF", 1: "Mod.", 2: "Fat."}
_CLASS_FULL = {0: "Non-fatigued", 1: "Moderate", 2: "Fatigued"}


def _paper_rc() -> Dict[str, object]:
    """rcParams for publication-quality figures (Elsevier 2-column format)."""
    return {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "legend.framealpha": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }


def _save_fig(fig, stem: Path) -> None:
    """Save figure as both PDF (vector) and PNG (300 dpi)."""
    for ext in ("pdf", "png"):
        fig.savefig(
            stem.with_suffix(f".{ext}"),
            bbox_inches="tight",
            dpi=300 if ext == "png" else None,
        )


def _abbrev_feature_name(name: str) -> str:
    """Shorten all channel prefixes to 2-letter abbreviations and replace __ with ·."""
    for muscle, ab in [
        ("BICEPS_BRACHII", "BB"),
        ("DELTOID_ANTERIOR", "DA"),
        ("DELTOID_MEDIUS", "DM"),
        ("DELTOID_POSTERIOR", "DP"),
    ]:
        name = name.replace(muscle, ab)
    return name.replace("__", "·")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests(
    fold_df: pd.DataFrame,
    num_classes: int,
    model_name: str,
) -> Dict[str, object]:
    """
    Friedman test (+ Kendall's W effect size) across raw / mono / progress variants,
    pairwise Wilcoxon signed-rank with Bonferroni correction (+ rank-biserial r and
    95% CI on the mean difference), and each variant tested against chance with
    95% CI and rank-biserial effect size.
    """
    chance = 1.0 / num_classes
    out: Dict[str, object] = {"model": model_name, "n_folds": len(fold_df)}
    variant_cols = {
        "raw": "test_macro_f1",
        "mono": "mono_test_macro_f1",
        "progress": "progress_test_macro_f1",
    }
    present_cols = {key: col for key, col in variant_cols.items() if col in fold_df.columns}
    numeric_metrics = fold_df[list(present_cols.values())].apply(pd.to_numeric, errors="coerce")

    variants: Dict[str, np.ndarray] = {}
    for key, col in present_cols.items():
        vals = numeric_metrics[col].dropna().to_numpy(dtype=np.float64)
        if len(vals) >= 5:
            variants[key] = vals

    friedman_keys = [key for key in ("raw", "mono", "progress") if key in present_cols]
    if len(friedman_keys) >= 3:
        friedman_df = numeric_metrics[[present_cols[key] for key in friedman_keys]].dropna()
        n = len(friedman_df)
        try:
            if n < 5:
                raise ValueError(f"Need at least 5 aligned folds, got {n}.")
            friedman_values = friedman_df[
                [present_cols[key] for key in friedman_keys]
            ].to_numpy(dtype=np.float64)
            k_groups = len(friedman_keys)
            if np.allclose(friedman_values, friedman_values[:, [0]]):
                stat, p, kendall_w = 0.0, 1.0, 0.0
            else:
                stat, p = friedmanchisquare(
                    *[friedman_values[:, i] for i in range(friedman_values.shape[1])]
                )
                # Kendall's W = χ²_F / (N · (k−1))
                kendall_w = float(stat) / (n * (k_groups - 1)) if n * (k_groups - 1) > 0 else float("nan")
            out["friedman"] = {
                "statistic": float(stat),
                "p_value": float(p),
                "kendall_w": kendall_w,
                "variants_tested": friedman_keys,
                "n_aligned_folds": n,
            }
        except Exception as exc:
            out["friedman"] = {"error": str(exc)}

    keys = [key for key in ("raw", "mono", "progress") if key in variants]
    pairs = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]
    n_pairs = max(len(pairs), 1)
    pairwise: Dict[str, object] = {}
    for a, b in pairs:
        pair_df = numeric_metrics[[present_cols[a], present_cols[b]]].dropna()
        n = len(pair_df)
        try:
            if n < 5:
                raise ValueError(f"Need at least 5 aligned folds, got {n}.")
            va = pair_df[present_cols[a]].to_numpy(dtype=np.float64)
            vb = pair_df[present_cols[b]].to_numpy(dtype=np.float64)
            diff = va - vb
            mean_diff = float(np.mean(diff))
            se_diff = float(np.std(diff, ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
            if np.allclose(diff, 0.0):
                stat, p = 0.0, 1.0
            else:
                stat, p = wilcoxon(va, vb, alternative="two-sided")
            p_adj = min(1.0, float(p) * n_pairs)
            # rank-biserial r = sign(mean_diff) × (1 − T / E[T|H₀])
            # where E[T|H₀] = n(n+1)/4 for min(T+, T−)
            max_t = n * (n + 1) / 4.0
            r_rb = float(np.sign(mean_diff)) * (1.0 - float(stat) / max_t) if max_t > 0 else 0.0
            pairwise[f"{a}_vs_{b}"] = {
                "statistic": float(stat),
                "p_raw": float(p),
                "p_bonferroni": p_adj,
                "significant_at_0.05": p_adj < 0.05,
                "mean_diff": mean_diff,
                "median_diff": float(np.median(diff)),
                "ci_95_lower_diff": mean_diff - 1.96 * se_diff,
                "ci_95_upper_diff": mean_diff + 1.96 * se_diff,
                "rank_biserial_r": r_rb,
                "n_aligned_folds": n,
            }
        except Exception as exc:
            pairwise[f"{a}_vs_{b}"] = {"error": str(exc)}
    out["pairwise_wilcoxon_bonferroni"] = pairwise

    against_chance: Dict[str, object] = {}
    for k, v in variants.items():
        try:
            diff_from_chance = v - chance
            if np.allclose(diff_from_chance, 0.0):
                stat, p = 0.0, 1.0
            else:
                stat, p = wilcoxon(diff_from_chance, alternative="greater")
            n_v = len(v)
            mean_f1 = float(np.mean(v))
            std_f1 = float(np.std(v, ddof=1)) if n_v > 1 else float("nan")
            se_f1 = std_f1 / np.sqrt(n_v) if n_v > 1 and not np.isnan(std_f1) else float("nan")
            # For 'greater', scipy returns T+ (sum of positive ranks); range [0, n(n+1)/2]
            t_plus_max = n_v * (n_v + 1) / 2.0
            r_rb_chance = 2.0 * float(stat) / t_plus_max - 1.0 if t_plus_max > 0 else float("nan")
            against_chance[k] = {
                "statistic": float(stat),
                "p_value": float(p),
                "significant_at_0.05": float(p) < 0.05,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "ci_95_lower": mean_f1 - 1.96 * se_f1 if not np.isnan(se_f1) else float("nan"),
                "ci_95_upper": mean_f1 + 1.96 * se_f1 if not np.isnan(se_f1) else float("nan"),
                "chance_level": chance,
                "mean_above_chance": mean_f1 - chance,
                "rank_biserial_r": r_rb_chance,
            }
        except Exception as exc:
            against_chance[k] = {"error": str(exc)}
    out["against_chance"] = against_chance

    return out


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def aggregate_feature_importances(
    model_dir: Path,
    feature_names: List[str],
) -> pd.DataFrame:
    """Aggregate estimator feature_importances_ values across all fold models."""
    model_paths = sorted(model_dir.glob("*.joblib"))
    if not model_paths:
        return pd.DataFrame(columns=["feature_name", "mean_importance", "std_importance", "n_folds"])

    fi_rows: List[np.ndarray] = []
    for path in model_paths:
        try:
            model = joblib.load(path)
            fi = np.asarray(model.feature_importances_, dtype=np.float64)
            if len(fi) == len(feature_names):
                fi_rows.append(fi)
        except Exception:
            pass

    if not fi_rows:
        return pd.DataFrame(columns=["feature_name", "mean_importance", "std_importance", "n_folds"])

    matrix = np.stack(fi_rows, axis=0)
    mean_fi = matrix.mean(axis=0)
    std_fi = matrix.std(axis=0, ddof=1) if len(fi_rows) > 1 else np.zeros(len(feature_names))

    return (
        pd.DataFrame({
            "feature_name": feature_names,
            "mean_importance": mean_fi,
            "std_importance": std_fi,
            "n_folds": len(fi_rows),
        })
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Ablation table
# ---------------------------------------------------------------------------

def ablation_table(fold_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary table (mean ± std) for raw / mono / progress variants.
    Returns a tidy DataFrame suitable for direct LaTeX export.
    """
    rows = []
    for _, row in fold_df.iterrows():
        for variant, f1_col, kappa_col, mcc_col in [
            ("raw", "test_macro_f1", "test_cohen_kappa", "test_mcc"),
            ("mono", "mono_test_macro_f1", "mono_test_cohen_kappa", "mono_test_mcc"),
            ("progress", "progress_test_macro_f1", "progress_test_cohen_kappa", "progress_test_mcc"),
        ]:
            f1 = row.get(f1_col)
            if pd.isna(f1):
                continue
            rows.append({
                "variant": variant,
                "subject": row["test_subject"],
                "macro_f1": float(f1),
                "cohen_kappa": float(row.get(kappa_col) or 0.0),
                "mcc": float(row.get(mcc_col) or 0.0),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return (
        df.groupby("variant")[["macro_f1", "cohen_kappa", "mcc"]]
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", lambda x: x.std(ddof=1)),
            mean_cohen_kappa=("cohen_kappa", "mean"),
            std_cohen_kappa=("cohen_kappa", lambda x: x.std(ddof=1)),
            mean_mcc=("mcc", "mean"),
            std_mcc=("mcc", lambda x: x.std(ddof=1)),
        )
        .round(4)
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Paper tables (LaTeX / booktabs)
# ---------------------------------------------------------------------------

def _fmt(mean: float, std: float) -> str:
    return f"${mean:.3f} \\pm {std:.3f}$"


def _col_stats(fold_df: pd.DataFrame, col: str):
    if col not in fold_df.columns:
        return None, None
    v = fold_df[col].dropna()
    if v.empty:
        return None, None
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def generate_paper_tables(
    fold_df: pd.DataFrame,
    num_classes: int,
    output_dir: Path,
    model_name: str,
) -> None:
    """Write booktabs-style LaTeX tables for inclusion in the paper."""

    # --- Table 1: main performance metrics ---
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Decoder & Accuracy & Macro $F_1$ & Cohen's $\kappa$ & MCC \\",
        r"\midrule",
    ]
    for display, prefix in [("Raw", "test"), ("Monotonic", "mono_test"), ("Progress", "progress_test")]:
        f1_col = f"{prefix}_macro_f1"
        if f1_col not in fold_df.columns or fold_df[f1_col].dropna().empty:
            continue
        cells = []
        for col in [f"{prefix}_accuracy", f"{prefix}_macro_f1",
                    f"{prefix}_cohen_kappa", f"{prefix}_mcc"]:
            mean, std = _col_stats(fold_df, col)
            cells.append(_fmt(mean, std) if mean is not None else "---")
        lines.append(f"{display} & {' & '.join(cells)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (output_dir / f"{model_name}_results.tex").write_text("\n".join(lines))

    # --- Table 2: per-class recall ---
    class_headers = " & ".join(_CLASS_FULL.get(c, f"Class {c}") for c in range(num_classes))
    lines2 = [
        r"\begin{tabular}{l" + "c" * num_classes + "}",
        r"\toprule",
        f"Decoder & {class_headers} \\\\",
        r"\midrule",
    ]
    for display, prefix in [("Raw", "test"), ("Monotonic", "mono"), ("Progress", "progress")]:
        cells = []
        for c in range(num_classes):
            col = f"{prefix}_recall_c{c}"
            mean, std = _col_stats(fold_df, col)
            cells.append(_fmt(mean, std) if mean is not None else "---")
        # skip row entirely if no recall columns found
        if any(c != "---" for c in cells):
            lines2.append(f"{display} & {' & '.join(cells)} \\\\")
    lines2 += [r"\bottomrule", r"\end{tabular}"]
    (output_dir / f"{model_name}_recall.tex").write_text("\n".join(lines2))


# ---------------------------------------------------------------------------
# Dataset summary table
# ---------------------------------------------------------------------------

def generate_dataset_table(
    predictions_df: pd.DataFrame,
    num_classes: int,
    output_dir: Path,
) -> None:
    """
    Write a per-subject window-count and class-distribution table as booktabs LaTeX.
    Uses the LOSO test predictions (each subject appears as test subject exactly once),
    so the distribution reflects the full dataset.
    """
    class_labels = [_CLASS_ABBREV.get(c, f"C{c}") for c in range(num_classes)]
    header_pct = " & ".join(f"\\%{lbl}" for lbl in class_labels)

    lines = [
        r"\begin{tabular}{l" + "c" * (num_classes + 2) + "}",
        r"\toprule",
        f"Subject & Windows & {header_pct} & Trials \\\\",
        r"\midrule",
    ]

    subjects = sorted(
        predictions_df["test_subject"].unique(),
        key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 0,
    )
    total_windows = 0
    total_counts = np.zeros(num_classes, dtype=np.int64)

    for subj in subjects:
        sub = predictions_df[predictions_df["test_subject"] == subj]
        n_windows = len(sub)
        counts = np.array([(sub["y_true"] == c).sum() for c in range(num_classes)], dtype=np.int64)
        n_trials = sub["trial_id"].nunique() if "trial_id" in sub.columns else "---"
        total_windows += n_windows
        total_counts += counts
        pct_cells = " & ".join(f"{100 * counts[c] / n_windows:.1f}" for c in range(num_classes))
        subj_label = subj.replace("subject_", "S")
        lines.append(f"{subj_label} & {n_windows} & {pct_cells} & {n_trials} \\\\")

    lines.append(r"\midrule")
    total_pct = " & ".join(
        f"{100 * total_counts[c] / total_windows:.1f}" if total_windows > 0 else "---"
        for c in range(num_classes)
    )
    lines.append(f"Total & {total_windows} & {total_pct} & --- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (output_dir / "dataset_summary.tex").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _average_confusion_matrix(
    predictions_df: pd.DataFrame,
    num_classes: int,
    pred_col: str = "progress_pred",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean and SD of per-subject recall-normalised confusion matrices.
    Each subject contributes equally (macro-average over subjects, not windows).
    """
    cms: List[np.ndarray] = []
    for subject in sorted(predictions_df["test_subject"].unique()):
        sub = predictions_df[predictions_df["test_subject"] == subject]
        cm = confusion_matrix(
            sub["y_true"], sub[pred_col], labels=list(range(num_classes))
        ).astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cms.append(cm / row_sums)
    if not cms:
        z = np.zeros((num_classes, num_classes))
        return z, z
    stack = np.stack(cms)
    mean_cm = stack.mean(axis=0)
    std_cm = stack.std(axis=0, ddof=1) if len(cms) > 1 else np.zeros_like(mean_cm)
    return mean_cm, std_cm


def generate_figures(
    fold_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    num_classes: int,
    output_dir: Path,
    model_name: str,
    top_n_features: int = 20,
) -> None:
    """
    Generate all paper-quality figures. Titles are omitted — supply them
    via LaTeX \\caption{}. PDFs are vector; PNGs are 300 dpi.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed; skipping figure generation. Run: pip install matplotlib")
        return

    plt.rcParams.update(_paper_rc())
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    class_labels = [_CLASS_ABBREV.get(i, f"C{i}") for i in range(num_classes)]

    # --- Figure 1a: Mean normalised CM; Figure 1b: SD across subjects ---
    for pred_col, _ in [("progress_pred", "progress"), ("mono_pred", "mono"), ("raw_pred", "raw")]:
        if pred_col not in predictions_df.columns:
            continue
        avg_cm, std_cm = _average_confusion_matrix(predictions_df, num_classes, pred_col)
        std_thresh = 0.5 * (float(std_cm.max()) or 1.0)

        for mat, cmap, clabel, stem_suffix, txt_thresh, vmax in [
            (avg_cm, "Blues",   "Recall (mean)", "confusion_matrix",    0.5,        1.0),
            (std_cm, "Oranges", "Recall (SD)",   "confusion_matrix_sd", std_thresh, None),
        ]:
            fig, ax = plt.subplots(figsize=(3.8, 3.2))
            im = ax.imshow(mat, vmin=0.0, vmax=vmax, cmap=cmap)
            cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.06)
            cbar.set_label(clabel, fontsize=9)
            cbar.ax.tick_params(labelsize=8)
            ax.set_xticks(range(num_classes))
            ax.set_yticks(range(num_classes))
            ax.set_xticklabels(class_labels)
            ax.set_yticklabels(class_labels)
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            for i in range(num_classes):
                for j in range(num_classes):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=9, color="white" if mat[i, j] > txt_thresh else "black")
            fig.tight_layout()
            _save_fig(fig, fig_dir / f"{model_name}_{stem_suffix}")
            plt.close(fig)

        break  # only the best available variant

    # --- Figure 2: Per-subject macro F1 (grouped bar) ---
    subjects = fold_df["test_subject"].tolist()
    subj_labels = [s.replace("subject_", "S") for s in subjects]
    x = np.arange(len(subjects))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.yaxis.grid(True, linewidth=0.5, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    bar_candidates = [
        ("test_macro_f1", "Raw"),
        ("mono_test_macro_f1", "Monotonic"),
        ("progress_test_macro_f1", "Progress"),
    ]
    present_bars = [(col, lbl) for col, lbl in bar_candidates
                    if col in fold_df.columns and fold_df[col].notna().any()]
    n_bars = len(present_bars)
    if n_bars > 0:
        bar_offsets = np.linspace(-(n_bars - 1) * width / 2, (n_bars - 1) * width / 2, n_bars)
        for boff, (col, label) in zip(bar_offsets, present_bars):
            ax.bar(x + boff, fold_df[col].fillna(0), width,
                   label=label, color=_PALETTE[label], zorder=3)
    ax.axhline(1.0 / num_classes, color="#D55E00", linestyle="--", linewidth=1.0,
               zorder=4, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(subj_labels)
    ax.set_xlim(-0.5, len(subjects) - 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Test subject (LOSO fold)")
    ax.set_ylabel(r"Macro $F_1$")
    ax.legend(loc="upper left", ncol=4)
    fig.tight_layout()
    _save_fig(fig, fig_dir / f"{model_name}_per_subject_f1")
    plt.close(fig)

    # --- Figure 3: Feature importances (top N, horizontal bar) ---
    if not feature_importance_df.empty:
        top = feature_importance_df.head(top_n_features).copy()
        top["display_name"] = top["feature_name"].apply(_abbrev_feature_name)
        fig, ax = plt.subplots(figsize=(3.5, max(3.0, len(top) * 0.33)))
        y_pos = np.arange(len(top))
        err = top["std_importance"].fillna(0).to_numpy() if "std_importance" in top else None
        ax.barh(y_pos, top["mean_importance"], xerr=err, color="#0072B2",
                error_kw={"capsize": 2.5, "elinewidth": 0.8, "ecolor": "0.4"}, zorder=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top["display_name"], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("Mean feature importance")
        ax.xaxis.grid(True, linewidth=0.5, color="0.85", zorder=0)
        ax.set_axisbelow(True)
        fig.tight_layout()
        _save_fig(fig, fig_dir / f"{model_name}_feature_importances")
        plt.close(fig)

    # --- Figure 4: Ablation variant comparison (mean ± SD) ---
    variants, means, stds = [], [], []
    for col, label in [
        ("test_macro_f1", "Raw"),
        ("mono_test_macro_f1", "Monotonic"),
        ("progress_test_macro_f1", "Progress"),
    ]:
        if col in fold_df.columns and fold_df[col].notna().any():
            v = fold_df[col].dropna()
            variants.append(label)
            means.append(float(v.mean()))
            stds.append(float(v.std(ddof=1)) if len(v) > 1 else 0.0)

    if variants:
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        ax.yaxis.grid(True, linewidth=0.5, color="0.85", zorder=0)
        ax.set_axisbelow(True)
        bars = ax.bar(variants, means, yerr=stds, capsize=5,
                      color=[_PALETTE[v] for v in variants],
                      error_kw={"elinewidth": 1.2, "ecolor": "0.3"}, zorder=3)
        ax.axhline(1.0 / num_classes, color="#D55E00", linestyle="--", linewidth=1.0,
                   zorder=4, label=f"Chance (1/{num_classes})")
        top_val = max(m + s for m, s in zip(means, stds))
        ax.set_ylim(0.0, min(1.05, top_val + 0.15))
        ax.set_ylabel(r"Macro $F_1$ (mean $\pm$ SD)")
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2.0, mean + std + 0.015,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        _save_fig(fig, fig_dir / f"{model_name}_ablation")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_post_analysis(
    fold_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    model_dir: Path,
    feature_names: List[str],
    num_classes: int,
    output_dir: Path,
    model_name: str,
) -> Dict[str, object]:
    """Run all post-hoc analyses and write paper-ready outputs to output_dir."""
    results: Dict[str, object] = {}

    stats = run_statistical_tests(fold_df, num_classes, model_name)
    results["statistical_tests"] = stats
    (output_dir / f"{model_name}_statistical_tests.json").write_text(json.dumps(stats, indent=2))

    fi_df = aggregate_feature_importances(model_dir, feature_names)
    if not fi_df.empty:
        fi_df.to_csv(output_dir / f"{model_name}_feature_importances.csv", index=False)
        results["top_20_features"] = fi_df.head(20).to_dict(orient="records")

    abl = ablation_table(fold_df)
    if not abl.empty:
        abl.to_csv(output_dir / f"{model_name}_ablation_table.csv", index=False)
        results["ablation"] = abl.to_dict(orient="records")

    # Progress prior tuning parameter distribution across folds (issue 16/22)
    if "progress_prior_weight" in fold_df.columns:
        weights = fold_df["progress_prior_weight"].dropna()
        temps = fold_df["progress_prior_temperature"].dropna()
        if not weights.empty:
            tuning: Dict[str, object] = {
                "mean_weight": float(weights.mean()),
                "std_weight": float(weights.std(ddof=1)) if len(weights) > 1 else 0.0,
                "mean_temperature": float(temps.mean()),
                "std_temperature": float(temps.std(ddof=1)) if len(temps) > 1 else 0.0,
                "weights_per_fold": [float(w) for w in weights],
                "temperatures_per_fold": [float(t) for t in temps],
            }
            results["progress_tuning"] = tuning
            (output_dir / f"{model_name}_progress_tuning.json").write_text(
                json.dumps(tuning, indent=2)
            )

    generate_paper_tables(fold_df, num_classes, output_dir, model_name)
    generate_dataset_table(predictions_df, num_classes, output_dir)

    generate_figures(
        fold_df=fold_df,
        predictions_df=predictions_df,
        feature_importance_df=fi_df,
        num_classes=num_classes,
        output_dir=output_dir,
        model_name=model_name,
    )

    return results
