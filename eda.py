"""
eda.py
Exploratory Data Analysis for the unified toxic comment dataset.
Generates all figures and statistics needed for the report.

Run after preprocessing_pipeline.py:
    python eda.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from collections import Counter

from config import DATA_UNIFIED, LABELS, ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eda")

FIGURES_DIR = ROOT / "notebooks" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Color palette ───────────────────────────────────────────
COLORS = {
    "TOX":  "#1A56A6",
    "SEV":  "#7C3AED",
    "HATE": "#DC2626",
    "IDNT": "#D97706",
    "THR":  "#B91C1C",
    "INS":  "#6D28D9",
    "OBS":  "#475569",
    "NEU":  "#059669",
}
LABEL_COLORS = [COLORS[l] for l in LABELS]


# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════

def load_unified(split: str = "train") -> pd.DataFrame:
    path = DATA_UNIFIED / f"{split}.csv"
    if not path.exists():
        # Try full dataset
        path = DATA_UNIFIED / "unified_full.csv"
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {split}: {len(df):,} rows")
    return df


# ═══════════════════════════════════════════════════════════════
# 1. OVERALL CLASS DISTRIBUTION
# ═══════════════════════════════════════════════════════════════

def plot_class_distribution(df: pd.DataFrame, title_suffix: str = ""):
    counts = {l: int(df[l].sum()) for l in LABELS if l in df.columns}
    total  = len(df)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(
        list(counts.keys()),
        list(counts.values()),
        color=LABEL_COLORS,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Number of samples", fontsize=12)
    ax.set_title(f"Label distribution{' — ' + title_suffix if title_suffix else ''}",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    # Percentage labels
    for bar, (label, count) in zip(bars, counts.items()):
        pct = count / total * 100
        ax.text(count + total * 0.002, bar.get_y() + bar.get_height() / 2,
                f"{count:,}  ({pct:.1f}%)",
                va="center", fontsize=9, color="#333")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = FIGURES_DIR / "class_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out}")

    # Print table
    logger.info("\nClass distribution:")
    logger.info(f"{'Label':<8} {'Count':>10} {'%':>8}")
    logger.info("-" * 30)
    for label, count in counts.items():
        logger.info(f"{label:<8} {count:>10,} {count/total*100:>7.1f}%")


# ═══════════════════════════════════════════════════════════════
# 2. PER-DATASET SAMPLE COUNTS
# ═══════════════════════════════════════════════════════════════

def plot_dataset_sizes(df: pd.DataFrame):
    if "source_dataset" not in df.columns:
        return
    counts = df["source_dataset"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        counts.index.tolist(),
        counts.values.tolist(),
        color="#1A56A6",
        edgecolor="white",
    )
    ax.set_xlabel("Number of samples", fontsize=12)
    ax.set_title("Samples per source dataset", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    for bar, count in zip(bars, counts.values):
        ax.text(count + counts.max() * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=9)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = FIGURES_DIR / "dataset_sizes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════
# 3. PLATFORM BREAKDOWN (pie)
# ═══════════════════════════════════════════════════════════════

def plot_platform_pie(df: pd.DataFrame):
    if "platform" not in df.columns:
        return
    counts = df["platform"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 7))
    wedge_props = {"edgecolor": "white", "linewidth": 1.5}
    ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops=wedge_props,
        colors=["#0D1B3E","#1A56A6","#0891B2","#059669","#D97706","#7C3AED"],
    )
    ax.set_title("Sample distribution by platform", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "platform_pie.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════
# 4. MULTI-LABEL CO-OCCURRENCE HEATMAP
# ═══════════════════════════════════════════════════════════════

def plot_label_cooccurrence(df: pd.DataFrame):
    avail  = [l for l in LABELS if l in df.columns]
    matrix = df[avail].T.dot(df[avail]).values.astype(float)

    # Normalize to conditional probability P(col|row)
    diag = matrix.diagonal()
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(diag[:, None] > 0, matrix / diag[:, None], 0)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cond, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(avail)))
    ax.set_yticks(range(len(avail)))
    ax.set_xticklabels(avail, rotation=45, ha="right")
    ax.set_yticklabels(avail)

    for i in range(len(avail)):
        for j in range(len(avail)):
            ax.text(j, i, f"{cond[i,j]:.2f}",
                    ha="center", va="center",
                    fontsize=9,
                    color="white" if cond[i,j] > 0.6 else "black")

    plt.colorbar(im, ax=ax, label="P(col | row)")
    ax.set_title("Label co-occurrence (conditional probability)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "label_cooccurrence.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out}")


# ═══════════════════════════════════════════════════════════════
# 5. TEXT LENGTH DISTRIBUTION
# ═══════════════════════════════════════════════════════════════

def plot_text_lengths(df: pd.DataFrame):
    lengths = df["text"].dropna().apply(lambda t: len(t.split()))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths.clip(upper=300), bins=60, color="#1A56A6",
            edgecolor="white", linewidth=0.4)
    ax.axvline(lengths.mean(),   color="#DC2626", linestyle="--",
               label=f"Mean: {lengths.mean():.0f}")
    ax.axvline(lengths.median(), color="#D97706", linestyle=":",
               label=f"Median: {lengths.median():.0f}")
    ax.set_xlabel("Token count (words)", fontsize=12)
    ax.set_ylabel("Samples", fontsize=12)
    ax.set_title("Text length distribution (clipped at 300 tokens)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = FIGURES_DIR / "text_lengths.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out}")

    logger.info(f"\nText length stats: mean={lengths.mean():.1f} "
                f"median={lengths.median():.1f} "
                f"p95={lengths.quantile(0.95):.0f}")


# ═══════════════════════════════════════════════════════════════
# 6. CONTEXT AVAILABILITY
# ═══════════════════════════════════════════════════════════════

def report_context_availability(df: pd.DataFrame):
    if "context_text" not in df.columns:
        return
    has_ctx   = df["context_text"].fillna("").str.strip().ne("").sum()
    pct       = has_ctx / len(df) * 100
    logger.info(f"\nContext availability: {has_ctx:,}/{len(df):,} "
                f"({pct:.1f}%) samples have non-empty context")

    if "source_dataset" in df.columns:
        ctx_by_src = (
            df.groupby("source_dataset")
            .apply(lambda g: g["context_text"].fillna("").str.strip().ne("").mean())
            .round(3)
        )
        logger.info("\nContext availability by source:")
        for src, pct in ctx_by_src.items():
            logger.info(f"  {src:<25}: {pct*100:.1f}%")


# ═══════════════════════════════════════════════════════════════
# 7. SUMMARY STATISTICS TABLE
# ═══════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame):
    total = len(df)
    logger.info("\n" + "="*55)
    logger.info("UNIFIED DATASET SUMMARY")
    logger.info("="*55)
    logger.info(f"Total rows          : {total:,}")

    if "source_dataset" in df.columns:
        logger.info(f"Source datasets     : {df['source_dataset'].nunique()}")

    avail_labels = [l for l in LABELS if l in df.columns]
    logger.info(f"\n{'Label':<8} {'Count':>10} {'%':>8} {'Samples/label':>14}")
    logger.info("-"*44)
    for label in avail_labels:
        n   = int(df[label].sum())
        pct = n / total * 100
        logger.info(f"{label:<8} {n:>10,} {pct:>7.2f}%")

    if "split" in df.columns:
        logger.info("\nSplit distribution:")
        for split, cnt in df["split"].value_counts().items():
            logger.info(f"  {split:<8}: {cnt:,}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run():
    df = load_unified("train")

    print_summary(df)
    plot_class_distribution(df, "training set")
    plot_dataset_sizes(df)
    plot_platform_pie(df)
    plot_label_cooccurrence(df)
    plot_text_lengths(df)
    report_context_availability(df)

    logger.info(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    run()
