"""
preprocessing_pipeline.py
Full pipeline: load → clean → map labels → dedup → split → save unified dataset.

Usage:
    python preprocessing_pipeline.py
"""

import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_RAW, DATA_PROC, DATA_UNIFIED,
    LABELS, METADATA_COLS,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    SEED,
)
from dataset_loaders import load_dataset
from label_mapping import apply_label_mapping
from text_utils import clean_text, adversarial_normalize, is_valid
from deduplication import exact_dedup, near_dedup, check_leakage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ─── Dataset file registry ───────────────────────────────────
# Edit paths to match your actual download locations
DATASET_FILES = {
    "jigsaw_2018":   DATA_RAW / "jigsaw_2018_train_data.csv",
    "civil_comments":DATA_RAW / "civil_comments_1.parquet",
    "davidson":      DATA_RAW / "Davidson.csv",
    "olid":          DATA_RAW / "olid-training-v1.0.tsv",
    "hasoc":         DATA_RAW / "HASOC-english_dataset.tsv",
    "toxigen":       DATA_RAW / "toxigen-train-00000-of-00001.parquet",
    "hatebench":     DATA_RAW / "HateBench-Label-train-00000-of-00001.parquet",
    "hatexplain":    DATA_RAW / "HateXplain-dataset.json",
    "implicit_hate": DATA_RAW / "implicit_hate_v1_stg3_posts.tsv",
    "reddit_hs":     DATA_RAW / "reddit_dataset.csv",
    "measuring_hs":  DATA_RAW / "measuring-hate-speech.parquet",
    "ethos":         DATA_RAW / "Ethos_Dataset_Multi_Label.csv",
}


# ═══════════════════════════════════════════════════════════════
# STEP 1 — Load and preprocess each dataset independently
# ═══════════════════════════════════════════════════════════════

def process_single_dataset(name: str, path: Path) -> pd.DataFrame:
    """Load, clean, map labels, filter, add unique IDs."""
    if not path.exists():
        logger.warning(f"File not found, skipping: {path}")
        return pd.DataFrame()

    # Load
    df = load_dataset(name, path)

    # Clean text
    df["text"] = df["text"].fillna("").apply(clean_text)
    if "context_text" in df.columns:
        df["context_text"] = df["context_text"].fillna("").apply(
            lambda t: clean_text(t) if t else ""
        )

    # Remove empty / too-short texts
    df = df[df["text"].apply(is_valid)].copy()

    # Apply label mapping
    df = apply_label_mapping(df, name)

    # Assign globally unique IDs
    df["id"] = [f"{name}_{i}" for i in range(len(df))]

    # Ensure all LABEL columns exist
    for label in LABELS:
        if label not in df.columns:
            df[label] = 0

    # Ensure metadata columns exist
    for col in ["context_text", "gender", "race", "religion",
                "sexual_orientation", "nationality", "disability", "target_group"]:
        if col not in df.columns:
            df[col] = ""

    # Exact dedup within this dataset
    df = exact_dedup(df, text_col="text")

    logger.info(f"  {name}: {len(df):,} clean samples")
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 2 — Merge all processed datasets
# ═══════════════════════════════════════════════════════════════

def merge_all(processed: list) -> pd.DataFrame:
    combined = pd.concat(processed, ignore_index=True)
    logger.info(f"Combined (pre-dedup): {len(combined):,} rows")
    return combined


# ═══════════════════════════════════════════════════════════════
# STEP 3 — Cross-dataset deduplication
# ═══════════════════════════════════════════════════════════════

def cross_dedup(df: pd.DataFrame) -> tuple:
    """Returns (clean_df, ambiguous_df)."""
    # Split ambiguous samples before dedup
    ambig_mask = df.get("ambiguous", pd.Series([False]*len(df)))
    df_ambig   = df[ambig_mask].copy()
    df_clean   = df[~ambig_mask].copy()

    # Exact dedup across all datasets
    df_clean = exact_dedup(df_clean, text_col="text")

    # MinHash near-dedup SKIPPED — too slow for large datasets
    #df_clean = near_dedup(df_clean, text_col="text")

    # Run separately offline if needed: python src/deduplication.py

    logger.info(f"After cross-dedup — clean: {len(df_clean):,}, "
                f"ambiguous (excluded): {len(df_ambig):,}")
    return df_clean, df_ambig

# ═══════════════════════════════════════════════════════════════
# STEP 4 — Stratified train / val / test split
# ═══════════════════════════════════════════════════════════════

def make_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stratified split by source_dataset × dominant_label.
    Adds 'split' column: 'train' / 'val' / 'test'.
    """
    # Dominant label = label with highest frequency for stratification key
    label_cols = LABELS[:-1]  # exclude NEU
    df["_dominant"] = df[label_cols].idxmax(axis=1)
    df.loc[df[label_cols].sum(axis=1) == 0, "_dominant"] = "NEU"

    strat_key = df["source_dataset"] + "_" + df["_dominant"]

    # First split: (train+val) vs test
    idx_trainval, idx_test = train_test_split(
        df.index,
        test_size=TEST_RATIO,
        stratify=strat_key,
        random_state=SEED,
    )

    strat_trainval = strat_key.loc[idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
        stratify=strat_trainval,
        random_state=SEED,
    )

    df["split"] = "train"
    df.loc[idx_val,  "split"] = "val"
    df.loc[idx_test, "split"] = "test"
    df = df.drop(columns=["_dominant"])

    counts = df["split"].value_counts()
    logger.info(f"Split — train: {counts['train']:,}  "
                f"val: {counts['val']:,}  test: {counts['test']:,}")
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 5 — Save outputs
# ═══════════════════════════════════════════════════════════════

FINAL_COLS = (
    ["id", "text", "context_text", "source_dataset", "platform", "split",
     "label_confidence", "annotator_count",
     "gender", "race", "religion", "sexual_orientation",
     "nationality", "disability", "target_group"]
    + LABELS
)

def save_splits(df: pd.DataFrame):
    for split in ["train", "val", "test"]:
        subset = df[df["split"] == split]
        out    = DATA_UNIFIED / f"{split}.csv"
        subset[FINAL_COLS].to_csv(out, index=False)
        logger.info(f"Saved {out}  ({len(subset):,} rows)")

    # Full unified dataset
    df[FINAL_COLS].to_csv(DATA_UNIFIED / "unified_full.csv", index=False)
    logger.info(f"Saved unified_full.csv  ({len(df):,} rows total)")

    # Schema documentation
    schema = {
        "columns": FINAL_COLS,
        "labels": LABELS,
        "label_encoding": "multi-label binary (0/1 per column)",
        "total_rows": len(df),
        "split_counts": df["split"].value_counts().to_dict(),
        "source_counts": df["source_dataset"].value_counts().to_dict(),
    }
    import json
    with open(DATA_UNIFIED / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    logger.info("Saved schema.json")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run():
    logger.info("═" * 60)
    logger.info("UNIFIED TOXIC COMMENT DATASET PIPELINE")
    logger.info("═" * 60)

    # Step 1 — per-dataset
    processed = []
    for name, path in DATASET_FILES.items():
        df = process_single_dataset(name, path)
        if not df.empty:
            out_path = DATA_PROC / f"{name}_processed.csv"
            df.to_csv(out_path, index=False)
            processed.append(df)

    if not processed:
        logger.error("No datasets loaded. Check file paths in DATASET_FILES.")
        return

    # Step 2 — merge
    combined = merge_all(processed)

    # Step 3 — cross-dedup
    clean_df, ambig_df = cross_dedup(combined)
    ambig_df.to_csv(DATA_UNIFIED / "ambiguous.csv", index=False)

    # Step 4 — splits
    clean_df = make_splits(clean_df)

    # Step 5 — leakage check
    train = clean_df[clean_df["split"] == "train"]
    test  = clean_df[clean_df["split"] == "test"]
    check_leakage(train, test)

    # Step 6 — save
    save_splits(clean_df)

    logger.info("Pipeline complete.")
    logger.info(f"Final unified dataset: {len(clean_df):,} samples")


if __name__ == "__main__":
    run()
