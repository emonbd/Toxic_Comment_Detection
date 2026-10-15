"""
dataset_loaders.py
Individual loader functions for each of the 12 datasets.
Each loader returns a DataFrame with at minimum:
  text, context_text, source_dataset, platform
plus original label columns (mapping applied separately).
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────
def _add_meta(df: pd.DataFrame, source: str, platform: str) -> pd.DataFrame:
    df["source_dataset"] = source
    df["platform"]       = platform
    if "context_text" not in df.columns:
        df["context_text"] = ""
    if "label_confidence" not in df.columns:
        df["label_confidence"] = 1.0
    if "annotator_count" not in df.columns:
        df["annotator_count"] = 1
    return df


# ═══════════════════════════════════════════════════════════════
# A1. JIGSAW 2018
# ═══════════════════════════════════════════════════════════════

def load_jigsaw_2018(path: Path) -> pd.DataFrame:
    """
    CSV with columns: id, comment_text, toxic, severe_toxic,
                      obscene, threat, insult, identity_hate
    """
    df = pd.read_csv(path, dtype=str)
    df = df.rename(columns={"comment_text": "text"})
    for col in ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df = _add_meta(df, "jigsaw_2018", "wikipedia")
    return df[["id","text","context_text","source_dataset","platform",
               "toxic","severe_toxic","obscene","threat","insult","identity_hate",
               "label_confidence","annotator_count"]]


# ═══════════════════════════════════════════════════════════════
# A2. CIVIL COMMENTS (parquet)
# ═══════════════════════════════════════════════════════════════

def load_civil_comments(path: Path) -> pd.DataFrame:
    """
    Parquet with: text, toxicity, severe_toxicity, obscene, threat,
                  insult, identity_attack, sexual_explicit + identity cols.
    """
    df = pd.read_parquet(path)

    float_cols = ["toxicity","severe_toxicity","obscene","threat",
                  "insult","identity_attack","sexual_explicit"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Preserve demographic columns
    identity_cols = [c for c in df.columns if c not in
                     ["text","toxicity","severe_toxicity","obscene",
                      "threat","insult","identity_attack","sexual_explicit"]]
    df["annotator_count"] = df.get("annotator_count",
                                   pd.Series([1]*len(df))).fillna(1).astype(int)
    df = _add_meta(df, "civil_comments", "news_forums")

    # Map annotator columns to unified demographic names
    rename_map = {
        "female": "gender", "male": "gender",
        "black": "race", "white": "race", "asian": "race", "latino": "race",
        "muslim": "religion", "jewish": "religion", "christian": "religion",
        "hindu": "religion", "buddhist": "religion",
        "homosexual_gay_or_lesbian": "sexual_orientation",
        "bisexual": "sexual_orientation",
        "transgender": "gender",
        "intellectual_or_learning_disability": "disability",
        "physical_disability": "disability",
        "psychiatric_or_mental_illness": "disability",
    }
    return df


# ═══════════════════════════════════════════════════════════════
# B1. DAVIDSON 2017
# ═══════════════════════════════════════════════════════════════

def load_davidson(path: Path) -> pd.DataFrame:
    """
    CSV with: count, hate_speech, offensive_language, neither, class, tweet
    """
    df = pd.read_csv(path, dtype=str)
    df = df.rename(columns={"tweet": "text"})
    df["class"] = pd.to_numeric(df["class"], errors="coerce").fillna(2).astype(int)
    # annotator count from vote columns
    df["annotator_count"] = (
        pd.to_numeric(df.get("count", 3), errors="coerce").fillna(3).astype(int)
    )
    df = _add_meta(df, "davidson", "twitter")
    return df[["text","class","context_text","source_dataset","platform",
               "label_confidence","annotator_count"]]


# ═══════════════════════════════════════════════════════════════
# B2. OLID 2019
# ═══════════════════════════════════════════════════════════════

def load_olid(path: Path) -> pd.DataFrame:
    """TSV with: id, tweet, subtask_a, subtask_b, subtask_c"""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.rename(columns={"tweet": "text"})
    for col in ["subtask_a","subtask_b","subtask_c"]:
        if col not in df.columns:
            df[col] = "NaN"
        df[col] = df[col].fillna("NaN")
    df = _add_meta(df, "olid", "twitter")
    return df[["id","text","subtask_a","subtask_b","subtask_c",
               "context_text","source_dataset","platform",
               "label_confidence","annotator_count"]]


# ═══════════════════════════════════════════════════════════════
# B3. HASOC 2019
# ═══════════════════════════════════════════════════════════════

def load_hasoc(path: Path) -> pd.DataFrame:
    """TSV with: text_id, text, task_1, task_2, task_3"""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.rename(columns={"text_id": "id"})
    for col in ["task_1","task_2","task_3"]:
        if col not in df.columns:
            df[col] = "NONE"
        df[col] = df[col].fillna("NONE")
    df = _add_meta(df, "hasoc", "twitter_facebook")
    return df[["id","text","task_1","task_2","task_3",
               "context_text","source_dataset","platform",
               "label_confidence","annotator_count"]]


# ═══════════════════════════════════════════════════════════════
# D1. TOXIGEN  (parquet or CSV)
# ═══════════════════════════════════════════════════════════════

def load_toxigen(path: Path) -> pd.DataFrame:
    """
    Parquet/CSV with: prompt, generation, generation_method, group,
                      prompt_label, roberta_prediction, toxicity_ai,
                      toxicity_human, target_group, label
    """
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Use 'generation' as primary text
    if "generation" in df.columns:
        df = df.rename(columns={"generation": "text"})
    elif "text" not in df.columns:
        df["text"] = ""

    # Decode byte-string artifacts
    df["text"] = df["text"].astype(str).str.replace(r"^b['\"]|['\"]$", "", regex=True)

    # Map group → target_group
    if "group" in df.columns and "target_group" not in df.columns:
        df["target_group"] = df["group"]

    for col in ["toxicity_human","toxicity_ai"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "label" not in df.columns:
        df["label"] = "neutral"

    df = _add_meta(df, "toxigen", "synthetic")
    return df


# ═══════════════════════════════════════════════════════════════
# D2. HATEBENCH  (parquet)
# ═══════════════════════════════════════════════════════════════

def load_hatebench(path: Path) -> pd.DataFrame:
    """
    Parquet with: model, status, main_target, sub_target,
                  target_name, pid, prompt, text, hate_label, …
    """
    df = pd.read_parquet(path)
    if "hate_label" in df.columns:
        df["hate_label"] = pd.to_numeric(df["hate_label"], errors="coerce").fillna(0.0)
    df = _add_meta(df, "hatebench", "llm_generated")
    return df


# ═══════════════════════════════════════════════════════════════
# E1. HATEXPLAIN  (JSON)
# ═══════════════════════════════════════════════════════════════

def load_hatexplain(path: Path) -> pd.DataFrame:
    """
    JSON with post_id as keys.  Each value: {post_id, annotators, rationales, post_tokens}
    annotators: [{label, annotator_id, target}]
    """
    from label_mapping import majority_vote_label

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for post_id, data in raw.items():
        tokens    = data.get("post_tokens", [])
        text      = " ".join(tokens)
        annotators= data.get("annotators", [])

        if not annotators:
            continue

        majority_label, confidence, targets = majority_vote_label(annotators)

        records.append({
            "id":              post_id,
            "text":            text,
            "majority_label":  majority_label,
            "target_group":    ", ".join(targets) if targets else "",
            "label_confidence":confidence,
            "annotator_count": len(annotators),
        })

    df = pd.DataFrame(records)
    df = _add_meta(df, "hatexplain", "twitter_gab")
    return df


# ═══════════════════════════════════════════════════════════════
# F1. IMPLICIT HATE SPEECH  (TSV)
# ═══════════════════════════════════════════════════════════════

def load_implicit_hate(path: Path) -> pd.DataFrame:
    """TSV with: post, class, target, implied_statement"""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.rename(columns={"post": "text"})
    if "class" not in df.columns:
        df["class"] = "not_hate"
    df["class"] = df["class"].fillna("not_hate")
    if "target" in df.columns:
        df["target_group"] = df["target"].fillna("")
    df = _add_meta(df, "implicit_hate", "twitter")
    return df


# ═══════════════════════════════════════════════════════════════
# G1. REDDIT HATE SPEECH
# ═══════════════════════════════════════════════════════════════

def load_reddit_hs(path: Path) -> pd.DataFrame:
    """CSV with: reddit_comment, label"""
    df = pd.read_csv(path, dtype=str)
    df = df.rename(columns={"reddit_comment": "text"})
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df = _add_meta(df, "reddit_hs", "reddit")
    return df


# ═══════════════════════════════════════════════════════════════
# H1. MEASURING HATE SPEECH  (parquet — aggregated per comment)
# ═══════════════════════════════════════════════════════════════

def load_measuring_hs(path: Path) -> pd.DataFrame:
    """
    Parquet with per-annotator rows.  Aggregate to per-comment mean scores.
    Preserves annotator demographic columns.
    """
    df = pd.read_parquet(path)

    score_cols = ["sentiment","respect","insult","humiliate","status",
                  "dehumanize","violence","genocide","attack_defend","hatespeech"]
    existing   = [c for c in score_cols if c in df.columns]

    # Per-comment aggregation
    agg_dict   = {c: "mean" for c in existing}
    agg_dict["annotator_id"] = "count"
    if "text" in df.columns:
        agg_dict["text"] = "first"

    grouped = df.groupby("comment_id").agg(agg_dict).reset_index()
    grouped = grouped.rename(columns={"annotator_id": "annotator_count"})

    grouped = _add_meta(grouped, "measuring_hs", "multi_platform")
    return grouped


# ═══════════════════════════════════════════════════════════════
# I1. ETHOS
# ═══════════════════════════════════════════════════════════════

def load_ethos(path: Path) -> pd.DataFrame:
    """
    CSV with semicolon separator.
    Cols: comment; violence; directed_vs_generalized; gender; race;
          national_origin; disability; religion; sexual_orientation
    """
    df = pd.read_csv(path, sep=";", dtype=str)
    df = df.rename(columns={"comment": "text"})
    float_cols = ["violence","directed_vs_generalized","gender","race",
                  "national_origin","disability","religion","sexual_orientation"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Map to unified demographic names
    df["nationality"]       = df.get("national_origin", 0.0)
    df["sexual_orientation"]= df.get("sexual_orientation", 0.0)

    df = _add_meta(df, "ethos", "youtube_reddit")
    return df


# ═══════════════════════════════════════════════════════════════
# DISPATCH TABLE
# ═══════════════════════════════════════════════════════════════

LOADER = {
    "jigsaw_2018":   load_jigsaw_2018,
    "civil_comments":load_civil_comments,
    "davidson":      load_davidson,
    "olid":          load_olid,
    "hasoc":         load_hasoc,
    "toxigen":       load_toxigen,
    "hatebench":     load_hatebench,
    "hatexplain":    load_hatexplain,
    "implicit_hate": load_implicit_hate,
    "reddit_hs":     load_reddit_hs,
    "measuring_hs":  load_measuring_hs,
    "ethos":         load_ethos,
}


def load_dataset(name: str, path: Path) -> pd.DataFrame:
    loader = LOADER.get(name)
    if loader is None:
        raise ValueError(f"No loader registered for '{name}'")
    logger.info(f"Loading {name} from {path} …")
    df = loader(path)
    logger.info(f"  → {len(df):,} rows loaded")
    return df
