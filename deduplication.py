"""
deduplication.py
Exact and near-duplicate removal using MinHash LSH.
Removes ~4% of samples that appear across multiple datasets.
"""

import hashlib
import logging
from typing import List, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ─── Try importing datasketch; graceful fallback ─────────
try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_OK = True
except ImportError:
    DATASKETCH_OK = False
    logger.warning("datasketch not installed. Run: pip install datasketch")


from config import MINHASH_NUM_PERM, MINHASH_NGRAM, LSH_JACCARD_THRESHOLD


# ═══════════════════════════════════════════════════════════════
# 1. EXACT DEDUPLICATION
# ═══════════════════════════════════════════════════════════════

def exact_dedup(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Remove rows with identical cleaned text.
    Keeps first occurrence per group.
    """
    before = len(df)
    df = df.drop_duplicates(subset=[text_col], keep="first")
    after = len(df)
    logger.info(f"Exact dedup: {before - after:,} rows removed "
                f"({(before-after)/before*100:.1f}%)")
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# 2. NEAR-DUPLICATE DETECTION  (MinHash LSH)
# ═══════════════════════════════════════════════════════════════

def _shingles(text: str, n: int = MINHASH_NGRAM) -> List[str]:
    """Generate character-level n-gram shingles from text."""
    text = text.lower().strip()
    return [text[i:i+n] for i in range(max(1, len(text) - n + 1))]


def _make_minhash(text: str, num_perm: int = MINHASH_NUM_PERM) -> "MinHash":
    m = MinHash(num_perm=num_perm)
    for s in _shingles(text):
        m.update(s.encode("utf-8"))
    return m


def near_dedup(
    df: pd.DataFrame,
    text_col: str = "text",
    threshold: float = LSH_JACCARD_THRESHOLD,
    num_perm: int = MINHASH_NUM_PERM,
) -> pd.DataFrame:
    """
    Identify near-duplicate pairs using MinHash LSH.
    For each cluster, keep the sample with the highest label_confidence
    (or the first occurrence if confidence is unavailable).

    Returns deduplicated DataFrame.
    """
    if not DATASKETCH_OK:
        logger.warning("Skipping near-dedup: datasketch not available.")
        return df

    texts  = df[text_col].fillna("").tolist()
    n      = len(texts)
    logger.info(f"Building MinHash signatures for {n:,} texts …")

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}

    for i, text in enumerate(texts):
        if not text.strip():
            continue
        m = _make_minhash(text, num_perm)
        key = str(i)
        lsh.insert(key, m)
        minhashes[key] = m

    # Union-Find for clustering duplicates
    parent = {str(i): str(i) for i in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    logger.info("Running LSH queries …")
    for key, m in minhashes.items():
        neighbours = lsh.query(m)
        for nb in neighbours:
            if nb != key:
                union(key, nb)

    # For each cluster, pick the representative to keep
    clusters: dict = {}
    for i in range(n):
        root = find(str(i))
        clusters.setdefault(root, []).append(i)

    keep_indices = set()
    for root, members in clusters.items():
        if len(members) == 1:
            keep_indices.add(members[0])
            continue
        # Prefer sample with highest label_confidence
        if "label_confidence" in df.columns:
            confs = df.iloc[members]["label_confidence"].fillna(0.0)
            best  = members[confs.argmax()]
        else:
            best = members[0]
        keep_indices.add(best)

    before = len(df)
    df     = df.iloc[sorted(keep_indices)].reset_index(drop=True)
    after  = len(df)
    logger.info(f"Near-dedup: {before - after:,} rows removed "
                f"({(before-after)/before*100:.1f}%)")
    return df


# ═══════════════════════════════════════════════════════════════
# 3. CROSS-SPLIT LEAKAGE CHECK
# ═══════════════════════════════════════════════════════════════

def check_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "text",
) -> int:
    """
    Verify that no test text appears verbatim in train.
    Returns the number of leaking rows (should be 0).
    """
    train_hashes = set(
        hashlib.md5(t.encode()).hexdigest()
        for t in train_df[text_col].fillna("")
    )
    leaking = sum(
        1 for t in test_df[text_col].fillna("")
        if hashlib.md5(t.encode()).hexdigest() in train_hashes
    )
    if leaking > 0:
        logger.warning(f"DATA LEAKAGE: {leaking} test rows found in train!")
    else:
        logger.info("Leakage check passed — no test rows in train.")
    return leaking
