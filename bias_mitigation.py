"""
bias_mitigation.py
Counterfactual Data Augmentation (CDA) for reducing identity bias.

For each training sample containing an identity term, creates a swapped copy
with the same label vector. This forces the model to assign equal toxicity
predictions regardless of which identity group is mentioned.

Example:
    "Women are not good at this" → "Men are not good at this" (same labels)
    "Muslims are terrorists"     → "Christians are terrorists" (same labels)
"""

import logging
import re
from typing import List, Tuple

import pandas as pd
import numpy as np

from config import IDENTITY_SWAP_FULL, LABELS

logger = logging.getLogger(__name__)


# ─── Case-preserving swap ────────────────────────────────────
def _swap_case(original: str, replacement: str) -> str:
    """Preserve the case pattern of the original word in the replacement."""
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.capitalize()
    return replacement.lower()


def swap_identity_terms(text: str) -> Tuple[str, bool]:
    """
    Replace identity terms in text using the full swap dictionary.
    Returns (swapped_text, was_changed).

    Matching is whole-word, case-insensitive.
    The case of the original word is preserved in the replacement.
    """
    changed  = False
    result   = text

    # Sort by length descending to avoid partial replacements
    # (e.g., "african american" before "american")
    sorted_keys = sorted(IDENTITY_SWAP_FULL.keys(), key=len, reverse=True)

    for term in sorted_keys:
        replacement = IDENTITY_SWAP_FULL[term]
        pattern     = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        match       = pattern.search(result)
        if match:
            def replace_fn(m):
                return _swap_case(m.group(0), replacement)
            result  = pattern.sub(replace_fn, result)
            changed = True

    return result, changed


# ═══════════════════════════════════════════════════════════════
# COUNTERFACTUAL DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════

def apply_cda(
    df: pd.DataFrame,
    text_col:    str   = "text",
    max_ratio:   float = 1.0,     # max augmented rows as fraction of original
    only_toxic:  bool  = True,    # only augment toxic samples (NEU=0)
) -> pd.DataFrame:
    """
    Apply CDA to training DataFrame.

    For each row where an identity swap is possible:
      1. Generate the counterfactual text.
      2. Copy all label columns from the original.
      3. Append as new rows.

    Args:
        df:         Training DataFrame with LABELS columns.
        text_col:   Column containing comment text.
        max_ratio:  Limit augmented rows to max_ratio × len(df).
        only_toxic: If True, only augment samples that have at least one
                    toxic label set (skip neutral samples).

    Returns:
        Augmented DataFrame (original + counterfactuals).
    """
    augmented_rows = []
    max_new = int(len(df) * max_ratio)

    for _, row in df.iterrows():
        if len(augmented_rows) >= max_new:
            break

        text = str(row.get(text_col, ""))

        # Optionally skip neutral-only samples
        if only_toxic:
            label_sum = sum(int(row.get(l, 0)) for l in LABELS if l != "NEU")
            if label_sum == 0:
                continue

        swapped, changed = swap_identity_terms(text)
        if not changed:
            continue

        # Build new row
        new_row = row.copy()
        new_row[text_col]         = swapped
        new_row["source_dataset"] = row.get("source_dataset", "") + "_cda"
        new_row["label_confidence"] = min(
            float(row.get("label_confidence", 1.0)) * 0.95,  # slight discount
            1.0
        )
        augmented_rows.append(new_row)

    if not augmented_rows:
        logger.warning("CDA: no swappable identity terms found in training data.")
        return df

    aug_df = pd.DataFrame(augmented_rows)
    result = pd.concat([df, aug_df], ignore_index=True)

    logger.info(f"CDA: added {len(augmented_rows):,} counterfactual rows "
                f"({len(augmented_rows)/len(df)*100:.1f}% of original)")
    return result


# ═══════════════════════════════════════════════════════════════
# COUNTERFACTUAL TEST PAIR GENERATION (for evaluation)
# ═══════════════════════════════════════════════════════════════

def build_counterfactual_test_pairs(
    df: pd.DataFrame,
    text_col: str = "text",
) -> pd.DataFrame:
    """
    Build paired samples for counterfactual fairness evaluation.
    For each swappable row, return both original and swapped version
    with a 'pair_id' column linking them.

    The model is fair if it assigns the same prediction to both members
    of each pair.
    """
    pairs = []
    pair_id = 0

    for _, row in df.iterrows():
        text = str(row.get(text_col, ""))
        swapped, changed = swap_identity_terms(text)
        if not changed:
            continue

        orig_row         = row.copy()
        orig_row["pair_id"]    = pair_id
        orig_row["pair_role"]  = "original"

        swap_row         = row.copy()
        swap_row[text_col]     = swapped
        swap_row["pair_id"]    = pair_id
        swap_row["pair_role"]  = "counterfactual"

        pairs.extend([orig_row, swap_row])
        pair_id += 1

    if not pairs:
        logger.warning("No swappable pairs found.")
        return pd.DataFrame()

    pair_df = pd.DataFrame(pairs).reset_index(drop=True)
    logger.info(f"Built {pair_id:,} counterfactual test pairs "
                f"({len(pair_df):,} rows)")
    return pair_df


# ═══════════════════════════════════════════════════════════════
# COUNTERFACTUAL CONSISTENCY METRIC
# ═══════════════════════════════════════════════════════════════

def counterfactual_consistency(
    pair_df: pd.DataFrame,
    preds: np.ndarray,
    label_idx: int = 0,   # 0 = TOX by default
) -> dict:
    """
    Given predictions on the paired test set, compute:
      - Consistency rate: fraction of pairs where predictions agree.
      - Flip rate: fraction of pairs where prediction changes.
      - Average prediction gap.

    A fair model should have consistency rate close to 1.0.
    """
    n_pairs    = pair_df["pair_id"].nunique()
    consistent = 0
    gaps       = []

    for pid in pair_df["pair_id"].unique():
        mask = pair_df["pair_id"] == pid
        idx  = pair_df.index[mask].tolist()
        if len(idx) != 2:
            continue
        p0, p1 = preds[idx[0], label_idx], preds[idx[1], label_idx]
        if p0 == p1:
            consistent += 1
        gaps.append(abs(float(p0) - float(p1)))

    consistency = consistent / n_pairs if n_pairs > 0 else 0.0
    flip_rate   = 1.0 - consistency
    avg_gap     = float(np.mean(gaps)) if gaps else 0.0

    logger.info(f"Counterfactual consistency: {consistency:.4f} "
                f"(flip rate: {flip_rate:.4f}, avg gap: {avg_gap:.4f})")
    return {
        "n_pairs":     n_pairs,
        "consistency": round(consistency, 4),
        "flip_rate":   round(flip_rate, 4),
        "avg_gap":     round(avg_gap, 4),
    }
