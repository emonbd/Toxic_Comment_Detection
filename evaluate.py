"""
evaluate.py
Full evaluation suite:
  1. Standard metrics  — Macro-F1, per-class P/R/F1, ROC-AUC
  2. Cross-domain      — hold-one-out per source dataset
  3. Fairness          — FPR parity gap across identity groups
  4. Adversarial       — Attack Success Rate on HateBench / adversarial set

Usage:
    python evaluate.py --checkpoint models/single_model \
                       --test_csv data/unified/test.csv  \
                       --adv_csv  data/unified/test_adversarial.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import LABELS, NUM_LABELS, DATA_UNIFIED, MODELS_DIR, MAX_SEQ_LEN
from models import ToxicClassifier, ToxicDataset
from text_utils import adversarial_normalize

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate")


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def load_model(checkpoint: Path, device: torch.device) -> tuple:
    tok   = AutoTokenizer.from_pretrained(str(checkpoint))
    model = ToxicClassifier()
    state = torch.load(checkpoint / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, tok


# ═══════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def get_predictions(
    model,
    tok,
    df: pd.DataFrame,
    device: torch.device,
    threshold: float = 0.5,
    adv_norm: bool = False,
) -> tuple:
    """Returns (probs, preds, labels) all as np.ndarray (N, 8)."""
    ds     = ToxicDataset(df, tok, MAX_SEQ_LEN, adversarial_normalize=adv_norm)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    all_logits, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels").numpy()
        batch  = {k: v.to(device) for k, v in batch.items() if k != "idx"}
        out    = model(**batch)
        all_logits.append(out["logits"].cpu().numpy())
        all_labels.append(labels)

    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)
    probs  = 1 / (1 + np.exp(-logits))
    preds  = (probs >= threshold).astype(int)
    return probs, preds, labels


# ═══════════════════════════════════════════════════════════════
# 1. STANDARD METRICS
# ═══════════════════════════════════════════════════════════════

def standard_metrics(probs, preds, labels) -> dict:
    macro_f1  = f1_score(labels, preds, average="macro",  zero_division=0)
    micro_f1  = f1_score(labels, preds, average="micro",  zero_division=0)

    per_class = {}
    for i, label in enumerate(LABELS):
        per_class[label] = {
            "precision": round(precision_score(labels[:,i], preds[:,i], zero_division=0), 4),
            "recall":    round(recall_score(labels[:,i],    preds[:,i], zero_division=0), 4),
            "f1":        round(f1_score(labels[:,i],        preds[:,i], zero_division=0), 4),
        }
        try:
            per_class[label]["auc"] = round(roc_auc_score(labels[:,i], probs[:,i]), 4)
        except ValueError:
            per_class[label]["auc"] = None

    return {
        "macro_f1":  round(macro_f1, 4),
        "micro_f1":  round(micro_f1, 4),
        "per_class": per_class,
    }


def print_standard(metrics: dict):
    logger.info(f"\n{'='*60}")
    logger.info(f"STANDARD EVALUATION")
    logger.info(f"  Macro-F1 : {metrics['macro_f1']}")
    logger.info(f"  Micro-F1 : {metrics['micro_f1']}")
    logger.info(f"\n  {'Label':<8} {'Precision':>10} {'Recall':>8} "
                f"{'F1':>8} {'AUC':>8}")
    logger.info(f"  {'-'*46}")
    for label, m in metrics["per_class"].items():
        auc = m['auc'] if m['auc'] is not None else "N/A"
        logger.info(f"  {label:<8} {m['precision']:>10.4f} {m['recall']:>8.4f} "
                    f"{m['f1']:>8.4f} {str(auc):>8}")


# ═══════════════════════════════════════════════════════════════
# 2. CROSS-DOMAIN (hold-one-out)
# ═══════════════════════════════════════════════════════════════

def cross_domain_eval(
    model, tok, df: pd.DataFrame, device: torch.device
) -> dict:
    """
    For each source dataset, evaluate the model on samples from that source.
    Measures how well a model trained on ALL OTHER sources generalizes.
    (Since model is already trained on everything here, this measures
     per-domain performance — a proxy for generalization.)
    """
    results = {}
    for source in df["source_dataset"].unique():
        subset = df[df["source_dataset"] == source].copy()
        if len(subset) < 10:
            continue
        probs, preds, labels = get_predictions(model, tok, subset, device)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        results[source] = round(f1, 4)
        logger.info(f"  Cross-domain {source}: Macro-F1={f1:.4f}  (n={len(subset):,})")

    avg = round(np.mean(list(results.values())), 4)
    results["AVERAGE"] = avg
    logger.info(f"  Cross-domain AVERAGE: {avg:.4f}")
    return results


# ═══════════════════════════════════════════════════════════════
# 3. FAIRNESS — FPR PARITY GAP
# ═══════════════════════════════════════════════════════════════

IDENTITY_GROUPS = {
    "gender":             ["female","male","transgender"],
    "race":               ["black","white","asian","latino"],
    "religion":           ["muslim","jewish","christian","hindu"],
    "sexual_orientation": ["homosexual_gay_or_lesbian","bisexual"],
}

def fpr_for_group(
    labels_col: np.ndarray,  # true binary labels for TOX
    preds_col: np.ndarray,   # predicted binary for TOX
) -> float:
    """False Positive Rate = FP / (FP + TN)"""
    tn = ((preds_col == 0) & (labels_col == 0)).sum()
    fp = ((preds_col == 1) & (labels_col == 0)).sum()
    if tn + fp == 0:
        return 0.0
    return float(fp / (fp + tn))


def fairness_eval(
    df: pd.DataFrame,
    probs: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Compute FPR per identity group and report FPR gap.
    Uses Civil Comments identity columns (float presence scores).
    """
    tox_idx = LABELS.index("TOX")
    results = {}

    identity_cols = [c for c in df.columns
                     if c in ["muslim","female","black","white","gay",
                               "homosexual_gay_or_lesbian","transgender",
                               "jewish","christian","asian","latino",
                               "hindu","bisexual"]]

    for col in identity_cols:
        if col not in df.columns:
            continue
        mask = df[col].fillna(0).astype(float) >= 0.5
        if mask.sum() < 20:
            continue
        fpr = fpr_for_group(labels[mask, tox_idx], preds[mask, tox_idx])
        results[col] = round(fpr, 4)

    if not results:
        logger.info("  No identity columns found for fairness eval.")
        return {}

    fpr_values = list(results.values())
    gap = round(max(fpr_values) - min(fpr_values), 4)
    results["FPR_GAP"] = gap

    logger.info(f"\n  FAIRNESS EVALUATION")
    for grp, fpr in results.items():
        logger.info(f"    {grp:<35}: FPR={fpr:.4f}")
    logger.info(f"    FPR GAP (max-min): {gap:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════
# 4. ADVERSARIAL ROBUSTNESS — Attack Success Rate
# ═══════════════════════════════════════════════════════════════

def adversarial_eval(
    model, tok, adv_df: pd.DataFrame, device: torch.device,
    apply_norm: bool = True,
) -> dict:
    """
    Attack Success Rate (ASR):
      = fraction of TRULY TOXIC adversarial samples the model classifies as non-toxic.

    Apply normalization to see the effect of our preprocessing.
    """
    # Only evaluate on samples that are actually hateful/toxic
    tox_mask = adv_df[[l for l in LABELS if l != "NEU"]].sum(axis=1) > 0
    toxic_df  = adv_df[tox_mask].copy()

    if len(toxic_df) == 0:
        logger.warning("No toxic samples in adversarial set.")
        return {}

    # Without normalization
    probs_raw, preds_raw, labels_raw = get_predictions(
        model, tok, toxic_df, device, adv_norm=False
    )
    tox_idx  = LABELS.index("TOX")
    hate_idx = LABELS.index("HATE")

    # ASR: model predicts NEU (both TOX=0 and HATE=0) on a toxic sample
    tox_or_hate_pred  = (preds_raw[:, tox_idx] | preds_raw[:, hate_idx])
    asr_raw = float(1 - tox_or_hate_pred.mean())

    result = {"ASR_no_norm": round(asr_raw, 4)}
    logger.info(f"\n  ADVERSARIAL EVALUATION")
    logger.info(f"    Adversarial samples evaluated: {len(toxic_df):,}")
    logger.info(f"    ASR (no normalization) : {asr_raw:.4f}")

    if apply_norm:
        probs_n, preds_n, _ = get_predictions(
            model, tok, toxic_df, device, adv_norm=True
        )
        toh_norm = (preds_n[:, tox_idx] | preds_n[:, hate_idx])
        asr_norm = float(1 - toh_norm.mean())
        result["ASR_with_norm"] = round(asr_norm, 4)
        result["ASR_reduction"]  = round(asr_raw - asr_norm, 4)
        logger.info(f"    ASR (with char norm)   : {asr_norm:.4f}")
        logger.info(f"    ASR reduction          : {asr_raw - asr_norm:.4f}")

    return result


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run_evaluation(
    checkpoint: Path,
    test_csv:   Path,
    adv_csv:    Optional[Path] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading checkpoint: {checkpoint}")
    model, tok = load_model(checkpoint, device)

    test_df = pd.read_csv(test_csv)
    logger.info(f"Test set: {len(test_df):,} rows")

    probs, preds, labels = get_predictions(model, tok, test_df, device)

    # 1. Standard
    std = standard_metrics(probs, preds, labels)
    print_standard(std)

    # 2. Cross-domain
    logger.info("\n  CROSS-DOMAIN EVALUATION")
    cross = cross_domain_eval(model, tok, test_df, device)

    # 3. Fairness
    fair = fairness_eval(test_df, probs, preds, labels)

    # 4. Adversarial
    adv_result = {}
    if adv_csv and adv_csv.exists():
        adv_df     = pd.read_csv(adv_csv)
        adv_result = adversarial_eval(model, tok, adv_df, device)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info(f"  Macro-F1      : {std['macro_f1']}")
    logger.info(f"  Cross-dom avg : {cross.get('AVERAGE','N/A')}")
    logger.info(f"  FPR Gap       : {fair.get('FPR_GAP','N/A')}")
    if adv_result:
        logger.info(f"  ASR (no norm) : {adv_result.get('ASR_no_norm','N/A')}")
        logger.info(f"  ASR (normed)  : {adv_result.get('ASR_with_norm','N/A')}")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(MODELS_DIR / "single_model"))
    parser.add_argument("--test_csv",   default=str(DATA_UNIFIED / "test.csv"))
    parser.add_argument("--adv_csv",    default=str(DATA_UNIFIED / "test_adversarial.csv"))
    args = parser.parse_args()

    run_evaluation(
        checkpoint = Path(args.checkpoint),
        test_csv   = Path(args.test_csv),
        adv_csv    = Path(args.adv_csv) if args.adv_csv else None,
    )
