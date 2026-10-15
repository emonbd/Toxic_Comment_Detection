"""
train.py
Training loop for ToxicClassifier and DualEncoderClassifier.
Supports:
  - Class-weighted BCEWithLogitsLoss
  - Early stopping
  - Per-epoch Macro-F1 evaluation
  - Checkpoint saving

Usage:
    python train.py --model_type single --data_dir data/unified
    python train.py --model_type dual   --data_dir data/unified
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report

from config import (
    MODEL_NAME, NUM_LABELS, LABELS,
    BATCH_SIZE, GRAD_ACCUM_STEPS, LEARNING_RATE,
    WEIGHT_DECAY, WARMUP_RATIO, NUM_EPOCHS,
    EARLY_STOPPING_PAT, ADAM_EPSILON,
    DATA_UNIFIED, MODELS_DIR, SEED,
    MAX_SEQ_LEN,
)
from models import (
    ToxicClassifier, DualEncoderClassifier,
    ToxicDataset, DualEncoderDataset,
    compute_class_weights,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("train")

torch.manual_seed(SEED)
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device, model_type="single") -> dict:
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch  = {k: v.to(device) for k, v in batch.items()
                  if k not in ("idx",)}

        if model_type == "dual":
            out = model(
                comment_input_ids      = batch["comment_input_ids"],
                comment_attention_mask = batch["comment_attention_mask"],
                context_input_ids      = batch["context_input_ids"],
                context_attention_mask = batch["context_attention_mask"],
            )
        else:
            out = model(
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
            )

        all_logits.append(out["logits"].cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)
    probs  = 1 / (1 + np.exp(-logits))     # sigmoid
    preds  = (probs >= 0.5).astype(int)

    macro_f1  = f1_score(labels, preds, average="macro",  zero_division=0)
    micro_f1  = f1_score(labels, preds, average="micro",  zero_division=0)

    per_class = f1_score(labels, preds, average=None, zero_division=0)
    per_class_dict = {l: round(float(f), 4)
                      for l, f in zip(LABELS, per_class)}

    return {
        "macro_f1":     round(macro_f1, 4),
        "micro_f1":     round(micro_f1, 4),
        "per_class_f1": per_class_dict,
    }


# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train(
    model_type:     str  = "single",   # "single" | "dual"
    data_dir:       Path = DATA_UNIFIED,
    model_name:     str  = MODEL_NAME,
    output_dir:     Path = MODELS_DIR,
    use_context:    bool = False,
    adv_normalize:  bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  model_type: {model_type}")

    # ── Load data ─────────────────────────────────────────────
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df   = pd.read_csv(data_dir / "val.csv")

    # Filter ambiguous if present
    if "ambiguous" in train_df.columns:
        train_df = train_df[~train_df["ambiguous"].fillna(False)].copy()

    logger.info(f"Train: {len(train_df):,}  Val: {len(val_df):,}")

    # ── Tokenizer ─────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(model_name)

    # ── Datasets & loaders ────────────────────────────────────
    if model_type == "dual":
        train_ds = DualEncoderDataset(train_df, tok, MAX_SEQ_LEN - 32, 32)
        val_ds   = DualEncoderDataset(val_df,   tok, MAX_SEQ_LEN - 32, 32)
    else:
        train_ds = ToxicDataset(train_df, tok, MAX_SEQ_LEN,
                                use_context=use_context,
                                adversarial_normalize=adv_normalize)
        val_ds   = ToxicDataset(val_df, tok, MAX_SEQ_LEN,
                                use_context=use_context)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2,
                              shuffle=False, num_workers=4)

    # ── Class weights ─────────────────────────────────────────
    weights = compute_class_weights(train_df).to(device)

    # ── Model ─────────────────────────────────────────────────
    if model_type == "dual":
        model = DualEncoderClassifier(model_name=model_name,
                                       class_weights=weights)
    else:
        model = ToxicClassifier(model_name=model_name,
                                 class_weights=weights)
    model.to(device)

    # ── Optimizer & scheduler ─────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps   = len(train_loader) * NUM_EPOCHS // GRAD_ACCUM_STEPS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # ── Training ──────────────────────────────────────────────
    best_macro_f1 = 0.0
    patience      = 0
    save_path     = output_dir / f"{model_type}_model"
    save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device)
            batch  = {k: v.to(device) for k, v in batch.items()
                      if k not in ("idx",)}

            if model_type == "dual":
                out = model(
                    comment_input_ids      = batch["comment_input_ids"],
                    comment_attention_mask = batch["comment_attention_mask"],
                    context_input_ids      = batch["context_input_ids"],
                    context_attention_mask = batch["context_attention_mask"],
                    labels                 = labels,
                )
            else:
                out = model(
                    input_ids      = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                    labels         = labels,
                )

            loss = out["loss"] / GRAD_ACCUM_STEPS
            loss.backward()
            running_loss += loss.item()

            if step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 500 == 0:
                logger.info(f"Epoch {epoch} step {step} "
                            f"loss={running_loss/500:.4f}")
                running_loss = 0.0

        # ── Validation ────────────────────────────────────────
        metrics = evaluate(model, val_loader, device, model_type)
        macro   = metrics["macro_f1"]
        logger.info(f"Epoch {epoch}  Val Macro-F1={macro:.4f}  "
                    f"per-class={metrics['per_class_f1']}")

        if macro > best_macro_f1:
            best_macro_f1 = macro
            patience      = 0
            torch.save(model.state_dict(), save_path / "best_model.pt")
            tok.save_pretrained(save_path)
            logger.info(f"  ✓ New best saved → {save_path}")
        else:
            patience += 1
            logger.info(f"  No improvement ({patience}/{EARLY_STOPPING_PAT})")
            if patience >= EARLY_STOPPING_PAT:
                logger.info("Early stopping triggered.")
                break

    logger.info(f"Training complete. Best Val Macro-F1 = {best_macro_f1:.4f}")
    return best_macro_f1


# ═══════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",   default="single",
                        choices=["single","dual"])
    parser.add_argument("--data_dir",     default=str(DATA_UNIFIED))
    parser.add_argument("--model_name",   default=MODEL_NAME)
    parser.add_argument("--use_context",  action="store_true")
    parser.add_argument("--adv_normalize",action="store_true")
    args = parser.parse_args()

    train(
        model_type    = args.model_type,
        data_dir      = Path(args.data_dir),
        model_name    = args.model_name,
        use_context   = args.use_context,
        adv_normalize = args.adv_normalize,
    )
