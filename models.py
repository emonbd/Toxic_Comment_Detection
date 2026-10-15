"""
models.py
PyTorch Dataset, baseline classifier, dual-encoder context-aware model,
and teacher model for label prediction on single-column datasets.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Dict, List

from config import (
    LABELS, NUM_LABELS, MODEL_NAME,
    MAX_SEQ_LEN, DROPOUT,
)


# ═══════════════════════════════════════════════════════════════
# 1. DATASET CLASS
# ═══════════════════════════════════════════════════════════════

class ToxicDataset(Dataset):
    """
    Multi-label toxic comment dataset for single-encoder training.
    Optionally prepends context to text using SEP token.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_len: int = MAX_SEQ_LEN,
        use_context: bool = False,
        adversarial_normalize: bool = False,
    ):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_len    = max_len
        self.use_context= use_context
        self.adv_norm   = adversarial_normalize

        if adversarial_normalize:
            from text_utils import adversarial_normalize as adv_fn
            self._adv_fn = adv_fn
        else:
            self._adv_fn = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        text = str(row.get("text", ""))

        if self._adv_fn:
            text = self._adv_fn(text)

        if self.use_context:
            ctx = str(row.get("context_text", ""))
            if ctx.strip():
                text = f"{ctx} {self.tokenizer.sep_token} {text}"

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        labels = torch.tensor(
            [float(row.get(l, 0)) for l in LABELS],
            dtype=torch.float,
        )

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         labels,
            "idx":            idx,
        }


class DualEncoderDataset(Dataset):
    """
    Dataset for dual-encoder context-aware model.
    Encodes comment and context separately.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_len_comment: int = 96,
        max_len_context: int = 64,
    ):
        self.df              = df.reset_index(drop=True)
        self.tok             = tokenizer
        self.max_c           = max_len_comment
        self.max_ctx         = max_len_context

    def __len__(self):
        return len(self.df)

    def _encode(self, text: str, max_len: int) -> dict:
        enc = self.tok(
            text if text.strip() else "[PAD]",
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        comment = str(row.get("text", ""))
        context = str(row.get("context_text", ""))

        enc_c   = self._encode(comment, self.max_c)
        enc_ctx = self._encode(context, self.max_ctx)

        labels  = torch.tensor(
            [float(row.get(l, 0)) for l in LABELS],
            dtype=torch.float,
        )

        return {
            "comment_input_ids":      enc_c["input_ids"],
            "comment_attention_mask": enc_c["attention_mask"],
            "context_input_ids":      enc_ctx["input_ids"],
            "context_attention_mask": enc_ctx["attention_mask"],
            "labels":                 labels,
        }


# ═══════════════════════════════════════════════════════════════
# 2. BASELINE SINGLE-ENCODER CLASSIFIER
# ═══════════════════════════════════════════════════════════════

class ToxicClassifier(nn.Module):
    """
    RoBERTa / BERT with a multi-label classification head.
    Loss: BCEWithLogitsLoss with class weights.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_labels: int = NUM_LABELS,
        dropout: float = DROPOUT,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.loss_fn    = nn.BCEWithLogitsLoss(
            pos_weight=class_weights
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        out     = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)
        pooled  = out.last_hidden_state[:, 0, :]   # [CLS] token
        pooled  = self.dropout(pooled)
        logits  = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# ═══════════════════════════════════════════════════════════════
# 3. DUAL-ENCODER CONTEXT-AWARE MODEL
# ═══════════════════════════════════════════════════════════════

class DualEncoderClassifier(nn.Module):
    """
    Two transformer encoders sharing weights.
    Comment [CLS] and context [CLS] are concatenated,
    then projected to 8-class output.

    h_comment  = Encoder(comment)[CLS]       shape: (B, H)
    h_context  = Encoder(context)[CLS]       shape: (B, H)
    h_fused    = concat([h_comment, h_ctx])   shape: (B, 2H)
    logits     = Linear(2H → 8)
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_labels: int = NUM_LABELS,
        dropout: float = DROPOUT,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        # Fusion projection: 2H → H → num_labels
        self.fusion     = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )
        self.loss_fn    = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    def _encode(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]   # [CLS]

    def forward(
        self,
        comment_input_ids,
        comment_attention_mask,
        context_input_ids,
        context_attention_mask,
        labels=None,
    ):
        h_c   = self._encode(comment_input_ids, comment_attention_mask)
        h_ctx = self._encode(context_input_ids, context_attention_mask)
        h_c   = self.dropout(h_c)
        h_ctx = self.dropout(h_ctx)

        fused  = torch.cat([h_c, h_ctx], dim=-1)
        logits = self.fusion(fused)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# ═══════════════════════════════════════════════════════════════
# 4. TEACHER MODEL — label prediction for single-column datasets
# ═══════════════════════════════════════════════════════════════

class TeacherPredictor:
    """
    Wraps a trained ToxicClassifier checkpoint and applies it to
    predict 8-class labels for datasets that only have a single
    original label column.

    Usage:
        teacher = TeacherPredictor("models/teacher_roberta")
        df = teacher.predict_and_fill(df, source_col="class",
                                      hard_constraint={0: {"HATE":1,"TOX":1},
                                                       1: {"TOX":1,"INS":1},
                                                       2: {"NEU":1}})
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device  = torch.device(device)
        self.tok     = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model   = ToxicClassifier.from_pretrained(checkpoint_path) \
                       if hasattr(ToxicClassifier, "from_pretrained") \
                       else self._load_model(checkpoint_path)
        self.model.to(self.device).eval()

    def _load_model(self, path: str) -> ToxicClassifier:
        m = ToxicClassifier()
        state = torch.load(f"{path}/pytorch_model.bin",
                           map_location=self.device)
        m.load_state_dict(state)
        return m

    @torch.no_grad()
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 64,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Returns (N, 8) binary numpy array."""
        all_preds = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc   = self.tok(
                batch,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            out    = self.model(**enc)
            probs  = torch.sigmoid(out["logits"]).cpu().numpy()
            preds  = (probs >= threshold).astype(int)
            all_preds.append(preds)
        return np.vstack(all_preds)

    def predict_and_fill(
        self,
        df: pd.DataFrame,
        hard_constraints: Optional[Dict[int, Dict[str, int]]] = None,
        source_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Predict 8-class labels for df["text"] and fill LABELS columns.
        hard_constraints: override specific classes based on original label.
            e.g. {0: {"HATE": 1, "TOX": 1}, 1: {"TOX": 1}, 2: {"NEU": 1}}
        """
        texts  = df["text"].fillna("").tolist()
        preds  = self.predict_batch(texts)

        for i, label in enumerate(LABELS):
            df[label] = preds[:, i]

        # Apply hard constraints (original label → force certain bits)
        if hard_constraints and source_col and source_col in df.columns:
            for orig_val, forced in hard_constraints.items():
                mask = df[source_col] == orig_val
                for label, bit in forced.items():
                    if label in LABELS:
                        df.loc[mask, label] = bit

        return df


# ═══════════════════════════════════════════════════════════════
# 5. CLASS WEIGHT CALCULATION
# ═══════════════════════════════════════════════════════════════

def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss.
    weight_i = (N - N_pos_i) / N_pos_i
    This is equivalent to setting higher loss for rare positive classes.
    """
    n_total = len(df)
    weights = []
    for label in LABELS:
        n_pos = df[label].sum()
        if n_pos == 0:
            weights.append(1.0)
        else:
            w = (n_total - n_pos) / n_pos
            weights.append(float(w))

    wt = torch.tensor(weights, dtype=torch.float)
    return wt
