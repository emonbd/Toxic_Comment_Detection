"""
text_utils.py
Text cleaning, adversarial normalization, and context augmentation utilities.
"""

import re
import unicodedata
from html import unescape
from typing import Optional

from config import LEET_PATTERNS, LEET_CHAR_MAP, LEVENSHTEIN_MAX_DIST


# ═══════════════════════════════════════════════════════════════
# 1. BASIC TEXT CLEANING
# ═══════════════════════════════════════════════════════════════

def clean_text(text: str, keep_case: bool = True) -> str:
    """
    Standard cleaning pipeline applied to every sample.
    Preserves case by default — ALL CAPS carries aggression signal.
    """
    if not isinstance(text, str):
        return ""

    # Decode HTML entities  (&amp; → & etc.)
    text = unescape(text)

    # Normalize unicode (NFKC collapses ligatures, full-width chars, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Remove byte-string prefix artifacts from some datasets  b'...'
    text = re.sub(r"^b['\"]|['\"]$", "", text.strip())

    # Replace URLs with [URL]
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "[URL]",
        text
    )

    # Replace @usernames with [USER]
    text = re.sub(r"@\w+", "[USER]", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove zero-width / control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse repeated punctuation  !!!!! → !
    text = re.sub(r"([!?.]){3,}", r"\1\1", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if not keep_case:
        text = text.lower()

    return text


def is_valid(text: str, min_tokens: int = 3) -> bool:
    """Return True if text has enough content to be useful."""
    if not isinstance(text, str) or text.strip() == "":
        return False
    tokens = text.split()
    return len(tokens) >= min_tokens


# ═══════════════════════════════════════════════════════════════
# 2. ADVERSARIAL NORMALIZATION
# ═══════════════════════════════════════════════════════════════

def apply_leet_patterns(text: str) -> str:
    """Stage 1: apply regex-based leet-speak substitutions."""
    for pattern, replacement in LEET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def normalize_leet_chars(token: str) -> str:
    """
    Replace individual leet characters within a token.
    E.g.  'h4t3' → 'hate'
    """
    return "".join(LEET_CHAR_MAP.get(ch, ch) for ch in token)


def levenshtein(s1: str, s2: str) -> int:
    """Standard dynamic-programming Levenshtein distance."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for ch1 in s1:
        curr = [prev[0] + 1]
        for j, ch2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (ch1 != ch2)))
        prev = curr
    return prev[-1]


# Curated toxic vocabulary for Levenshtein screening
TOXIC_VOCAB = {
    "kill", "hate", "fuck", "shit", "bitch", "die", "stupid",
    "idiot", "moron", "slut", "whore", "nigger", "nigga",
    "faggot", "retard", "cunt", "ass", "bastard", "damn",
    "rape", "murder", "terrorist", "bomb",
}


def levenshtein_screen(token: str, max_dist: int = LEVENSHTEIN_MAX_DIST) -> str:
    """
    Stage 2: if a token is within max_dist edits of a toxic vocab word,
    return the vocab word; otherwise return the token unchanged.
    Only applied to tokens that look obfuscated (contain non-alpha chars
    or are near-matches after leet normalization).
    """
    normed = normalize_leet_chars(token.lower())
    if normed in TOXIC_VOCAB:
        return normed
    for word in TOXIC_VOCAB:
        # Only compare tokens of similar length to avoid false positives
        if abs(len(normed) - len(word)) <= max_dist:
            if levenshtein(normed, word) <= max_dist:
                return word
    return token


def adversarial_normalize(text: str) -> str:
    """
    Full two-stage adversarial normalization pipeline:
      Stage 1 — regex pattern matching (fast, high precision)
      Stage 2 — per-token Levenshtein screening (catches residuals)

    Apply at inference time and optionally at training time for
    adversarial augmentation samples.
    """
    # Stage 1
    text = apply_leet_patterns(text)

    # Stage 2 — only screen tokens that contain non-alpha characters
    tokens = text.split()
    normalized = []
    for tok in tokens:
        if re.search(r"[^a-zA-Z\s\[\]]", tok):
            normalized.append(levenshtein_screen(tok))
        else:
            normalized.append(tok)
    return " ".join(normalized)


# ═══════════════════════════════════════════════════════════════
# 3. CONTEXT AUGMENTATION
# ═══════════════════════════════════════════════════════════════

def build_context_input(
    comment: str,
    context: Optional[str] = None,
    sep_token: str = "</s>",
    max_context_tokens: int = 64,
) -> str:
    """
    Prepend context to comment for single-encoder fallback.
    For dual-encoder models, comment and context are tokenized separately.

    Args:
        comment:     The main comment text (already cleaned).
        context:     Parent tweet / thread summary / talk page section.
        sep_token:   Separator between context and comment.
        max_context_tokens: Rough word-level truncation for context.

    Returns:
        Concatenated string  "context </s> comment"
        or just "comment" if context is None / empty.
    """
    if not context or not context.strip():
        return comment

    # Rough truncation of context by whitespace tokens
    ctx_tokens = context.split()
    if len(ctx_tokens) > max_context_tokens:
        context = " ".join(ctx_tokens[:max_context_tokens]) + " ..."

    return f"{context} {sep_token} {comment}"


def format_dual_encoder_inputs(
    comment: str,
    context: Optional[str],
    tokenizer,
    max_len_comment: int = 96,
    max_len_context: int = 64,
) -> dict:
    """
    Tokenize comment and context separately for the dual-encoder model.
    Returns two sets of input_ids / attention_mask tensors.
    """
    enc_comment = tokenizer(
        comment,
        truncation=True,
        max_length=max_len_comment,
        padding="max_length",
        return_tensors="pt",
    )
    ctx = context if (context and context.strip()) else ""
    enc_context = tokenizer(
        ctx,
        truncation=True,
        max_length=max_len_context,
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "comment_input_ids":      enc_comment["input_ids"],
        "comment_attention_mask": enc_comment["attention_mask"],
        "context_input_ids":      enc_context["input_ids"],
        "context_attention_mask": enc_context["attention_mask"],
    }
