"""
config.py
Central configuration for the Unified Toxic Comment Detection project.
All paths, label constants, and hyperparameters live here.
"""

import os
from pathlib import Path

# ─── Project root ────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
DATA_UNIFIED= ROOT / "data" / "unified"
MODELS_DIR  = ROOT / "models"

for d in [DATA_RAW, DATA_PROC, DATA_UNIFIED, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Unified label schema (8 classes) ────────────────────
LABELS = ["TOX", "SEV", "HATE", "IDNT", "THR", "INS", "OBS", "NEU"]
NUM_LABELS = len(LABELS)

# Label index lookup
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
IDX2LABEL = {i: l for i, l in enumerate(LABELS)}

# ─── Metadata columns preserved in unified dataset ───────
METADATA_COLS = [
    "id",
    "text",
    "context_text",
    "source_dataset",
    "platform",
    "split",
    "label_confidence",
    "annotator_count",
    # Demographic / identity group columns
    "gender",
    "race",
    "religion",
    "sexual_orientation",
    "nationality",
    "disability",
    "target_group",
]

# ─── Dataset registry ────────────────────────────────────
DATASET_INFO = {
    "jigsaw_2018": {
        "label_type": "multilabel",
        "platform": "wikipedia",
        "text_col": "comment_text",
        "download": "https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data",
    },
    "civil_comments": {
        "label_type": "continuous",
        "platform": "news_forums",
        "text_col": "text",
        "download": "https://huggingface.co/datasets/civil_comments",
    },
    "davidson": {
        "label_type": "multiclass",
        "platform": "twitter",
        "text_col": "tweet",
        "download": "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv",
    },
    "olid": {
        "label_type": "hierarchical",
        "platform": "twitter",
        "text_col": "tweet",
        "download": "https://sites.google.com/site/offensevalsharedtask/offenseval2019",
    },
    "hasoc": {
        "label_type": "hierarchical",
        "platform": "twitter_facebook",
        "text_col": "text",
        "download": "https://hasocfire.github.io/hasoc/2021/dataset.html",
    },
    "toxigen": {
        "label_type": "continuous",
        "platform": "synthetic",
        "text_col": "generation",
        "download": "https://huggingface.co/datasets/skg/toxigen-data",
    },
    "hatebench": {
        "label_type": "binary",
        "platform": "llm_generated",
        "text_col": "text",
        "download": "https://github.com/HateBench/HateBench",
    },
    "hatexplain": {
        "label_type": "multiclass",
        "platform": "twitter_gab",
        "text_col": "post_tokens",
        "download": "https://huggingface.co/datasets/hatexplain",
    },
    "implicit_hate": {
        "label_type": "multiclass",
        "platform": "twitter",
        "text_col": "post",
        "download": "https://github.com/GT-SALT/implicit-hate",
    },
    "reddit_hs": {
        "label_type": "binary",
        "platform": "reddit",
        "text_col": "reddit_comment",
        "download": "https://huggingface.co/datasets/reddit_hate_speech",
    },
    "measuring_hs": {
        "label_type": "continuous_multidim",
        "platform": "multi_platform",
        "text_col": "text",
        "download": "https://huggingface.co/datasets/ucberkeley-dsp/measuring-hate-speech",
    },
    "ethos": {
        "label_type": "multilabel",
        "platform": "youtube_reddit",
        "text_col": "comment",
        "download": "https://huggingface.co/datasets/ethos",
    },
}

# ─── Continuous score thresholds (Civil Comments etc.) ───
THRESHOLD_SEV = 0.8    # score >= this → SEV
THRESHOLD_TOX = 0.5    # score >= this → TOX
THRESHOLD_NEU = 0.3    # score < this  → NEU
THRESHOLD_AMBIG_LOW  = 0.3
THRESHOLD_AMBIG_HIGH = 0.5

# ─── Near-dedup settings ─────────────────────────────────
MINHASH_NUM_PERM      = 128
MINHASH_NGRAM         = 5       # character n-gram for shingles
LSH_JACCARD_THRESHOLD = 0.8     # duplicate if Jaccard >= this

# ─── Training hyperparameters ────────────────────────────
MODEL_NAME          = "roberta-base"
TEACHER_MODEL_NAME  = "roberta-base"    # for label prediction
MAX_SEQ_LEN         = 128
BATCH_SIZE          = 32
GRAD_ACCUM_STEPS    = 4               # effective batch = 128
LEARNING_RATE       = 2e-5
WEIGHT_DECAY        = 0.01
WARMUP_RATIO        = 0.1
NUM_EPOCHS          = 5
EARLY_STOPPING_PAT  = 2
ADAM_EPSILON        = 1e-8
DROPOUT             = 0.1

# ─── Split ratios ─────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20

# ─── Adversarial augmentation ────────────────────────────
ADV_AUGMENT_RATIO   = 0.25   # fraction of adversarial samples in train
LEVENSHTEIN_MAX_DIST= 2      # max edit distance for leet-speak matching

# ─── Bias mitigation (CDA) ───────────────────────────────
IDENTITY_SWAP_DICT = {
    "women":       "men",
    "woman":       "man",
    "female":      "male",
    "girl":        "boy",
    "girls":       "boys",
    "she":         "he",
    "her":         "his",
    "hers":        "his",
    "herself":     "himself",
    "muslim":      "christian",
    "muslims":     "christians",
    "islam":       "christianity",
    "islamic":     "christian",
    "jewish":      "christian",
    "jews":        "christians",
    "black":       "white",
    "blacks":      "whites",
    "african american": "european american",
    "gay":         "straight",
    "gays":        "straight people",
    "lesbian":     "heterosexual",
    "lgbt":        "heterosexual",
    "transgender": "cisgender",
    "trans":       "cis",
    "immigrant":   "citizen",
    "immigrants":  "citizens",
    "asian":       "western",
    "asians":      "westerners",
}
# Build reverse mapping too
IDENTITY_SWAP_REVERSE = {v: k for k, v in IDENTITY_SWAP_DICT.items()}
IDENTITY_SWAP_FULL = {**IDENTITY_SWAP_DICT, **IDENTITY_SWAP_REVERSE}

# ─── Leet-speak normalization dictionary ─────────────────
LEET_CHAR_MAP = {
    "!": "i", "1": "i", "|": "i",
    "4": "a", "@": "a",
    "3": "e",
    "0": "o",
    "$": "s", "5": "s",
    "7": "t",
    "+": "t",
    "8": "b",
    "9": "g",
    "6": "g",
    "2": "z",
}

# ─── Adversarial regex patterns (raw → canonical) ────────
import re
LEET_PATTERNS = [
    (re.compile(r'\bk[!1i|]l+[s]?\b', re.I),   "kill"),
    (re.compile(r'\bh[4@a]t[3e]\b',   re.I),   "hate"),
    (re.compile(r'\bf[\*@u][c\*]k\b', re.I),   "fuck"),
    (re.compile(r'\b[@a]\$\$\b',      re.I),   "ass"),
    (re.compile(r'\b[s\$]h[!1i]t\b',  re.I),   "shit"),
    (re.compile(r'\bb[!1i]tc?h\b',    re.I),   "bitch"),
    (re.compile(r'\bst[u\*]p[!1i]d\b',re.I),   "stupid"),
    (re.compile(r'\bd[!1i]e\b',        re.I),  "die"),
    (re.compile(r'\bm[0o]r[0o]n\b',   re.I),   "moron"),
    (re.compile(r'\bi[d][!1i][0o]t\b',re.I),   "idiot"),
]

# ─── Seed ────────────────────────────────────────────────
SEED = 42
