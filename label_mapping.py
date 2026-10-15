"""
label_mapping.py
Maps every dataset's original label format to the unified 8-class binary vector.

Unified schema:  [TOX, SEV, HATE, IDNT, THR, INS, OBS, NEU]
Index:              0    1    2     3    4    5    6    7
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from config import (
    LABELS, NUM_LABELS, LABEL2IDX,
    THRESHOLD_SEV, THRESHOLD_TOX, THRESHOLD_NEU,
    THRESHOLD_AMBIG_LOW, THRESHOLD_AMBIG_HIGH,
)


# ─── Helpers ────────────────────────────────────────────────
def _vec(**kwargs) -> np.ndarray:
    """
    Build an 8-dim binary vector from keyword arguments.
    Usage:  _vec(TOX=1, HATE=1) → [1,0,1,0,0,0,0,0]
    """
    v = np.zeros(NUM_LABELS, dtype=np.int8)
    for label, val in kwargs.items():
        v[LABEL2IDX[label]] = int(val)
    return v


def _neu() -> np.ndarray:
    """Pure neutral vector."""
    return _vec(NEU=1)


def _tox(**kwargs) -> np.ndarray:
    """Toxic + any additional flags. Always sets TOX=1."""
    return _vec(TOX=1, **kwargs)


def is_ambiguous(score: float) -> bool:
    return THRESHOLD_AMBIG_LOW <= score < THRESHOLD_AMBIG_HIGH


# ═══════════════════════════════════════════════════════════════
# 1.  JIGSAW 2018  (direct multi-label mapping)
# ═══════════════════════════════════════════════════════════════

def map_jigsaw_2018(row: pd.Series) -> np.ndarray:
    """
    Original cols: toxic, severe_toxic, obscene, threat, insult, identity_hate
    All binary.  Direct 1-to-1 mapping.
    """
    toxic         = int(row.get("toxic", 0))
    severe_toxic  = int(row.get("severe_toxic", 0))
    obscene       = int(row.get("obscene", 0))
    threat        = int(row.get("threat", 0))
    insult        = int(row.get("insult", 0))
    identity_hate = int(row.get("identity_hate", 0))

    # All zeros → neutral
    if not any([toxic, severe_toxic, obscene, threat, insult, identity_hate]):
        return _neu()

    return _vec(
        TOX  = toxic or severe_toxic or obscene or insult or identity_hate,
        SEV  = severe_toxic,
        HATE = identity_hate,      # identity_hate maps to both HATE and IDNT
        IDNT = identity_hate,
        THR  = threat,
        INS  = insult,
        OBS  = obscene,
    )


# ═══════════════════════════════════════════════════════════════
# 2.  CIVIL COMMENTS  (continuous float scores)
# ═══════════════════════════════════════════════════════════════

def map_civil_comments(row: pd.Series) -> Optional[np.ndarray]:
    """
    Original cols: toxicity (float), severe_toxicity, obscene, threat,
                   insult, identity_attack, sexual_explicit
    Also: identity attribute cols (muslim, female, black …) as floats.

    Returns None for ambiguous samples (should be excluded or flagged).
    """
    tox  = float(row.get("toxicity", 0.0))
    sev  = float(row.get("severe_toxicity", 0.0))
    obs  = float(row.get("obscene", 0.0))
    thr  = float(row.get("threat", 0.0))
    ins  = float(row.get("insult", 0.0))
    idnt = float(row.get("identity_attack", 0.0))

    if is_ambiguous(tox):
        return None   # caller should flag this row

    if tox < THRESHOLD_NEU:
        return _neu()

    # Check if any identity column + toxicity co-occur
    identity_cols = [
        "asian", "atheist", "bisexual", "black", "buddhist", "christian",
        "female", "heterosexual", "hindu", "homosexual_gay_or_lesbian",
        "intellectual_or_learning_disability", "jewish", "latino",
        "male", "muslim", "other_disability", "other_gender",
        "other_race_or_ethnicity", "other_religion",
        "other_sexual_orientation", "physical_disability",
        "psychiatric_or_mental_illness", "transgender", "white",
    ]
    has_identity_target = any(
        float(row.get(col, 0.0)) >= 0.5 for col in identity_cols
        if col in row.index
    )

    return _vec(
        TOX  = tox >= THRESHOLD_TOX,
        SEV  = sev >= THRESHOLD_SEV or tox >= THRESHOLD_SEV,
        HATE = has_identity_target and tox >= THRESHOLD_TOX,
        IDNT = has_identity_target and tox >= THRESHOLD_TOX,
        THR  = thr >= THRESHOLD_TOX,
        INS  = ins >= THRESHOLD_TOX,
        OBS  = obs >= THRESHOLD_TOX,
    )


# ═══════════════════════════════════════════════════════════════
# 3.  DAVIDSON 2017  (3-class integer)
# ═══════════════════════════════════════════════════════════════

# Original: 0=hate_speech, 1=offensive_language, 2=neither
DAVIDSON_MAP: Dict[int, np.ndarray] = {
    0: _vec(TOX=1, HATE=1),          # hate_speech → HATE + TOX
    1: _vec(TOX=1, INS=1),           # offensive   → TOX  + INS
    2: _neu(),                        # neither     → NEU
}

def map_davidson(row: pd.Series) -> np.ndarray:
    cls = int(row.get("class", 2))
    return DAVIDSON_MAP.get(cls, _neu()).copy()


# ═══════════════════════════════════════════════════════════════
# 4.  OLID 2019  (hierarchical 3-task)
# ═══════════════════════════════════════════════════════════════

def map_olid(row: pd.Series) -> np.ndarray:
    """
    Task A: OFF / NOT
    Task B: TIN (targeted) / UNT (untargeted)  — only if A=OFF
    Task C: IND / GRP / OTH                    — only if B=TIN
    """
    a = str(row.get("subtask_a", "NOT")).strip().upper()
    b = str(row.get("subtask_b", "NaN")).strip().upper()
    c = str(row.get("subtask_c", "NaN")).strip().upper()

    if a == "NOT":
        return _neu()

    # A = OFF
    if b in ("NAN", "NONE", ""):
        return _tox()                              # OFF with no task B

    if b == "UNT":
        return _tox()                              # untargeted offensive

    # B = TIN
    if c == "GRP":
        return _vec(TOX=1, HATE=1, IDNT=1)        # targeted group
    if c == "IND":
        return _vec(TOX=1, INS=1)                 # targeted individual
    return _tox()                                  # OTH or missing


# ═══════════════════════════════════════════════════════════════
# 5.  HASOC 2019
# ═══════════════════════════════════════════════════════════════

def map_hasoc(row: pd.Series) -> np.ndarray:
    """
    task_1: HOF / NOT
    task_2: HATE / OFFN / PRFN   (if HOF)
    task_3: TIN / UNT            (targeting)
    """
    t1 = str(row.get("task_1", "NOT")).strip().upper()
    t2 = str(row.get("task_2", "NONE")).strip().upper()

    if t1 == "NOT":
        return _neu()

    if t2 == "HATE":
        return _vec(TOX=1, HATE=1)
    if t2 == "OFFN":
        return _tox(INS=1)
    if t2 == "PRFN":
        return _vec(TOX=1, OBS=1)
    return _tox()


# ═══════════════════════════════════════════════════════════════
# 6.  TOXIGEN  (continuous score 1–5 + target group)
# ═══════════════════════════════════════════════════════════════

def map_toxigen(row: pd.Series) -> np.ndarray:
    """
    toxicity_human / toxicity_ai: 1–5 scale
    target_group: string (asian, jewish, lgbtq …)
    label: hate / neutral
    """
    # Prefer human score; fall back to AI score
    score_h = float(row.get("toxicity_human", 0.0))
    score_a = float(row.get("toxicity_ai", 0.0))
    score   = score_h if score_h > 0 else score_a
    label   = str(row.get("label", "neutral")).strip().lower()
    group   = str(row.get("target_group", "")).strip()

    has_group = bool(group and group not in ("", "nan"))

    if label == "neutral" or score <= 2:
        return _neu()
    if score >= 4 or label == "hate":
        return _vec(TOX=1, HATE=1, IDNT=int(has_group))
    # score == 3 → moderately toxic
    return _vec(TOX=1, IDNT=int(has_group))


# ═══════════════════════════════════════════════════════════════
# 7.  HATEBENCH
# ═══════════════════════════════════════════════════════════════

def map_hatebench(row: pd.Series) -> np.ndarray:
    """
    hate_label: float (0.0 / 1.0)
    main_target: race / religion / gender / sexual_orientation …
    """
    hate  = float(row.get("hate_label", 0.0))
    target= str(row.get("main_target", "")).strip().lower()
    has_t = bool(target and target not in ("", "nan"))

    if hate < 0.5:
        return _neu()
    return _vec(TOX=1, HATE=1, IDNT=int(has_t))


# ═══════════════════════════════════════════════════════════════
# 8.  HATEXPLAIN  (majority-vote across 3 annotators)
# ═══════════════════════════════════════════════════════════════

def map_hatexplain_label(label: str, target_communities: List[str]) -> np.ndarray:
    """
    label: 'hatespeech' / 'offensive' / 'normal'
    target_communities: list of strings (e.g. ['African', 'Women'])
    """
    has_target = len(target_communities) > 0
    label = label.lower().strip()

    if label == "normal":
        return _neu()
    if label == "hatespeech":
        return _vec(TOX=1, HATE=1, IDNT=int(has_target))
    if label == "offensive":
        return _vec(TOX=1, INS=1)
    return _neu()


def majority_vote_label(annotators: List[dict]) -> tuple:
    """
    Compute majority-vote label from a list of annotator dicts.
    Returns (winning_label, confidence, target_communities).
    """
    from collections import Counter
    labels   = [a["label"] for a in annotators]
    counts   = Counter(labels)
    winner, top_count = counts.most_common(1)[0]
    confidence = top_count / len(labels)

    targets = []
    for a in annotators:
        targets.extend(a.get("target", []))
    targets = list(set(targets))

    return winner, confidence, targets


# ═══════════════════════════════════════════════════════════════
# 9.  IMPLICIT HATE SPEECH
# ═══════════════════════════════════════════════════════════════

IMPLICIT_HATE_MAP = {
    "explicit_hate": _vec(TOX=1, HATE=1),
    "implicit_hate": _vec(TOX=1, HATE=1),
    "not_hate":      _neu(),
}

def map_implicit_hate(row: pd.Series) -> np.ndarray:
    cls = str(row.get("class", "not_hate")).strip().lower()
    return IMPLICIT_HATE_MAP.get(cls, _neu()).copy()


# ═══════════════════════════════════════════════════════════════
# 10.  REDDIT HATE SPEECH
# ═══════════════════════════════════════════════════════════════

def map_reddit_hs(row: pd.Series) -> np.ndarray:
    """Binary: label 1 = hate, 0 = not hate."""
    label = int(row.get("label", 0))
    if label == 1:
        return _vec(TOX=1, HATE=1)
    return _neu()


# ═══════════════════════════════════════════════════════════════
# 11.  MEASURING HATE SPEECH (continuous multi-dimensional)
# ═══════════════════════════════════════════════════════════════

def map_measuring_hs(row: pd.Series) -> np.ndarray:
    """
    10 continuous dimensions per annotator (0–4 scale).
    Aggregated per comment_id before calling this function.
    Columns: sentiment, respect, insult, humiliate, status,
             dehumanize, violence, genocide, attack_defend, hatespeech
    """
    insult     = float(row.get("insult",     0.0))
    humiliate  = float(row.get("humiliate",  0.0))
    dehumanize = float(row.get("dehumanize", 0.0))
    violence   = float(row.get("violence",   0.0))
    hatespeech = float(row.get("hatespeech", 0.0))
    status     = float(row.get("status",     0.0))

    # Scale from 0–4 to 0–1
    def n(x): return x / 4.0

    is_hate = n(hatespeech) >= 0.5 or n(dehumanize) >= 0.5
    is_thr  = n(violence) >= 0.5
    is_ins  = n(insult) >= 0.5 or n(humiliate) >= 0.5
    is_tox  = is_hate or is_thr or is_ins or n(status) >= 0.5

    if not is_tox:
        return _neu()

    return _vec(TOX=is_tox, HATE=is_hate, THR=is_thr, INS=is_ins)


# ═══════════════════════════════════════════════════════════════
# 12.  ETHOS  (multi-label binary identity cols)
# ═══════════════════════════════════════════════════════════════

def map_ethos(row: pd.Series) -> np.ndarray:
    """
    Cols: violence (float), directed_vs_generalized (float),
          gender, race, national_origin, disability, religion,
          sexual_orientation  — all floats in [0,1].
    isHate is derived: any identity col >= 0.5.
    """
    identity_cols = ["gender", "race", "national_origin",
                     "disability", "religion", "sexual_orientation"]
    thr   = 0.5
    is_hate  = any(float(row.get(c, 0.0)) >= thr for c in identity_cols)
    violence = float(row.get("violence", 0.0)) >= thr

    if not is_hate and not violence:
        return _neu()

    return _vec(
        TOX  = True,
        HATE = is_hate,
        IDNT = is_hate,
        THR  = violence,
    )



# ═══════════════════════════════════════════════════════════════
# 8b.  HATEXPLAIN — row-level mapper
#
# load_hatexplain() already runs majority_vote_label() and stores:
#   majority_label   : 'hatespeech' / 'offensive' / 'normal'
#   target_group     : comma-separated string of target communities
#   label_confidence : fraction of annotators who agreed
#
# This mapper reads those pre-computed columns so it works with
# the standard apply_label_mapping() row-by-row loop.
# ═══════════════════════════════════════════════════════════════

def map_hatexplain(row: pd.Series) -> np.ndarray:
    label   = str(row.get("majority_label", "normal")).strip().lower()
    targets = str(row.get("target_group",   "")).strip()

    target_list = [t.strip() for t in targets.split(",") if t.strip()] \
                  if targets else []
    return map_hatexplain_label(label, target_list)


# ═══════════════════════════════════════════════════════════════
# DISPATCH TABLE — maps dataset name → row-level function
# ═══════════════════════════════════════════════════════════════

MAPPER = {
    "jigsaw_2018":   map_jigsaw_2018,
    "civil_comments":map_civil_comments,
    "davidson":      map_davidson,
    "olid":          map_olid,
    "hasoc":         map_hasoc,
    "toxigen":       map_toxigen,
    "hatebench":     map_hatebench,
    "hatexplain":    map_hatexplain,    # row-level wrapper
    "implicit_hate": map_implicit_hate,
    "reddit_hs":     map_reddit_hs,
    "measuring_hs":  map_measuring_hs,
    "ethos":         map_ethos,
}


def apply_label_mapping(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Apply the correct label mapping to a DataFrame and return it
    with 8 new binary columns [TOX, SEV, HATE, IDNT, THR, INS, OBS, NEU].

    Rows with None return from the mapper (ambiguous) are flagged in
    a 'ambiguous' column and excluded from training by default.
    """
    mapper = MAPPER.get(dataset_name)
    if mapper is None:
        raise ValueError(f"No mapper found for dataset '{dataset_name}'")

    vectors = []
    ambig   = []

    for _, row in df.iterrows():
        vec = mapper(row)
        if vec is None:
            vec = np.zeros(NUM_LABELS, dtype=np.int8)
            ambig.append(True)
        else:
            ambig.append(False)
        vectors.append(vec)

    label_matrix = np.vstack(vectors)
    for i, label in enumerate(LABELS):
        df[label] = label_matrix[:, i]

    df["ambiguous"] = ambig
    return df