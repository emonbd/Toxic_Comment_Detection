# Unified Multi-Source Toxic Comment Detection

### Dataset Construction, Multi-Label Classification, and Adversarial Robustness

A research project focused on building a unified toxic comment detection framework by combining 12 publicly available datasets, harmonizing their labels, fine-tuning transformer-based models, and improving cross-domain generalization and adversarial robustness.

**Author:** Ariful Islam Emon  
**Research Area:** Natural Language Processing (NLP), Machine Learning, Responsible AI  
**Status:** Preliminary research results; context-aware modeling and additional fairness and adversarial training experiments are planned.

---

## Overview

Toxic comment detection is an important NLP task for identifying harmful, abusive, and hateful content across online platforms. However, models trained on a single dataset often struggle to generalize across different platforms, detect underrepresented harmful categories, and resist adversarial text manipulation.

This project addresses these challenges by consolidating 12 publicly available datasets from sources including Wikipedia discussions, Twitter, Reddit, Gab, news forums, and LLM-generated content into a unified multi-label classification framework.

The unified dataset contains approximately **2.315 million samples** and uses an eight-class label schema. Preliminary experiments demonstrate that training RoBERTa-base on the combined dataset improves classification performance compared with training on a single source.

The project also investigates character-level text normalization to reduce the effectiveness of adversarial attacks against toxic comment classifiers.

## Research Objectives

- **Dataset unification:** Combine 12 heterogeneous datasets into a consistent, training-ready format.
- **Label harmonization:** Map different annotation schemes into a unified eight-class multi-label taxonomy.
- **Cross-domain generalization:** Improve performance across platforms and data distributions.
- **Multi-label classification:** Identify multiple types of harmful content within a single comment.
- **Adversarial robustness:** Detect obfuscated toxic language using character-level normalization.
- **Bias mitigation:** Investigate counterfactual data augmentation to reduce unintended demographic bias.
- **Context-aware detection:** Explore the use of conversational context to improve classification.

## Unified Label Schema

The project uses the following eight labels:

| Label | Description |
|---|---|
| `TOX` | General toxic, rude, or disrespectful content |
| `SEV` | Severe toxicity, including extreme attacks and dehumanizing language |
| `HATE` | Hate speech targeting protected characteristics |
| `IDNT` | Identity attacks targeting an identity group |
| `THR` | Threats or expressions of intent to cause harm |
| `INS` | Insults, personal attacks, and ridicule |
| `OBS` | Obscene or sexually explicit language and profanity |
| `NEU` | Neutral or non-toxic content |

The task is formulated as multi-label binary classification. A comment may receive several harmful-content labels simultaneously. The `NEU` label is assigned when none of the seven harmful labels applies.

## Datasets

The research integrates the following 12 datasets:

| Dataset | Primary contribution |
|---|---|
| Jigsaw 2018 | Core multi-label baseline |
| Civil Comments | Large-scale toxicity and bias analysis |
| Davidson 2017 | Twitter hate speech and offensive language |
| OLID 2019 | Hierarchical offensive-language classification |
| HASOC 2019 | Hate, offensive, and profane content |
| ToxiGen | LLM-generated and adversarial content |
| HateBench | Adversarial robustness evaluation |
| HateXplain | Hate speech labels and explanatory rationales |
| Implicit Hate | Implicit and explicit hate speech |
| Reddit HS | Forum-based hate speech classification |
| Measuring Hate Speech | Continuous annotations and fairness analysis |
| ETHOS | Identity-related hate speech and violence labels |

The source datasets use different label formats, including binary labels, continuous scores, hierarchical annotations, and multi-label vectors. The project harmonizes these formats using rule-based mapping, thresholding, hierarchical label conversion, and annotator aggregation.

**Important:** Dataset access, redistribution permissions, and individual dataset licenses should be verified before publishing the merged corpus.

## Methodology

### 1. Data Preprocessing

Each source dataset undergoes preprocessing before merging:

- HTML tag removal.
- URL normalization using a `[URL]` token.
- Username anonymization using a `[USER]` token.
- Whitespace normalization.
- Exact duplicate removal.
- Ambiguous sample identification.
- Conversational context collection where available.
- Mapping of source-specific labels into the unified schema.

### 2. Label Harmonization

Different annotation schemes are converted into the eight-class taxonomy. Datasets with rich multi-label annotations provide direct supervision, while datasets with single-label or limited-label formats use mapping rules and teacher-model predictions.

The teacher model is based on RoBERTa-base, fine-tuned on richly annotated datasets. Predictions are thresholded independently for each class, while applicable source labels are retained as hard constraints.

### 3. Dataset Splitting

The unified dataset contains approximately 2.315 million samples after merging and deduplication.

| Split | Approximate samples | Percentage |
|---|---:|---:|
| Training | 1.62 million | 70% |
| Validation | 231,000 | 10% |
| Test | 463,000 | 20% |

The split is stratified jointly by source dataset and dominant label class. Additional evaluation designs include held-out-domain testing, adversarial testing, and counterfactual fairness evaluation.

### 4. Transformer-Based Classification

The project evaluates transformer models for toxic comment classification.

The principal model is RoBERTa-base, fine-tuned on the unified dataset using the Hugging Face Transformers framework and PyTorch.

The reported training configuration includes:

- Learning rate: `2e-5`
- Effective batch size: `128`
- Maximum sequence length: `128` tokens
- Training epochs: Up to `5`
- Loss function: `BCEWithLogitsLoss`
- Optimizer: `AdamW`
- Evaluation: Macro-F1, per-class metrics, cross-domain F1, and adversarial Attack Success Rate

### 5. Adversarial Character Normalization

The adversarial preprocessing pipeline targets character substitutions commonly used to evade text classifiers.

**Stage 1 — Rule-based normalization**

Regular expressions and a leet-speak substitution dictionary normalize patterns such as `h4te` and `k!ll` into canonical forms.

**Stage 2 — Edit-distance screening**

Tokens within an edit distance of two from a curated toxic vocabulary are mapped to their canonical forms, helping identify character transpositions, repetitions, and mixed obfuscation strategies.

The normalization is applied at inference time and is evaluated by comparing adversarial Attack Success Rates before and after preprocessing.

### 6. Planned Research Extensions

The following components remain part of the broader experimental program:

- **Context-aware dual encoder:** Encode a comment and its conversational context using a shared transformer encoder.
- **Counterfactual data augmentation:** Create identity-swapped training examples to investigate demographic bias and false-positive disparities.
- **Adversarial training augmentation:** Introduce adversarial examples into training to improve resistance to LLM-generated and paraphrased attacks.

These extensions require further implementation and experimental validation; they should not be interpreted as completed results.

## Preliminary Results

### Model Comparison

The following results are reported in the preliminary research report.

| Model | Training data | Macro-F1 | Cross-domain F1 |
|---|---|---:|---:|
| BERT-base | Jigsaw 2018 | 0.614 | 0.488 |
| RoBERTa-base | Jigsaw 2018 | 0.641 | 0.512 |
| toxic-bert | Jigsaw 2018 | 0.658 | 0.531 |
| **RoBERTa-base** | **Unified 12-dataset corpus** | **0.703** | **0.584** |

### Key Findings

- **6.2 percentage-point improvement in Macro-F1:** RoBERTa-base improved from `0.641` on Jigsaw-only training to `0.703` on the unified corpus.
- **7.2 percentage-point improvement in cross-domain F1:** Performance increased from `0.512` to `0.584`.
- **Improved minority-class detection:** The unified model's reported F1 increased by `0.08` for identity attacks and `0.07` for hate speech.
- **Improved adversarial robustness:** Character-level normalization reduced Attack Success Rate on HateBench from `38.4%` to `29.1%`, a reduction of `9.3` percentage points without model retraining.

These are preliminary findings from the report. Additional validation, hyperparameter optimization, and completion of the planned experiments are needed before drawing definitive conclusions.

## Technology Stack

- **Programming language:** Python
- **Natural Language Processing:** Hugging Face Transformers
- **Deep Learning:** PyTorch
- **Machine Learning:** scikit-learn
- **Dataset processing:** Hugging Face Datasets
- **Evaluation:** Precision, Recall, F1-score, ROC-AUC, and Attack Success Rate
- **Fairness analysis:** Fairlearn
- **Experimental environment:** Kaggle GPU environments, including NVIDIA T4 or P100

## Repository Structure

The following is a suggested structure for organizing the GitHub repository. Adapt the filenames to match the actual implementation.

```text
unified-toxic-comment-detection/
├── README.md
├── data/
│   └── README.md
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── dataset_unification.ipynb
│   └── model_evaluation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── label_mapping.py
│   ├── train.py
│   ├── evaluate.py
│   └── adversarial_normalization.py
├── results/
│   └── README.md
├── requirements.txt
└── LICENSE
```

This structure is illustrative and does not imply that every listed file or module has already been implemented.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/unified-toxic-comment-detection.git
cd unified-toxic-comment-detection
```

Replace `YOUR-USERNAME` and the repository name with the actual GitHub details.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

After creating a `requirements.txt` file with the verified project dependencies, install them using:

```bash
pip install -r requirements.txt
```

The research report lists PyTorch, Transformers, Datasets, scikit-learn, and Fairlearn among the libraries used. Exact versions should be aligned with the actual implementation.

### 4. Prepare the datasets

Obtain the source datasets from their official or authorized distribution channels. Follow their respective terms of use, then apply the preprocessing and label-mapping pipeline.

Do not assume that the complete merged dataset is included in the repository.

### 5. Train and evaluate

Training and evaluation commands depend on the final implementation. Once the scripts are available, document their exact arguments, dataset paths, model checkpoints, and expected outputs here.

## Evaluation Metrics

The project uses multiple metrics to assess model performance:

- **Macro-F1:** Measures performance across all eight classes with equal class weighting.
- **Precision and Recall:** Measures classification correctness and the ability to identify positive examples.
- **ROC-AUC:** Evaluates class-wise discrimination.
- **Cross-domain F1:** Assesses generalization to held-out source datasets.
- **False-positive-rate parity gap:** Measures differences in false-positive rates across identity groups.
- **Attack Success Rate:** Measures vulnerability to adversarial examples.

These metrics provide complementary views of classification quality, domain generalization, fairness, and robustness.

## Limitations

The preliminary research identifies several limitations:

- Label noise may arise from inconsistent annotations and teacher-model predictions.
- Threat detection remains challenging because threats are underrepresented.
- Conversational context is unavailable for many samples.
- Overlap between source datasets may introduce evaluation leakage despite deduplication.
- Preliminary model results require further tuning and validation.
- The planned context-aware model, counterfactual augmentation, and adversarial training experiments remain to be completed.

Toxicity classification is inherently context-dependent. Model predictions should not be treated as definitive judgments about a person or as a substitute for careful human review in high-impact moderation decisions.

## Future Work

The next phase of research will focus on:

1. Implementing and evaluating the context-aware dual-encoder architecture.
2. Measuring fairness improvements through counterfactual data augmentation.
3. Evaluating adversarial training with a 25% adversarial-sample mix.
4. Improving threat detection and other underrepresented classes.
5. Conducting additional cross-domain and robustness experiments.
6. Documenting data provenance, reproducibility, and model limitations.

## Research Contribution

This project investigates how multi-source dataset construction and label harmonization can improve toxic comment detection across different online platforms. By combining multi-label transformer classification with adversarial text normalization and planned fairness-aware extensions, the research aims to support more robust, generalizable, and responsible content moderation systems.

## References

- Risch, J., Stoll, A., Ziegele, M., & Krestel, R. (2021). *Toxic Comment Collection: Making More Than 30 Datasets Easily Accessible in One Unified Format.* [ACL Anthology](https://aclanthology.org/2021.woah-1.17/)
- Hartvigsen, T., et al. (2022). *ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection.* [arXiv](https://arxiv.org/abs/2203.09509)
- Mathew, B., et al. (2021). *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection.* [arXiv](https://arxiv.org/abs/2012.10289)
- Elsherief, M., et al. (2021). *Latent Hatred: A Benchmark for Understanding Implicit Hate Speech.* [arXiv](https://arxiv.org/abs/2109.05322)
- Dixon, L., et al. (2018). *Measuring and Mitigating Unintended Bias in Text Classification.*
- Zmigrod, R., et al. (2019). *Counterfactual Data Augmentation for Mitigating Gender Stereotypes in Languages with Rich Morphology.*

---

**Disclaimer:** This repository documents ongoing research and preliminary experimental results. Reported metrics reflect the current experimental stage and may change as further experiments are completed.
