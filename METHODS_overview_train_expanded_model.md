# Methods: Expanded NLI-Feature Hallucination Detection Pipeline

**Full in-depth overview of the pipeline implemented in `train_expanded_model.py` for use in research papers.**

---

## 1. Task and Dataset

**Task.** Sentence-level binary classification for hallucination detection: each sentence is labeled as **factual** (0) or **hallucinated** (1).

**Dataset.** We use the **WikiBio GPT-3 Hallucination** evaluation set (Manakul et al., 2023), loaded from Hugging Face as `potsawee/wiki_bio_gpt3_hallucination` (split: `evaluation`). Each example consists of:
- **gpt3_sentences**: list of sentences from a GPT-3–generated biography.
- **annotation**: per-sentence human labels in `{accurate, minor_inaccurate, major_inaccurate}`.
- **gpt3_text_samples**: a fixed set of 20 stochastically sampled alternative generations for the same prompt (used as “evidence” in SelfCheck-style scoring).

**Label mapping.** We binarize annotations as follows: `accurate → 0` (factual), `minor_inaccurate → 1`, `major_inaccurate → 1` (both treated as hallucinated). The dataset is flattened to one row per sentence, with document indices preserved so that all sentences from the same biography share the same 20 sampled passages.

---

## 2. NLI-Based Feature Extraction

**Rationale.** We follow the SelfCheckGPT-NLI idea: treat the model’s own samples as evidence and use a natural language inference (NLI) model to score whether each **sentence** (premise) is entailed or contradicted by each **sampled passage** (hypothesis). High contradiction across samples is treated as a signal of hallucination.

**NLI model.** We use a pre-trained NLI model (default: **MoritzLaurer/DeBERTa-v3-base-mnli**, i.e., DeBERTa-v3-base fine-tuned on MultiNLI). The implementation supports any Hugging Face model that outputs entailment/neutral/contradiction (or entailment/contradiction only). For each (sentence, passage) pair, the model is run with **sentence as premise** and **passage as hypothesis**.

**E/C renormalization.** Raw NLI logits are restricted to the entailment (E) and contradiction (C) classes; the neutral class is dropped and probabilities are renormalized over E and C only. Thus for each pair we obtain \(P(E)\) and \(P(C)\) with \(P(E)+P(C)=1\). This matches the original SelfCheckGPT-NLI setup.

**Batching.** All (sentence, sample) pairs for a document are processed in batches (default batch size 32 on CPU, 64 on GPU) for efficiency. The result is two matrices per document: **entailment scores** and **contradiction scores**, each of shape (number of sentences in document × 20 samples).

---

## 3. Fifteen-Dimensional Feature Vector

For each sentence we obtain 20 entailment values and 20 contradiction values (one per sampled passage). We summarize these into a **15-dimensional feature vector** as follows.

**Seven contradiction (C) statistics** (indices 0–6), computed over the 20 contradiction scores for that sentence:

1. **mean(C)** — mean contradiction probability.
2. **max(C)** — maximum contradiction over samples.
3. **std(C)** — standard deviation of contradiction scores.
4. **frac(C > 0.5)** — fraction of samples with contradiction probability above 0.5.
5. **entropy(C)** — mean over the 20 samples of the binary entropy \( -p\log p - (1-p)\log(1-p) \) with \(p\) the contradiction probability (clamped to avoid \(\log 0\)).
6. **IQR(C)** — interquartile range (75th percentile − 25th percentile) of the contradiction scores.
7. **top3_mean(C)** — mean of the three largest contradiction scores.

**Seven entailment (E) statistics** (indices 7–13): the same seven statistics (mean, max, std, frac(E>0.5), entropy(E), IQR(E), top3_mean(E)) computed over the 20 entailment scores.

**Support margin** (index 14): **mean(E) − mean(C)**. This captures the average tendency of the evidence to support rather than contradict the sentence; more negative values indicate stronger contradiction and thus higher hallucination signal.

**Caching.** Extracted 15-feature matrices can be saved to disk (default directory `.feature_cache`) keyed by dataset and NLI model. When the same data and model are used, the script loads from cache and skips NLI inference to avoid recomputation.

---

## 4. Train/Test Split

We use a **single 70% train / 30% test** split, applied at the **document level** so that no document appears in both train and test (avoiding leakage). The split is **stratified** by document-level majority label (factual vs. hallucinated) to keep class balance. The random seed is fixed (default 42). This procedure is **identical** to that in `train_on_wikibio.py`, so results are directly comparable to the logistic regression baseline trained on 7 contradiction-only features.

**Feature extraction** is performed on the **full** dataset (all documents) before splitting; the 70/30 split is then applied to the feature matrix and labels. Thus both train and test sentences use the same NLI model and the same feature definition.

---

## 5. Baseline: Unsupervised Mean Contradiction

**Baseline method.** The **unsupervised** baseline is the SelfCheckGPT-NLI score: for each sentence, the score is the **mean contradiction probability** over the 20 sampled passages. Higher score indicates higher likelihood of hallucination. No training is involved.

**Implementation.** By default we use the first dimension of the 15-feature vector, **mean(C)** (column 0), which is identical to this baseline when computed from the same NLI model and E/C renormalization. Optionally, the script can run the original per-sentence SelfCheckNLI predictor for an exact match to the original implementation (`--exact_baseline`).

The baseline is evaluated on the **same test set** as the supervised methods (same 30% of documents / sentences).

---

## 6. Supervised Methods: Random Forest and XGBoost

**Input.** Both classifiers take the **15-dimensional** feature vector (all 15 features above) as input.

**Random Forest (RF-15feat).** We train a Random Forest classifier with the following settings: 300 trees, no maximum depth, minimum samples per leaf = 5, class weight = “balanced” (to account for label imbalance), random state 42, parallelized over cores (`n_jobs=-1`). No feature scaling is applied.

**XGBoost (XGB-15feat).** If the XGBoost library is available, we also train an XGBoost classifier: 300 estimators, maximum depth 6, learning rate 0.1, scale_pos_weight set to the ratio (number of negative examples) / (number of positive examples) in the training set to handle imbalance, evaluation metric “logloss”, random state 42, parallelized. Again, no scaling.

**Training.** Both models are trained **only on the 70% training set** (same document-level split as above). No hyperparameter search or cross-validation is applied in this pipeline; the reported metrics are on the held-out 30% test set.

---

## 7. Evaluation Metrics

All methods (baseline, RF, XGBoost) are evaluated on the **same test set** using the same metrics.

**Primary metric: AUC-PR.** Area under the precision–recall curve (average precision), computed from the continuous scores (mean(C) for baseline, predicted probability of the positive class for RF and XGBoost). AUC-PR is preferred over ROC-AUC when the positive class (hallucinated) is minority, as in this dataset.

**Secondary metric: F1-max.** The maximum F1 score achievable at any threshold on the precision–recall curve: for each threshold we compute precision and recall, then F1 = 2·P·R/(P+R), and report the maximum over thresholds. This reflects the best possible F1 for a given score function.

**Precision–recall curves.** Precision–recall curves are computed for each method and plotted on a single figure for visual comparison, with AUC-PR reported in the legend.

---

## 8. Model Selection and Final Saved Model

**Selection.** Among the supervised methods (RF and XGBoost), the method with the **higher test AUC-PR** is selected as the best.

**Final model.** The selected classifier is then **retrained on the full dataset** (all 70% + 30% sentences) and saved to disk (default: `hallucination_detector_expanded.pkl`). The saved artifact includes: the trained classifier, the list of 15 feature names, the number of features, the method name (RF-15feat or XGB-15feat), and the test AUC-PR and F1-max achieved by that method before retraining. This final model can be used for inference on new sentences once their 15 NLI-derived features are computed with the same NLI model and feature extraction procedure.

---

## 9. Reproducibility

- **Random seeds:** NumPy and PyTorch seeds are set to the same value (default 42) at the start of the script.
- **Train/test split:** Document-level stratified split with fixed seed (default 42) and test fraction 0.3.
- **Classifiers:** Random state 42 for RF and XGBoost.
- **Caching:** Feature extraction cache is keyed by NLI model identifier and dataset size so that re-runs with the same data and model skip NLI inference and reproduce the same feature matrix.

---

## 10. Summary Table for Paper

| Component | Specification |
|-----------|----------------|
| **Dataset** | WikiBio GPT-3 Hallucination (Hugging Face: `potsawee/wiki_bio_gpt3_hallucination`, split `evaluation`) |
| **Labels** | Binary: accurate → 0 (factual); minor_inaccurate, major_inaccurate → 1 (hallucinated) |
| **Evidence** | 20 sampled passages per document (same for all sentences in the document) |
| **NLI** | DeBERTa-v3-base-mnli (default); E/C renormalization (neutral dropped) |
| **Features** | 15-D: 7 stats on C, 7 stats on E, plus support margin mean(E)−mean(C) |
| **Split** | 70% train / 30% test, document-level, stratified, seed 42 |
| **Baseline** | Unsupervised: mean(C) per sentence |
| **Supervised** | Random Forest (300 trees, min_leaf=5, balanced) and XGBoost (300 trees, depth 6, scale_pos_weight), both on 15 features |
| **Metrics** | AUC-PR (primary), F1-max; PR curves for comparison |
| **Output** | Best supervised model retrained on full data and saved; PR plot and summary table |

---

## References

- Manakul, P., Liusie, A., & Gales, M. J. F. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. *EMNLP 2023*.
- WikiBio GPT-3 Hallucination dataset: https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination
