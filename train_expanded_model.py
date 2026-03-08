"""
Expanded-feature training and multi-method comparison for hallucination detection.

Uses the SAME 70/30 document-level stratified split as train_on_wikibio.py
(seed=42, test_size=0.3, split_by_doc) so results are directly comparable.

Compares methods on WikiBio GPT-3 Hallucination data:
  1. Baseline      – unsupervised mean(C) from original SelfCheckNLI
  2. RF-15feat     – Random Forest on 15 features (7C + 7E + margin)
  3. XGB-15feat    – XGBoost on 15 features (if xgboost is installed)

Pipeline:
  - Loads WikiBio from Hugging Face (potsawee/wiki_bio_gpt3_hallucination)
  - Splits 70/30 by document (same as train_on_wikibio.py)
  - Extracts 15 features per sentence via predict_matrix (E/C renormalized)
  - Trains RF (and XGB if available) on train set, evaluates on test set
  - Plots PR curves for every method on a single figure
  - Prints a summary table of AUC-PR and F1-max
  - Saves the best-performing supervised model to disk
  - Checkpoints after each method (Baseline + RF + XGB); on interrupt, re-run to resume.

Defaults to deberta-v3-base for ~3-4x faster CPU inference vs deberta-v3-large.
"""

import numpy as np
import pickle
import argparse
import json
import hashlib
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI as SelfCheckNLIBaseline
from selfcheckgpt.modeling_selfcheck_extended import SelfCheckNLI as SelfCheckNLIExtended


# ── Feature names ────────────────────────────────────────────────────────────

FEATURE_NAMES_15 = [
    "mean(C)", "max(C)", "std(C)", "frac(C>0.5)",
    "entropy(C)", "IQR(C)", "top3_mean(C)",
    "mean(E)", "max(E)", "std(E)", "frac(E>0.5)",
    "entropy(E)", "IQR(E)", "top3_mean(E)",
    "mean(E)-mean(C)",
]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_wikibio_data():
    """Load WikiBio GPT-3 Hallucination dataset and flatten to sentence level."""
    print("Loading WikiBio GPT-3 Hallucination dataset from Hugging Face...")
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")["evaluation"]

    all_sentences = []
    all_labels = []
    all_sampled_passages = []
    doc_indices = []

    label_mapping = {"accurate": 0, "minor_inaccurate": 1, "major_inaccurate": 1}

    for doc_idx, example in enumerate(dataset):
        passages = example["gpt3_text_samples"]
        for sentence, annotation in zip(example["gpt3_sentences"], example["annotation"]):
            all_sentences.append(sentence)
            all_labels.append(label_mapping[annotation])
            all_sampled_passages.append(passages)
            doc_indices.append(doc_idx)

    all_labels = np.array(all_labels)
    doc_indices = np.array(doc_indices)

    n_fact = np.sum(all_labels == 0)
    n_hall = np.sum(all_labels == 1)
    print(f"  Documents: {len(dataset)}  |  Sentences: {len(all_sentences)}")
    print(f"  Factual: {n_fact} ({n_fact/len(all_labels)*100:.1f}%)  "
          f"|  Hallucinated: {n_hall} ({n_hall/len(all_labels)*100:.1f}%)")

    return all_sentences, all_labels, all_sampled_passages, doc_indices


# ── Feature extraction ───────────────────────────────────────────────────────

def _compute_7_stats(scores: np.ndarray):
    """Return 7 summary statistics from a 1-D array of per-sample probabilities."""
    if len(scores) == 0:
        return [0.0] * 7

    mean_val = float(np.mean(scores))
    max_val = float(np.max(scores))
    std_val = float(np.std(scores))
    frac_above = float(np.mean(scores > 0.5))

    p = np.clip(scores, 1e-10, 1.0 - 1e-10)
    entropy_val = float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p))))

    iqr_val = float(np.percentile(scores, 75) - np.percentile(scores, 25))

    sorted_desc = np.sort(scores)[::-1]
    top3_mean = float(np.mean(sorted_desc[: min(3, len(sorted_desc))]))

    return [mean_val, max_val, std_val, frac_above, entropy_val, iqr_val, top3_mean]


def _cache_path(cache_dir, cache_key):
    if cache_dir is None:
        return None
    return Path(cache_dir) / f"features15_{cache_key}.npz"


def extract_features_15(nli_model, sentences, sampled_passages, doc_indices,
                         cache_dir=None, cache_key=None):
    """
    Extract a 15-dim feature vector per sentence via predict_matrix.

    Layout (indices):
      0-6   : 7 Contradiction stats  (mean, max, std, frac>0.5, entropy, IQR, top3_mean)
      7-13  : 7 Entailment stats     (same 7 stats computed over E scores)
      14    : Support margin          mean(E) - mean(C)
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fp = _cache_path(cache_dir, cache_key)
        if fp is not None and fp.exists():
            try:
                features = np.load(fp)["features"]
                if len(features) == len(sentences):
                    print(f"  Loaded cached 15-features ({len(features)} sentences)")
                    return features
            except Exception:
                pass

    doc_to_sents = defaultdict(list)
    doc_to_passages = {}
    doc_to_global = defaultdict(list)

    for idx, (sent, passages, doc_idx) in enumerate(
        zip(sentences, sampled_passages, doc_indices)
    ):
        doc_to_sents[doc_idx].append(sent)
        doc_to_passages[doc_idx] = passages
        doc_to_global[doc_idx].append(idx)

    all_features = [None] * len(sentences)

    num_docs = len(doc_to_sents)
    try:
        from tqdm import tqdm
        iterator = tqdm(doc_to_sents.items(), desc="Extracting 15-features", total=num_docs)
    except ImportError:
        iterator = doc_to_sents.items()
        print(f"  Processing {num_docs} documents (install tqdm for a progress bar)...")

    for doc_i, (doc_idx, doc_sents) in enumerate(iterator):
        if not hasattr(iterator, "set_postfix") and (doc_i + 1) % 50 == 0:
            print(f"    documents {doc_i + 1}/{num_docs} ...")
        entail, _, contra = nli_model.predict_matrix(
            doc_sents, doc_to_passages[doc_idx]
        )
        for local_i, global_i in enumerate(doc_to_global[doc_idx]):
            c = contra[local_i]
            e = entail[local_i]
            c_stats = _compute_7_stats(c)
            e_stats = _compute_7_stats(e)
            margin = e_stats[0] - c_stats[0]
            all_features[global_i] = c_stats + e_stats + [margin]

    features = np.array(all_features, dtype=np.float32)

    if cache_dir is not None:
        fp = _cache_path(cache_dir, cache_key)
        try:
            np.savez_compressed(fp, features=features)
        except Exception:
            pass

    print(f"  Feature matrix shape: {features.shape}")
    return features


# ── Baseline unsupervised scores ─────────────────────────────────────────────

def compute_baseline_scores_exact(nli_model_name, device, sentences, sampled_passages):
    """Run the original SelfCheckNLI.predict() per sentence (slow but exact)."""
    baseline_nli = SelfCheckNLIBaseline(nli_model=nli_model_name, device=device)
    scores = np.empty(len(sentences))

    try:
        from tqdm import tqdm
        iterator = tqdm(
            enumerate(zip(sentences, sampled_passages)),
            total=len(sentences), desc="Baseline (exact)"
        )
    except ImportError:
        iterator = enumerate(zip(sentences, sampled_passages))

    for i, (sent, passages) in iterator:
        scores[i] = baseline_nli.predict([sent], passages)[0]

    del baseline_nli
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return scores


# ── Checkpoint (resume after interrupt) ───────────────────────────────────────

def _checkpoint_path(cache_dir, nli_model, test_size, random_seed):
    """Path for checkpoint file; same args => same path so resume works."""
    if cache_dir is None:
        return None
    safe = (nli_model or "default").replace("/", "_").replace(" ", "_")
    key = f"{safe}_ts{test_size}_seed{random_seed}"
    return Path(cache_dir) / f"checkpoint_expanded_{key}.pkl"


def _load_checkpoint(path, n_test):
    """
    Load checkpoint if valid. Returns dict with key "test_scores" (name -> array)
    or None if missing/invalid.
    """
    if path is None or not Path(path).exists():
        return None
    try:
        with open(path, "rb") as f:
            ck = pickle.load(f)
        if ck.get("version") != 1:
            return None
        for name, arr in ck.get("test_scores", {}).items():
            if not isinstance(arr, np.ndarray) or len(arr) != n_test:
                return None
        return ck
    except Exception:
        return None


def _save_checkpoint(path, test_scores, nli_model, test_size, random_seed):
    """Save checkpoint after each completed method."""
    if path is None:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ck = {
            "version": 1,
            "nli_model": nli_model,
            "test_size": test_size,
            "random_seed": random_seed,
            "n_test": len(next(iter(test_scores.values()))) if test_scores else 0,
            "test_scores": {k: np.asarray(v) for k, v in test_scores.items()},
        }
        with open(path, "wb") as f:
            pickle.dump(ck, f)
    except Exception:
        pass


# ── Evaluation helpers ───────────────────────────────────────────────────────

def f1_max_from_pr(y_true, y_score):
    """Maximum achievable F1 across all thresholds on the PR curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1s = 2 * precision * recall / (precision + recall)
    return float(np.nanmax(f1s))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train expanded-feature models and compare against baselines"
    )
    parser.add_argument("--nli_model", type=str, default="MoritzLaurer/DeBERTa-v3-base-mnli",
                        help="NLI model (default: deberta-v3-base for fast CPU; "
                             "use potsawee/deberta-v3-large-mnli for higher accuracy)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda / cpu / auto")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.3,
                        help="Test split ratio (default: 0.3, same as train_on_wikibio.py)")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--output_plot", type=str, default="pr_curve_comparison.png",
                        help="File path for the PR-curve overlay plot")
    parser.add_argument("--output_model", type=str, default="hallucination_detector_expanded.pkl",
                        help="File path for the saved best model")
    parser.add_argument("--cache_dir", type=str, default=".feature_cache",
                        help="Cache directory for features and checkpoint (empty to disable)")
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Disable checkpointing; do not resume from previous run")
    parser.add_argument("--exact_baseline", action="store_true",
                        help="Run the original per-sentence baseline (slow). "
                             "Default: use mean(C) from predict_matrix as proxy.")
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        d = args.device.lower().strip()
        if d.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA not available — falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(d)
    print(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────────────────────
    sentences, labels, sampled_passages, doc_indices = load_wikibio_data()

    # ── Extended NLI model ───────────────────────────────────────────────────
    batch_size = max(args.batch_size, 64) if device.type == "cuda" else args.batch_size
    print("\nInitializing SelfCheckNLI (extended)...")
    nli_ext = SelfCheckNLIExtended(
        nli_model=args.nli_model, device=device, batch_size=batch_size,
    )

    # ── 15-feature extraction ────────────────────────────────────────────────
    cache_dir = args.cache_dir if args.cache_dir else None
    print("\nExtracting 15-feature vectors for all sentences...")
    X_15 = extract_features_15(
        nli_ext, sentences, sampled_passages, doc_indices,
        cache_dir=cache_dir, cache_key="full_15feat",
    )
    y = labels.copy()

    # ── 70/30 document-level stratified split (same as train_on_wikibio.py) ──
    unique_docs = np.unique(doc_indices)
    doc_majority_label = np.array([
        int(np.bincount(y[doc_indices == d]).argmax()) for d in unique_docs
    ])
    train_docs, test_docs = train_test_split(
        unique_docs,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=doc_majority_label,
    )
    test_docs_set = set(test_docs)
    train_mask = np.array([d not in test_docs_set for d in doc_indices])
    test_mask = ~train_mask

    X_train, X_test = X_15[train_mask], X_15[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"\n  Split (same as train_on_wikibio.py):")
    print(f"    Train docs: {len(train_docs)}  |  Train sentences: {train_mask.sum()}")
    print(f"    Test docs:  {len(test_docs)}  |  Test sentences:  {test_mask.sum()}")
    print(f"    Test factual: {np.sum(y_test == 0)}  |  Test hallucinated: {np.sum(y_test == 1)}")

    # ── Baseline scores (test set only) ──────────────────────────────────────
    if args.exact_baseline:
        print("\nComputing exact baseline scores (original SelfCheckNLI)...")
        test_sentences = [s for s, m in zip(sentences, test_mask) if m]
        test_passages = [p for p, m in zip(sampled_passages, test_mask) if m]
        baseline_test = compute_baseline_scores_exact(
            args.nli_model, device, test_sentences, test_passages,
        )
    else:
        print("\nUsing mean(C) from predict_matrix as baseline proxy.")
        baseline_test = X_test[:, 0].copy()

    # ── Classifier definitions ───────────────────────────────────────────────
    classifiers = {
        "RF-15feat": {
            "make": lambda: RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            ),
        },
    }

    try:
        from xgboost import XGBClassifier
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        classifiers["XGB-15feat"] = {
            "make": lambda: XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                scale_pos_weight=spw, eval_metric="logloss",
                random_state=42, n_jobs=-1,
            ),
        }
        print("XGBoost available — included in comparison.\n")
    except Exception as e:
        print("XGBoost not available — skipping XGB-15feat.")
        if "libomp" in str(e).lower() or "openmp" in str(e).lower():
            print("  (On macOS install OpenMP: brew install libomp)\n")
        else:
            print(f"  ({e})\n")

    # ── Checkpoint path and resume from previous run if any ─────────────────
    checkpoint_path = None if args.no_checkpoint else _checkpoint_path(
        cache_dir, args.nli_model, args.test_size, args.random_seed
    )
    test_scores = {"Baseline": baseline_test}
    ck = _load_checkpoint(checkpoint_path, n_test=len(y_test))
    if ck is not None and ck.get("test_scores"):
        for k, v in ck["test_scores"].items():
            # Restore only classifier results (not Baseline; we keep current run's baseline)
            if k != "Baseline" and k in classifiers and len(v) == len(y_test):
                test_scores[k] = np.asarray(v)
        resumed = [m for m in test_scores if m != "Baseline"]
        if resumed:
            print(f"  Resumed: loaded from checkpoint for {resumed}")
    _save_checkpoint(
        checkpoint_path, test_scores,
        args.nli_model, args.test_size, args.random_seed,
    )

    # ── Train on 70%, predict on 30% (skip if already in checkpoint) ──────────
    print("Training classifiers on train set, evaluating on test set...")
    for name, cfg in classifiers.items():
        if name in test_scores:
            print(f"  {name}: using saved checkpoint (skipping training)")
            continue
        try:
            clf = cfg["make"]()
            clf.fit(X_train, y_train)
            proba = clf.predict_proba(X_test)[:, 1]
            test_scores[name] = proba
            cfg["trained_clf"] = clf
            _save_checkpoint(
                checkpoint_path, test_scores,
                args.nli_model, args.test_size, args.random_seed,
            )
            print(f"  {name}: trained on {len(y_train)} sentences, "
                  f"predicted on {len(y_test)} sentences (checkpoint saved)")
        except Exception as e:
            print(f"  {name}: failed ({e}); checkpoint saved for completed methods.")
            _save_checkpoint(
                checkpoint_path, test_scores,
                args.nli_model, args.test_size, args.random_seed,
            )
            raise

    # ── Compute metrics (all methods on same test set) ───────────────────────
    method_order = ["Baseline", "RF-15feat"]
    if "XGB-15feat" in classifiers:
        method_order.append("XGB-15feat")

    results = {}
    for name in method_order:
        scores = test_scores[name]
        auc_pr = average_precision_score(y_test, scores)
        f1m = f1_max_from_pr(y_test, scores)
        results[name] = {"AUC-PR": auc_pr, "F1-max": f1m}

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY  (70/30 doc-level split, seed={})".format(args.random_seed))
    print("=" * 60)
    header = f"  {'Method':<20s} {'AUC-PR':>10s} {'F1-max':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in method_order:
        r = results[name]
        print(f"  {name:<20s} {r['AUC-PR']:>10.4f} {r['F1-max']:>10.4f}")
    print("=" * 60)

    baseline_auc = results["Baseline"]["AUC-PR"]
    best_sup = max(
        (n for n in method_order if n != "Baseline"),
        key=lambda n: results[n]["AUC-PR"],
    )
    best_auc = results[best_sup]["AUC-PR"]
    delta = best_auc - baseline_auc
    print(f"\n  Best supervised method: {best_sup}")
    print(f"  AUC-PR improvement over baseline: {delta:+.4f} "
          f"({delta / baseline_auc * 100:+.1f}%)" if baseline_auc > 0 else "")

    # ── PR curve plot ────────────────────────────────────────────────────────
    colors = {
        "Baseline": "#888888",
        "RF-15feat": "#4CAF50",
        "XGB-15feat": "#FF9800",
    }
    linestyles = {
        "Baseline": "--",
        "RF-15feat": "-",
        "XGB-15feat": "-",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for name in method_order:
        scores = test_scores[name]
        prec, rec, _ = precision_recall_curve(y_test, scores)
        auc_pr = results[name]["AUC-PR"]
        ax.plot(
            rec, prec,
            color=colors.get(name, "#000"),
            linestyle=linestyles.get(name, "-"),
            linewidth=2,
            label=f"{name}  (AUC-PR = {auc_pr:.4f})",
        )

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision–Recall Curves: Hallucination Detection Methods", fontsize=14)
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_plot, dpi=150)
    plt.close(fig)
    print(f"\n  PR curve plot saved to {args.output_plot}")

    # ── Retrain best model on full data and save ────────────────────────────
    best_cfg = classifiers[best_sup]
    final_clf = best_cfg["make"]()
    final_clf.fit(X_15, y)

    model_data = {
        "model": final_clf,
        "feature_names": FEATURE_NAMES_15,
        "num_features": X_15.shape[1],
        "method": best_sup,
        "test_auc_pr": results[best_sup]["AUC-PR"],
        "test_f1_max": results[best_sup]["F1-max"],
    }
    with open(args.output_model, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  Final model ({best_sup}) saved to {args.output_model}")
    print("\nDone.")


if __name__ == "__main__":
    main()
