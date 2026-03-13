"""
Full NER-NLI pipeline
"""

import argparse
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .fact_extraction import extract_atomic_claims, load_spacy_model
from .nli_scorer import NLIScorer


# EXPERIMENT LR 8 FEATURES -- not in our final report

FEATURE_NAMES = [
    "max_contra_weighted",
    "mean_contra_weighted",
    "max_calibrated",
    "mean_calibrated",
    "max_contra_raw",
    "max_neutral",
    "n_claims",
    "has_high_stakes",
]

def evaluate_auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute AUC-PR (average precision) and max-F1."""
    auc_pr = float(average_precision_score(y_true, y_score))
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1s = 2 * precision * recall / (precision + recall)
    f1_max = float(np.nanmax(f1s))
    return {"auc_pr": auc_pr, "f1_max": f1_max}



def load_wikibio_with_reference():
    """
    Load WikiBio GPT-3 Hallucination dataset
    """
    from datasets import load_dataset

    print("Loading WikiBio GPT-3 Hallucination dataset...")
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")["evaluation"]

    sentences, labels, references, passages, doc_ids = [], [], [], [], []
    label_map = {"accurate": 0, "minor_inaccurate": 1, "major_inaccurate": 1}

    for doc_idx, example in enumerate(dataset):
        ref = example["wiki_bio_text"]
        for sent, ann in zip(example["gpt3_sentences"], example["annotation"]):
            sentences.append(sent)
            labels.append(label_map[ann])
            references.append(ref)
            passages.append(example["gpt3_text_samples"])
            doc_ids.append(doc_idx)

    labels = np.array(labels)
    doc_ids = np.array(doc_ids)
    n_f = int(np.sum(labels == 0))
    n_h = int(np.sum(labels == 1))
    print(f"  Documents: {len(dataset)}  |  Sentences: {len(sentences)}")
    print(f"  Factual: {n_f} ({n_f / len(labels) * 100:.1f}%)  "
          f"|  Hallucinated: {n_h} ({n_h / len(labels) * 100:.1f}%)")
    return sentences, labels, references, passages, doc_ids


def _save_checkpoint(
    path: Path,
    features: np.ndarray,
    completed_docs: Set[int],
    n_total: int,
    n_feat: int,
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "version": 2,
                "n_total": n_total,
                "n_feat": n_feat,
                "completed_docs": completed_docs,
                "features": features,
            }, f)
    except Exception:
        pass


def extract_features_with_checkpoint(
    nlp,
    scorer: NLIScorer,
    sentences: List[str],
    references: List[str],
    doc_ids: np.ndarray,
    entity_weight: float = 1.2,
    neutral_weight: float = 0.5,
    checkpoint_path: Optional[Path] = None,
    save_every: int = 10,
) -> np.ndarray:
    """
    Extract LR 8 dimesnion NER-NLI feature vector for every sentence.
    """
    n = len(sentences)
    n_feat = len(FEATURE_NAMES)

    doc_to_global: Dict[int, List[int]] = defaultdict(list)
    for idx, d in enumerate(doc_ids):
        doc_to_global[int(d)].append(idx)

    features = np.full((n, n_feat), np.nan, dtype=np.float32)
    completed_docs: Set[int] = set()

    if checkpoint_path is not None and checkpoint_path.exists():
        try:
            with open(checkpoint_path, "rb") as f:
                ck = pickle.load(f)
            if (ck.get("version") == 2
                    and ck["n_total"] == n
                    and ck["n_feat"] == n_feat):
                features = ck["features"]
                completed_docs = ck["completed_docs"]
                n_done = len(completed_docs)
                print(f"  Resumed checkpoint: {n_done}/{len(doc_to_global)} "
                      f"docs complete")
        except Exception:
            pass

    remaining = [d for d in doc_to_global if d not in completed_docs]
    if not remaining:
        print(f"  All {len(doc_to_global)} documents already complete")
        return features

    print(f"  Processing {len(remaining)} remaining documents "
          f"(of {len(doc_to_global)} total)...")

    try:
        from tqdm import tqdm
        iterator = tqdm(remaining, desc="NER-NLI features (by doc)")
    except ImportError:
        iterator = remaining

    docs_since_save = 0
    t0 = time.time()

    for doc_id in iterator:
        global_indices = doc_to_global[doc_id]
        doc_ref = references[global_indices[0]]

        for gi in global_indices:
            sent = sentences[gi]
            claims = extract_atomic_claims(sent, nlp)
            result = scorer.score_sentence(
                doc_ref, claims,
                entity_weight=entity_weight,
                neutral_weight=neutral_weight,
            )
            features[gi] = [
                result["max_contra_weighted"],
                result["mean_contra_weighted"],
                result["max_calibrated"],
                result["mean_calibrated"],
                result["max_contra_raw"],
                result["max_neutral"],
                result["n_claims"],
                1.0 if result["has_high_stakes"] else 0.0,
            ]

        completed_docs.add(doc_id)
        docs_since_save += 1

        if checkpoint_path is not None and docs_since_save >= save_every:
            _save_checkpoint(
                checkpoint_path, features, completed_docs, n, n_feat,
            )
            docs_since_save = 0
            scorer.clear_cache()

    elapsed = time.time() - t0
    print(f"  Feature extraction: {elapsed:.1f}s "
          f"({elapsed / max(len(remaining), 1):.2f}s/doc)")

    if checkpoint_path is not None:
        _save_checkpoint(
            checkpoint_path, features, completed_docs, n, n_feat,
        )

    return features


def train_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
) -> LogisticRegression:
    """Train logistic regression on 8 NER-NLI features."""
    clf = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def main():
    parser = argparse.ArgumentParser(
        description="NER-NLI pipeline for hallucination detection on WikiBio"
    )
    parser.add_argument(
        "--nli_model", type=str,
        default="microsoft/deberta-v3-base",
    )
    parser.add_argument(
        "--spacy_model", type=str, default="en_core_web_trf",
        help="spaCy model for NER (en_core_web_trf | en_core_web_sm)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--entity_weight", type=float, default=1.2
    )
    parser.add_argument(
        "--neutral_weight", type=float, default=0.5,
    )
    parser.add_argument("--C", type=float, default=1.0, help="LR regularization")
    parser.add_argument(
        "--cache_dir", type=str, default=".feature_cache",
    )
    parser.add_argument(
        "--output_plot", type=str, default="pr_curve_ner_nli.png",
    )
    parser.add_argument(
        "--output_model", type=str, default="ner_nli_model.pkl",
    )
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

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

    sentences, labels, references, passages, doc_ids = \
        load_wikibio_with_reference()

    print(f"\nLoading spaCy model ({args.spacy_model})...")
    nlp = load_spacy_model(args.spacy_model)

    batch_size = max(args.batch_size, 64) if device.type == "cuda" else args.batch_size
    print(f"Initializing NLIScorer ({args.nli_model})...")
    scorer = NLIScorer(
        nli_model=args.nli_model, device=device, batch_size=batch_size,
    )

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    checkpoint_file = None
    final_cache = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe = args.nli_model.replace("/", "_")
        final_cache = cache_dir / f"ner_nli_{safe}.npz"
        checkpoint_file = cache_dir / f"ner_nli_{safe}_checkpoint.pkl"

    X = None
    if final_cache is not None and final_cache.exists():
        try:
            X_loaded = np.load(final_cache)["features"]
            if len(X_loaded) == len(sentences):
                print(f"\n  Loaded cached NER-NLI features ({len(X_loaded)} rows)")
                X = X_loaded
        except Exception:
            pass

    if X is None:
        print(f"\nExtracting NER-NLI features for {len(sentences)} sentences...")
        print("  (checkpoints saved every 10 docs; safe to interrupt and resume)")
        X = extract_features_with_checkpoint(
            nlp, scorer, sentences, references, doc_ids,
            entity_weight=args.entity_weight,
            neutral_weight=args.neutral_weight,
            checkpoint_path=checkpoint_file,
            save_every=10,
        )
        if final_cache is not None:
            np.savez_compressed(final_cache, features=X)
            print(f"  Final features cached to {final_cache}")
        if checkpoint_file is not None and checkpoint_file.exists():
            checkpoint_file.unlink()
            print(f"  Checkpoint removed (no longer needed)")

    print(f"  Feature matrix shape: {X.shape}")
    y = labels.copy()

    # Use 70/30
    unique_docs = np.unique(doc_ids)
    doc_majority = np.array([
        int(np.bincount(y[doc_ids == d]).argmax()) for d in unique_docs
    ])
    train_docs, test_docs = train_test_split(
        unique_docs,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=doc_majority,
    )
    test_set = set(test_docs)
    train_mask = np.array([d not in test_set for d in doc_ids])
    test_mask = ~train_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"\n  Train: {train_mask.sum()} sentences ({len(train_docs)} docs)")
    print(f"  Test:  {test_mask.sum()} sentences ({len(test_docs)} docs)")

    rule_test = X_test[:, 0] # max_contra_weighted
    res_rule = evaluate_auc_pr(y_test, rule_test)
    print(f"\n  Rule (Step 3, W={args.entity_weight}):      "
          f"AUC-PR={res_rule['auc_pr']:.4f}  F1-max={res_rule['f1_max']:.4f}")

    cal_test = X_test[:, 2] # max_calibrated
    res_cal = evaluate_auc_pr(y_test, cal_test)
    print(f"  Calibrated (Step 4, γ={args.neutral_weight}):  "
          f"AUC-PR={res_cal['auc_pr']:.4f}  F1-max={res_cal['f1_max']:.4f}")

    clf = train_lr(X_train, y_train, C=args.C)
    lr_test = clf.predict_proba(X_test)[:, 1]
    res_lr = evaluate_auc_pr(y_test, lr_test)
    print(f"  LR-8ner (C={args.C}):                "
          f"AUC-PR={res_lr['auc_pr']:.4f}  F1-max={res_lr['f1_max']:.4f}")

    # baseline
    print("\n  Computing baseline (mean sentence-vs-sample contradiction)...")
    from selfcheckgpt.modeling_selfcheck_extended import (
        SelfCheckNLI as SelfCheckNLIExt,
    )
    nli_ext = SelfCheckNLIExt(
        nli_model=args.nli_model, device=device, batch_size=batch_size,
    )
    baseline_scores = np.zeros(test_mask.sum(), dtype=np.float32)
    test_indices = np.where(test_mask)[0]
    doc_to_test = defaultdict(list)
    for pos, gi in enumerate(test_indices):
        doc_to_test[int(doc_ids[gi])].append((pos, gi))

    from tqdm import tqdm
    doc_iter = tqdm(doc_to_test.items(), desc="Baseline (by doc)")

    for doc_id, pairs in doc_iter:
        doc_sents = [sentences[gi] for _, gi in pairs]
        doc_passages = passages[pairs[0][1]]
        _, _, contra = nli_ext.predict_matrix(doc_sents, doc_passages)
        for local_i, (pos, _) in enumerate(pairs):
            baseline_scores[pos] = float(np.mean(contra[local_i]))

    res_base = evaluate_auc_pr(y_test, baseline_scores)
    print(f"  Baseline (mean C):               "
          f"AUC-PR={res_base['auc_pr']:.4f}  F1-max={res_base['f1_max']:.4f}")

    del nli_ext
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n  LR-8ner coefficients:")
    for name, coef in zip(FEATURE_NAMES, clf.coef_[0]):
        print(f"    {name:<24s} {coef:+.4f}")
    print(f"    {'intercept':<24s} {clf.intercept_[0]:+.4f}")

    print("\n" + "=" * 62)
    print(f"  RESULTS  (70/30 doc-level split, seed={args.random_seed})")
    print("=" * 62)
    print(f"  {'Method':<30s} {'AUC-PR':>10s} {'F1-max':>10s}")
    print("  " + "-" * 54)
    for name, res in [
        ("Baseline (mean C)", res_base),
        (f"Rule (W={args.entity_weight})", res_rule),
        (f"Calibrated (γ={args.neutral_weight})", res_cal),
        (f"LR-8ner (C={args.C})", res_lr),
    ]:
        print(f"  {name:<30s} {res['auc_pr']:>10.4f} {res['f1_max']:>10.4f}")
    print("=" * 62)

    best_name, best_auc = max(
        [("Baseline", res_base["auc_pr"]),
         ("Rule", res_rule["auc_pr"]),
         ("Calibrated", res_cal["auc_pr"]),
         ("LR-8ner", res_lr["auc_pr"])],
        key=lambda x: x[1],
    )
    delta = best_auc - res_base["auc_pr"]
    print(f"\n  Best: {best_name}  (delta vs baseline: {delta:+.4f})")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, scores, color, ls in [
        ("Baseline (mean C)", baseline_scores, "#888888", "--"),
        (f"Rule (W={args.entity_weight})", rule_test, "#FF9800", "-"),
        (f"Calibrated (γ={args.neutral_weight})", cal_test, "#4CAF50", "-"),
        ("LR-8ner", lr_test, "#2196F3", "-"),
    ]:
        prec, rec, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        ax.plot(rec, prec, color=color, linestyle=ls, linewidth=2,
                label=f"{label}  (AUC-PR={ap:.4f})")

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("NER-NLI Pipeline: PR Curves on WikiBio", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_plot, dpi=150)
    plt.close(fig)
    print(f"  PR curve saved to {args.output_plot}")

    final_clf = train_lr(X, y, C=args.C)
    model_data = {
        "model": final_clf,
        "feature_names": FEATURE_NAMES,
        "num_features": len(FEATURE_NAMES),
        "method": "LR-8ner",
        "entity_weight": args.entity_weight,
        "neutral_weight": args.neutral_weight,
        "test_auc_pr": res_lr["auc_pr"],
        "test_f1_max": res_lr["f1_max"],
    }
    with open(args.output_model, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  Model saved to {args.output_model}")
    print("\nDone.")


if __name__ == "__main__":
    main()
