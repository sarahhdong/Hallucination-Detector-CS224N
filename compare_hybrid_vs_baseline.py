"""
Compare baseline SelfCheckNLI against a trained hybrid detector.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI as SelfCheckNLIBaseline
from selfcheckgpt.modeling_selfcheck_extended import SelfCheckNLI as SelfCheckNLIExtended
from train_hybrid_paper_features import (extract_hybrid_features, load_wikibio_data, split_data, TokenUncertaintyScorer,)


def evaluate_baseline(test_sentences, test_samples, test_labels, nli_model, device):
    print("\n" + "=" * 60)
    print("BASELINE: Original SelfCheckNLI")
    print("=" * 60)
    baseline_nli = SelfCheckNLIBaseline(nli_model=nli_model, device=device)

    scores = []
    for i, (sent, passages) in enumerate(zip(test_sentences, test_samples), 1):
        score = baseline_nli.predict([sent], passages)[0]
        scores.append(score)
    scores = np.array(scores)
    preds = (scores > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(test_labels, preds),
        "precision": precision_score(test_labels, preds, zero_division=0),
        "recall": recall_score(test_labels, preds, zero_division=0),
        "f1": f1_score(test_labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(test_labels, scores) if len(np.unique(test_labels)) > 1 else 0.0,
        "auc_pr": average_precision_score(test_labels, scores),
    }
    return metrics


def evaluate_hybrid(test_labels, X_test, model):
    print("\n" + "=" * 60)
    print("HYBRID MODEL")
    print("=" * 60)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(test_labels, preds),
        "precision": precision_score(test_labels, preds, zero_division=0),
        "recall": recall_score(test_labels, preds, zero_division=0),
        "f1": f1_score(test_labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(test_labels, probs) if len(np.unique(test_labels)) > 1 else 0.0,
        "auc_pr": average_precision_score(test_labels, probs),
    }
    return metrics


def print_metrics(title, m):
    print(f"\n{title}")
    # print(f"  Accuracy:  {m['accuracy']:.4f}")
    # print(f"  Precision: {m['precision']:.4f}")
    # print(f"  Recall:    {m['recall']:.4f}")
    # print(f"  F1 Score:  {m['f1']:.4f}")
    # print(f"  ROC-AUC:   {m['roc_auc']:.4f}")
    print(f"  AUC-PR:    {m['auc_pr']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compare hybrid model vs baseline")
    parser.add_argument("--model", type=str, required=True, help="Path to hybrid model pkl")
    parser.add_argument("--nli_model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--split_by_doc", action="store_true")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--uncertainty_model", type=str, default="distilgpt2")
    parser.add_argument("--disable_uncertainty", action="store_true")
    parser.add_argument("--n_entropy_clusters", type=int, default=3)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Hybrid model not found: {model_path}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sentences, labels, sampled_passages, doc_indices = load_wikibio_data()
    (_train_sentences,
        test_sentences,
        _y_train,
        y_test,
        _train_sampled,
        test_sampled,
        _train_doc_indices,
        _test_doc_indices,
    ) = split_data( sentences, labels, sampled_passages, doc_indices, test_size=args.test_size, random_seed=args.random_seed, split_by_doc=args.split_by_doc,)
    print(f"Test size: {len(test_sentences)}")

    baseline_metrics = None
    if not args.skip_baseline:
        baseline_metrics = evaluate_baseline(
            test_sentences=test_sentences,
            test_sampled_passages=test_sampled,
            test_labels=y_test,
            nli_model=args.nli_model,
            device=device,
        )
        print_metrics("Baseline Metrics", baseline_metrics)

    nli = SelfCheckNLIExtended(nli_model=args.nli_model, device=device, batch_size=args.batch_size)
    uncertainty_scoring = None
    if not args.disable_uncertainty:
        uncertainty_scoring = TokenUncertaintyScorer(args.uncertainty_model, device=device)

    X_test = extract_hybrid_features( test_sentences,
        test_sampled,
        nli_model=nli,
        uncertainty_scoring=uncertainty_scoring,
        n_entropy_clusters=args.n_entropy_clusters,
        random_seed=args.random_seed,
    )

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    hybrid_model = model_data["model"]
    hybrid_metrics = evaluate_hybrid(y_test, X_test, hybrid_model)
    print_metrics("Hybrid Metrics", hybrid_metrics)

    if baseline_metrics is not None:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"{'Metric':<12} {'Baseline':<10} {'Hybrid':<10} {'Delta':<10}")
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "auc_pr"]:
            b = baseline_metrics[metric]
            h = hybrid_metrics[metric]
            print(f"{metric:<12} {b:<10.4f} {h:<10.4f} {h - b:+.4f}")


if __name__ == "__main__":
    main()

