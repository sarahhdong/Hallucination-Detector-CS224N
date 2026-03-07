"""
Compare baseline (original SelfCheckNLI) vs trained model on test set.

Computes AUC-PR (Area Under Precision-Recall Curve) for both methods.
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
from datasets import load_dataset
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import torch

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI as SelfCheckNLIBaseline
from selfcheckgpt.modeling_selfcheck_extended import SelfCheckNLI as SelfCheckNLIExtended


def load_wikibio_test_data(test_size=0.3, random_seed=42, split_by_doc=False):
    """
    Load WikiBio test data (same split as training).
    
    Returns:
        test_sentences: List of sentences
        test_labels: Array of labels (0=factual, 1=hallucinated)
        test_sampled_passages: List of sampled passages
    """
    print("Loading WikiBio dataset...")
    dataset = load_dataset('potsawee/wiki_bio_gpt3_hallucination')
    dataset = dataset['evaluation']
    
    all_sentences = []
    all_labels = []
    all_sampled_passages = []
    doc_indices = []
    
    # Label mapping
    label_mapping = {
        'accurate': 0,
        'minor_inaccurate': 1,
        'major_inaccurate': 1
    }
    
    for doc_idx, example in enumerate(dataset):
        sentences = example['gpt3_sentences']
        annotations = example['annotation']
        sampled_passages = example['gpt3_text_samples']
        
        for sent_idx, (sentence, annotation) in enumerate(zip(sentences, annotations)):
            all_sentences.append(sentence)
            all_labels.append(label_mapping[annotation])
            all_sampled_passages.append(sampled_passages)
            doc_indices.append(doc_idx)
    
    all_labels = np.array(all_labels)
    
    # Split data (same as training)
    if split_by_doc:
        from sklearn.model_selection import train_test_split
        unique_docs = np.unique(doc_indices)
        doc_labels = []
        for doc_idx in unique_docs:
            doc_mask = np.array(doc_indices) == doc_idx
            doc_label_dist = all_labels[doc_mask]
            majority_label = int(np.bincount(doc_label_dist).argmax())
            doc_labels.append(majority_label)
        doc_labels = np.array(doc_labels)
        
        train_docs, test_docs = train_test_split(
            unique_docs,
            test_size=test_size,
            random_state=random_seed,
            stratify=doc_labels
        )
        test_docs = set(test_docs)
        test_mask = np.array([doc_idx in test_docs for doc_idx in doc_indices])
    else:
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(all_sentences))
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_seed,
            stratify=all_labels
        )
        test_mask = np.zeros(len(all_sentences), dtype=bool)
        test_mask[test_indices] = True
    
    test_sentences = [sent for sent, mask in zip(all_sentences, test_mask) if mask]
    test_labels = all_labels[test_mask]
    test_sampled_passages = [samp for samp, mask in zip(all_sampled_passages, test_mask) if mask]
    
    print(f"Test set: {len(test_sentences)} sentences")
    print(f"  Factual (0): {np.sum(test_labels == 0)} ({np.mean(test_labels == 0)*100:.1f}%)")
    print(f"  Hallucinated (1): {np.sum(test_labels == 1)} ({np.mean(test_labels == 1)*100:.1f}%)")
    
    return test_sentences, test_labels, test_sampled_passages


def evaluate_baseline(test_sentences, test_sampled_passages, test_labels, nli_model, device, batch_size=32):
    """
    Evaluate baseline SelfCheckNLI (original, unsupervised method).
    
    Returns:
        scores: Array of contradiction scores (higher = more likely hallucinated)
    """
    print("\n" + "="*60)
    print("BASELINE: Original SelfCheckNLI (Unsupervised)")
    print("="*60)
    
    # Use original SelfCheckNLI class
    baseline_nli = SelfCheckNLIBaseline(nli_model=nli_model, device=device)
    
    # Process sentences one by one (original predict() handles one sentence at a time)
    all_scores = []
    
    print(f"Processing {len(test_sentences)} sentences...")
    for i, (sent, passages) in enumerate(zip(test_sentences, test_sampled_passages)):
        # Original predict() takes: [sentence] and list of passages
        # Returns: average contradiction probability across all passages
        score = baseline_nli.predict([sent], passages)
        all_scores.append(score[0])
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(test_sentences)} sentences...")
    
    scores = np.array(all_scores)
    
    # Compute metrics
    # For baseline: higher score = more likely hallucinated (contradiction probability)
    # So scores directly correspond to hallucination probability
    predictions = (scores > 0.5).astype(int)
    
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    
    # AUC-ROC
    try:
        roc_auc = roc_auc_score(test_labels, scores)
    except:
        roc_auc = 0.0
    
    # AUC-PR (Precision-Recall)
    pr_auc = average_precision_score(test_labels, scores)
    
    print(f"\nBaseline Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  AUC-PR:    {pr_auc:.4f}")
    
    return scores, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }


def evaluate_trained(test_sentences, test_sampled_passages, test_labels, model_path, nli_model, device, batch_size=32):
    """
    Evaluate trained supervised model.
    
    Returns:
        probabilities: Array of hallucination probabilities
    """
    print("\n" + "="*60)
    print("TRAINED: Supervised Logistic Regression Model")
    print("="*60)
    
    # Load trained model
    print(f"Loading trained model from {model_path}...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        classifier = model_data['model']
        feature_names = model_data['feature_names']
    print(f"Model loaded. Features: {feature_names}")
    
    # Initialize extended NLI model for feature extraction
    extended_nli = SelfCheckNLIExtended(nli_model=nli_model, device=device, batch_size=batch_size)
    
    # Extract features
    # Note: extract_features expects sampled_passages to be a list of strings (shared),
    # but we have a list of lists (one per sentence). Process each sentence individually.
    print(f"Extracting features for {len(test_sentences)} sentences...")
    
    all_features = []
    for i, (sent, passages) in enumerate(zip(test_sentences, test_sampled_passages)):
        # Extract features for this sentence with its passages
        sent_features = extended_nli.extract_features([sent], passages)
        all_features.append(sent_features[0])  # Get single feature vector
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(test_sentences)} sentences...")
    
    features = np.array(all_features)
    
    # Predict
    predictions = classifier.predict(features)
    probabilities = classifier.predict_proba(features)[:, 1]  # Probability of class 1 (hallucinated)
    
    # Compute metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    
    # AUC-ROC
    try:
        roc_auc = roc_auc_score(test_labels, probabilities)
    except:
        roc_auc = 0.0
    
    # AUC-PR (Precision-Recall)
    pr_auc = average_precision_score(test_labels, probabilities)
    
    print(f"\nTrained Model Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  AUC-PR:    {pr_auc:.4f}")
    
    return probabilities, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }


def main():
    parser = argparse.ArgumentParser(description='Compare baseline vs trained model')
    parser.add_argument('--model', type=str, default='model.pkl',
                        help='Path to trained model file')
    parser.add_argument('--nli_model', type=str, default=None,
                        help='NLI model name (default: from NLIConfig)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda, cpu, or None for auto)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Test split ratio (must match training)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed (must match training)')
    parser.add_argument('--no_split_by_doc', action='store_true',
                        help='Use sentence-level split (default: split by document to match training)')
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip baseline evaluation (only evaluate trained model)')
    
    args = parser.parse_args()
    args.split_by_doc = not args.no_split_by_doc

    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        requested_device = args.device.lower().strip()
        if requested_device == "cuda" or requested_device.startswith("cuda"):
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                print("⚠️  Warning: CUDA requested but not available. Falling back to CPU.")
                device = torch.device("cpu")
        else:
            device = torch.device(requested_device)
    print(f"Using device: {device}")
    
    # Load test data
    test_sentences, test_labels, test_sampled_passages = load_wikibio_test_data(
        test_size=args.test_size,
        random_seed=args.random_seed,
        split_by_doc=args.split_by_doc
    )
    
    # Evaluate baseline (if not skipped)
    if not args.skip_baseline:
        baseline_scores, baseline_metrics = evaluate_baseline(
            test_sentences, test_sampled_passages, test_labels,
            args.nli_model, device, args.batch_size
        )
    else:
        baseline_metrics = None
        print("\n⚠️  Skipping baseline evaluation (--skip_baseline flag set)")
    
    # Evaluate trained model
    trained_probs, trained_metrics = evaluate_trained(
        test_sentences, test_sampled_passages, test_labels,
        args.model, args.nli_model, device, args.batch_size
    )
    
    # Comparison (if baseline was evaluated)
    if baseline_metrics is not None:
        print("\n" + "="*60)
        print("COMPARISON: Baseline vs Trained Model")
        print("="*60)
        print(f"\n{'Metric':<15} {'Baseline':<12} {'Trained':<12} {'Improvement':<12}")
        print("-" * 60)
        
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        for metric in metrics_to_compare:
            baseline_val = baseline_metrics[metric]
            trained_val = trained_metrics[metric]
            improvement = trained_val - baseline_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
            print(f"{metric.capitalize():<15} {baseline_val:<12.4f} {trained_val:<12.4f} {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        print("\n" + "="*60)
        print("Key Results:")
        print(f"  Baseline AUC-PR:  {baseline_metrics['pr_auc']:.4f}")
        print(f"  Trained AUC-PR:   {trained_metrics['pr_auc']:.4f}")
        print(f"  Improvement:      {trained_metrics['pr_auc'] - baseline_metrics['pr_auc']:+.4f}")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("TRAINED MODEL RESULTS")
        print("="*60)
        print(f"\nAUC-PR: {trained_metrics['pr_auc']:.4f}")
        print(f"ROC-AUC: {trained_metrics['roc_auc']:.4f}")
        print(f"Accuracy: {trained_metrics['accuracy']:.4f}")
        print(f"Precision: {trained_metrics['precision']:.4f}")
        print(f"Recall: {trained_metrics['recall']:.4f}")
        print(f"F1 Score: {trained_metrics['f1']:.4f}")
        print("="*60)


if __name__ == '__main__':
    main()
