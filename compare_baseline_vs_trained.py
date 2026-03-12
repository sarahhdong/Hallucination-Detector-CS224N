"""
compare baseline vs trained model on test set
"""

import numpy as np
import pickle
import argparse
import json
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
    print("\n" + "="*60)
    print("BASELINE: Original SelfCheckNLI (Unsupervised)")
    print("="*60)
    
    baseline_nli = SelfCheckNLIBaseline(nli_model=nli_model, device=device)
    
    all_scores = []
    
    print(f"Processing {len(test_sentences)} sentences...")
    for i, (sent, passages) in enumerate(zip(test_sentences, test_sampled_passages)):
        score = baseline_nli.predict([sent], passages)
        all_scores.append(score[0])
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(test_sentences)} sentences...")
    
    scores = np.array(all_scores)
    predictions = (scores > 0.5).astype(int)
    
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(test_labels, scores)
    except:
        roc_auc = 0.0
    
    auc_pr = average_precision_score(test_labels, scores)
    
    print(f"  AUC-PR:    {auc_pr:.4f}")
    
    return scores, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'auc_pr': auc_pr
    }


def compute_metrics_from_scores(test_labels, scores):
    predictions = (scores > 0.5).astype(int)
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    try:
        roc_auc = roc_auc_score(test_labels, scores)
    except Exception:
        roc_auc = 0.0
    auc_pr = average_precision_score(test_labels, scores)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'auc_pr': auc_pr
    }


def evaluate_trained(
    test_sentences,
    test_sampled_passages,
    test_labels,
    model_path,
    nli_model,
    device,
    batch_size=32,
    cached_features=None,
):
    print("\n" + "="*60)
    print("TRAINED: Supervised Logistic Regression Model")
    print("="*60)
    
    print(f"Loading trained model from {model_path}...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        classifier = model_data['model']
        feature_names = model_data['feature_names']
    print(f"Model loaded. Features: {feature_names}")
    
    if cached_features is not None:
        print(f"Using cached test features: shape={cached_features.shape}")
        features = cached_features
    else:
        extended_nli = SelfCheckNLIExtended(nli_model=nli_model, device=device, batch_size=batch_size)
        print(f"Extracting features for {len(test_sentences)} sentences...")
        
        all_features = []
        for i, (sent, passages) in enumerate(zip(test_sentences, test_sampled_passages)):
            sent_features = extended_nli.extract_features([sent], passages)
            all_features.append(sent_features[0])  
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(test_sentences)} sentences...")
        
        features = np.array(all_features)
    
    # Predict
    predictions = classifier.predict(features)
    probabilities = classifier.predict_proba(features)[:, 1] 
    
    # Compute metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(test_labels, probabilities)
    except:
        roc_auc = 0.0
    
    auc_pr = average_precision_score(test_labels, probabilities)
    
    print(f"  AUC-PR:    {auc_pr:.4f}")
    
    return probabilities, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'auc_pr': auc_pr
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
    parser.add_argument('--cache_dir', type=str, default='.feature_cache',
                        help='Directory containing cached features (features_test.npz). Set to empty string to disable.')
    
    args = parser.parse_args()
    args.split_by_doc = not args.no_split_by_doc

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        requested_device = args.device.lower().strip()
        if requested_device == "cuda" or requested_device.startswith("cuda"):
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                print("Warning: CUDA requested but not available. Falling back to CPU.")
                device = torch.device("cpu")
        else:
            device = torch.device(requested_device)
    print(f"Using device: {device}")
    
    test_sentences, test_labels, test_sampled_passages = load_wikibio_test_data(
        test_size=args.test_size,
        random_seed=args.random_seed,
        split_by_doc=args.split_by_doc
    )
    
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_baseline:
        baseline_scores = None
        baseline_cache_file = None
        baseline_meta_file = None
        if cache_dir is not None:
            baseline_cache_file = cache_dir / 'baseline_scores_test.npz'
            baseline_meta_file = cache_dir / 'baseline_scores_test.json'
            if baseline_cache_file.exists():
                try:
                    cached_scores = np.load(baseline_cache_file)['scores']
                    if len(cached_scores) == len(test_labels):
                        baseline_scores = cached_scores
                        print(f"Loaded cached baseline scores from {baseline_cache_file}")
                    else:
                        print(
                            f"Cached baseline score count mismatch: "
                            f"{len(cached_scores)} vs {len(test_labels)} labels. Recomputing baseline."
                        )
                except Exception as e:
                    print(f"Could not load cached baseline scores ({e}). Recomputing baseline.")

        if baseline_scores is None:
            baseline_scores, _ = evaluate_baseline(
                test_sentences, test_sampled_passages, test_labels,
                args.nli_model, device, args.batch_size
            )
            if baseline_cache_file is not None:
                try:
                    np.savez_compressed(baseline_cache_file, scores=baseline_scores)
                    with open(baseline_meta_file, 'w') as f:
                        json.dump({
                            'num_test_sentences': len(test_labels),
                            'test_size': args.test_size,
                            'random_seed': args.random_seed,
                            'split_by_doc': args.split_by_doc,
                            'nli_model': args.nli_model or 'default',
                        }, f, indent=2)
                    print(f"Saved baseline scores to cache: {baseline_cache_file}")
                except Exception as e:
                    print(f"Could not save baseline scores cache: {e}")

        baseline_metrics = compute_metrics_from_scores(test_labels, baseline_scores)
        print(f"\nBaseline Metrics:")
        print(f"  AUC-PR:    {baseline_metrics['auc_pr']:.4f}")
    else:
        baseline_metrics = None
        print("\nSkipping baseline evaluation (--skip_baseline flag set)")
    
    cached_test_features = None
    if cache_dir is not None:
        test_cache_file = cache_dir / 'features_test.npz'
        if test_cache_file.exists():
            try:
                cached_test_features = np.load(test_cache_file)['features']
                if len(cached_test_features) != len(test_labels):
                    print(
                        f"Cached test feature count mismatch: "
                        f"{len(cached_test_features)} vs {len(test_labels)} labels. "
                        "Falling back to on-the-fly extraction."
                    )
                    cached_test_features = None
                else:
                    print(f"Loaded cached test features from {test_cache_file}")
            except Exception as e:
                print(f"Could not load cached test features ({e}). Falling back to extraction.")
                cached_test_features = None
        else:
            print(f"No cached test features found at {test_cache_file}. Falling back to extraction.")

    trained_probs, trained_metrics = evaluate_trained(
        test_sentences, test_sampled_passages, test_labels,
        args.model, args.nli_model, device, args.batch_size,
        cached_features=cached_test_features
    )
    
    if baseline_metrics is not None:
        print("\n" + "="*60)
        print("COMPARISON: Baseline vs Trained Model")
        print("="*60)
        print(f"\n{'Metric':<15} {'Baseline':<12} {'Trained':<12} {'Improvement':<12}")
        print("-" * 60)
        
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'auc_pr']
        for metric in metrics_to_compare:
            baseline_val = baseline_metrics[metric]
            trained_val = trained_metrics[metric]
            improvement = trained_val - baseline_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
            print(f"{metric.capitalize():<15} {baseline_val:<12.4f} {trained_val:<12.4f} {improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        print("\n" + "="*60)
        print("Key Results:")
        print(f"  Baseline AUC-PR:  {baseline_metrics['auc_pr']:.4f}")
        print(f"  Trained AUC-PR:   {trained_metrics['auc_pr']:.4f}")
        print(f"  Improvement:      {trained_metrics['auc_pr'] - baseline_metrics['auc_pr']:+.4f}")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("TRAINED MODEL RESULTS")
        print("="*60)
        print(f"\nAUC-PR: {trained_metrics['auc_pr']:.4f}")
        print("="*60)


if __name__ == '__main__':
    main()
