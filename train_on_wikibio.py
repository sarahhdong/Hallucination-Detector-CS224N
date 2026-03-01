"""
Training script for supervised hallucination detection on WikiBio GPT-3 Hallucination dataset.

This script:
1. Loads data from Hugging Face dataset: potsawee/wiki_bio_gpt3_hallucination
2. Flattens sentences and converts annotations to binary labels
   - Accurate → 0 (factual)
   - Minor Inaccurate → 1 (non-factual/hallucinated)
   - Major Inaccurate → 1 (non-factual/hallucinated)
3. Extracts 7 contradiction-derived features using SelfCheckNLI
4. Splits data: 70% train, 30% test (default)
5. Trains a logistic regression classifier
6. Evaluates on test set and saves the model
"""

import numpy as np
import pickle
import argparse
import json
import hashlib
from pathlib import Path
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
import torch

from selfcheckgpt.modeling_selfcheck_extended import SelfCheckNLI


def load_wikibio_data():
    """
    Load WikiBio GPT-3 Hallucination dataset from Hugging Face.
    
    Returns:
        all_sentences: List of all sentences (flattened)
        all_labels: Array of binary labels (0=factual, 1=hallucinated)
        all_sampled_passages: List of sampled passages (same for all sentences in a document)
        doc_indices: List mapping each sentence to its document index
    """
    print("Loading WikiBio GPT-3 Hallucination dataset from Hugging Face...")
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
    dataset = dataset['evaluation']
    
    all_sentences = []
    all_labels = []
    all_sampled_passages = []
    doc_indices = []
    
    # Label mapping: 
    # - Accurate → 0 (factual)
    # - Minor Inaccurate → 1 (non-factual/hallucinated)
    # - Major Inaccurate → 1 (non-factual/hallucinated)
    label_mapping = {
        'accurate': 0,           # Factual
        'minor_inaccurate': 1,   # Non-factual (hallucinated)
        'major_inaccurate': 1    # Non-factual (hallucinated)
    }
    
    for doc_idx, example in enumerate(dataset):
        sentences = example['gpt3_sentences']
        annotations = example['annotation']
        sampled_passages = example['gpt3_text_samples']  # List of 20 sampled passages
        
        # Flatten: each sentence gets its own entry
        for sent_idx, (sentence, annotation) in enumerate(zip(sentences, annotations)):
            all_sentences.append(sentence)
            all_labels.append(label_mapping[annotation])
            all_sampled_passages.append(sampled_passages)  # Same passages for all sentences in doc
            doc_indices.append(doc_idx)
    
    all_labels = np.array(all_labels)
    
    print(f"\nDataset Statistics:")
    print(f"  Total documents: {len(dataset)}")
    print(f"  Total sentences: {len(all_sentences)}")
    print(f"  Label distribution:")
    print(f"    Factual (0): {np.sum(all_labels == 0)} ({np.mean(all_labels == 0)*100:.1f}%)")
    print(f"    Non-factual/Hallucinated (1): {np.sum(all_labels == 1)} ({np.mean(all_labels == 1)*100:.1f}%)")
    print(f"  Sampled passages per sentence: {len(all_sampled_passages[0])}")
    print(f"\nLabel mapping:")
    print(f"  'accurate' → 0 (factual)")
    print(f"  'minor_inaccurate' → 1 (non-factual)")
    print(f"  'major_inaccurate' → 1 (non-factual)")
    
    return all_sentences, all_labels, all_sampled_passages, doc_indices


def compute_cache_key(sentences, sampled_passages, nli_model_name, batch_size):
    """Compute a hash key for caching features."""
    # Create a string representation of the data
    data_str = json.dumps({
        'sentences': sentences[:10],  # First 10 for quick check
        'num_sentences': len(sentences),
        'num_passages_per_sent': len(sampled_passages[0]) if sampled_passages else 0,
        'nli_model': nli_model_name,
        'batch_size': batch_size
    }, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()


def extract_features(sentences, sampled_passages, doc_indices, nli_model, device, batch_size=32, cache_dir=None, cache_key=None):
    """
    Extract contradiction-derived features for all sentences.
    Supports caching to avoid re-extraction on subsequent runs.
    
    Args:
        sentences: List of sentences
        sampled_passages: List of lists (each inner list has sampled passages for that sentence)
        doc_indices: List mapping each sentence to its document index
        nli_model: SelfCheckNLI model instance
        device: torch device
        batch_size: Batch size for processing
        cache_dir: Directory to save/load cached features (None to disable caching)
        cache_key: Optional cache key (auto-generated if None)
    
    Returns:
        features: numpy array of shape (num_sentences, 7)
    """
    # Try to load from cache
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        if cache_key is None:
            nli_model_name = getattr(nli_model, 'model_name', 'unknown')
            cache_key = compute_cache_key(sentences, sampled_passages, nli_model_name, batch_size)
        
        cache_file = cache_dir / f"features_{cache_key}.npz"
        cache_meta_file = cache_dir / f"features_{cache_key}.json"
        
        if cache_file.exists() and cache_meta_file.exists():
            try:
                print(f"\n📦 Loading cached features from {cache_file}...")
                data = np.load(cache_file)
                features = data['features']
                
                # Verify cache is valid
                with open(cache_meta_file, 'r') as f:
                    meta = json.load(f)
                if meta['num_sentences'] == len(sentences):
                    print(f"✅ Loaded {len(features)} cached feature vectors")
                    return features
                else:
                    print(f"⚠️  Cache mismatch (expected {meta['num_sentences']} sentences, got {len(sentences)}). Re-extracting...")
            except Exception as e:
                print(f"⚠️  Error loading cache: {e}. Re-extracting...")
    
    # Extract features (cache miss or cache disabled)
    print("\nExtracting features...")
    
    # Group sentences by document (sentences in same doc share sampled passages)
    from collections import defaultdict
    doc_to_sentences = defaultdict(list)
    doc_to_passages = {}
    doc_to_indices = defaultdict(list)
    
    for idx, (sent, passages, doc_idx) in enumerate(zip(sentences, sampled_passages, doc_indices)):
        doc_to_sentences[doc_idx].append(sent)
        doc_to_passages[doc_idx] = passages  # Same for all sentences in doc
        doc_to_indices[doc_idx].append(idx)
    
    # Process each document's sentences together (they share sampled passages)
    all_features = [None] * len(sentences)
    
    num_docs = len(doc_to_sentences)
    print(f"Processing {num_docs} documents...")
    
    # Use parallel processing for CPU, sequential for GPU (GPU can't be shared across processes)
    use_gpu = str(nli_model.device) != 'cpu'
    
    if num_docs > 1 and not use_gpu:
        # CPU: Use parallel processing
        try:
            from joblib import Parallel, delayed
            from tqdm import tqdm
            
            def process_doc(doc_idx):
                doc_sentences = doc_to_sentences[doc_idx]
                doc_passages = doc_to_passages[doc_idx]
                doc_sentence_indices = doc_to_indices[doc_idx]
                doc_features = nli_model.extract_features(doc_sentences, doc_passages)
                return doc_sentence_indices, doc_features
            
            # Use multiple workers for CPU (but not too many to avoid memory issues)
            n_workers = min(4, num_docs)
            
            results = Parallel(n_jobs=n_workers)(
                delayed(process_doc)(doc_idx)
                for doc_idx in tqdm(doc_to_sentences.keys(), desc="Extracting features")
            )
            
            # Store features in original order
            for doc_sentence_indices, doc_features in results:
                for feat_idx, orig_idx in enumerate(doc_sentence_indices):
                    all_features[orig_idx] = doc_features[feat_idx]
        except ImportError:
            # Fallback to sequential if joblib not available
            print("⚠️  joblib not available, processing sequentially...")
            from tqdm import tqdm
            for doc_idx in tqdm(doc_to_sentences.keys(), desc="Extracting features"):
                doc_sentences = doc_to_sentences[doc_idx]
                doc_passages = doc_to_passages[doc_idx]
                doc_sentence_indices = doc_to_indices[doc_idx]
                
                doc_features = nli_model.extract_features(doc_sentences, doc_passages)
                
                for feat_idx, orig_idx in enumerate(doc_sentence_indices):
                    all_features[orig_idx] = doc_features[feat_idx]
    else:
        # GPU or single document: process sequentially with progress bar
        try:
            from tqdm import tqdm
            iterator = tqdm(doc_to_sentences.items(), desc="Extracting features")
        except ImportError:
            iterator = doc_to_sentences.items()
        
        for doc_idx, doc_sentences in iterator:
            doc_passages = doc_to_passages[doc_idx]
            doc_sentence_indices = doc_to_indices[doc_idx]
            
            doc_features = nli_model.extract_features(doc_sentences, doc_passages)
            
            for feat_idx, orig_idx in enumerate(doc_sentence_indices):
                all_features[orig_idx] = doc_features[feat_idx]
    
    features = np.array(all_features)
    print(f"Extracted features shape: {features.shape}")
    
    # Save to cache if enabled
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        if cache_key is None:
            nli_model_name = getattr(nli_model, 'model_name', 'unknown')
            cache_key = compute_cache_key(sentences, sampled_passages, nli_model_name, batch_size)
        
        cache_file = cache_dir / f"features_{cache_key}.npz"
        cache_meta_file = cache_dir / f"features_{cache_key}.json"
        
        try:
            print(f"💾 Saving features to cache: {cache_file}")
            np.savez_compressed(cache_file, features=features)
            with open(cache_meta_file, 'w') as f:
                json.dump({
                    'num_sentences': len(sentences),
                    'num_features': features.shape[1],
                    'nli_model': getattr(nli_model, 'model_name', 'unknown'),
                    'batch_size': batch_size
                }, f, indent=2)
            print(f"✅ Features cached successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not save cache: {e}")
    
    return features


def train_model(X_train, y_train, X_test, y_test, C=1.0, max_iter=1000):
    """
    Train logistic regression model and evaluate on test set.
    
    Args:
        X_train: Training features (num_samples, 7)
        y_train: Training labels (num_samples,)
        X_test: Test features
        y_test: Test labels
        C: Regularization strength (inverse of regularization)
        max_iter: Maximum iterations
    
    Returns:
        model: Trained LogisticRegression model
        metrics: Dictionary of test metrics
    """
    print("\nTraining logistic regression...")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=42,
        solver='lbfgs',  # Good for small datasets
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0,
    }
    
    print("\n" + "="*60)
    print("TEST SET METRICS:")
    print("="*60)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Factual', 'Hallucinated']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("="*60)
    
    return model, metrics


def save_model(model, output_path, feature_names=None):
    """
    Save trained model and metadata.
    
    Args:
        model: Trained LogisticRegression model
        output_path: Path to save model
        feature_names: Optional list of feature names
    """
    model_data = {
        'model': model,
        'feature_names': feature_names or [
            'mean(C)', 'max(C)', 'std(C)', 'frac(C>0.5)', 
            'entropy(C)', 'iqr(C)', 'top3_mean(C)'
        ],
        'num_features': 7
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train supervised hallucination detection on WikiBio dataset')
    parser.add_argument('--output', type=str, default='hallucination_detector_wikibio.pkl',
                        help='Output path for trained model')
    parser.add_argument('--nli_model', type=str, default=None,
                        help='NLI model name (default: from NLIConfig)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda, cpu, or None for auto)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--test_size', type=float, default=0.3,
                        help='Test split ratio (default: 0.3 for 30% test, 70% train)')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Logistic regression regularization strength')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='Maximum iterations for logistic regression')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--split_by_doc', action='store_true',
                        help='Split by document (ensures sentences from same doc in same split)')
    parser.add_argument('--cache_dir', type=str, default='.feature_cache',
                        help='Directory to cache extracted features (speeds up re-runs). Set to empty string to disable caching.')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # Setup device with proper CUDA validation
    if args.device is None:
        # Auto-detect: use CUDA if available, else CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        requested_device = args.device.lower().strip()
        # Check if CUDA is requested
        if requested_device == "cuda" or requested_device.startswith("cuda"):
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                print("⚠️  Warning: CUDA requested but not available. Falling back to CPU.")
                print("   (PyTorch was not compiled with CUDA support)")
                device = torch.device("cpu")
        else:
            # CPU or other device
            device = torch.device(requested_device)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    
    # Load data
    sentences, labels, sampled_passages, doc_indices = load_wikibio_data()
    
    # Split data
    if args.split_by_doc:
        # Split by document to ensure sentences from same doc stay together
        # Use stratified splitting to maintain class distribution
        from sklearn.model_selection import train_test_split
        
        unique_docs = np.unique(doc_indices)
        
        # Compute document-level labels for stratification
        # Use majority label per document (or proportion-based stratification)
        doc_labels = []
        for doc_idx in unique_docs:
            doc_mask = doc_indices == doc_idx
            doc_label_dist = labels[doc_mask]
            # Use majority label for stratification
            majority_label = int(np.bincount(doc_label_dist).argmax())
            doc_labels.append(majority_label)
        
        doc_labels = np.array(doc_labels)
        
        # Stratified split on documents
        train_docs, test_docs = train_test_split(
            unique_docs,
            test_size=args.test_size,
            random_state=args.random_seed,
            stratify=doc_labels  # Maintain class distribution at document level
        )
        
        test_docs = set(test_docs)
        train_mask = np.array([doc_idx not in test_docs for doc_idx in doc_indices])
        test_mask = ~train_mask
        
        y_train = labels[train_mask]
        y_test = labels[test_mask]
        train_sentences = [sent for sent, mask in zip(sentences, train_mask) if mask]
        test_sentences = [sent for sent, mask in zip(sentences, test_mask) if mask]
        train_sampled = [samp for samp, mask in zip(sampled_passages, train_mask) if mask]
        test_sampled = [samp for samp, mask in zip(sampled_passages, test_mask) if mask]
        train_doc_indices = [doc_idx for doc_idx, mask in zip(doc_indices, train_mask) if mask]
        test_doc_indices = [doc_idx for doc_idx, mask in zip(doc_indices, test_mask) if mask]
        
        print(f"\nSplit by document (stratified):")
        print(f"  Train documents: {len(train_docs)}")
        print(f"  Test documents: {len(test_docs)}")
        print(f"  Train sentences: {len(train_sentences)}")
        print(f"  Test sentences: {len(test_sentences)}")
    else:
        # Random split at sentence level
        from sklearn.model_selection import train_test_split
        
        indices = np.arange(len(sentences))
        train_indices, test_indices = train_test_split(
            indices,
            test_size=args.test_size,
            random_state=args.random_seed,
            stratify=labels  # Maintain class distribution
        )
        
        train_sentences = [sentences[i] for i in train_indices]
        test_sentences = [sentences[i] for i in test_indices]
        y_train = labels[train_indices]
        y_test = labels[test_indices]
        train_sampled = [sampled_passages[i] for i in train_indices]
        test_sampled = [sampled_passages[i] for i in test_indices]
        train_doc_indices = [doc_indices[i] for i in train_indices]
        test_doc_indices = [doc_indices[i] for i in test_indices]
        
        print(f"\nRandom split (stratified):")
        print(f"  Train sentences: {len(train_sentences)}")
        print(f"  Test sentences: {len(test_sentences)}")
    
    # Print class distribution for both splits
    print(f"\nClass Distribution:")
    print(f"  Train - Factual (0): {np.sum(y_train == 0)} ({np.mean(y_train == 0)*100:.1f}%)")
    print(f"  Train - Non-factual (1): {np.sum(y_train == 1)} ({np.mean(y_train == 1)*100:.1f}%)")
    print(f"  Test - Factual (0): {np.sum(y_test == 0)} ({np.mean(y_test == 0)*100:.1f}%)")
    print(f"  Test - Non-factual (1): {np.sum(y_test == 1)} ({np.mean(y_test == 1)*100:.1f}%)")
    
    # Initialize NLI model
    print("\nInitializing SelfCheckNLI model...")
    
    # Increase batch size if GPU is available (faster processing)
    effective_batch_size = args.batch_size
    if device.type == 'cuda':
        # Use larger batches on GPU
        effective_batch_size = max(args.batch_size, 64)
        print(f"GPU detected: using batch_size={effective_batch_size} (increased from {args.batch_size})")
    
    nli_model = SelfCheckNLI(
        nli_model=args.nli_model,
        device=device,
        batch_size=effective_batch_size
    )
    
    # Setup cache directory
    cache_dir = args.cache_dir if args.cache_dir else None
    
    # Extract features for training set
    print("\n" + "="*60)
    print("TRAINING SET")
    print("="*60)
    X_train = extract_features(
        train_sentences, train_sampled, train_doc_indices, nli_model, device, args.batch_size,
        cache_dir=cache_dir, cache_key='train'
    )
    
    # Extract features for test set
    print("\n" + "="*60)
    print("TEST SET")
    print("="*60)
    X_test = extract_features(
        test_sentences, test_sampled, test_doc_indices, nli_model, device, args.batch_size,
        cache_dir=cache_dir, cache_key='test'
    )
    
    # Train model
    model, metrics = train_model(
        X_train, y_train, X_test, y_test,
        C=args.C, max_iter=args.max_iter
    )
    
    # Save model
    output_path = Path(args.output)
    save_model(model, output_path)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model saved to: {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()
