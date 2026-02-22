"""
Training script for supervised hallucination detection using SelfCheckNLI features.

This script:
1. Loads sentences and labels (hallucinated vs factual)
2. Generates sampled passages (or loads pre-generated ones)
3. Extracts 7 contradiction-derived features using SelfCheckNLI
4. Trains a logistic regression classifier
5. Evaluates and saves the model
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
import torch

from selfcheckgpt.modeling_selfcheck_extended import SelfCheckNLI


def load_data(sentences_path, labels_path, sampled_passages_path=None):
    """
    Load training data.
    
    Args:
        sentences_path: Path to file with sentences (one per line, or pickle)
        labels_path: Path to file with labels (0=factual, 1=hallucinated, or pickle)
        sampled_passages_path: Optional path to pre-generated sampled passages
    
    Returns:
        sentences: List of sentences
        labels: Array of labels (0 or 1)
        sampled_passages: List of sampled passages (or None if needs generation)
    """
    # Load sentences
    if sentences_path.suffix == '.pkl':
        with open(sentences_path, 'rb') as f:
            sentences = pickle.load(f)
    else:
        with open(sentences_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
    
    # Load labels
    if labels_path.suffix == '.pkl':
        with open(labels_path, 'rb') as f:
            labels = pickle.load(f)
    else:
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = [int(line.strip()) for line in f if line.strip()]
    
    labels = np.array(labels)
    
    # Load sampled passages if provided
    sampled_passages = None
    if sampled_passages_path and sampled_passages_path.exists():
        if sampled_passages_path.suffix == '.pkl':
            with open(sampled_passages_path, 'rb') as f:
                sampled_passages = pickle.load(f)
        else:
            with open(sampled_passages_path, 'r', encoding='utf-8') as f:
                sampled_passages = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(sentences)} sentences with {len(labels)} labels")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return sentences, labels, sampled_passages


def extract_features(sentences, sampled_passages, nli_model, device, batch_size=32):
    """
    Extract contradiction-derived features for all sentences.
    
    Args:
        sentences: List of sentences
        sampled_passages: List of sampled passages
        nli_model: SelfCheckNLI model instance
        device: torch device
        batch_size: Batch size for processing
    
    Returns:
        features: numpy array of shape (num_sentences, 7)
    """
    print("Extracting features...")
    features = nli_model.extract_features(sentences, sampled_passages)
    print(f"Extracted features shape: {features.shape}")
    return features


def train_model(X_train, y_train, X_val, y_val, C=1.0, max_iter=1000):
    """
    Train logistic regression model.
    
    Args:
        X_train: Training features (num_samples, 7)
        y_train: Training labels (num_samples,)
        X_val: Validation features
        y_val: Validation labels
        C: Regularization strength (inverse of regularization)
        max_iter: Maximum iterations
    
    Returns:
        model: Trained LogisticRegression model
        metrics: Dictionary of validation metrics
    """
    print("Training logistic regression...")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Train model
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=42,
        solver='lbfgs',  # Good for small datasets
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1': f1_score(y_val, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.0,
    }
    
    print("\nValidation Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Factual', 'Hallucinated']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    
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
    parser = argparse.ArgumentParser(description='Train supervised hallucination detection model')
    parser.add_argument('--sentences', type=str, required=True,
                        help='Path to sentences file (txt or pkl)')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to labels file (txt or pkl, 0=factual, 1=hallucinated)')
    parser.add_argument('--sampled_passages', type=str, default=None,
                        help='Path to sampled passages file (optional, txt or pkl)')
    parser.add_argument('--output', type=str, default='hallucination_detector.pkl',
                        help='Output path for trained model')
    parser.add_argument('--nli_model', type=str, default=None,
                        help='NLI model name (default: from NLIConfig)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda, cpu, or None for auto)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test/validation split ratio')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Logistic regression regularization strength')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='Maximum iterations for logistic regression')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    sentences_path = Path(args.sentences)
    labels_path = Path(args.labels)
    sampled_passages_path = Path(args.sampled_passages) if args.sampled_passages else None
    
    sentences, labels, sampled_passages = load_data(
        sentences_path, labels_path, sampled_passages_path
    )
    
    # Validate data
    assert len(sentences) == len(labels), "Sentences and labels must have same length"
    assert set(labels) <= {0, 1}, "Labels must be 0 (factual) or 1 (hallucinated)"
    
    # Initialize NLI model
    print("\nInitializing SelfCheckNLI model...")
    nli_model = SelfCheckNLI(
        nli_model=args.nli_model,
        device=device,
        batch_size=args.batch_size
    )
    
    # Extract features
    features = extract_features(
        sentences, sampled_passages, nli_model, device, args.batch_size
    )
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=labels  # Maintain class distribution
    )
    
    # Train model
    model, metrics = train_model(
        X_train, y_train, X_val, y_val,
        C=args.C, max_iter=args.max_iter
    )
    
    # Save model
    output_path = Path(args.output)
    save_model(model, output_path)
    
    print("\nTraining complete!")
    print(f"Model saved to: {output_path}")


if __name__ == '__main__':
    main()
