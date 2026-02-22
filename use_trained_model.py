"""
Script to use a trained supervised hallucination detection model.

Example usage:
    python use_trained_model.py \
        --model hallucination_detector.pkl \
        --sentences test_sentences.txt \
        --sampled_passages sampled_passages.txt \
        --output predictions.txt
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
import torch

from selfcheckgpt.modeling_selfcheck_extended import SelfCheckNLI


def load_model(model_path):
    """Load trained model."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['feature_names']


def predict(sentences, sampled_passages, nli_model, classifier_model, device, batch_size=32):
    """
    Predict hallucination probabilities for sentences.
    
    Args:
        sentences: List of sentences
        sampled_passages: List of sampled passages
        nli_model: SelfCheckNLI model instance
        classifier_model: Trained logistic regression model
        device: torch device
        batch_size: Batch size for feature extraction
    
    Returns:
        predictions: Array of predicted labels (0 or 1)
        probabilities: Array of hallucination probabilities (0 to 1)
    """
    # Extract features
    features = nli_model.extract_features(sentences, sampled_passages)
    
    # Predict
    predictions = classifier_model.predict(features)
    probabilities = classifier_model.predict_proba(features)[:, 1]  # Probability of class 1 (hallucinated)
    
    return predictions, probabilities


def main():
    parser = argparse.ArgumentParser(description='Use trained hallucination detection model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model file')
    parser.add_argument('--sentences', type=str, required=True,
                        help='Path to sentences file (txt or pkl)')
    parser.add_argument('--sampled_passages', type=str, required=True,
                        help='Path to sampled passages file (txt or pkl)')
    parser.add_argument('--output', type=str, default='predictions.txt',
                        help='Output path for predictions')
    parser.add_argument('--nli_model', type=str, default=None,
                        help='NLI model name (default: from NLIConfig)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda, cpu, or None for auto)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for binary prediction')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    classifier_model, feature_names = load_model(args.model)
    print(f"Model loaded. Features: {feature_names}")
    
    # Load sentences
    sentences_path = Path(args.sentences)
    if sentences_path.suffix == '.pkl':
        with open(sentences_path, 'rb') as f:
            sentences = pickle.load(f)
    else:
        with open(sentences_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
    
    # Load sampled passages
    sampled_passages_path = Path(args.sampled_passages)
    if sampled_passages_path.suffix == '.pkl':
        with open(sampled_passages_path, 'rb') as f:
            sampled_passages = pickle.load(f)
    else:
        with open(sampled_passages_path, 'r', encoding='utf-8') as f:
            sampled_passages = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(sentences)} sentences")
    print(f"Loaded {len(sampled_passages)} sampled passages")
    
    # Initialize NLI model
    print("\nInitializing SelfCheckNLI model...")
    nli_model = SelfCheckNLI(
        nli_model=args.nli_model,
        device=device,
        batch_size=args.batch_size
    )
    
    # Predict
    print("\nExtracting features and predicting...")
    predictions, probabilities = predict(
        sentences, sampled_passages, nli_model, classifier_model, device, args.batch_size
    )
    
    # Save predictions
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("sentence\tprediction\tprobability\n")
        for sent, pred, prob in zip(sentences, predictions, probabilities):
            label = "hallucinated" if pred == 1 else "factual"
            f.write(f"{sent}\t{label}\t{prob:.4f}\n")
    
    print(f"\nPredictions saved to {output_path}")
    print(f"\nSummary:")
    print(f"  Total sentences: {len(sentences)}")
    print(f"  Predicted factual: {np.sum(predictions == 0)}")
    print(f"  Predicted hallucinated: {np.sum(predictions == 1)}")
    print(f"  Mean probability: {np.mean(probabilities):.4f}")


if __name__ == '__main__':
    main()
