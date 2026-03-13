"""
Use a trained confidence-weighted NLI model for inference, uses the 9 feature SelfCheckNLI 
"""

import sys
from pathlib import Path
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
from modeling_selfcheck_confidence_weighted import SelfCheckNLI
import numpy as np
import pickle
import argparse
import torch

def load_model(model_path):
    #load trained model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['feature_names']

def predict(sentences, sampled_passages, nli_model, classifier_model, device, batch_size=32):
    #predict hallucination probabilities for sentences
    features= nli_model.extract_features(sentences, sampled_passages)
    predictions= classifier_model.predict(features)
    probabilities= classifier_model.predict_proba(features)[:, 1]
    return predictions, probabilities


def main():
    parser = argparse.ArgumentParser(description='Use trained confidence-weighted hallucination detection model')
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
    
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Loading model from {args.model}")
    classifier_model, feature_names = load_model(args.model)
    print(f"Model loaded. features: {feature_names}")
    
    sentences_path = Path(args.sentences)
    if sentences_path.suffix== '.pkl':
        with open(sentences_path, 'rb') as f:
            sentences = pickle.load(f)
    else:
        with open(sentences_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
    
    sampled_passages_path = Path(args.sampled_passages)
    if sampled_passages_path.suffix == '.pkl':
        with open(sampled_passages_path, 'rb') as f:
            sampled_passages = pickle.load(f)
    else:
        with open(sampled_passages_path, 'r', encoding='utf-8') as f:
            sampled_passages = [line.strip() for line in f if line.strip()]
    
    print(f"loaded {len(sentences)} sentences")
    print(f"loaded {len(sampled_passages)} sampled passages")
    
    print("\nInitializing SelfCheckNLI (confidence-weighted)")
    nli_model = SelfCheckNLI(
        nli_model=args.nli_model,
        device=device,
        batch_size=args.batch_size
    )
    
    print("\nextracting features and predicting")
    predictions, probabilities = predict(
        sentences, sampled_passages, nli_model, classifier_model, device, args.batch_size
    )
    
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("sentence\tprediction\tprobability\n")
        for sent, pred, prob in zip(sentences, predictions, probabilities):
            label = "hallucinated" if pred == 1 else "factual"
            f.write(f"{sent}\t{label}\t{prob:.4f}\n")
    
    print(f"\nPredictions saved to {output_path}")
    print(f"\nSummary:")
    print(f"Total sentences: {len(sentences)}")
    print(f"Predicted factual: {np.sum(predictions == 0)}")
    print(f"Predicted hallucinated: {np.sum(predictions == 1)}")
    print(f" Mean prob: {np.mean(probabilities):.4f}")


if __name__ == '__main__':
    main()
