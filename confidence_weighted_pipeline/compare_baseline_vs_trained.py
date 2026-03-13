"""
compare baseline (original SelfCheckNLI) vs trained confidence weighted model on test set
"""

import sys
from pathlib import Path
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
from modeling_selfcheck_confidence_weighted import SelfCheckNLI as SelfCheckNLIExtended

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

def load_wikibio_test_data(test_size=0.3, random_seed=42, split_by_doc=False):
    #load WikiBio test data, return test_sentencesm,test_labels ,  test_sampled_passage
    dataset = load_dataset('potsawee/wiki_bio_gpt3_hallucination')
    dataset = dataset['evaluation']
    all_sentences= []
    all_labels= []
    all_sampled_passages= []
    doc_indices= []
    label_mapping= {
        'accurate': 0,
        'minor_inaccurate': 1,
        'major_inaccurate': 1
    }
    
    for doc_idx, example in enumerate(dataset):
        sentences =example['gpt3_sentences']
        annotations =example['annotation']
        sampled_passages =example['gpt3_text_samples']
        
        for sent_idx, (sentence, annotation) in enumerate(zip(sentences, annotations)):
            all_sentences.append(sentence)
            all_labels.append(label_mapping[annotation])
            all_sampled_passages.append(sampled_passages)
            doc_indices.append(doc_idx)
    all_labels = np.array(all_labels)
    
    if split_by_doc:
        from sklearn.model_selection import train_test_split
        unique_docs = np.unique(doc_indices)
        doc_labels=[]
        for doc_idx in unique_docs:
            doc_mask= np.array(doc_indices) == doc_idx
            doc_label_dist= all_labels[doc_mask]
            majority_label= int(np.bincount(doc_label_dist).argmax())
            doc_labels.append(majority_label)
        doc_labels=np.array(doc_labels)
        train_docs, test_docs = train_test_split(unique_docs,test_size=test_size,random_state=random_seed,stratify=doc_labels )
        test_docs= set(test_docs)
        test_mask= np.array([doc_idx in test_docs for doc_idx in doc_indices])
    else:
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(all_sentences))
        train_indices, test_indices = train_test_split(indices,test_size=test_size,random_state=random_seed,stratify=all_labels)
        test_mask = np.zeros(len(all_sentences), dtype=bool)
        test_mask[test_indices]= True
    
    test_sentences = [sent for sent, mask in zip(all_sentences, test_mask) if mask]
    test_labels = all_labels[test_mask]
    test_sampled_passages = [samp for samp, mask in zip(all_sampled_passages, test_mask) if mask]
    
    print(f"test set: {len(test_sentences)} sentences")
    print(f"  factual (0): {np.sum(test_labels==0)} ({np.mean(test_labels == 0)*100:.1f}%)")
    print(f"  hallucinated (1): {np.sum(test_labels==1)} ({np.mean(test_labels == 1)*100:.1f}%)")
    
    return test_sentences, test_labels, test_sampled_passages


def evaluate_baseline(test_sentences, test_sampled_passages, test_labels, nli_model, device, batch_size=32):
    #evaluate baseline SelfCheckNLI (original, unsupervised)
    print("baseline: original SelfCheckNLI (unsupervised)")
    baseline_nli = SelfCheckNLIBaseline(nli_model=nli_model, device=device)
    all_scores=[]
    print(f"processing {len(test_sentences)} sentences")
    for i, (sent, passages) in enumerate(zip(test_sentences, test_sampled_passages)):
        score = baseline_nli.predict([sent], passages)
        all_scores.append(score[0])
        if (i+ 1)%50 == 0:
            print(f"  processed {i + 1}/{len(test_sentences)} sentences")
    
    scores= np.array(all_scores)
    predictions= (scores>0.5).astype(int)
    accuracy= accuracy_score(test_labels, predictions)
    precision= precision_score(test_labels, predictions, zero_division=0)
    recall= recall_score(test_labels, predictions, zero_division=0)
    f1= f1_score(test_labels, predictions, zero_division=0)
    try:
        roc_auc = roc_auc_score(test_labels, scores)
    except:
        roc_auc= 0.0
    pr_auc= average_precision_score(test_labels, scores)
    
    print(f"\nbaseline metrics:")
    print(f"  accuracy:  {accuracy:.4f}")
    print(f"  precision: {precision:.4f}")
    print(f"  recall:{recall:.4f}")
    print(f"  f1 Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  AUC-PR:  {pr_auc:.4f}")
    
    return scores, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

def evaluate_trained(test_sentences, test_sampled_passages, test_labels, model_path, nli_model, device, batch_size=32):
    # evaluate trained confidence weighted model (9-feature)
    print("trained: supervised logistic regression (confidence weighted)")
    
    print(f"loading trained model from {model_path}...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        classifier = model_data['model']
        feature_names = model_data['feature_names']
    print(f"model loaded. features: {feature_names}")
    extended_nli = SelfCheckNLIExtended(nli_model=nli_model, device=device, batch_size=batch_size)
    
    all_features = []
    for i, (sent, passages) in enumerate(zip(test_sentences, test_sampled_passages)):
        sent_features = extended_nli.extract_features([sent], passages)
        all_features.append(sent_features[0])
        if (i+ 1)%50 == 0:
            print(f" processed {i+1}/{len(test_sentences)} sentences")
    
    features= np.array(all_features)
    predictions= classifier.predict(features)
    probabilities= classifier.predict_proba(features)[:, 1]
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    try:
        roc_auc = roc_auc_score(test_labels, probabilities)
    except:
        roc_auc = 0.0
    pr_auc = average_precision_score(test_labels, probabilities)
    
    print(f"\ntrained model metrics:")
    print(f"  accuracy:  {accuracy:.4f}")
    print(f"  precision: {precision:.4f}")
    print(f"  recall:    {recall:.4f}")
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
    parser = argparse.ArgumentParser(description='Compare baseline vs confidence-weighted trained model')
    parser.add_argument('--model', type=str, default='hallucination_detector_wikibio.pkl',
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
                        help='Use sentence-level split (default: split by document)')
    parser.add_argument('--split_by_doc', '--split-by-doc', action='store_true', dest='_doc_split_alias',
                        help='Document-level split (default). Optional.')
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip baseline evaluation')
    
    args = parser.parse_args()
    args.split_by_doc = not args.no_split_by_doc

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        requested_device = args.device.lower().strip()
        if requested_device == "cuda" or requested_device.startswith("cuda"):
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = torch.device(requested_device)
    print(f"using device: {device}")
    
    test_sentences, test_labels, test_sampled_passages = load_wikibio_test_data(
        test_size=args.test_size,
        random_seed=args.random_seed,
        split_by_doc=args.split_by_doc
    )
    
    if not args.skip_baseline:
        baseline_scores, baseline_metrics = evaluate_baseline(
            test_sentences, test_sampled_passages, test_labels,
            args.nli_model, device, args.batch_size
        )
    else:
        baseline_metrics = None
        print("\nskipping baseline eval")
    
    trained_probs, trained_metrics = evaluate_trained(
        test_sentences, test_sampled_passages, test_labels,
        args.model, args.nli_model, device, args.batch_size
    )
    
    if baseline_metrics is not None:
        print("comparison: baseline vs trained (Confidence weighted)")
        print(f"\n{'Metric':<15} {'Baseline':<12} {'Trained':<12} {'Improvement':<12}")
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        for metric in metrics_to_compare:
            baseline_val = baseline_metrics[metric]
            trained_val = trained_metrics[metric]
            improvement = trained_val - baseline_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
            print(f"{metric.capitalize():<15} {baseline_val:<12.4f} {trained_val:<12.4f} {improvement:+.4f} ({improvement_pct:+.1f}%)")
        print("Key Results:")
        print(f"  baseline AUC-PR: {baseline_metrics['pr_auc']:.4f}")
        print(f"  trained AUC-PR: {trained_metrics['pr_auc']:.4f}")
        print(f"  improvement: {trained_metrics['pr_auc'] - baseline_metrics['pr_auc']:+.4f}")
    else:
        print("trained model results")
        print(f"\nAUC-PR: {trained_metrics['pr_auc']:.4f}")
        print(f"ROC-AUC: {trained_metrics['roc_auc']:.4f}")


if __name__ == '__main__':
    main()
