"""
This is training script for the confidence -weighted NLI pipeline
uses the 9 feature SelfCheckNLI
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
import json
import hashlib
from pathlib import Path
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
import torch

def load_wikibio_data():
    #load WikiBio GPT3 hallucination dataset, return all_sentences, all_labels, all_sampled_passages,doc_indices
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
    dataset = dataset['evaluation']
    all_sentences= []
    all_labels= []
    all_sampled_passages= []
    doc_indices= []
    
    label_mapping = {
        'accurate': 0,  #factual
        'minor_inaccurate': 1,#nonfactual (hallucinated)
        'major_inaccurate': 1    #nonfactual (hallucinated)
    }
    for doc_idx, example in enumerate(dataset):
        sentences = example['gpt3_sentences']
        annotations = example['annotation']
        sampled_passages = example['gpt3_text_samples']  
        
        # flatten, each sentence gets own entry
        for sent_idx, (sentence, annotation) in enumerate(zip(sentences, annotations)):
            all_sentences.append(sentence)
            all_labels.append(label_mapping[annotation])
            all_sampled_passages.append(sampled_passages)#same passages for all sentences in doc
            doc_indices.append(doc_idx)
    all_labels = np.array(all_labels)
    
    print(f"\nDataset stats:")
    print(f"  total docs: {len(dataset)}")
    print(f"  total sentences: {len(all_sentences)}")
    print(f" label distribution:")
    print(f"  Factual (0): {np.sum(all_labels == 0)} ({np.mean(all_labels == 0)*100:.1f}%)")
    print(f" Non-factual/Hallucinated (1): {np.sum(all_labels == 1)} ({np.mean(all_labels == 1)*100:.1f}%)")
    print(f" Sampled passages per sentence: {len(all_sampled_passages[0])}")
    print(f"\nlabel mapping:")
    print(f"  'accurate' -> 0 (factual)")
    print(f" 'minor_inaccurate' -> 1 (non-factual)")
    print(f" 'major_inaccurate' -> 1 (non-factual)")

    return all_sentences, all_labels, all_sampled_passages, doc_indices

def compute_cache_key(sentences, sampled_passages, nli_model_name, batch_size):
    #compute hash key for caching features
    # create a string representation of the data
    data_str = json.dumps({
        'sentences': sentences[:10], 
        'num_sentences': len(sentences),
        'num_passages_per_sent': len(sampled_passages[0]) if sampled_passages else 0,
        'nli_model': nli_model_name,
        'batch_size': batch_size
    }, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()


def extract_features(sentences, sampled_passages, doc_indices, nli_model, device, batch_size=32, cache_dir=None, cache_key=None):
    """
    extract contradiction-derived features for all sentences
    returns features, which is numpy array of shape (num_sentences, num_features)
    """
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
                print(f"\n loading cached features from {cache_file}...")
                data = np.load(cache_file)
                features = data['features']
                with open(cache_meta_file, 'r') as f:
                    meta = json.load(f)
                if meta['num_sentences'] == len(sentences):
                    print(f"loaded {len(features)} cached feature vectors")
                    return features
                else:
                    print(f"cache mismatch (expected {meta['num_sentences']} sentences, got {len(sentences)}).")
            except Exception as e:
                print(f"error loading cache: {e}.")
    
    # Extract features (cache miss or cache disabled)
    print("\nextracting features")
    
    #group sentences by doc (sentences in same doc share sampled passages)
    from collections import defaultdict
    doc_to_sentences= defaultdict(list)
    doc_to_passages= {}
    doc_to_indices= defaultdict(list)
    
    for idx, (sent, passages, doc_idx) in enumerate(zip(sentences, sampled_passages, doc_indices)):
        doc_to_sentences[doc_idx].append(sent)
        doc_to_passages[doc_idx] = passages  # Same for all sentences in doc
        doc_to_indices[doc_idx].append(idx)
    
    #process each doc's sentences together(they share sampled passages)
    all_features = [None]*len(sentences)

    num_docs = len(doc_to_sentences)
    print(f"processing {num_docs} docs")
    
    #use parallel processing for CPU
    use_gpu = str(nli_model.device) != 'cpu'
    
    if num_docs>1 and not use_gpu:
        try:
            from joblib import Parallel, delayed
            from tqdm import tqdm
            def process_doc(doc_idx):
                doc_sentences = doc_to_sentences[doc_idx]
                doc_passages = doc_to_passages[doc_idx]
                doc_sentence_indices = doc_to_indices[doc_idx]
                doc_features = nli_model.extract_features(doc_sentences, doc_passages)
                return doc_sentence_indices, doc_features
            n_workers = min(4, num_docs)
            results = Parallel(n_jobs=n_workers)(
                delayed(process_doc)(doc_idx)
                for doc_idx in tqdm(doc_to_sentences.keys(), desc="Extracting features")
            )
            
            #store features in original order
            for doc_sentence_indices, doc_features in results:
                for feat_idx, orig_idx in enumerate(doc_sentence_indices):
                    all_features[orig_idx] = doc_features[feat_idx]
        except ImportError:
            print("joblib not available")
            from tqdm import tqdm
            for doc_idx in tqdm(doc_to_sentences.keys(), desc="Extracting features"):
                doc_sentences = doc_to_sentences[doc_idx]
                doc_passages = doc_to_passages[doc_idx]
                doc_sentence_indices = doc_to_indices[doc_idx]
                doc_features = nli_model.extract_features(doc_sentences, doc_passages)
                
                for feat_idx, orig_idx in enumerate(doc_sentence_indices):
                    all_features[orig_idx] = doc_features[feat_idx]
    else:
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
    
    #save to cache if enabled
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        if cache_key is None:
            nli_model_name = getattr(nli_model, 'model_name', 'unknown')
            cache_key = compute_cache_key(sentences, sampled_passages, nli_model_name, batch_size)
        
        cache_file = cache_dir / f"features_{cache_key}.npz"
        cache_meta_file = cache_dir / f"features_{cache_key}.json"
        
        try:
            print(f"saving features to cache: {cache_file}")
            np.savez_compressed(cache_file, features=features)
            with open(cache_meta_file, 'w') as f:
                json.dump({
                    'num_sentences': len(sentences),
                    'num_features': features.shape[1],
                    'nli_model': getattr(nli_model, 'model_name', 'unknown'),
                    'batch_size': batch_size
                }, f, indent=2)
            print(f"features cached successfully")
        except Exception as e:
            print(f"warning: could not save cache: {e}")
    
    return features


def train_model(X_train, y_train, X_test, y_test, C=1.0, max_iter=1000):
    """
    train log regression model and eval on test set
    returns model trained LogisticRegression model and dict of test metrics
    """
    print("\ntraining logistic regression...")
    print(f"training set: {X_train.shape[0]} samples")
    print(f"test set: {X_test.shape[0]} samples")
    #train model
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=42,
        solver='lbfgs',  # good for small datasets
        class_weight='balanced'  #handle class imbalance
    )
    model.fit(X_train, y_train)
    #eval on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0,
        'pr_auc': pr_auc,
    }
    print("test set metrics:")
    print(f" accuracy:  {metrics['accuracy']:.4f}")
    print(f" precision: {metrics['precision']:.4f}")
    print(f" recall: {metrics['recall']:.4f}")
    print(f"F1 Score:{metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"AUC-PR:  {metrics['pr_auc']:.4f}")
    print("\nclassification report:")
    print(classification_report(y_test, y_pred, target_names=['Factual', 'Hallucinated']))
    print("\nconfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    return model, metrics

def save_model(model, output_path, feature_names=None):
    #save trained model and metadata.
    if feature_names is None:
        try:
            n_f = model.coef_.shape[1]
            feature_names = [f"feature_{i}" for i in range(n_f)]
        except Exception:
            feature_names = []
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'num_features': len(feature_names),
    }
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train confidence-weighted NLI pipeline on WikiBio')
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
    parser.add_argument('--no_split_by_doc', action='store_true',
                        help='Use sentence-level split (default: split by document to avoid leakage)')
    parser.add_argument('--split_by_doc', '--split-by-doc', action='store_true', dest='_doc_split_alias',
                        help='Document-level split (default). Optional.')
    parser.add_argument('--cv_folds', type=int, default=0,
                        help='K-fold CV at document level (e.g. 5). 0 = single train/test split.')
    parser.add_argument('--tune_C', action='store_true',
                        help='Grid-search C over [0.01, 0.1, 1.0, 10.0] using CV; use with --cv_folds >= 2')
    parser.add_argument('--cache_dir', type=str, default='.feature_cache',
                        help='Directory to cache extracted features. Set to empty string to disable.')
    
    args = parser.parse_args()
    args.split_by_doc = not args.no_split_by_doc
    
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        requested_device = args.device.lower().strip()
        if requested_device == "cuda" or requested_device.startswith("cuda"):
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(requested_device)
    
    if device.type == "cuda":
        print(f"  CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    
    #load data
    sentences, labels, sampled_passages, doc_indices = load_wikibio_data()
    doc_indices = np.array(doc_indices)

    use_cv = args.cv_folds >= 2
    if args.tune_C and not use_cv:
        args.cv_folds = 5
        use_cv = True
        print(f"  C tuning requested: using {args.cv_folds} fold CV for selection")

    print("\ninitializing SelfCheckNLI (confidence-weighted, 9 features)")
    effective_batch_size = args.batch_size
    if device.type == 'cuda':
        effective_batch_size = max(args.batch_size, 64)
        print(f"GPU detected: using batch_size={effective_batch_size}")
    nli_model = SelfCheckNLI(
        nli_model=args.nli_model,
        device=device,
        batch_size=effective_batch_size
    )
    cache_dir = args.cache_dir if args.cache_dir else None

    if use_cv:
        print("\nextracting features for full dataset (for CV)")
        X_all = extract_features(
            sentences, sampled_passages, doc_indices, nli_model, device, args.batch_size,
            cache_dir=cache_dir, cache_key='full'
        )
        y_all = np.array(labels)
        unique_docs = np.unique(doc_indices)
        doc_labels = np.array([
            int(np.bincount(labels[doc_indices == d]).argmax()) for d in unique_docs
        ])
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_seed)
        doc_to_sent_idx = {}
        for i, d in enumerate(doc_indices):
            doc_to_sent_idx.setdefault(d, []).append(i)
        fold_doc_splits = list(kf.split(unique_docs, doc_labels))

        C_candidates = [0.01, 0.1, 1.0, 10.0] if args.tune_C else [args.C]
        best_C = args.C
        best_mean_pr = -1.0
        cv_results = []

        for C in C_candidates:
            fold_pr_aucs = []
            for fold_idx, (train_doc_idx, test_doc_idx) in enumerate(fold_doc_splits):
                train_docs = set(unique_docs[train_doc_idx])
                test_docs = set(unique_docs[test_doc_idx])
                train_sent_idx = [i for i in range(len(sentences)) if doc_indices[i] in train_docs]
                test_sent_idx = [i for i in range(len(sentences)) if doc_indices[i] in test_docs]
                X_tr, y_tr = X_all[train_sent_idx], y_all[train_sent_idx]
                X_te, y_te = X_all[test_sent_idx], y_all[test_sent_idx]
                model_fold = LogisticRegression(
                    C=C, max_iter=args.max_iter, random_state=42, solver='lbfgs', class_weight='balanced'
                )
                model_fold.fit(X_tr, y_tr)
                proba = model_fold.predict_proba(X_te)[:, 1]
                pr = average_precision_score(y_te, proba) if len(np.unique(y_te)) > 1 else 0.0
                fold_pr_aucs.append(pr)
            mean_pr = np.mean(fold_pr_aucs)
            std_pr = np.std(fold_pr_aucs)
            cv_results.append((C, mean_pr, std_pr, fold_pr_aucs))
            if mean_pr > best_mean_pr:
                best_mean_pr = mean_pr
                best_C = C
            print(f"  C={C:.2f}  AUC-PR (mean +- std): {mean_pr:.4f} +- {std_pr:.4f}")

 
        print("cross-val results")
        for C, mean_pr, std_pr, _ in cv_results:
            print(f"  C={C:.2f}: AUC-PR = {mean_pr:.4f} ± {std_pr:.4f}")
        if args.tune_C:
            print(f"best C (by AUC-PR): {best_C}")


        print(f"\ntraining final model on full dataset with C={best_C}...")
        model = LogisticRegression(
            C=best_C, max_iter=args.max_iter, random_state=42, solver='lbfgs', class_weight='balanced'
        )
        model.fit(X_all, y_all)
        output_path = Path(args.output)
        feature_names = getattr(nli_model, 'feature_names', None)
        save_model(model, output_path, feature_names=feature_names)
        print("\ntraining complete! (final model trained on all data with best C.)")
        return

    #single train/test split path below
    if args.split_by_doc:
        from sklearn.model_selection import train_test_split
        unique_docs = np.unique(doc_indices)
        doc_labels = []
        for doc_idx in unique_docs:
            doc_mask = doc_indices == doc_idx
            doc_label_dist = labels[doc_mask]
            majority_label = int(np.bincount(doc_label_dist).argmax())
            doc_labels.append(majority_label)
        doc_labels = np.array(doc_labels)
        train_docs, test_docs = train_test_split(
            unique_docs,
            test_size=args.test_size,
            random_state=args.random_seed,
            stratify=doc_labels
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
        print(f" train documents: {len(train_docs)}")
        print(f"test documents: {len(test_docs)}")
        print(f"train sentences: {len(train_sentences)}")
        print(f"test sentences: {len(test_sentences)}")
    else:
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(sentences))
        train_indices, test_indices = train_test_split(
            indices,
            test_size=args.test_size,
            random_state=args.random_seed,
            stratify=labels
        )
        train_sentences = [sentences[i] for i in train_indices]
        test_sentences = [sentences[i] for i in test_indices]
        y_train = labels[train_indices]
        y_test = labels[test_indices]
        train_sampled = [sampled_passages[i] for i in train_indices]
        test_sampled = [sampled_passages[i] for i in test_indices]
        train_doc_indices = list(doc_indices[train_indices])
        test_doc_indices = list(doc_indices[test_indices])
        print(f"\nrandom split (stratified, sentence-level):")
        print(f"  train sentences: {len(train_sentences)}")
        print(f"  test sentences: {len(test_sentences)}")
    
    print(f"\nClass Distribution:")
    print(f" Train- Factual (0): {np.sum(y_train == 0)} ({np.mean(y_train == 0)*100:.1f}%)")
    print(f" Train- Non-factual (1): {np.sum(y_train == 1)} ({np.mean(y_train == 1)*100:.1f}%)")
    print(f"Test- Factual (0): {np.sum(y_test == 0)} ({np.mean(y_test == 0)*100:.1f}%)")
    print(f"Test- Non-factual (1): {np.sum(y_test == 1)} ({np.mean(y_test == 1)*100:.1f}%)")
    

    print("training set")
    X_train = extract_features(
        train_sentences, train_sampled, train_doc_indices, nli_model, device, args.batch_size,
        cache_dir=cache_dir, cache_key='train')
    
    print("test set")
    X_test = extract_features(
        test_sentences, test_sampled, test_doc_indices, nli_model, device, args.batch_size,
        cache_dir=cache_dir, cache_key='test')
    
    model, metrics = train_model(
        X_train, y_train, X_test, y_test,
        C=args.C, max_iter=args.max_iter)
    
    output_path = Path(args.output)
    feature_names = getattr(nli_model, 'feature_names', None)
    save_model(model, output_path, feature_names=feature_names)
    
    print("training complete yay")
    print(f"model saved to: {output_path}")
 

if __name__ == '__main__':
    main()
