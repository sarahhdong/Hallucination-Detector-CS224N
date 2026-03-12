"""
Bidirectional NLI + semantic entropy + token uncertainty.
Major Attemppt #1 
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from selfcheckgpt.modeling_selfcheck_extended import SelfCheckNLI


FEATURE_NAMES = [
    "fwd_mean_c",
    "fwd_max_c",
    "fwd_std_c",
    "fwd_frac_c_gt_05",
    "rev_mean_c",
    "rev_max_c",
    "rev_std_c",
    "bidir_mean_abs_diff_c",
    "bidir_gap_mean_c",
    "sem_entropy_clustered",
    "fwd_entropy_ec",
    "rev_entropy_ec",
    "token_avg_nll",
]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_wikibio_data() -> Tuple[List[str], np.ndarray, List[List[str]], np.ndarray]:
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")["evaluation"]
    label_mapping = {"accurate": 0, "minor_inaccurate": 1, "major_inaccurate": 1}

    sentences: List[str] = []
    labels: List[int] = []
    passages: List[List[str]] = []
    doc_indices: List[int] = []

    for doc_idx, example in enumerate(dataset):
        for sentence, annotation in zip(example["gpt3_sentences"], example["annotation"]):
            sentences.append(sentence)
            labels.append(label_mapping[annotation])
            passages.append(example["gpt3_text_samples"])
            doc_indices.append(doc_idx)

    return sentences, np.array(labels), passages, np.array(doc_indices)


def split_data(
    sentences: List[str],
    labels: np.ndarray,
    passages: List[List[str]],
    doc_indices: np.ndarray,
    test_size: float,
    random_seed: int,
    split_by_doc: bool,
):
    from sklearn.model_selection import train_test_split

    if split_by_doc:
        unique_docs = np.unique(doc_indices)
        doc_labels = []
        for doc_idx in unique_docs:
            mask = doc_indices == doc_idx
            doc_labels.append(int(np.bincount(labels[mask]).argmax()))
        doc_labels = np.array(doc_labels)

        train_docs, test_docs = train_test_split(
            unique_docs,
            test_size=test_size,
            random_state=random_seed,
            stratify=doc_labels,
        )
        train_mask = np.isin(doc_indices, train_docs)
        test_mask = ~train_mask
    else:
        indices = np.arange(len(sentences))
        from sklearn.model_selection import train_test_split

        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_seed,
            stratify=labels,
        )
        train_mask = np.zeros(len(sentences), dtype=bool)
        train_mask[train_idx] = True
        test_mask = ~train_mask

    train_sentences = [s for s, m in zip(sentences, train_mask) if m]
    test_sentences = [s for s, m in zip(sentences, test_mask) if m]
    y_train = labels[train_mask]
    y_test = labels[test_mask]
    train_sampled = [p for p, m in zip(passages, train_mask) if m]
    test_sampled = [p for p, m in zip(passages, test_mask) if m]
    train_doc_indices = [int(d) for d, m in zip(doc_indices, train_mask) if m]
    test_doc_indices = [int(d) for d, m in zip(doc_indices, test_mask) if m]

    return ( train_sentences, test_sentences, y_train, y_test, train_sampled, test_sampled, train_doc_indices, test_doc_indices)


class TokenUncertaintyScorer:
    def __init__(self, model_name: str, device: torch.device, max_length: int = 256):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._cache: Dict[str, float] = {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def score(self, text: str) -> float:
        if text in self._cache:
            return self._cache[text]

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        out = self.model(**enc, labels=enc["input_ids"])
        score = float(out.loss.item())
        self._cache[text] = score
        return score


def _entropy_binary(p: np.ndarray) -> float:
    p = np.clip(p, 1e-10, 1.0 - 1e-10)
    q = 1.0 - p
    ent = -(p * np.log(p) + q * np.log(q))
    return float(np.mean(ent))


def _semantic_entropy_from_clusters(points: np.ndarray, n_clusters: int, random_seed: int) -> float:
    if len(points) <= 1:
        return 0.0

    k = max(2, min(n_clusters, len(points)))
    if np.allclose(points, points[0]):
        return 0.0

    km = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
    labels = km.fit_predict(points)
    counts = np.bincount(labels, minlength=k).astype(float)
    probs = counts / counts.sum()
    probs = np.clip(probs, 1e-10, 1.0)
    ent = -np.sum(probs * np.log(probs))
    return float(ent / np.log(k))


def extract_hybrid_features(
    sentences: List[str],
    passages_list: List[List[str]],
    nli_model: SelfCheckNLI,
    uncertainty_scorer: Optional[TokenUncertaintyScorer],
    n_entropy_clusters: int,
    random_seed: int,
    disable_progress: bool = False,
) -> np.ndarray:
    features = []
    iterator = zip(sentences, passages_list)
    if not disable_progress:
        iterator = tqdm(iterator, total=len(sentences), desc="Extracting hybrid features")

    for sentence, passages in iterator:
        # forward 
        f_ent, _, f_con = nli_model.predict_matrix([sentence], passages)
        f_ent = f_ent[0]
        f_con = f_con[0]

        # reverse
        r_ent, _, r_con = nli_model.predict_matrix(passages, [sentence])
        r_ent = r_ent[:, 0]
        r_con = r_con[:, 0]

        # semantic entropy 
        points = np.stack([f_ent, f_con], axis=1)
        sem_entropy = _semantic_entropy_from_clusters(points, n_entropy_clusters, random_seed)

        token_nll = 0.0
        if uncertainty_scorer is not None:
            token_nll = uncertainty_scorer.score(sentence)

        feature = [
            float(np.mean(f_con)), float(np.max(f_con)), float(np.std(f_con)), float(np.mean(f_con > 0.5)), float(np.mean(r_con)),
            float(np.max(r_con)), float(np.std(r_con)), float(np.mean(np.abs(f_con - r_con))), float(abs(np.mean(f_con) - np.mean(r_con))),
            sem_entropy, _entropy_binary(f_con),
            _entropy_binary(r_con),
            float(token_nll),
        ]
        features.append(feature)

    return np.array(features, dtype=np.float32)


def compute_cache_key(name: str, params: Dict) -> str:
    key = json.dumps({"name": name, **params}, sort_keys=True)
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def try_load_cached_features(cache_dir: Path, key: str, expected_dim: int) -> Optional[np.ndarray]:
    f_path = cache_dir / f"hybrid_features_{key}.npz"
    if not f_path.exists():
        return None
    data = np.load(f_path)
    features = data["features"]
    if features.shape[0] != expected_dim:
        return None
    return features


def save_cached_features(cache_dir: Path, key: str, features: np.ndarray) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / f"hybrid_features_{key}.npz", features=features)


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.0,
        "auc_pr": average_precision_score(y, probs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hybrid hallucination detector")
    parser.add_argument("--output", type=str, default="hybrid_hallucination_detector.pkl")
    parser.add_argument("--nli_model", type=str, default=None, help="NLI model (default: SelfCheckNLI default)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--split_by_doc", action="store_true")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--cache_dir", type=str, default=".hybrid_feature_cache")
    parser.add_argument("--n_entropy_clusters", type=int, default=3)
    parser.add_argument("--uncertainty_model", type=str, default="distilgpt2")
    parser.add_argument("--disable_uncertainty", action="store_true")
    args = parser.parse_args()

    set_seed(args.random_seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    print(f"Using device: {device}")

    sentences, labels, passages, doc_indices = load_wikibio_data()
    (   train_sentences, test_sentences, y_train, y_test, train_sampled, test_sampled,
        _,
        _,
    ) = split_data(sentences,
        labels,
        passages,
        doc_indices,
        test_size=args.test_size,
        random_seed=args.random_seed,
        split_by_doc=args.split_by_doc,
    )

    print(f"Train size: {len(train_sentences)} | Test size: {len(test_sentences)}")

    nli = SelfCheckNLI(nli_model=args.nli_model, device=device, batch_size=args.batch_size)
    uncertainty_scorer = None
    if not args.disable_uncertainty:
        # print(f"Initializing token uncertainty model: {args.uncertainty_model}")
        uncertainty_scorer = TokenUncertaintyScorer(args.uncertainty_model, device=device)
    # else:
        # print("Token uncertainty disabled.")

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    cache_params = {
        "nli_model": args.nli_model or "default",
        "batch_size": args.batch_size,
        "test_size": args.test_size,
        "random_seed": args.random_seed,
        "split_by_doc": args.split_by_doc,
        "n_entropy_clusters": args.n_entropy_clusters,
        "uncertainty_model": None if args.disable_uncertainty else args.uncertainty_model,
        "num_train": len(train_sentences),
        "num_test": len(test_sentences),
    }

    X_train, X_test = None, None
    if cache_dir is not None:
        train_key = compute_cache_key("train", cache_params)
        test_key = compute_cache_key("test", cache_params)
        X_train = try_load_cached_features(cache_dir, train_key, len(train_sentences))
        X_test = try_load_cached_features(cache_dir, test_key, len(test_sentences))

    if X_train is None:
        X_train = extract_hybrid_features( train_sentences,
            train_sampled,
            nli_model=nli,
            uncertainty_scorer=uncertainty_scorer,
            n_entropy_clusters=args.n_entropy_clusters,
            random_seed=args.random_seed,
        )
        if cache_dir is not None:
            save_cached_features(cache_dir, compute_cache_key("train", cache_params), X_train)

    if X_test is None:
        X_test = extract_hybrid_features( test_sentences,
            test_sampled,
            nli_model=nli,
            uncertainty_scorer=uncertainty_scorer,
            n_entropy_clusters=args.n_entropy_clusters,
            random_seed=args.random_seed,
        )
        if cache_dir is not None:
            save_cached_features(cache_dir, compute_cache_key("test", cache_params), X_test)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=args.C,
            max_iter=args.max_iter,
            random_state=args.random_seed,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    print("\nHybrid model metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    out = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "metrics": metrics,
        "config": { "nli_model": args.nli_model,
            "uncertainty_model": None if args.disable_uncertainty else args.uncertainty_model,
            "n_entropy_clusters": args.n_entropy_clusters,
            "split_by_doc": args.split_by_doc,
            "test_size": args.test_size,
            "random_seed": args.random_seed,
        },
    }
    with open(args.output, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved hybrid model to: {args.output}")


if __name__ == "__main__":
    main()

