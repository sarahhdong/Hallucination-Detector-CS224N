"""
confidence weighted NLI aggregation for hallucination detection
"""

import numpy as np
import torch
from typing import List
from transformers import logging
logging.set_verbosity_error()
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification

DEFAULT_NLI_MODEL = "sotsawee/deberta-v3-large-mnli"

class SelfCheckNLI:
    """
same NLI backbone; extract_features returns 9D: baseline mean(C), max/std/frac/entropy/iqr/top3(C), and confidence-weighted mean(C), mean(E)
    """
    FEATURE_NAMES = ["baseline_selfcheck_nli", "max(C)", "std(C)", "frac(C>0.5)","entropy(C)", "iqr(C)", "top3_mean(C)", "conf_weighted_mean(C)", "conf_weighted_mean(E)", ]

    def __init__(self, nli_model: str = None, device=None, batch_size: int = 32):
        nli_model = nli_model if nli_model is not None else DEFAULT_NLI_MODEL
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(nli_model, trust_remote_code=True)
        except Exception as e:
            error_msg = str(e)
            if "sentencepiece" in error_msg.lower() or "protobuf" in error_msg.lower() or "spm.model" in error_msg.lower():
                print(f"\nerror loading tokenizer: {error_msg}")
            raise
        self.model = DebertaV2ForSequenceClassification.from_pretrained(nli_model)
        self.model.eval()

        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device_str = device.lower().strip()
            if device_str == "cuda" or device_str.startswith("cuda"):
                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            else:
                device = torch.device(device_str)
        elif isinstance(device, torch.device) and device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

        self.model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.id2label = self.model.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.entailment_idx =None
        self.neutral_idx =None
        self.contradiction_idx =None

        for idx, label in self.id2label.items():
            label_lower = label.lower()
            if 'entail' in label_lower:
                self.entailment_idx = idx
            elif 'neutral' in label_lower:
                self.neutral_idx = idx
            elif 'contradict' in label_lower:
                self.contradiction_idx = idx

        if self.entailment_idx is None or self.contradiction_idx is None:
            num_labels = len(self.id2label)
            labels_list = list(self.id2label.values())
            all_generic = all(label.upper().startswith('LABEL_') for label in labels_list)
            if all_generic and num_labels >= 2:
                if num_labels == 2:
                    self.entailment_idx = 0
                    self.contradiction_idx = 1
                    self.neutral_idx = None
                elif num_labels == 3:
                    self.entailment_idx = 0
                    self.neutral_idx = 1
                    self.contradiction_idx = 2
                else:
                    sorted_indices = sorted(self.id2label.keys())
                    self.entailment_idx = sorted_indices[0]
                    self.contradiction_idx = sorted_indices[-1]
                    self.neutral_idx = sorted_indices[1] if len(sorted_indices) == 3 else None
            else:
                raise ValueError(
                    f"could not find entailment/contradiction in model. labels: {labels_list}"
                )
        self.model_name = nli_model
        self.feature_names = list(self.FEATURE_NAMES)

    @torch.no_grad()
    def predict(self, sentences: List[str], sampled_passages: List[str]):
        #sentence level score=mean contradiction (E/C renormalized)
        _, _, contra_scores = self.predict_matrix(sentences, sampled_passages)
        return contra_scores.mean(axis=-1)

    @torch.no_grad()
    def predict_matrix(self, sentences: List[str], sampled_passages: List[str]):
        #return (entailment_scores,None,contradiction_scores),shapes (num_sentences,num_samples)
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        if num_sentences == 0 or num_samples == 0:
            return (
                np.zeros((num_sentences, num_samples), dtype=np.float32),
                None,
                np.zeros((num_sentences, num_samples), dtype=np.float32),
            )
        pairs = []
        pair_indices = []
        for sent_i, sentence in enumerate(sentences):
            for sample_i, sample in enumerate(sampled_passages):
                pairs.append((sentence, sample))
                pair_indices.append((sent_i, sample_i))

        all_logits = []
        total_pairs = len(pairs)
        for batch_start in range(0, total_pairs, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_pairs)
            batch_pairs = pairs[batch_start:batch_end]
            inputs = self.tokenizer(
                [s for s, _ in batch_pairs],
                [p for _, p in batch_pairs],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits
            all_logits.append(logits.cpu())

        all_logits = torch.cat(all_logits, dim=0).cpu()
        sent_indices = np.array([s for s, _ in pair_indices])
        sample_indices = np.array([p for _, p in pair_indices])

        entailment_scores = np.zeros((num_sentences, num_samples), dtype=np.float32)
        contradiction_scores = np.zeros((num_sentences, num_samples), dtype=np.float32)
        ec_logits = all_logits[:, [self.entailment_idx, self.contradiction_idx]]
        ec_probs = torch.softmax(ec_logits, dim=1)
        entailment_scores[sent_indices, sample_indices] = ec_probs[:, 0].cpu().numpy()
        contradiction_scores[sent_indices, sample_indices] = ec_probs[:, 1].cpu().numpy()
        return entailment_scores, None, contradiction_scores

    def extract_features(self, sentences: List[str], sampled_passages: List[str]):
        # 9D feature vector per sentence:baseline mean(C), max/std/frac/entropy/iqr/top3(C),conf_weighted_mean(C), conf_weighted_mean(E)
        entailment_scores, _,contradiction_scores = self.predict_matrix(sentences, sampled_passages)
        features = []
        for sent_i in range(len(sentences)):
            e= entailment_scores[sent_i, :]
            c= contradiction_scores[sent_i, :]
            contras = np.asarray(c, dtype=np.float64)
            entails = np.asarray(e, dtype=np.float64)
            baseline_score = float(np.mean(contras)) if contras.size>0 else 0.0
            contra_feat = [
                baseline_score,
                np.max(contras) if contras.size > 0 else 0.0,
                np.std(contras) if contras.size > 0 else 0.0,
                np.mean(contras > 0.5) if contras.size > 0 else 0.0,
            ]
            if contras.size>0:
                c_probs = np.clip(contras, 1e-10, 1.0 - 1e-10)
                not_c = 1.0 - c_probs
                entropies = -np.sum(
                    np.stack([c_probs, not_c], axis=0) * (np.log(np.stack([c_probs, not_c], axis=0) + 1e-10)),
                    axis=0,
                )
                entropy_feat = [np.mean(entropies)]
            else:
                entropy_feat = [0.0]

            iqr_c = np.percentile(contras, 75) - np.percentile(contras, 25) if contras.size > 0 else 0.0
            iqr_feat = [float(iqr_c)]

            if contras.size > 0:
                top_k = min(3, contras.size)
                topk_feat = [float(np.mean(np.sort(contras)[::-1][:top_k]))]
            else:
                topk_feat = [0.0]

            if contras.size>0:
                conf = np.maximum(entails, contras)
                conf = np.clip(conf, 1e-10, 1.0)
                w_sum = np.sum(conf)
                if w_sum > 0:
                    conf_weighted_mean_c = float(np.sum(conf * contras) / w_sum)
                    conf_weighted_mean_e = float(np.sum(conf * entails) / w_sum)
                else:
                    conf_weighted_mean_c = float(np.mean(contras))
                    conf_weighted_mean_e = float(np.mean(entails))
            else:
                conf_weighted_mean_c = 0.0
                conf_weighted_mean_e = 0.0
            conf_feat = [conf_weighted_mean_c, conf_weighted_mean_e]
            feat = contra_feat + entropy_feat + iqr_feat + topk_feat + conf_feat
            features.append(feat)

        return np.array(features, dtype=np.float32)
