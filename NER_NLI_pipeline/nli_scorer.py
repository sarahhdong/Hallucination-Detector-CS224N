"""
Does NLI verification, entity-weighted aggregation, threshold calibration.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from .fact_extraction import AtomicClaim

class NLIScorer:
    """
    Scores atomic claims against a reference text via DeBERTa NLI.
    """

    def __init__(
        self,
        nli_model: str = "microsoft/deberta-v3-base",
        device=None,
        batch_size: int = 32,
    ):
        from transformers import AutoTokenizer, DebertaV2ForSequenceClassification

        self.model_name = nli_model
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model)
        self.model = DebertaV2ForSequenceClassification.from_pretrained(nli_model)
        self.model.eval()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            d = device.lower().strip()
            if d.startswith("cuda") and not torch.cuda.is_available():
                print("CUDA not available — falling back to CPU.")
                device = torch.device("cpu")
            else:
                device = torch.device(d)

        self.model.to(device)
        self.device = device
        self.batch_size = batch_size

        id2label = self.model.config.id2label
        self.entailment_idx: Optional[int] = None
        self.neutral_idx: Optional[int] = None
        self.contradiction_idx: Optional[int] = None

        for idx, label in id2label.items():
            ll = label.lower()
            if "entail" in ll:
                self.entailment_idx = idx
            elif "neutral" in ll:
                self.neutral_idx = idx
            elif "contradict" in ll:
                self.contradiction_idx = idx

        if self.entailment_idx is None or self.contradiction_idx is None:
            raise ValueError(
                f"Cannot find entailment/contradiction in model labels: "
                f"{list(id2label.values())}"
            )

        print(f"NLIScorer initialized on {device}  "
              f"(E={self.entailment_idx}, N={self.neutral_idx}, "
              f"C={self.contradiction_idx})")

        self._cache: Dict[Tuple[str, str], np.ndarray] = {}

    def clear_cache(self):
        """Free NLI cache."""
        self._cache.clear()


    @torch.no_grad()
    def score_pairs(
        self,
        premises: List[str],
        hypotheses: List[str],
    ) -> np.ndarray:
        """
        Score (premise, hypothesis) pairs via softmax.
        """
        n = len(premises)
        results = np.zeros((n, 3), dtype=np.float32)
        uncached: List[int] = []

        for i in range(n):
            cached = self._cache.get((premises[i], hypotheses[i]))
            if cached is not None:
                results[i] = cached
            else:
                uncached.append(i)

        if not uncached:
            return results

        for start in range(0, len(uncached), self.batch_size):
            batch_idx = uncached[start : start + self.batch_size]
            batch_p = [premises[i] for i in batch_idx]
            batch_h = [hypotheses[i] for i in batch_idx]

            inputs = self.tokenizer(
                batch_p, batch_h,
                padding=True, truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for j, idx in enumerate(batch_idx):
                row = np.array([
                    probs[j, self.entailment_idx],
                    probs[j, self.neutral_idx],
                    probs[j, self.contradiction_idx],
                ], dtype=np.float32)
                results[idx] = row
                self._cache[(premises[idx], hypotheses[idx])] = row

        return results


    def score_sentence(
        self,
        reference: str,
        claims: List[AtomicClaim],
        entity_weight: float = 1.2,
        neutral_weight: float = 0.5,
    ) -> dict:
        """
        Score sentence by aggregating NLI results from atomic claims.
        """
        empty = {
            "max_contra_weighted": 0.0,
            "mean_contra_weighted": 0.0,
            "max_calibrated": 0.0,
            "mean_calibrated": 0.0,
            "max_contra_raw": 0.0,
            "max_neutral": 0.0,
            "n_claims": 0,
            "n_entities": 0,
            "has_high_stakes": False,
            "claim_scores": [],
        }
        if not claims:
            return empty

        # Batch-score
        premises = [reference] * len(claims)
        hypotheses = [c.text for c in claims]
        all_probs = self.score_pairs(premises, hypotheses)

        claim_results = []
        for i, claim in enumerate(claims):
            p_e = float(all_probs[i, 0])
            p_n = float(all_probs[i, 1])
            p_c = float(all_probs[i, 2])
            w = entity_weight if claim.has_high_stakes_entity else 1.0

            claim_results.append({
                "p_entail": p_e,
                "p_neutral": p_n,
                "p_contra": p_c,
                "entity_multiplier": w,
                "weighted_contra": p_c * w,
                "calibrated_score": (p_c + neutral_weight * p_n) * w,
                "claim_text": claim.text,
                "entity_types": claim.entity_types,
            })

        w_contras = [r["weighted_contra"] for r in claim_results]
        cal_scores = [r["calibrated_score"] for r in claim_results]
        raw_contras = [r["p_contra"] for r in claim_results]
        neutrals = [r["p_neutral"] for r in claim_results]
        all_ent_types = [t for c in claims for t in c.entity_types]

        return {
            "max_contra_weighted": max(w_contras),
            "mean_contra_weighted": float(np.mean(w_contras)),
            "max_calibrated": max(cal_scores),
            "mean_calibrated": float(np.mean(cal_scores)),
            "max_contra_raw": max(raw_contras),
            "max_neutral": max(neutrals),
            "n_claims": len(claims),
            "n_entities": len(all_ent_types),
            "has_high_stakes": any(c.has_high_stakes_entity for c in claims),
            "claim_scores": claim_results,
        }
