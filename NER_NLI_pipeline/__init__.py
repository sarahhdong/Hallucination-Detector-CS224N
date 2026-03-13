"""
Setting up init file for NER-NLI Pipeline.

Four-steps:
    1. NER Fact Extraction
    2. NLI Verification
    3. Entity-Weighted Aggregation
    4. Threshold Calibration
"""

from .fact_extraction import AtomicClaim, extract_atomic_claims, load_spacy_model
from .nli_scorer import NLIScorer
from .pipeline import FEATURE_NAMES, evaluate_auc_pr

__all__ = [
    "AtomicClaim",
    "extract_atomic_claims",
    "load_spacy_model",
    "NLIScorer",
    "FEATURE_NAMES",
    "evaluate_auc_pr",
]
