"""
My code below breaks sentences down into atomic claims (entity isolated facts)
using NER and dependency parsing.
"""

import spacy
from dataclasses import dataclass, field
from typing import List, Tuple

# GPE = Geopolitical Entity
# DATE = Date
# LOC = Location
# PERSON = Person
# ORG = Organization
# NORP = Nationality
# FAC = Facility
# GPE = Geopolitical Entity
HIGH_STAKES_ENTITY_TYPES = frozenset({
    "DATE", "GPE", "LOC", "PERSON", "ORG", "NORP", "FAC",
})


@dataclass
class AtomicClaim:
    """A single verifiable factual claim extracted from a sentence."""
    text: str
    entity_types: List[str] = field(default_factory=list)
    entities: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def has_high_stakes_entity(self) -> bool:
        """Whether this claim involves entity."""
        return bool(set(self.entity_types) & HIGH_STAKES_ENTITY_TYPES)


def load_spacy_model(model_name: str = "en_core_web_trf"):
    """
    Load a spaCy model
    """
    try:
        return spacy.load(model_name)
    except OSError:
        raise RuntimeError(
            f"No spaCy English model found. Install one"
        )


# Atomic extraction helper functions

def _get_subject_text(root, doc):
    """Extract textual subject from dependency tree rooted at root"""
    for child in root.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subtree = sorted(child.subtree, key=lambda t: t.i)
            return " ".join(t.text for t in subtree)
    # Fallback
    if root.i > 0:
        return doc[:root.i].text.strip()
    return ""


def _get_verb_phrase(root):
    """Build the verb phrase: auxiliaries + negation + root"""
    aux_deps = {"aux", "auxpass", "neg"}
    parts = [child for child in root.children if child.dep_ in aux_deps]
    parts.append(root)
    parts.sort(key=lambda t: t.i)
    return " ".join(t.text for t in parts)


def extract_atomic_claims(sentence: str, nlp) -> List[AtomicClaim]:
    """
    Decompose sentence into atomic claims using NER + dependency parsing.
    """
    doc = nlp(sentence)
    entities = list(doc.ents)

    # No entities: full sentence only claim
    if not entities:
        return [AtomicClaim(text=sentence)]

    # Find syntactic root
    root = None
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
            break

    if root is None:
        return [AtomicClaim(
            text=sentence,
            entity_types=[e.label_ for e in entities],
            entities=[(e.text, e.label_) for e in entities],
        )]

    subj_text = _get_subject_text(root, doc)
    verb_text = _get_verb_phrase(root)

    # Dependency labels to skip
    skip_deps = {"nsubj", "nsubjpass", "aux", "auxpass", "neg", "punct", "cc"}

    claims = []
    covered_ent_spans = set()

    for child in root.children:
        if child.dep_ in skip_deps:
            continue

        subtree = sorted(child.subtree, key=lambda t: t.i)
        sub_start = subtree[0].i
        sub_end = subtree[-1].i + 1
        sub_text = " ".join(t.text for t in subtree)

        # Find entities in subtree
        sub_ents = [
            ent for ent in entities
            if ent.start >= sub_start and ent.end <= sub_end
        ]

        if sub_ents:
            claim_text = f"{subj_text} {verb_text} {sub_text}".strip()
            claims.append(AtomicClaim(
                text=claim_text,
                entity_types=[e.label_ for e in sub_ents],
                entities=[(e.text, e.label_) for e in sub_ents],
            ))
            for e in sub_ents:
                covered_ent_spans.add((e.start, e.end))

    uncovered = [
        e for e in entities
        if (e.start, e.end) not in covered_ent_spans
    ]
    if uncovered:
        claims.append(AtomicClaim(
            text=sentence,
            entity_types=[e.label_ for e in uncovered],
            entities=[(e.text, e.label_) for e in uncovered],
        ))

    if not claims:
        claims.append(AtomicClaim(
            text=sentence,
            entity_types=[e.label_ for e in entities],
            entities=[(e.text, e.label_) for e in entities],
        ))

    return claims
