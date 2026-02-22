# Why These Features Are Useful for Hallucination Detection

## Overview

The 14 features extracted from NLI scores capture different aspects of how a sentence relates to multiple sampled passages. Hallucinated sentences tend to show different patterns than factual ones when compared against multiple samples.

---

## Feature Categories and Their Utility

### 1. Entailment Features (Features 0-3)

**What they capture:** How well the sentence is supported by the sampled passages

| Feature | Why It's Useful |
|---------|-----------------|
| **mean(entailment)** | **Factual sentences** should have high average entailment across samples (they're consistently supported). **Hallucinated sentences** have lower entailment (not well supported). |
| **max(entailment)** | Even if average is low, one sample might strongly support it. High max suggests the sentence might be factual but only in specific contexts. |
| **std(entailment)** | **High std** = inconsistent support (some samples support, others don't) → potential hallucination. **Low std** = consistent support → likely factual. |
| **prop(entailment > 0.5)** | Proportion of samples that strongly support the sentence. **High proportion** = well-supported (factual). **Low proportion** = poorly supported (hallucinated). |

**Intuition:** Factual sentences should be consistently entailed across multiple samples. Hallucinated sentences are less likely to be entailed.

---

### 2. Contradiction Features (Features 4-7)

**What they capture:** How much the sentence contradicts the sampled passages

| Feature | Why It's Useful |
|---------|-----------------|
| **mean(contradiction)** | **High mean** = sentence contradicts many samples → likely hallucinated. **Low mean** = few contradictions → likely factual. |
| **max(contradiction)** | Captures the strongest contradiction signal. Even if average is low, one strong contradiction can indicate hallucination. |
| **std(contradiction)** | **High std** = inconsistent (some samples contradict strongly, others don't) → uncertainty/hallucination. **Low std** = consistent → clearer signal. |
| **prop(contradiction > 0.5)** | Proportion of samples that strongly contradict. **High proportion** = many contradictions → likely hallucinated. |

**Intuition:** Hallucinated sentences are more likely to contradict what the model generates in other samples, since they contain false information.

---

### 3. Entropy Feature (Feature 8)

**What it captures:** Uncertainty in the E/C distribution

**Why it's useful:**
- **High entropy** (≈0.693) = model is uncertain, E and C are balanced → **ambiguous/hallucinated**
- **Low entropy** (≈0) = model is certain, strongly favors E or C → **clearer signal**

**Intuition:** When the model is uncertain (high entropy), it suggests the sentence might be hallucinated because the model can't confidently determine if it's entailed or contradicted.

**Example:**
- Entropy ≈ 0.69: P(E)=0.5, P(C)=0.5 → very uncertain → likely hallucination
- Entropy ≈ 0.1: P(E)=0.95, P(C)=0.05 → certain entailment → likely factual

---

### 4. Margin Features (Features 9-10)

**What they capture:** The relationship between entailment and contradiction

| Feature | Why It's Useful |
|---------|-----------------|
| **mean(E - C)** | **Positive** = more entailment than contradiction → likely factual. **Negative** = more contradiction → likely hallucinated. **Near zero** = ambiguous. |
| **max(C - E)** | Captures the strongest contradiction signal. Even if average margin is positive, one strong contradiction can indicate hallucination. |

**Intuition:** 
- **Factual sentences:** E >> C (strongly entailed, rarely contradicted)
- **Hallucinated sentences:** C >> E (strongly contradicted, rarely entailed)
- **Ambiguous sentences:** E ≈ C (uncertain)

**Example:**
- mean(E-C) = +0.6 → Strong entailment, likely factual
- mean(E-C) = -0.6 → Strong contradiction, likely hallucinated
- max(C-E) = 0.8 → At least one sample strongly contradicts → potential hallucination

---

### 5. Disagreement Features (Features 11-12)

**What they capture:** How much the samples disagree about contradiction

| Feature | Why It's Useful |
|---------|-----------------|
| **std(contradiction)** | **High std** = samples disagree (some say contradict, others don't) → **uncertainty/hallucination**. **Low std** = samples agree → clearer signal. |
| **IQR(contradiction)** | Robust measure of spread. **High IQR** = wide disagreement → potential hallucination. **Low IQR** = consistent → clearer signal. |

**Intuition:** 
- **Factual sentences:** Samples should agree (low disagreement) - consistently low contradiction
- **Hallucinated sentences:** Samples may disagree (high disagreement) - some samples contradict, others don't, creating uncertainty

**Why disagreement matters:**
- If samples consistently agree (low std/IQR), we have a clear signal
- If samples disagree (high std/IQR), it suggests uncertainty, which is common with hallucinations

---

### 6. Top-k Feature (Feature 13)

**What it captures:** Mean of top 3 highest contradiction probabilities

**Why it's useful:**
- Captures **strong contradiction signals** even if most samples are neutral/entailed
- Useful when only a few samples strongly contradict (outlier detection)
- **High top3_contradiction** = at least 3 samples strongly contradict → likely hallucinated

**Intuition:** 
- Sometimes most samples are neutral, but a few strongly contradict
- This feature captures those strong signals that might be averaged out in mean(contradiction)
- **Example:** If 17 samples have contradiction=0.1, but 3 have contradiction=0.9, mean=0.2 (low), but top3_mean=0.9 (high) → strong signal

---

## How Features Work Together

### Pattern 1: Clear Factual Sentence
- **High entailment features** (mean, max, prop)
- **Low contradiction features** (mean, max, prop)
- **Low entropy** (certain)
- **Positive margin** (E >> C)
- **Low disagreement** (samples agree)
- **Low top3_contradiction**

### Pattern 2: Clear Hallucination
- **Low entailment features**
- **High contradiction features**
- **Low entropy** (certain, but wrong direction)
- **Negative margin** (C >> E)
- **Low disagreement** (samples agree it's contradicted)
- **High top3_contradiction**

### Pattern 3: Ambiguous/Uncertain (Often Hallucinated)
- **Medium entailment/contradiction**
- **High entropy** (uncertain)
- **Near-zero margin** (E ≈ C)
- **High disagreement** (samples disagree)
- **Variable top3_contradiction**

---

## Why Multiple Features Matter

1. **Redundancy:** Multiple features capture similar signals, making the model robust
2. **Complementarity:** Different features capture different aspects (central tendency vs. spread vs. outliers)
3. **Non-linearity:** Complex interactions between features help detect subtle patterns
4. **Robustness:** If one feature is noisy, others can compensate

---

## Real-World Example

**Sentence:** "Barack Obama was born in Kenya."

**Factual sentence pattern:**
- High entailment (0.8), low contradiction (0.1)
- Low entropy (0.2)
- Positive margin (+0.7)
- Low disagreement (std=0.05)

**Hallucinated sentence pattern:**
- Low entailment (0.2), high contradiction (0.7)
- Low entropy (0.3) but wrong direction
- Negative margin (-0.5)
- Low disagreement (std=0.1) - samples agree it's contradicted

**Ambiguous sentence pattern:**
- Medium entailment (0.5), medium contradiction (0.5)
- High entropy (0.69) - very uncertain
- Near-zero margin (0.0)
- High disagreement (std=0.3) - samples disagree

---

## Summary: Why These Features Work

1. **Entailment features** → Measure support (factual = well-supported)
2. **Contradiction features** → Measure conflict (hallucinated = contradicted)
3. **Entropy** → Measures uncertainty (hallucinated = uncertain)
4. **Margin** → Measures E-C relationship (factual = E>>C, hallucinated = C>>E)
5. **Disagreement** → Measures consistency (factual = consistent, hallucinated = inconsistent)
6. **Top-k** → Captures outlier contradictions (hallucinated = strong contradictions exist)

Together, these features create a rich representation that captures multiple dimensions of how a sentence relates to multiple samples, enabling a supervised model to learn complex patterns that distinguish factual from hallucinated content.
