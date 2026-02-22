# Training Guide for Supervised Hallucination Detection

## Overview

This guide explains how to train a logistic regression model on SelfCheckNLI features to detect hallucinations.

## Files Created

1. **`train_supervised_model.py`** - Training script
2. **`use_trained_model.py`** - Inference script
3. **`BATCHING_EXPLANATION.md`** - Explanation of how batching works

---

## How Batching Improves Performance

### Key Concept: Parallel Processing

**Sequential (Original):**
- Processes one (sentence, sample) pair at a time
- 20 sentences × 20 samples = **400 individual forward passes**
- GPU is underutilized (only processes 1 example at a time)

**Batched (Extended):**
- Processes multiple pairs simultaneously
- 20 sentences × 20 samples = 400 pairs in **~13 batches** (batch_size=32)
- GPU processes 32 examples in parallel
- **Result: 10-30x faster**

### Why It's Faster

1. **GPU Parallelization**: All GPU cores work simultaneously
2. **Reduced Overhead**: Fewer kernel launches (13 vs 400)
3. **Optimized Operations**: Batch matrix operations are highly optimized
4. **Better Memory Utilization**: More efficient memory transfers

See `BATCHING_EXPLANATION.md` for detailed explanation.

---

## Training the Model

### Step 1: Prepare Your Data

You need:
- **Sentences**: List of sentences to evaluate
- **Labels**: Binary labels (0=factual, 1=hallucinated)
- **Sampled Passages**: Multiple sampled passages per sentence (optional if you'll generate them)

**Data Format Options:**

**Option A: Text Files**
```
# sentences.txt (one sentence per line)
Barack Obama was born in Hawaii.
The capital of France is Paris.
...

# labels.txt (one label per line, 0 or 1)
0
0
...
```

**Option B: Pickle Files**
```python
import pickle

# Save sentences
sentences = ["Sentence 1", "Sentence 2", ...]
with open('sentences.pkl', 'wb') as f:
    pickle.dump(sentences, f)

# Save labels
labels = [0, 1, 0, ...]  # 0=factual, 1=hallucinated
with open('labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

# Save sampled passages (list of lists, or flat list)
sampled_passages = ["Sample 1", "Sample 2", ...]
with open('sampled_passages.pkl', 'wb') as f:
    pickle.dump(sampled_passages, f)
```

### Step 2: Run Training

**Basic Usage:**
```bash
python train_supervised_model.py \
    --sentences sentences.txt \
    --labels labels.txt \
    --sampled_passages sampled_passages.txt \
    --output hallucination_detector.pkl
```

**With Options:**
```bash
python train_supervised_model.py \
    --sentences sentences.pkl \
    --labels labels.pkl \
    --sampled_passages sampled_passages.pkl \
    --output hallucination_detector.pkl \
    --device cuda \
    --batch_size 32 \
    --test_size 0.2 \
    --C 1.0 \
    --max_iter 1000
```

**Arguments:**
- `--sentences`: Path to sentences file (txt or pkl)
- `--labels`: Path to labels file (txt or pkl)
- `--sampled_passages`: Path to sampled passages (optional)
- `--output`: Output path for trained model (default: `hallucination_detector.pkl`)
- `--nli_model`: NLI model name (default: from NLIConfig)
- `--device`: Device (cuda, cpu, or None for auto)
- `--batch_size`: Batch size for feature extraction (default: 32)
- `--test_size`: Validation split ratio (default: 0.2)
- `--C`: Logistic regression regularization (default: 1.0, higher = less regularization)
- `--max_iter`: Maximum iterations (default: 1000)
- `--random_seed`: Random seed (default: 42)

### Step 3: Check Results

The script will output:
- Validation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Classification report
- Confusion matrix
- Saved model file

---

## Using the Trained Model

### Inference

```bash
python use_trained_model.py \
    --model hallucination_detector.pkl \
    --sentences test_sentences.txt \
    --sampled_passages test_sampled_passages.txt \
    --output predictions.txt
```

**Output Format:**
```
sentence	prediction	probability
Barack Obama was born in Hawaii.	factual	0.1234
The capital of France is London.	hallucinated	0.8765
...
```

---

## Example Workflow

### 1. Prepare Data
```python
# example_prepare_data.py
import pickle

# Your sentences
sentences = [
    "Barack Obama was born in Hawaii.",
    "The capital of France is London.",  # Hallucinated
    "Python is a programming language.",
    # ... more sentences
]

# Your labels (0=factual, 1=hallucinated)
labels = [0, 1, 0, ...]

# Your sampled passages (same for all sentences, or per-sentence)
sampled_passages = [
    "Sample passage 1...",
    "Sample passage 2...",
    # ... more samples
]

# Save
with open('sentences.pkl', 'wb') as f:
    pickle.dump(sentences, f)

with open('labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

with open('sampled_passages.pkl', 'wb') as f:
    pickle.dump(sampled_passages, f)
```

### 2. Train
```bash
python train_supervised_model.py \
    --sentences sentences.pkl \
    --labels labels.pkl \
    --sampled_passages sampled_passages.pkl \
    --output model.pkl \
    --device cuda
```

### 3. Use
```bash
python use_trained_model.py \
    --model model.pkl \
    --sentences test_sentences.txt \
    --sampled_passages test_sampled_passages.txt \
    --output predictions.txt
```

---

## Model Details

### Features Used (7 features)
1. `mean(C)` - Average contradiction probability
2. `max(C)` - Maximum contradiction probability
3. `std(C)` - Standard deviation of contradiction probabilities
4. `frac(C>0.5)` - Proportion with contradiction > 0.5
5. `entropy(C)` - Entropy of [C, 1-C] distribution
6. `iqr(C)` - Interquartile range
7. `top3_mean(C)` - Mean of top 3 contradiction probabilities

### Model Type
- **Logistic Regression** (sklearn)
- Regularized with L2 penalty
- Class-balanced (handles imbalanced datasets)
- Solver: LBFGS (good for small-medium datasets)

---

## Tips

1. **Batch Size**: Start with 32, increase if you have GPU memory
2. **Regularization (C)**: 
   - Higher C (e.g., 10.0) = less regularization, more complex model
   - Lower C (e.g., 0.1) = more regularization, simpler model
   - Try different values if overfitting/underfitting
3. **Class Imbalance**: The model uses `class_weight='balanced'` to handle imbalanced data
4. **Feature Scaling**: Logistic regression benefits from scaled features, but our features are already in [0,1] range mostly

---

## Troubleshooting

**Out of Memory:**
- Reduce `--batch_size` (e.g., 16 or 8)
- Process data in chunks

**Poor Performance:**
- Check label distribution (should have both classes)
- Try different `--C` values
- Ensure sampled passages are diverse
- Check feature extraction is working correctly

**Slow Training:**
- Use GPU (`--device cuda`)
- Increase `--batch_size` if memory allows
- Consider using fewer samples per sentence

---

## Next Steps

After training, you can:
1. Evaluate on test set
2. Tune hyperparameters (C, batch_size)
3. Try other classifiers (Random Forest, SVM, etc.)
4. Add more features if needed
5. Deploy the model for inference
