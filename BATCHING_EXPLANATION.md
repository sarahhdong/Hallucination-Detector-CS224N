# How Batching Improves Performance

## The Problem: Sequential Processing

### Original Class (Sequential)
```python
for sent_i, sentence in enumerate(sentences):
    for sample_i, sample in enumerate(sampled_passages):
        # One forward pass per pair
        inputs = tokenizer(sentence, sample, ...)
        logits = model(**inputs).logits  # Single forward pass
        probs = softmax(logits)
        scores[sent_i, sample_i] = probs[0][1]
```

**Example: 20 sentences × 20 samples = 400 forward passes**
- Each forward pass has overhead (GPU kernel launch, memory transfer, etc.)
- GPU is underutilized (processing one example at a time)
- Total time ≈ 400 × (overhead + computation)

---

## The Solution: Batching

### Extended Class (Batched)
```python
# Build all pairs
pairs = [(sentence, sample) for sentence in sentences for sample in sampled_passages]

# Process in batches
for batch_start in range(0, total_pairs, batch_size=32):
    batch_pairs = pairs[batch_start:batch_start+32]
    
    # Tokenize entire batch at once
    inputs = tokenizer([sent for sent, _ in batch_pairs], 
                       [sample for _, sample in batch_pairs], ...)
    
    # Single forward pass for entire batch
    logits = model(**inputs).logits  # Shape: (32, num_labels)
    all_logits.append(logits)
```

**Example: 20 sentences × 20 samples = 400 pairs in ~13 batches (batch_size=32)**
- 13 forward passes instead of 400
- GPU processes 32 examples in parallel
- Total time ≈ 13 × (overhead + computation/32)

---

## Why Batching is Faster

### 1. **GPU Parallelization**
- **Sequential**: GPU processes 1 example → waits → processes next → waits
- **Batched**: GPU processes 32 examples in parallel (utilizes all cores)
- **Speedup**: ~10-30x depending on batch size and GPU

### 2. **Reduced Overhead**
- **Sequential**: 400 kernel launches (each has overhead)
- **Batched**: 13 kernel launches (much less overhead)
- **Overhead includes**: GPU kernel launch, memory transfer, synchronization

### 3. **Better Memory Utilization**
- **Sequential**: Small memory transfers (inefficient)
- **Batched**: Large memory transfers (more efficient)
- **GPU memory**: Better utilization of GPU memory bandwidth

### 4. **Optimized Operations**
- **Matrix operations**: GPUs are optimized for batch matrix operations
- **BLAS libraries**: Batch operations use highly optimized libraries (cuBLAS, etc.)
- **Vectorization**: Batch operations can be vectorized more effectively

---

## Performance Comparison

### Example: 20 sentences × 20 samples = 400 pairs

| Method | Forward Passes | GPU Utilization | Estimated Time |
|--------|----------------|-----------------|----------------|
| **Sequential** | 400 individual | ~5-10% | 400 × 10ms = **4 seconds** |
| **Batched (32)** | 13 batches | ~80-90% | 13 × 30ms = **0.39 seconds** |
| **Speedup** | 30.8x fewer | 8-9x better | **~10x faster** |

*Note: Actual speedup depends on GPU, model size, and batch size*

---

## How Batching Works Step-by-Step

### Step 1: Build All Pairs
```python
pairs = []
for sent_i, sentence in enumerate(sentences):
    for sample_i, sample in enumerate(sampled_passages):
        pairs.append((sentence, sample))
# Result: 400 pairs for 20×20
```

### Step 2: Process in Batches
```python
batch_size = 32
for batch_start in range(0, 400, 32):
    batch_end = min(batch_start + 32, 400)
    batch_pairs = pairs[batch_start:batch_end]  # 32 pairs
    
    # Batch tokenization
    inputs = tokenizer(
        [sent for sent, _ in batch_pairs],      # 32 sentences
        [sample for _, sample in batch_pairs],  # 32 samples
        padding=True,  # Pad to same length
        return_tensors="pt"
    )
    # Result: inputs['input_ids'] shape (32, max_length)
    
    # Batch forward pass
    logits = model(**inputs).logits
    # Result: logits shape (32, num_labels)
    
    all_logits.append(logits)
```

### Step 3: Concatenate Results
```python
all_logits = torch.cat(all_logits, dim=0)  # (400, num_labels)
```

### Step 4: Vectorized Assignment
```python
# Instead of loop:
for i, (sent_i, sample_i) in enumerate(pair_indices):
    scores[sent_i, sample_i] = all_logits[i][contradiction_idx]

# Use vectorized indexing:
scores[sent_indices, sample_indices] = all_logits[:, contradiction_idx].numpy()
```

---

## Key Concepts

### 1. **Padding**
- Batches need same-length sequences
- Tokenizer automatically pads shorter sequences
- Padding tokens are masked during attention

### 2. **Batch Size Trade-off**
- **Too small** (1-8): Underutilizes GPU, high overhead
- **Optimal** (16-64): Good balance of speed and memory
- **Too large** (128+): May run out of GPU memory, diminishing returns

### 3. **Memory Considerations**
- Larger batches use more GPU memory
- Need to balance batch size with available memory
- Default batch_size=32 is a good starting point

---

## Visual Comparison

### Sequential Processing
```
GPU: [Example 1] → wait → [Example 2] → wait → [Example 3] → ...
     ↑ Underutilized (only 1 core active)
```

### Batched Processing
```
GPU: [Example 1, 2, 3, ..., 32] → [Example 33, 34, ..., 64] → ...
     ↑ Fully utilized (all cores active in parallel)
```

---

## Real-World Impact

**Before (Sequential):**
- 1000 sentences × 20 samples = 20,000 forward passes
- Time: ~200 seconds (3.3 minutes)

**After (Batched, batch_size=32):**
- 1000 sentences × 20 samples = 20,000 pairs in 625 batches
- Time: ~20 seconds
- **Speedup: 10x faster**

---

## Summary

Batching improves performance by:
1. ✅ **Parallelization**: Process multiple examples simultaneously
2. ✅ **Reduced overhead**: Fewer GPU kernel launches
3. ✅ **Better GPU utilization**: Use all GPU cores effectively
4. ✅ **Optimized operations**: Leverage optimized batch matrix operations

The extended class uses batching to achieve **10-30x speedup** compared to sequential processing.
