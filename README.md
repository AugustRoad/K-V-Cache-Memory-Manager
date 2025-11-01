# Paged K-V Cache Memory Manager

A Python-based simulation of a "paged" K-V cache memory manager for Large Language Model (LLM) inference, inspired by the research paper [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180).

### The K-V Cache Challenge

In transformer-based LLMs, the Key-Value (K-V) cache stores attention keys and values for previously processed tokens. During autoregressive generation, this cache grows linearly with sequence length. For a typical model:

- **Size**: ~1 GB per sequence (32 layers, 32 heads, 2048 tokens)
- Memory is the primary bottleneck (not compute) for LLM inference

### Memory Fragmentation Issues

**Naive contiguous allocation** leads to two problems:

#### 1. Internal Fragmentation
```
Allocated: [████--------------------]  (4 tokens used, 20 slots reserved)
Wasted: 80% of allocated memory
```

#### 2. External Fragmentation (The Key Problem)
```
Initial:  [████████████████████████]  100 blocks free

Step 1:   [AAAA................]  Seq A uses 40 blocks → 60 free
Step 2:   [AAAABBBB............]  Seq B uses 40 blocks → 20 free
Step 3:   Cannot allocate Seq C (needs 30 blocks, only 20 free)

Seq A finishes:
Step 4:   [....BBBB............]  60 blocks free (40 freed + 20 existing)
Step 5:   Still cannot allocate Seq C
          Problem: 60 free blocks exist, but they're NON-CONTIGUOUS
```

**In naive systems**: Despite having 60 free blocks, Seq C fails because it requires 30 *contiguous* blocks.

## Solution: Paged Memory Management

Inspired by OS virtual memory, we separate **logical** (per-request view) from **physical** (actual GPU memory):

### Key Benefits

1. **No Contiguity Required**: Any free block can be assigned to any request
2. **Eliminates External Fragmentation**: 60 non-contiguous free blocks = 60 usable blocks
3. **Dynamic Growth**: Allocate blocks only as needed (no pre-allocation)
4. **Near 100% Utilization**: Minimal waste, maximum throughput

## Architecture

### BlockManager (Physical Memory)
Manages the actual GPU memory pool:
- Pre-allocates all GPU memory as fixed-size blocks
- Tracks free/allocated blocks
- Provides O(1) allocation and deallocation

```python
manager = BlockManager(
    num_blocks=100,      # Total blocks
    block_size=16,       # Tokens per block
    num_layers=4,        # Transformer layers
    num_heads=8,         # Attention heads
    head_size=64,        # Head dimension
    device="cuda"
)
```

### Sequence (Logical Memory)
Represents one inference request:
- Maintains block table (logical → physical mapping)
- Dynamically requests blocks as tokens are generated
- Returns blocks when request completes

```python
seq = Sequence("request_1")
seq.append_tokens(manager, 30)  # Allocates blocks as needed
seq.free_all_blocks(manager)    # Returns blocks to pool
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU support)

### Installation

```bash

git clone https://github.com/AugustRoad/K-V-Cache-Memory-Manager.git
cd K-V-Cache-Memory-Manager

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```


## Running the Simulation

```bash
python main.py
```

## Project Structure

```
K-V Cache Memory Manager/
├── block_manager.py          # Physical memory manager
├── sequence.py               # Logical memory per request
├── main.py                   # Simulation demonstrations
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Related Work

- **vLLM**: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)


