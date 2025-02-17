![NilVec Logo](NilVec.png)

## Overview

NilVec is a high-performance, memory-efficient vector search library designed for large-scale, real-time applications. It decouples metadata from core embeddings during distance calculations, ensuring minimal memory overhead while maintaining high search accuracy and speed.

Recent benchmarks demonstrate that NilVec outperforms leading vector databases, including Chroma, Qdrant, and Milvus, offering substantial improvements in query latency and insertion efficiency.

## Key Features

- **Optimized Memory Utilization:** Vectors are stored contiguously, and metadata is managed separately, preventing unnecessary memory bloat.
- **High-Speed Query Execution:** Benchmarked for superior query latency, outperforming Chroma, Qdrant, and Milvus in real-world workloads.
- **Flexible API:** Rust-powered backend with a Python interface, offering efficient insertion, bulk indexing, and metadata-aware filtering.
- **Multi-Backend Benchmarking:** Directly compared against Qdrant, Chroma, Milvus, Redis, and PyHNSW with real-time performance tracking.

## Benchmarks

NilVec has been benchmarked against multiple vector databases:

- **Chroma**: Lightweight embedding database optimized for LLM applications.
- **Qdrant**: High-performance vector search engine for production environments.
- **Milvus**: Open-source vector database designed for massive-scale search.
- **Redis**: Real-time vector similarity search with Redis AI.

### Configuration

- **Vector Dimension:** 128
- **Number of Insertions:** 10,000 vectors
- **Query Interval:** Every 100 insertions
- **Metadata Filtering:** Applied across different categories (`news`, `blog`, `report`)

### Results

- **Query Latency:** NilVec consistently delivers lower latency compared to Qdrant, Chroma, and Milvus.
- **Insertion Efficiency:** Handles bulk insertions with minimal overhead, maintaining stable performance as the dataset scales.
- **Scalability:** Performance remains stable even as the index grows, showcasing NilVec's robustness in large-scale deployments.

### Sample Benchmark Code

```python
import time
import random
import numpy as np
import nilvec

# Configuration
DIM = 128
NUM_INSERTS = 10_000
QUERY_INTERVAL = 100
CATEGORIES = ["news", "blog", "report"]

index = nilvec.PyHNSW(DIM, None, None, None, None, "inner_product", ["category"])

for i in range(NUM_INSERTS):
    vector = [random.random() for _ in range(DIM)]
    metadata = {"category": random.choice(CATEGORIES)}
    index.insert(vector, metadata)

    if (i + 1) % QUERY_INTERVAL == 0:
        query = [random.random() for _ in range(DIM)]
        results = index.search(query, 5, ("category", metadata["category"]))
        print(f"Query {i + 1}: {results}")
```

## Installation

NilVec can be installed via pip:

```bash
pip install nilvec
```

To build from source:

```bash
git clone https://github.com/cldrake01/nilvec.git
cd nilvec
maturin develop --release
```

## Usage

```python
import nilvec

# Create an index with metadata-aware filtering
index = nilvec.PyHNSW(128, None, None, None, None, "inner_product", ["color", "size"])

# Insert vectors with metadata
vector = [0.1] * 128
metadata = [("color", "blue"), ("size", 42)]
index.insert(vector, metadata)

# Perform a metadata-aware search
query = [0.1] * 128
results = index.search(query, k=5, filter=("color", "blue"))
for distance, vector in results:
    print("Distance:", distance, "Vector:", vector)
```

## Testing

Run the test suite with:

```bash
cargo test
```
