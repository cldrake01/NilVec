![NilVec Logo](NilVec.png)

## Overview

NilVec is a high-performance, memory-efficient vector search library designed to handle both embeddings and associated metadata without compromising query accuracy or speed. By decoupling metadata from the core embedding data during distance calculations, NilVec ensures that search accuracy remains high while keeping memory overhead minimal.

In our benchmarks, NilVec achieved a **95.5% improvement on query latency** compared to leading solutions like Chroma, making it an excellent choice for real-time applications and large-scale search deployments.

## Key Features

- **Memory Efficiency:**
  NilVec stores vectors in a contiguous block of memory and tracks metadata separately, avoiding unnecessary duplication and overhead.

- **High Performance:**
  Benchmarked to deliver a 95.5% improvement in query latency over comparable systems, ensuring rapid search responses.

- **Flexible and Ergonomic API:**
  Built in Rust with a Python interface, NilVec supports simple operations for inserting vectors, searching, and bulk index creation—all while handling metadata seamlessly.

## How It Works

NilVec separates the embedding components from metadata so that only the core vector elements contribute to distance calculations. Metadata is stored in parallel and associated via a schema that maps attribute names (as `String`s) to their corresponding positions in the metadata array. This design guarantees that metadata does not interfere with the accuracy of nearest neighbor searches.

## Benchmarks

Our benchmarks compare NilVec with Chroma using the following setup:

- **Configuration:**
  - Dimension: 10
  - Number of insertions: 100 vectors
  - Number of queries: 10 queries with metadata filtering
- **Results:**
  - NilVec demonstrated a **95.5% improvement on query latency** compared to Chroma.
  - Insertion latency is also highly optimized, ensuring minimal overhead during data ingestion.

Below is an excerpt from our benchmark script:

```python
import time
import random
import numpy as np
import nilvec
import chromadb

# Configuration
dim = 10
num_inserts = 100
num_queries = 10
categories = ["news", "blog", "report"]

# --- Chroma Benchmark ---
chroma_query_times = []
for i in range(num_queries):
    query = [random.random() for _ in range(dim)]
    filter_category = random.choice(categories)
    start_time = time.perf_counter()
    # Execute query on Chroma...
    elapsed = time.perf_counter() - start_time
    chroma_query_times.append(elapsed)

# --- NilVec Benchmark ---
nilvec_query_times = []
hnsw = nilvec.PyHNSW(dim, None, None, None, None, "inner_product", ["category"])
for i in range(num_queries):
    query = [random.random() for _ in range(dim)]
    filter_category = random.choice(categories)
    start_time = time.perf_counter()
    results = hnsw.search(query, 5, ("category", filter_category))
    elapsed = time.perf_counter() - start_time
    nilvec_query_times.append(elapsed)

```

## Usage

### Installation

NilVec is distributed as a Python package via its PyO3 bindings. You can install it using pip:

```bash
pip install nilvec
```

### Examples

Below is a quick example of how to use NilVec in your Python project:

```py
import nilvec

# Create an index with dimension 128 using inner product as the metric.
# Optionally, you can provide a schema for metadata.
index = nilvec.PyHNSW(128, None, None, None, None, "inner_product", ["color", "size"])

# Insert a vector with associated metadata.
vector = [0.1] * 128
metadata = [("color", "blue"), ("size", 42)]
index.insert(vector, metadata)

# Perform a search query with metadata filtering.
query = [0.1] * 128
results = index.search(query, k=5, filter=("color", "blue"))
for distance, vector in results:
    print("Distance:", distance, "Vector:", vector)

# Alternatively, bulk-create an index from a list of vectors.
vectors = [
    [0.1] * 128,
    [0.2] * 128,
    [0.3] * 128
]
index.create(vectors)
```

## Testing

To run the NilVec test suite, execute:

```bash
cargo test
```
