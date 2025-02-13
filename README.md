<!-- # $\vec0$ NilVec -->

<!-- <img src="./NilVec.png" alt="NilVec Logo" width="500"/> -->

![NilVec Logo](NilVec.png)

## Overview

Most vector databases consume a lot of memory, especially when handling metadata. NilVec is designed to be more
memory-efficient by embedding metadata directly within the vectors themselves.

In a traditional vector database, metadata **_should not_** be included within vectors, as it can significantly reduce
the accuracy of nearest neighbor searches by contributing to distance calculations. NilVec avoids this issue by indexing
only the core embedding components, thereby excluding metadata from the calculations and ensuring that metadata does not
affect search accuracy.

### How It Works

To achieve this separation, NilVec maintains a global map of metadata indexes. This map identifies where metadata is
stored within the vectors, allowing NilVec to mask metadata during indexing and searching.

_Conceptually_, a vector that contains metadata is represented as:

$$
\begin{pmatrix}
.0 \\
.1 \\
\vdots \\
.511 \\
\text{meta}_a \\
\text{meta}_b \\
\text{meta}_c \\
\end{pmatrix}
\begin{pmatrix}
1 \\
1 \\
\vdots \\
1 \\
0 \\
0 \\
0 \\
\end{pmatrix} = \begin{pmatrix}
.0 \\
.1 \\
\vdots \\
.511 \\
.0 \\
.0 \\
.0 \\
\end{pmatrix}
$$

Here, the second vector acts as a mask, zeroing out metadata components so that they are not considered in
the distance calculations. As a result, NilVec ignores metadata components during search operations, focusing solely on
the embedding values.

### Indexing and Metadata Retrieval

Metadata is retrieved using a global map of indexes that indicates which components of the vector correspond to
metadata. For example:

```python
index.map = {
    "embedding": 0,
    "meta_a": 512,
    "meta_b": 513,
    "meta_c": 514,
}

i = index.map["meta_a"]  # 512
meta_a = v[i]
```

## Implementational Philosophy

[Google's ScaNN](https://github.com/google-research/google-research/tree/master/scann) is one of the fastest and most
efficient libraries for approximate nearest neighbor search.
[Its rules of thumb](https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md) are:

- For a small dataset (fewer than $20 \text{k}$ points), use brute force.
- For a dataset with fewer than $100 \text{k}$ points, score with AH, then rescore.
- For datasets larger than $100 \text{k}$ points, partition, score with AH, then rescore.
- When scoring with AH, `dimensions_per_block` should be set to $2$.
- When partitioning, `num_leaves` should be roughly the square root of the number of data points.

[Pinecone](https://docs.pinecone.io/home) has the industry's most user-friendly interface. It's as easy as:

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="YOUR_API_KEY")

# Create a serverless index
# "dimension" needs to match the dimensions of the vectors you upsert
pc.create_index(
    name="products",
    dimension=1536,
    spec=ServerlessSpec(cloud='aws', region='us-east-1')
)

# Target the index
index = pc.Index("products")

# Mock vector and metadata objects (you would bring your own)
vector = [0.010, 2.34,...] # len(vector) = 1536
metadata = {"id": 3056, "description": "Networked neural adapter"}

# Upsert your vector(s)
index.upsert(
  vectors=[
    {"id": "some_id", "values": vector, "metadata": metadata}
  ]
)
```

## Testing

To test NilVec, run:

```bash
zig build test
```
