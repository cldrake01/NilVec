# $\vec0$ NilVec

## Big Idea

Most vector databses use too much memory. They use too much memory in how they handle metadata.
NilVec is a vector database that uses less memory by storing metadata within the vectors themselves.
Without a purpose-built vector database like NilVec, storing metadata directly within vectors adversely
affects the accuracy of nearest neighbor search since that metadata would contribute to the distance
calculation. NilVec's indexes consider only the vector components, not the metadata. It does this by
maintaining a global map of attributes to vector indices.

Mathematically, we can represent a vector with metadata and its resultant product before being passed
to the index as:

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
0 \\
0 \\
0 \\
\end{pmatrix}
$$

The primary index simply does not consider metadata when calculating distances as a result.
To retrieve metadata, we construct a global map of attributes to vector indices, e.g.,

```python
index.map = {
    "embedding": 0,
    "meta_a": 512,
    "meta_b": 513,
    "meta_c": 514,
}

i = index.map["meta_a"] # 512
meta_a = v[i]
```

## Implementational Philosophy

[Google's ScaNN](https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md)
is one of the fastest and most efficient libraries for approximate nearest neighbor search.
Its rules-of-thumb are:

- For a small dataset (fewer than 20k points), use brute-force.
- For a dataset with less than 100k points, score with AH, then rescore.
- For datasets larger than 100k points, partition, score with AH, then rescore.
- When scoring with AH, `dimensions_per_block` should be set to 2.
- When partitioning,` num_leaves should` be roughly the square root of the number of datapoints.

[Pinecone](https://docs.pinecone.io/home) has the industry's most user-friendly interface. It's as easy as:

```python
from pinecone import Pinecone, ServerlessSpec

# Create a serverless index
# "dimension" needs to match the dimensions of the vectors you upsert
pc = Pinecone(api_key="YOUR_API_KEY")

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
