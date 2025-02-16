import time
import random
import numpy as np
import matplotlib.pyplot as plt
import nilvec  # Your PyO3 module exposing PyHNSW and PyFlat
import chromadb
from tqdm import tqdm

# ------------------------------
# Global Configuration
# ------------------------------

dim = 1_000 # Dimension of each vector
num_inserts = 1_000_000 # Total number of vectors to insert
query_interval = 1_000 # Run a query every 1000 insertions
categories = ["news", "blog", "report"]

# ------------------------------
# Index Interface Definitions
# ------------------------------

# ---- Chroma Index ----
def create_chroma():
    client = chromadb.Client()
    try:
        client.delete_collection("test_collection")
    except Exception:
        pass
    return client.create_collection(name="test_collection")

def chroma_insert(index, vector, metadata, id_val):
    index.add(
        ids=[str(id_val)],
        embeddings=[vector],
        metadatas=[metadata],
        documents=[""]  # Documents are optional.
    )

def chroma_search(index, query, k, filter_value=None):
    if filter_value is not None:
        return index.query(query_embeddings=[query], n_results=k, where={"category": filter_value})
    else:
        return index.query(query_embeddings=[query], n_results=k)

# ---- NilVec (PyHNSW) Index ----
def create_pyhnsw():
    return nilvec.PyHNSW(dim, None, None, None, None, "inner_product", ["category"])

def pyhnsw_insert(index, vector, metadata, _):
    index.insert(vector, metadata)

def pyhnsw_search(index, query, k, filter_value=None):
    if filter_value is not None:
        return index.search(query, k, ("category", filter_value))
    else:
        return index.search(query, k)

# ---- PyFlat Index ----
def create_pyflat():
    return nilvec.PyFlat(dim, None)

def pyflat_insert(index, vector, metadata, _):
    index.insert(vector)

def pyflat_search(index, query, k, _):
    return index.search(query, k)

# ------------------------------
# List of Indexes to Benchmark
# ------------------------------

indexes = [
    # Uncomment other indexes as needed:
    # {
    #     "name": "Chroma",
    #     "create": create_chroma,
    #     "insert": chroma_insert,
    #     "search": chroma_search,
    # },
    # {
    #     "name": "PyHNSW",
    #     "create": create_pyhnsw,
    #     "insert": pyhnsw_insert,
    #     "search": pyhnsw_search,
    # },
    {
        "name": "PyFlat",
        "create": create_pyflat,
        "insert": pyflat_insert,
        "search": pyflat_search,
    }
]

# ------------------------------
# Benchmark Loop: Query at Given Intervals
# ------------------------------

# Dictionaries to hold timings.
insertion_timings = {}
query_scaling_timings = {}  # List of (insertion_count, query_time) pairs per index

for idx_entry in indexes:
    name = idx_entry["name"]
    print(f"\n==== Benchmarking {name} ====")
    index_instance = idx_entry["create"]()

    ins_times = []
    query_times = []
    query_indices = []  # Record at which insertion count the query was run

    for i in tqdm(range(num_inserts), desc=f"{name} insert+query"):
        # Generate a random vector.
        vector = [random.random() for _ in range(dim)]
        # For indexes that support metadata (Chroma, PyHNSW), create metadata.
        if name == "Chroma":
            metadata = {"category": random.choice(categories)}
        else:
            metadata = [("category", random.choice(categories))]

        # Time the insertion.
        start_ins = time.perf_counter()
        # For Chroma, pass the unique id.
        idx_entry["insert"](index_instance, vector, metadata, i)
        ins_elapsed = time.perf_counter() - start_ins
        ins_times.append(ins_elapsed)

        # Run a query only after every 'query_interval' insertions.
        if (i + 1) % query_interval == 0:
            query = [random.random() for _ in range(dim)]
            filter_value = None
            if name in ["Chroma", "PyHNSW"]:
                filter_value = random.choice(categories)
            start_query = time.perf_counter()
            results = idx_entry["search"](index_instance, query, 5, filter_value)
            query_elapsed = time.perf_counter() - start_query
            query_times.append(query_elapsed)
            query_indices.append(i + 1)

    total_ins = sum(ins_times)
    insertion_timings[name] = ins_times
    query_scaling_timings[name] = {"indices": query_indices, "times": query_times}

    print(f"[{name}] Total insertion time for {num_inserts} vectors: {total_ins:.6f} seconds.")
    if query_times:
        total_query = sum(query_times)
        print(f"[{name}] Total query time (sampled every {query_interval} insertions): {total_query:.6f} seconds.")
    else:
        print(f"[{name}] No queries were run (query_interval may be too high).")

# ------------------------------
# Plotting Query Time Scaling
# ------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

for idx_entry in indexes:
    name = idx_entry["name"]
    qs = query_scaling_timings[name]
    if not qs["indices"]:
        continue
    marker = 'o' if name == "Chroma" else ('s' if name == "PyHNSW" else '^')
    ax.scatter(qs["indices"], qs["times"], marker=marker, alpha=0.7, label=name)
    coeffs = np.polyfit(qs["indices"], qs["times"], 1)
    poly = np.poly1d(coeffs)
    color = 'blue' if name == "Chroma" else ('orange' if name == "PyHNSW" else 'green')
    ax.plot(qs["indices"], poly(qs["indices"]), linestyle='--', color=color, label=f"{name} Best Fit")

ax.set_title("Query Time Scaling as Index Grows")
ax.set_xlabel("Number of Insertions")
ax.set_ylabel("Query Time (seconds)")
ax.legend()
plt.tight_layout()
plt.savefig("query_scaling.png")
print("\nQuery scaling plot saved as 'query_scaling.png'.")
plt.show()
