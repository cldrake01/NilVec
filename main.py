import time
import random
import numpy as np
import matplotlib.pyplot as plt
import nilvec  # This is your PyO3 module exposing PyHNSW
import chromadb

# ------------------------------
# Chroma Test
# ------------------------------

# Setup Chroma client and collection (in-memory instance)
client = chromadb.Client()
collection = client.create_collection(name="test_collection")

# Configuration
dim = 10             # Dimension of each vector
num_inserts = 100    # Number of vectors to insert
num_queries = 10     # Number of search queries to time
categories = ["news", "blog", "report"]

# --- Insertion Timing for Chroma ---
chroma_insert_times = []
print("Chroma: Inserting vectors with metadata:")
for i in range(num_inserts):
    vector = [random.random() for _ in range(dim)]
    metadata = {"category": random.choice(categories)}

    start_time = time.perf_counter()
    collection.add(
        ids=[str(i)],
        embeddings=[vector],
        metadatas=[metadata],
        documents=[""]  # Documents are optional; here we use an empty string.
    )
    elapsed = time.perf_counter() - start_time
    chroma_insert_times.append(elapsed)
    print(f"[Chroma] Inserted vector {i+1}/{num_inserts} in {elapsed:.4f} seconds.")

# --- Query Timing for Chroma ---
chroma_query_times = []
print("\nChroma: Running search queries:")
for i in range(num_queries):
    query = [random.random() for _ in range(dim)]

    start_time = time.perf_counter()
    results = collection.query(
        query_embeddings=[query],
        n_results=5,
    )
    elapsed = time.perf_counter() - start_time
    chroma_query_times.append(elapsed)

    result_count = len(results.get("ids", [[]])[0])
    print(f"[Chroma] Query {i+1}/{num_queries} took {elapsed:.4f} seconds, returned {result_count} results.")

# ------------------------------
# NilVec Test
# ------------------------------

# Create an instance of PyHNSW with the given schema.
# The constructor signature is: PyHNSW(dim, layers, m, ef_construction, ef_search, metric, schema)
hnsw = nilvec.PyHNSW(dim, None, None, None, None, None, ["category"])

# --- Insertion Timing for NilVec ---
nilvec_insert_times = []
print("\nNilVec: Inserting vectors with metadata:")
for i in range(num_inserts):
    vector = [random.random() for _ in range(dim)]
    metadata = [("category", random.choice(categories))]

    start_time = time.perf_counter()
    hnsw.insert(vector, metadata)
    elapsed = time.perf_counter() - start_time
    nilvec_insert_times.append(elapsed)
    print(f"[NilVec] Inserted vector {i+1}/{num_inserts} in {elapsed:.4f} seconds.")

# --- Query Timing for NilVec ---
nilvec_query_times = []
print("\nNilVec: Running search queries:")
for i in range(num_queries):
    query = [random.random() for _ in range(dim)]

    start_time = time.perf_counter()
    results = hnsw.search(query, 5)
    elapsed = time.perf_counter() - start_time
    nilvec_query_times.append(elapsed)
    print(f"[NilVec] Query {i+1}/{num_queries} took {elapsed:.4f} seconds, returned {len(results)} results.")

# ------------------------------
# Combined Plotting & Saving
# ------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# X values for insertion and query operations
x_inserts = np.arange(1, num_inserts + 1)
x_queries = np.arange(1, num_queries + 1)

# --- Insertion Times Plot ---
# Scatter plots with slight transparency and no connecting lines.
ax1.scatter(x_inserts, chroma_insert_times, marker='o', alpha=0.7, label='Chroma Points')
ax1.scatter(x_inserts, nilvec_insert_times, marker='s', alpha=0.7, label='NilVec Points')

# Compute and plot best-fit lines.
# For Chroma:
coeffs = np.polyfit(x_inserts, chroma_insert_times, 1)
poly = np.poly1d(coeffs)
ax1.plot(x_inserts, poly(x_inserts), linestyle='--', color='blue', label='Chroma Best Fit')

# For NilVec:
coeffs = np.polyfit(x_inserts, nilvec_insert_times, 1)
poly = np.poly1d(coeffs)
ax1.plot(x_inserts, poly(x_inserts), linestyle='--', color='orange', label='NilVec Best Fit')

ax1.set_title("Insertion Times Comparison")
ax1.set_xlabel("Insert Operation")
ax1.set_ylabel("Time (seconds)")
ax1.legend()

# --- Query Times Plot ---
# Scatter plots with slight transparency.
ax2.scatter(x_queries, chroma_query_times, marker='o', alpha=0.7, label='Chroma Points')
ax2.scatter(x_queries, nilvec_query_times, marker='s', alpha=0.7, label='NilVec Points')

# Compute and plot best-fit lines.
# For Chroma:
coeffs = np.polyfit(x_queries, chroma_query_times, 1)
poly = np.poly1d(coeffs)
ax2.plot(x_queries, poly(x_queries), linestyle='--', color='blue', label='Chroma Best Fit')

# For NilVec:
coeffs = np.polyfit(x_queries, nilvec_query_times, 1)
poly = np.poly1d(coeffs)
ax2.plot(x_queries, poly(x_queries), linestyle='--', color='orange', label='NilVec Best Fit')

ax2.set_title("Query Times Comparison")
ax2.set_xlabel("Query Operation")
ax2.set_ylabel("Time (seconds)")
ax2.legend()

plt.tight_layout()
plt.savefig("combined_performance.png")
print("\nCombined plot saved as 'combined_performance.png'.")
plt.show()
