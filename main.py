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
dim = 10  # Dimension of each vector
num_inserts = 100  # Number of vectors to insert
num_queries = 10  # Number of search queries to time
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
        documents=[""],  # Documents are optional; here we use an empty string.
    )
    elapsed = time.perf_counter() - start_time
    chroma_insert_times.append(elapsed)
    print(f"[Chroma] Inserted vector {i+1}/{num_inserts} in {elapsed:.4f} seconds.")

total_chroma_insert = sum(chroma_insert_times)
print(
    f"\n[Chroma] Total insertion time for {num_inserts} vectors: {total_chroma_insert:.6f} seconds."
)

# --- Query Timing for Chroma ---
chroma_query_times = []
print("\nChroma: Running search queries with metadata filtering:")
for i in range(num_queries):
    query = [random.random() for _ in range(dim)]
    # Choose a random category filter for this query.
    filter_category = random.choice(categories)

    start_time = time.perf_counter()
    results = collection.query(
        query_embeddings=[query], n_results=5, where={"category": filter_category}
    )
    elapsed = time.perf_counter() - start_time
    chroma_query_times.append(elapsed)

    result_count = len(results.get("ids", [[]])[0])
    print(
        f"[Chroma] Query {i+1}/{num_queries} (where category = '{filter_category}') took {elapsed:.4f} seconds, returned {result_count} results."
    )

total_chroma_query = sum(chroma_query_times)
print(
    f"\n[Chroma] Total query time for {num_queries} queries: {total_chroma_query:.6f} seconds."
)

# ------------------------------
# NilVec Test
# ------------------------------

# Create an instance of PyHNSW with the given schema.
# Constructor: PyHNSW(dim, layers, m, ef_construction, ef_search, metric, schema)
hnsw = nilvec.PyHNSW(dim, None, None, None, None, "inner_product", ["category"])

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

total_nilvec_insert = sum(nilvec_insert_times)
print(
    f"\n[NilVec] Total insertion time for {num_inserts} vectors: {total_nilvec_insert:.6f} seconds."
)

# --- Query Timing for NilVec with metadata filtering ---
nilvec_query_times = []
print("\nNilVec: Running search queries with metadata filtering:")
for i in range(num_queries):
    query = [random.random() for _ in range(dim)]
    # Choose a random category filter for this query.
    filter_category = random.choice(categories)

    start_time = time.perf_counter()
    # The third parameter is the filter tuple (attribute, value)
    results = hnsw.search(query, 5, ("category", filter_category))
    elapsed = time.perf_counter() - start_time
    nilvec_query_times.append(elapsed)
    print(
        f"[NilVec] Query {i+1}/{num_queries} (where category = '{filter_category}') took {elapsed:.4f} seconds, returned {len(results)} results."
    )

total_nilvec_query = sum(nilvec_query_times)
print(
    f"\n[NilVec] Total query time for {num_queries} queries: {total_nilvec_query:.6f} seconds."
)

# ------------------------------
# Performance Improvement Calculation
# ------------------------------

# Compute average times.
avg_chroma_insert = np.mean(chroma_insert_times)
avg_nilvec_insert = np.mean(nilvec_insert_times)
avg_chroma_query = np.mean(chroma_query_times)
avg_nilvec_query = np.mean(nilvec_query_times)

# Calculate percentage improvement (lower time is better).
insert_improvement = ((avg_chroma_insert - avg_nilvec_insert) / avg_chroma_insert) * 100
query_improvement = ((avg_chroma_query - avg_nilvec_query) / avg_chroma_query) * 100
overall_improvement = (insert_improvement + query_improvement) / 2

print("\n--- Performance Improvement ---")
print(f"Average Insertion Time (Chroma): {avg_chroma_insert:.5f} seconds")
print(f"Average Insertion Time (NilVec):   {avg_nilvec_insert:.5f} seconds")
print(f"Insertion Improvement:           {insert_improvement:.2f}%")

print(f"Average Query Time (Chroma):     {avg_chroma_query:.5f} seconds")
print(f"Average Query Time (NilVec):       {avg_nilvec_query:.5f} seconds")
print(f"Query Improvement:               {query_improvement:.2f}%")

print(
    f"\nOverall Average Performance Improvement for NilVec relative to Chroma: {overall_improvement:.2f}%"
)

# ------------------------------
# Combined Plotting & Saving
# ------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# X values for insertion and query operations
x_inserts = np.arange(1, num_inserts + 1)
x_queries = np.arange(1, num_queries + 1)

# --- Insertion Times Plot ---
# Scatter plots with slight transparency and no connecting lines.
ax1.scatter(
    x_inserts, chroma_insert_times, marker="o", alpha=0.7, label="Chroma Points"
)
ax1.scatter(
    x_inserts, nilvec_insert_times, marker="s", alpha=0.7, label="NilVec Points"
)

# Compute and plot best-fit lines.
# For Chroma:
coeffs = np.polyfit(x_inserts, chroma_insert_times, 1)
poly = np.poly1d(coeffs)
ax1.plot(
    x_inserts, poly(x_inserts), linestyle="--", color="blue", label="Chroma Best Fit"
)

# For NilVec:
coeffs = np.polyfit(x_inserts, nilvec_insert_times, 1)
poly = np.poly1d(coeffs)
ax1.plot(
    x_inserts, poly(x_inserts), linestyle="--", color="orange", label="NilVec Best Fit"
)

ax1.set_title("Insertion Times Comparison")
ax1.set_xlabel("Insert Operation")
ax1.set_ylabel("Time (seconds)")
ax1.legend()

# --- Query Times Plot ---
# Scatter plots with slight transparency.
ax2.scatter(x_queries, chroma_query_times, marker="o", alpha=0.7, label="Chroma Points")
ax2.scatter(x_queries, nilvec_query_times, marker="s", alpha=0.7, label="NilVec Points")

# Compute and plot best-fit lines.
# For Chroma:
coeffs = np.polyfit(x_queries, chroma_query_times, 1)
poly = np.poly1d(coeffs)
ax2.plot(
    x_queries, poly(x_queries), linestyle="--", color="blue", label="Chroma Best Fit"
)

# For NilVec:
coeffs = np.polyfit(x_queries, nilvec_query_times, 1)
poly = np.poly1d(coeffs)
ax2.plot(
    x_queries, poly(x_queries), linestyle="--", color="orange", label="NilVec Best Fit"
)

ax2.set_title("Query Times Comparison")
ax2.set_xlabel("Query Operation")
ax2.set_ylabel("Time (seconds)")
ax2.legend()

plt.tight_layout()
plt.savefig("combined_performance.png")
print("\nCombined plot saved as 'combined_performance.png'.")
plt.show()
