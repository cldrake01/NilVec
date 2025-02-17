import time
import random
import numpy as np
import matplotlib.pyplot as plt
import nilvec  # Your PyO3 module exposing PyHNSW and PyFlat
import chromadb
import redis
import subprocess
import qdrant_client

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, Range
from tqdm import tqdm

# Global Configuration

DIM = 128  # Dimension of each vector
NUM_INSERTS = 10_000  # Total number of vectors to insert
QUERY_INTERVAL = 100  # Run a query every 1000 insertions
CATEGORIES = ["news", "blog", "report"]

# Helper Functions to Manage Docker


def run_docker_command(command):
    """Runs a shell command to start Docker containers if needed."""
    try:
        subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}\n{e.stderr}")


def check_container_running(name):
    """Returns True if a container with the given name is running."""
    result = subprocess.run(
        f"docker ps --filter 'name={name}' --format '{{{{.Names}}}}'",
        shell=True,
        capture_output=True,
        text=True,
    )
    return name in result.stdout.strip()


def remove_existing_container(name):
    """Removes an existing container if it exists (whether running or stopped)."""
    result = subprocess.run(
        f"docker ps -a --filter 'name={name}' --format '{{{{.Names}}}}'",
        shell=True,
        capture_output=True,
        text=True,
    )
    if name in result.stdout.strip():
        print(f"Removing existing container: {name}")
        subprocess.run(f"docker rm -f {name}", shell=True, check=True)


def kill_all_containers():
    """Kills all Docker containers created in this script."""
    print("\nStopping all database containers...")
    run_docker_command("docker ps -q | xargs -r docker rm -f")


# Index Interface Definitions


class _Test(ABC):
    @abstractmethod
    def insert(self, vector, metadata, id_val) -> None: ...

    @abstractmethod
    def search(
        self, query, k, filter_value=None
    ) -> List[Tuple[float, List[float]]]: ...


class _PyHNSW(_Test):
    def __init__(self, dim):
        self.index = nilvec.PyHNSW(
            dim, None, None, None, None, "inner_product", ["category"]
        )

    def insert(self, vector, metadata, id_val):
        self.index.insert(vector, metadata)

    def search(self, query, k, filter_value=None):
        if filter_value is not None:
            return self.index.search(query, k, ("category", filter_value))
        else:
            return self.index.search(query, k)


class _PyFlat(_Test):
    def __init__(self, dim):
        self.index = nilvec.PyFlat(dim, None, ["category"])

    def insert(self, vector, metadata, id_val):
        self.index.insert(vector, list(metadata.items()))

    def search(self, query, k, filter_value=None):
        return self.index.search(
            query, k, ("category", filter_value) if filter_value is not None else None
        )


class _Chroma(_Test):
    def __init__(self):
        client = chromadb.Client()
        try:
            client.delete_collection("test_collection")
        except ValueError as e:
            print(f"Error deleting collection: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        self.index = client.create_collection(name="test_collection")

    def insert(self, vector, metadata, id_val):
        # Ensure metadata is in dictionary format
        if not isinstance(metadata, dict):
            metadata = dict(metadata)  # Convert list of tuples to dictionary
        self.index.add(
            ids=[str(id_val)], embeddings=[vector], metadatas=[metadata], documents=[""]
        )

    def search(self, query, k, filter_value=None):
        return self.index.query(
            query_embeddings=[query],
            n_results=k,
            where={"category": filter_value} if filter_value else None,
        )


class _Qdrant(_Test):
    def __init__(self, dim):
        container_name = "qdrant_container"
        remove_existing_container(
            container_name
        )  # Remove any existing duplicate container

        if not check_container_running(container_name):
            print("Starting Qdrant...")
            run_docker_command(
                f"docker run -d --name {container_name} -p 6333:6333 -p 6334:6334 qdrant/qdrant"
            )

        client = QdrantClient("localhost", port=6333)
        client.recreate_collection(
            collection_name="qdrant_collection",
            vectors_config={"size": dim, "distance": "Cosine"},
        )
        self.index = client

    def insert(self, vector, metadata, id_val):
        # Ensure metadata is in dictionary format
        if not isinstance(metadata, dict):
            metadata = dict(metadata)  # Convert list of tuples to dictionary

        self.index.upsert(
            collection_name="qdrant_collection",
            points=[PointStruct(id=id_val, vector=vector, payload=metadata)],
        )

    def search(self, query, k, filter_value=None):
        filters = (
            qdrant_client.models.Filter(
                must=[
                    qdrant_client.models.FieldCondition(
                        key="category", match={"value": filter_value}
                    )
                ]
            )
            if filter_value
            else None
        )
        return self.index.search(
            collection_name="qdrant_collection",
            query_vector=query,
            limit=k,
            query_filter=filters,
        )


class _Milvus(_Test):
    def __init__(self):
        container_name = "milvus_container"
        remove_existing_container(container_name)  # Remove duplicates

        if not check_container_running(container_name):
            print("Starting Milvus...")
            run_docker_command(
                f"docker run -d --name {container_name} -p 19530:19530 milvusdb/milvus:latest"
            )

        # Wait for Milvus to be ready
        retries = 10
        for i in range(retries):
            try:
                connections.connect("default", host="localhost", port="19530")
                print("Milvus is ready!")
                break
            except Exception:
                print(f"Waiting for Milvus to start... ({i+1}/{retries})")
                time.sleep(3)
        else:
            raise RuntimeError("Milvus failed to start after multiple attempts.")

        self.index = Collection("milvus_collection")

    def insert(self, vector, metadata, id_val):
        self.index.insert([[None], [vector], [metadata["category"]]])

    def search(self, query, k, filter_value=None):
        expr = f'category == "{filter_value}"' if filter_value else ""
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        return self.index.search(
            data=[query],
            anns_field="vector",
            param=search_params,
            limit=k,
            expr=expr,
            output_fields=["category"],
        )


class _Redis(_Test):
    def __init__(self):
        container_name = "redis_container"
        remove_existing_container(container_name)  # Remove duplicates

        if not check_container_running(container_name):
            print("Starting Redis...")
            run_docker_command(
                f"docker run -d --name {container_name} -p 6379:6379 redis"
            )

        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.execute_command(
            "FT.CREATE redis_index ON HASH PREFIX 1 vec: SCHEMA vector VECTOR FLAT 6 TYPE FLOAT32 DIM 1000 DISTANCE_METRIC COSINE category TAG"
        )
        self.index = r

    def insert(self, vector, metadata, id_val):
        vector_binary = np.array(vector, dtype=np.float32).tobytes()
        self.index.hset(
            f"vec:{id_val}",
            mapping={"vector": vector_binary, "category": metadata["category"]},
        )

    def search(self, query, k, filter_value=None):
        query_vector = np.array(query, dtype=np.float32).tobytes()
        query_str = (
            f"(@category:{{{filter_value}}})=>[KNN {k} @vector $query_vector AS score]"
            if filter_value
            else f"[KNN {k} @vector $query_vector AS score]"
        )
        return self.index.execute_command(
            "FT.SEARCH",
            "redis_index",
            query_str,
            "PARAMS",
            2,
            "query_vector",
            query_vector,
        )


# Benchmarking

indexes = [
    {"name": "Qdrant", "index": _Qdrant(DIM)},
    # {"name": "Milvus", "index": _Milvus()},
    {"name": "Chroma", "index": _Chroma()},
    # {"name": "Redis", "index": _Redis()},
    {"name": "PyFlat", "index": _PyFlat(DIM)},
    # {"name": "PyHNSW", "index": _PyHNSW(dim)},
]

insertion_timings: Dict[str, List[float]] = {}
query_scaling_timings: Dict[str, Dict[str, List[float]]] = {
    idx["name"]: {"indices": [], "times": []} for idx in indexes
}

for idx_entry in indexes:
    name = idx_entry["name"]
    print(f"\n==== Benchmarking {name} ====")
    index_instance = idx_entry["index"]

    query_indices: List[int] = []
    query_times: List[float] = []

    for i in tqdm(range(NUM_INSERTS), desc=f"{name} insert+query"):
        vector = [random.random() for _ in range(DIM)]
        metadata = {"category": random.choice(CATEGORIES)}

        start_ins = time.perf_counter()
        index_instance.insert(vector, metadata, i)
        ins_elapsed = time.perf_counter() - start_ins

        if (i + 1) % QUERY_INTERVAL == 0:
            query = [random.random() for _ in range(DIM)]
            filter_value = random.choice(CATEGORIES)

            start_query = time.perf_counter()
            index_instance.search(query, 5, filter_value)
            query_elapsed = time.perf_counter() - start_query

            query_indices.append(i + 1)
            query_times.append(query_elapsed)

    query_scaling_timings[name]["indices"].extend(query_indices)
    query_scaling_timings[name]["times"].extend(query_times)

# Plot Query Scaling
fig, ax = plt.subplots(figsize=(10, 6))
for idx_entry in indexes:
    name = idx_entry["name"]
    qs = query_scaling_timings[name]
    if qs["indices"]:
        ax.scatter(qs["indices"], qs["times"], alpha=0.7, label=name)
        coeffs = np.polyfit(qs["indices"], qs["times"], 1)
        ax.plot(
            qs["indices"],
            np.poly1d(coeffs)(qs["indices"]),
            linestyle="--",
            label=f"{name} Best Fit",
        )

ax.set_title("Query Time Scaling as Index Grows")
ax.set_xlabel("Number of Insertions")
ax.set_ylabel("Query Time (seconds)")
ax.legend()
plt.tight_layout()
plt.show()
