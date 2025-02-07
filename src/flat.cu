#include <cuda_runtime.h>
#include <float.h>
#include <iostream>
#include <vector>

// Thrust is used here for sorting on the GPU.
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

// ----- Configuration -----
// #define DIM 2 // Change this to set the dimensionality

// ----- CUDA Error Checking Macro -----
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error in " << __FILE__ << "@" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << "\n";                            \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// For each vector (if not tombstoned), compute its Euclidean distance from the
// query.
template <int DIM>
__global__ void computeDistances(const float *d_vectors,
                                 const bool *d_tombstones, int numVectors,
                                 const float *query, float *distances) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numVectors) {
    if (d_tombstones[idx]) {
      // Mark “deleted” entries with a huge distance so they never rank as
      // nearest.
      distances[idx] = FLT_MAX;
      return;
    }
    float sum = 0.0f;
    for (int i = 0; i < DIM; i++) {
      float diff = query[i] - d_vectors[idx * DIM + i];
      sum += diff * diff;
    }
    distances[idx] = sqrtf(sum);
  }
}

// This class allocates a large device array for vectors and for a "tombstone"
// flag. It provides methods to insert vectors, search for the nearest
// neighbors, delete (mark tombstoned), and clean the index.
template <int DIM> class FlatIndex {
public:
  // Construct an index with a fixed capacity.
  FlatIndex(int capacity) : capacity(capacity), numVectors(0) {
    // Allocate device memory for vectors (each vector has DIM floats).
    CUDA_CHECK(cudaMalloc(&d_vectors, capacity * DIM * sizeof(float)));
    // Allocate device memory for tombstone flags.
    CUDA_CHECK(cudaMalloc(&d_tombstones, capacity * sizeof(bool)));
    // Initialize tombstones to false.
    CUDA_CHECK(cudaMemset(d_tombstones, 0, capacity * sizeof(bool)));
  }

  ~FlatIndex() {
    if (d_vectors)
      cudaFree(d_vectors);
    if (d_tombstones)
      cudaFree(d_tombstones);
  }

  // Insert a new vector (from host memory) into the index.
  void insert(const float *h_vector) {
    if (numVectors >= capacity) {
      std::cerr << "Index capacity exceeded!" << std::endl;
      return;
    }
    // Copy the vector (of DIM floats) from host to the device array.
    CUDA_CHECK(cudaMemcpy(d_vectors + numVectors * DIM, h_vector,
                          DIM * sizeof(float), cudaMemcpyHostToDevice));
    // Set the tombstone flag for this vector to false.
    bool falseVal = false;
    CUDA_CHECK(cudaMemcpy(d_tombstones + numVectors, &falseVal, sizeof(bool),
                          cudaMemcpyHostToDevice));
    numVectors++;
  }

  // Search for the k nearest neighbors to the given query vector.
  // The query is provided as a host array of DIM floats.
  // Returns a vector (host-side) of indices corresponding to the nearest
  // neighbors.
  std::vector<int> search(const float *h_query, int k) {
    // Allocate device memory for the query.
    float *d_query;
    CUDA_CHECK(cudaMalloc(&d_query, DIM * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_query, h_query, DIM * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Allocate device memory for distances (one per vector).
    float *d_distances;
    CUDA_CHECK(cudaMalloc(&d_distances, numVectors * sizeof(float)));

    // Launch kernel to compute distances.
    int threadsPerBlock = 256;
    int blocks = (numVectors + threadsPerBlock - 1) / threadsPerBlock;
    computeDistances<<<blocks, threadsPerBlock>>>(
        d_vectors, d_tombstones, numVectors, d_query, d_distances, DIM);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Wrap the distances in a Thrust device_vector for sorting.
    thrust::device_vector<float> d_vecDistances(numVectors);
    thrust::copy(thrust::device, d_distances, d_distances + numVectors,
                 d_vecDistances.begin());

    // Create a device_vector of indices [0, 1, 2, ..., numVectors-1].
    thrust::device_vector<int> d_indices(numVectors);
    thrust::sequence(d_indices.begin(), d_indices.end());

    // Sort indices by comparing the corresponding distance values.
    // (The lambda below is a __device__ comparator that accesses the
    // distances.)
    thrust::sort(
        d_indices.begin(), d_indices.end(),
        [d_ptr = thrust::raw_pointer_cast(d_vecDistances.data())] __device__(
            int a, int b) { return d_ptr[a] < d_ptr[b]; });

    // Copy the sorted indices back to the host.
    thrust::host_vector<int> h_indices = d_indices;
    std::vector<int> result;
    // Take the top k indices (ignoring any with FLT_MAX distance).
    for (int i = 0; i < k && i < numVectors; i++) {
      int idx = h_indices[i];
      // Optionally, you might copy the distance to check that it isn’t FLT_MAX.
      float dist;
      CUDA_CHECK(cudaMemcpy(&dist, d_distances + idx, sizeof(float),
                            cudaMemcpyDeviceToHost));
      if (dist < FLT_MAX)
        result.push_back(idx);
    }

    // Free temporary device memory.
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_distances));
    return result;
  }

  // Delete (mark as deleted) the nearest neighbor to the given query.
  void deleteNearest(const float *h_query) {
    std::vector<int> results = search(h_query, 1);
    if (!results.empty()) {
      int idx = results[0];
      bool trueVal = true;
      CUDA_CHECK(cudaMemcpy(d_tombstones + idx, &trueVal, sizeof(bool),
                            cudaMemcpyHostToDevice));
    }
  }

  // "Clean" the index by compacting out the tombstoned vectors.
  // (This is implemented on the host: we copy data back, remove deleted
  // entries, then re-upload the live vectors.)
  void clean() {
    // Allocate host-side temporary arrays.
    std::vector<float> h_vectors(numVectors * DIM);
    std::vector<bool> h_tombstones(numVectors);
    CUDA_CHECK(cudaMemcpy(h_vectors.data(), d_vectors,
                          numVectors * DIM * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_tombstones.data(), d_tombstones,
                          numVectors * sizeof(bool), cudaMemcpyDeviceToHost));

    int writeIndex = 0;
    for (int i = 0; i < numVectors; i++) {
      if (!h_tombstones[i]) {
        // Copy vector i to position writeIndex if it’s live.
        for (int d = 0; d < DIM; d++) {
          h_vectors[writeIndex * DIM + d] = h_vectors[i * DIM + d];
        }
        h_tombstones[writeIndex] = false;
        writeIndex++;
      }
    }
    numVectors = writeIndex;
    // Re-upload the compacted data.
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors.data(),
                          numVectors * DIM * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tombstones, h_tombstones.data(),
                          numVectors * sizeof(bool), cudaMemcpyHostToDevice));
  }

  // Return the current number of vectors in the index.
  int size() const { return numVectors; }

private:
  float *d_vectors;   // device pointer for vectors (size: capacity * DIM)
  bool *d_tombstones; // device pointer for tombstone flags (size: capacity)
  int numVectors;     // current number of inserted vectors
  int capacity;       // maximum capacity
};

// ----- Main -----
// This simple main demonstrates usage of the FlatIndex.
int main() {
  // Create an index that can store up to 10,000 vectors.
  FlatIndex index(10000);

  // Insert some 2D vectors.
  float vec1[DIM] = {-1.0f, -1.0f};
  float vec2[DIM] = {-1.0f, 1.0f};
  float vec3[DIM] = {1.0f, 1.0f};
  float vec4[DIM] = {1.0f, -1.0f};
  float vec5[DIM] = {0.0f, 0.0f};

  index.insert(vec1);
  index.insert(vec2);
  index.insert(vec3);
  index.insert(vec4);
  index.insert(vec5);

  std::cout << "Index size after insertion: " << index.size() << std::endl;

  // Search for the 2 nearest neighbors near (0.1, 0.1)
  float query[DIM] = {0.1f, 0.1f};
  std::vector<int> results = index.search(query, 2);

  std::cout << "Search results (indices): ";
  for (int idx : results)
    std::cout << idx << " ";
  std::cout << std::endl;

  // Delete the nearest neighbor to (0.1, 0.1)
  index.deleteNearest(query);
  std::cout << "Deleted nearest neighbor." << std::endl;

  // Search again
  results = index.search(query, 2);
  std::cout << "Search results after deletion (indices): ";
  for (int idx : results)
    std::cout << idx << " ";
  std::cout << std::endl;

  // Clean the index to remove tombstoned vectors.
  index.clean();
  std::cout << "Index size after cleaning: " << index.size() << std::endl;

  return 0;
}
