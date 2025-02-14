use ordered_float::OrderedFloat;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::Rng;
use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc; // requires the rand crate

/// Possible errors in HNSW operations.
#[derive(Debug, Clone)]
pub enum HNSWError {
    InvalidEF,
    InvalidLayer,
    NoSchema,
    EmptySchema,
    AttributeNotFound,
    EmptyVectors,
    InvalidNodeID,
}

/// A simple metadata type.
#[derive(Debug, Clone, PartialEq)]
pub enum Metadata {
    Str(String),
    Int(i64),
    Float(f64),
}

/// A candidate for nearest‐neighbor search.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Candidate {
    pub distance: OrderedFloat<f64>,
    pub id: usize,
}

/// Which distance metric to use.
#[derive(Debug, Clone)]
pub enum Metric {
    L2,
    Cosine,
    InnerProduct,
}

/// A filter that checks an attribute’s metadata.
pub struct Filter {
    pub attribute: String,
    // Use Arc so that the closure is cloneable, and require 'static.
    pub condition: Arc<dyn Fn(&Metadata) -> bool + Send + Sync + 'static>,
}

// Manually implement Clone (though Arc is cloneable)
impl Clone for Filter {
    fn clone(&self) -> Self {
        Filter {
            attribute: self.attribute.clone(),
            condition: self.condition.clone(),
        }
    }
}

/// The HNSW index structure.
/// Vectors are stored “flat” in a single Vec—each vector has length `dim`.
pub struct HNSW {
    pub dim: usize,
    pub layers: usize,
    pub m: usize,
    pub ml: f64,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub vectors: Vec<f64>, // flat list of all vectors (each of length `dim`)
    pub connections: Vec<usize>, // flat neighbor list
    pub offsets: Vec<usize>, // starting index in `connections` for each node
    pub levels: Vec<usize>, // level for each node
    pub tombstones: Vec<bool>, // deletion markers
    pub assignment_probabilities: Vec<f64>,
    pub metric: fn(&[f64], &[f64]) -> f64,
    pub schema: Option<Vec<String>>,
    pub metadata: Vec<Metadata>,
}

impl HNSW {
    /// Creates a new HNSW index.
    pub fn new(
        dim: usize,
        layers: Option<usize>,
        m: Option<usize>,
        ef_construction: Option<usize>,
        ef_search: Option<usize>,
        metric: Option<Metric>,
        schema: Option<Vec<String>>,
    ) -> Self {
        let layers = layers.unwrap_or(5);
        let m = m.unwrap_or(16);
        let ef_construction = ef_construction.unwrap_or(200);
        let ef_search = ef_search.unwrap_or(50);
        let ml = 1.0 - (1.0 / m as f64);

        // Precompute assignment probabilities (e.g. exp(-i)).
        let assignment_probabilities: Vec<f64> = (0..layers).map(|i| (-(i as f64)).exp()).collect();

        // Choose a metric function.
        let metric_fn: fn(&[f64], &[f64]) -> f64 = match metric {
            Some(Metric::Cosine) => Self::cosine_similarity,
            Some(Metric::InnerProduct) => Self::dot_product,
            _ => Self::euclidean_distance,
        };

        Self {
            dim,
            layers,
            m,
            ml,
            ef_construction,
            ef_search,
            vectors: Vec::new(),
            connections: Vec::new(),
            offsets: Vec::new(),
            levels: Vec::new(),
            tombstones: Vec::new(),
            assignment_probabilities,
            metric: metric_fn,
            schema,
            metadata: Vec::new(),
        }
    }

    /// Inserts `item` into a sorted Vec (sorted in ascending order by distance).
    fn insort(nns: &mut Vec<Candidate>, item: Candidate) {
        let pos = nns
            .binary_search_by(|c| c.distance.partial_cmp(&item.distance).unwrap())
            .unwrap_or_else(|e| e);
        nns.insert(pos, item);
    }

    /// Searches for the nearest neighbors (k‑NN) in a given layer.
    fn knn(
        &self,
        entry: usize,
        query: &[f64],
        ef: usize,
        layer: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<Candidate>, HNSWError> {
        if ef == 0 {
            return Err(HNSWError::InvalidEF);
        }
        if layer >= self.layers {
            return Err(HNSWError::InvalidLayer);
        }
        if filter.is_some() && self.schema.is_none() {
            return Err(HNSWError::NoSchema);
        }
        if let Some(f) = filter {
            let _ = self.attribute_index(&f.attribute)?;
        }

        let vector_entry = &self.vectors[entry * self.dim..(entry + 1) * self.dim];
        let best = Candidate {
            distance: OrderedFloat::from((self.metric)(query, vector_entry)),
            id: entry,
        };

        let mut nns = vec![best.clone()];
        let mut visited = HashSet::new();
        visited.insert(entry);

        let mut candidates = BinaryHeap::new();
        // Wrap in Reverse so that BinaryHeap pops the smallest distance first.
        candidates.push(std::cmp::Reverse(best));

        while let Some(std::cmp::Reverse(candidate)) = candidates.pop() {
            if candidate.distance > nns.last().unwrap().distance {
                break;
            }

            let start = self.offsets.get(candidate.id).copied().unwrap_or(0);
            let end = if candidate.id + 1 < self.offsets.len() {
                self.offsets[candidate.id + 1]
            } else {
                self.connections.len()
            };

            for i in start..end {
                let neighbor = self.connections[i];
                if self.levels[neighbor] < layer {
                    continue;
                }
                // If a filter is provided, check it.
                if let Some(filter) = filter {
                    let attr_index = self.attribute_index(&filter.attribute)?;
                    // Here we assume that the metadata for a node is stored at position `neighbor + attr_index`
                    // (adjust as needed for your layout).
                    let meta = &self.metadata[neighbor + attr_index];
                    if !(filter.condition)(meta) {
                        continue;
                    }
                }
                let neighbor_vector = &self.vectors[neighbor * self.dim..(neighbor + 1) * self.dim];
                let distance = (self.metric)(query, neighbor_vector);
                let current = Candidate {
                    distance: OrderedFloat(distance),
                    id: neighbor,
                };

                if !visited.contains(&current.id) {
                    visited.insert(current.id);
                    if nns.len() < ef || distance < *nns.last().unwrap().distance {
                        candidates.push(std::cmp::Reverse(current.clone()));
                        Self::insort(&mut nns, current);
                        if nns.len() > ef {
                            nns.pop(); // remove the worst candidate
                        }
                    }
                }
            }
        }
        Ok(nns)
    }

    fn vector_from_id(&self, id: usize) -> Vec<f64> {
        self.vectors[id * self.dim..(id + 1) * self.dim].to_vec()
    }

    /// Inserts a new vector (with optional metadata) into the index.
    pub fn insert<R: Rng>(
        &mut self,
        vector: &[f64],
        metadata: Option<&[Metadata]>,
        efc: Option<usize>,
        rng: &mut R,
    ) -> Result<(), HNSWError> {
        let efc = efc.unwrap_or(self.ef_construction);

        // If index is empty, add the vector and mark it as existing on all layers.
        if self.vectors.is_empty() {
            self.vectors.extend_from_slice(vector);
            self.levels.push(self.layers - 1);
            self.tombstones.push(false);
            self.offsets.push(0);
            if let Some(meta) = metadata {
                self.metadata.extend_from_slice(meta);
            }
            return Ok(());
        }

        // Determine new node’s level.
        let new_level = self.get_insert_layer(rng);
        self.vectors.extend_from_slice(vector);
        let new_node_id = self.vectors.len() / self.dim - 1;
        self.levels.push(new_level);
        self.tombstones.push(false);
        if let Some(meta) = metadata {
            self.metadata.extend_from_slice(meta);
        }

        let mut entry = 0; // global entry point
        for layer in 0..self.layers {
            if layer < new_level {
                let results = self.knn(entry, vector, 1, layer, None)?;
                if !results.is_empty() {
                    entry = results[0].id;
                }
            } else {
                let nns = self.knn(entry, vector, efc, layer, None)?;
                let mut new_connections = Vec::new();
                for candidate in &nns {
                    new_connections.push(candidate.id);
                    self.add_connection(candidate.id, new_node_id)?;
                }
                self.insert_connections_for_node(new_node_id, &new_connections)?;
                entry = new_node_id;
            }
        }
        Ok(())
    }

    /// Bulk-creates the index from a list of vectors (and optional metadata).
    pub fn create<R: Rng>(
        &mut self,
        vectors: Vec<&[f64]>,
        metadatas: Option<&[&[Metadata]]>,
        efc: Option<usize>,
        rng: &mut R,
    ) -> Result<(), HNSWError> {
        if vectors.is_empty() {
            return Err(HNSWError::EmptyVectors);
        }
        if let (Some(meta_list), Some(_)) = (metadatas, &self.schema) {
            if meta_list.len() != vectors.len() {
                return Err(HNSWError::AttributeNotFound);
            }
            for (vec, meta) in vectors.iter().zip(meta_list.iter()) {
                self.insert(vec, Some(meta), efc, rng)?;
            }
            return Ok(());
        }
        for vec in vectors {
            self.insert(vec, None, efc, rng)?;
        }
        Ok(())
    }

    /// Samples a level for a new node using the precomputed assignment probabilities.
    fn get_insert_layer<R: Rng>(&self, rng: &mut R) -> usize {
        let total: f64 = self.assignment_probabilities.iter().sum();
        let r_val = rng.random::<f64>() * total;
        let mut cumulative = 0.0;
        for (i, &p) in self.assignment_probabilities.iter().enumerate() {
            cumulative += p;
            if cumulative > r_val {
                return i;
            }
        }
        self.assignment_probabilities.len() - 1
    }

    /// Adds a neighbor connection from `node_id` to `neighbor`.
    fn add_connection(&mut self, node_id: usize, neighbor: usize) -> Result<(), HNSWError> {
        if node_id >= self.offsets.len() {
            return Err(HNSWError::InvalidNodeID);
        }
        let end = if node_id + 1 < self.offsets.len() {
            self.offsets[node_id + 1]
        } else {
            self.connections.len()
        };
        self.connections.insert(end, neighbor);
        for i in (node_id + 1)..self.offsets.len() {
            self.offsets[i] += 1;
        }
        Ok(())
    }

    /// Inserts the connection list for a new node.
    fn insert_connections_for_node(
        &mut self,
        node_id: usize,
        connections: &[usize],
    ) -> Result<(), HNSWError> {
        if node_id != self.vectors.len() / self.dim - 1 {
            return Err(HNSWError::InvalidNodeID);
        }
        let start = self.connections.len();
        self.offsets.push(start);
        self.connections.extend_from_slice(connections);
        Ok(())
    }

    /// Finds the nearest neighbor to `vector` and marks it as deleted.
    pub fn delete_nearest(
        &mut self,
        vector: &[f64],
        filter: Option<&Filter>,
    ) -> Result<(), HNSWError> {
        let results = self.search(vector, Some(1), filter)?;
        if !results.is_empty() {
            self.delete(results[0].id);
        }
        Ok(())
    }

    /// Marks a node as deleted.
    pub fn delete(&mut self, id: usize) {
        if id < self.tombstones.len() {
            self.tombstones[id] = true;
        }
    }

    /// Cleans the index by removing tombstoned nodes and re-mapping IDs.
    pub fn clean(&mut self) -> Result<(), HNSWError> {
        let old_count = self.vectors.len() / self.dim;
        let mut new_vectors = Vec::new();
        let mut new_levels = Vec::new();
        let mut new_tombstones = Vec::new();
        let mut new_offsets = Vec::new();
        let mut new_connections = Vec::new();
        let mut mapping = Vec::with_capacity(old_count);

        let mut new_id = 0;
        for i in 0..old_count {
            if !self.tombstones[i] {
                mapping.push(new_id);
                new_vectors.extend_from_slice(&self.vectors[i * self.dim..(i + 1) * self.dim]);
                new_levels.push(self.levels[i]);
                new_tombstones.push(false);
                new_id += 1;
            } else {
                mapping.push(usize::MAX);
            }
        }

        let mut current_offset = 0;
        for i in 0..old_count {
            if self.tombstones[i] {
                continue;
            }
            new_offsets.push(current_offset);
            let start = self.offsets.get(i).copied().unwrap_or(0);
            let end = if i + 1 < self.offsets.len() {
                self.offsets[i + 1]
            } else {
                self.connections.len()
            };
            for j in start..end {
                let old_neighbor = self.connections[j];
                if old_neighbor >= mapping.len() {
                    continue;
                }
                let new_neighbor = mapping[old_neighbor];
                if new_neighbor == usize::MAX {
                    continue;
                }
                new_connections.push(new_neighbor);
                current_offset += 1;
            }
        }
        self.vectors = new_vectors;
        self.levels = new_levels;
        self.tombstones = new_tombstones;
        self.offsets = new_offsets;
        self.connections = new_connections;
        Ok(())
    }

    /// Returns the index of an attribute in the schema.
    fn attribute_index(&self, name: &str) -> Result<usize, HNSWError> {
        if self.schema.is_none() {
            return Err(HNSWError::NoSchema);
        }
        let schema = self.schema.as_ref().unwrap();
        if schema.is_empty() {
            return Err(HNSWError::EmptySchema);
        }
        for (i, attribute) in schema.iter().enumerate() {
            if attribute == name {
                return Ok(i);
            }
        }
        Err(HNSWError::AttributeNotFound)
    }

    /// Updates the metadata for a given node and attribute.
    pub fn update_metadata(
        &mut self,
        id: usize,
        name: &str,
        value: Metadata,
    ) -> Result<(), HNSWError> {
        let index = self.attribute_index(name)?;
        let pos = id + index;
        if pos < self.metadata.len() {
            self.metadata[pos] = value;
            Ok(())
        } else {
            Err(HNSWError::InvalidNodeID)
        }
    }

    /// Euclidean distance.
    fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..a.len().min(b.len()) {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }

    /// Dot product.
    fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..a.len().min(b.len()) {
            sum += a[i] * b[i];
        }
        sum
    }

    /// Cosine similarity.
    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot = Self::dot_product(a, b);
        let norm_a = Self::dot_product(a, a).sqrt();
        let norm_b = Self::dot_product(b, b).sqrt();
        dot / (norm_a * norm_b)
    }

    /// Searches for k nearest neighbors to `query`.
    pub fn search(
        &self,
        query: &[f64],
        k: Option<usize>,
        filter: Option<&Filter>,
    ) -> Result<Vec<Candidate>, HNSWError> {
        let k = k.unwrap_or(1);
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }
        let mut current_layer = self.layers - 1;
        let mut entry = 0;
        let vector_entry = &self.vectors[entry * self.dim..(entry + 1) * self.dim];
        let mut candidate = Candidate {
            distance: OrderedFloat::from((self.metric)(query, vector_entry)),
            id: entry,
        };

        while current_layer > 0 {
            let layer_results = self.knn(entry, query, 1, current_layer, filter)?;
            if layer_results.is_empty() {
                break;
            }
            candidate = layer_results[0].clone();
            if self.tombstones[candidate.id] {
                break;
            }
            entry = candidate.id;
            current_layer -= 1;
        }

        let bottom_results = self.knn(entry, query, self.ef_search, 0, filter)?;
        let mut trimmed = Vec::new();
        let mut count = 0;
        for c in bottom_results {
            if !self.tombstones[c.id] {
                trimmed.push(c);
                count += 1;
                if count >= k {
                    break;
                }
            }
        }
        Ok(trimmed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_hnsw_basic_initialization() {
        let hnsw1 = HNSW::new(2, None, None, None, None, None, None);
        assert_eq!(hnsw1.layers, 5);
        assert_eq!(hnsw1.m, 16);
        assert_eq!(hnsw1.ef_construction, 200);
        assert_eq!(hnsw1.ef_search, 50);

        let hnsw2 = HNSW::new(2, Some(3), Some(8), Some(100), Some(10), None, None);
        assert_eq!(hnsw2.layers, 3);
        assert_eq!(hnsw2.m, 8);
        assert_eq!(hnsw2.ef_construction, 100);
        assert_eq!(hnsw2.ef_search, 10);
    }

    #[test]
    fn test_hnsw_insert_and_search() {
        let mut hnsw = HNSW::new(2, None, None, None, None, None, None);
        let seed: u64 = 42;
        let mut rng = StdRng::seed_from_u64(seed);

        // Points as an array of [2]f64.
        let points: &[[f64; 2]] = &[
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [0.0, 0.0],
        ];

        for pt in points {
            let vec = pt.to_vec();
            hnsw.insert(&vec, None, None, &mut rng).unwrap();
        }
        assert_eq!(hnsw.vectors.len() / hnsw.dim, 5);

        let query = [0.1, 0.0].to_vec();
        let results = hnsw.search(&query, Some(2), None).unwrap();
        assert!(results.len() >= 1);
        let best_candidate = &results[0];
        // Expect that the center [0.0, 0.0] is stored last (ID 4).
        assert_eq!(best_candidate.id, 4);
    }

    #[test]
    fn test_hnsw_delete_nearest_and_tombstone() {
        let mut hnsw = HNSW::new(2, None, None, None, None, None, None);
        let seed: u64 = 1234;
        let mut rng = StdRng::seed_from_u64(seed);

        let points: &[[f64; 2]] = &[
            [0.0, 0.0],   // ID 0
            [1.0, 1.0],   // ID 1
            [-1.0, -1.0], // ID 2
        ];

        for pt in points {
            let vec = pt.to_vec();
            hnsw.insert(&vec, None, None, &mut rng).unwrap();
        }
        assert_eq!(hnsw.vectors.len() / hnsw.dim, 3);

        let query = [1.0, 1.1].to_vec();
        let results = hnsw.search(&query, Some(1), None).unwrap();
        assert_eq!(results[0].id, 1);

        let query_del = [1.0, 1.0].to_vec();
        hnsw.delete_nearest(&query_del, None).unwrap();

        let query2 = [1.0, 1.1].to_vec();
        let results2 = hnsw.search(&query2, Some(1), None).unwrap();
        assert_ne!(results2[0].id, 1);
    }

    #[test]
    fn test_hnsw_clean_tombstoned_nodes() {
        let mut hnsw = HNSW::new(2, None, None, None, None, None, None);
        let seed: u64 = 9876;
        let mut rng = StdRng::seed_from_u64(seed);

        let points: &[[f64; 2]] = &[[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]];

        for pt in points {
            let vec = pt.to_vec();
            hnsw.insert(&vec, None, None, &mut rng).unwrap();
        }
        assert_eq!(hnsw.vectors.len() / hnsw.dim, 4);

        // Tombstone two nodes.
        hnsw.delete(1);
        hnsw.delete(2);
        assert!(hnsw.tombstones[1]);
        assert!(hnsw.tombstones[2]);

        hnsw.clean().unwrap();
        // After cleaning, only 2 active vectors remain.
        assert_eq!(hnsw.vectors.len() / hnsw.dim, 2);
        for flag in &hnsw.tombstones {
            assert!(!*flag);
        }
    }

    #[test]
    fn test_hnsw_distance_metrics() {
        let mut hnsw = HNSW::new(2, None, None, None, None, None, None);
        let seed: u64 = 1234;
        let mut rng = StdRng::seed_from_u64(seed);

        let points: &[[f64; 2]] = &[
            [1.0, 0.0],     // ID 0
            [-0.5, 0.866],  // ID 1
            [-0.5, -0.866], // ID 2
        ];

        for pt in points {
            let vec = pt.to_vec();
            hnsw.insert(&vec, None, None, &mut rng).unwrap();
        }
        assert_eq!(hnsw.vectors.len() / hnsw.dim, 3);

        // Euclidean distances
        for i in 0..3 {
            for j in (i + 1)..3 {
                let a = &hnsw.vectors[i * hnsw.dim..(i + 1) * hnsw.dim];
                let b = &hnsw.vectors[j * hnsw.dim..(j + 1) * hnsw.dim];
                let dist = HNSW::euclidean_distance(a, b);
                // Expect approximately 1.732
                assert!((dist - 1.732).abs() < 0.01);
            }
            let a = &hnsw.vectors[i * hnsw.dim..(i + 1) * hnsw.dim];
            let dist = HNSW::euclidean_distance(a, a);
            assert!((dist - 0.0).abs() < 0.001);
        }

        // Cosine similarity
        for i in 0..3 {
            for j in (i + 1)..3 {
                let a = &hnsw.vectors[i * hnsw.dim..(i + 1) * hnsw.dim];
                let b = &hnsw.vectors[j * hnsw.dim..(j + 1) * hnsw.dim];
                let sim = HNSW::cosine_similarity(a, b);
                assert!((sim + 0.5).abs() < 0.1);
            }
            let a = &hnsw.vectors[i * hnsw.dim..(i + 1) * hnsw.dim];
            let sim = HNSW::cosine_similarity(a, a);
            assert!((sim - 1.0).abs() < 0.001);
        }

        // Dot product
        for i in 0..3 {
            for j in (i + 1)..3 {
                let a = &hnsw.vectors[i * hnsw.dim..(i + 1) * hnsw.dim];
                let b = &hnsw.vectors[j * hnsw.dim..(j + 1) * hnsw.dim];
                let dot = HNSW::dot_product(a, b);
                assert!((dot + 0.5).abs() < 0.1);
            }
            let a = &hnsw.vectors[i * hnsw.dim..(i + 1) * hnsw.dim];
            let dot = HNSW::dot_product(a, a);
            assert!((dot - 1.0).abs() < 0.001);
        }
    }

    // A helper to build a simple color filter.
    fn color_filter(color: &str) -> Filter {
        let color_owned = color.to_owned();
        let condition: Arc<dyn Fn(&Metadata) -> bool + Send + Sync + 'static> =
            Arc::new(move |value: &Metadata| -> bool {
                if let Metadata::Str(ref s) = value {
                    s == &color_owned
                } else {
                    false
                }
            });
        Filter {
            attribute: "color".to_string(),
            condition,
        }
    }

    #[test]
    fn test_metadata_basic_insert_and_filter() {
        let schema = vec!["color".to_string()];
        let mut hnsw = HNSW::new(2, None, None, None, None, None, Some(schema));
        let seed: u64 = 1234;
        let mut rng = StdRng::seed_from_u64(seed);

        // Insert vector [1.0, 2.0] with color "blue"
        {
            let meta_a = vec![Metadata::Str("blue".to_string())];
            let vec_a = [1.0f64, 2.0].to_vec();
            hnsw.insert(&vec_a, Some(&meta_a), None, &mut rng).unwrap();
        }
        // Insert vector [2.0, 3.0] with color "red"
        {
            let meta_b = vec![Metadata::Str("red".to_string())];
            let vec_b = [2.0f64, 3.0].to_vec();
            hnsw.insert(&vec_b, Some(&meta_b), None, &mut rng).unwrap();
        }
        // Insert vector [10.0, 10.0] with color "blue"
        {
            let meta_c = vec![Metadata::Str("blue".to_string())];
            let vec_c = [10.0f64, 10.0].to_vec();
            hnsw.insert(&vec_c, Some(&meta_c), None, &mut rng).unwrap();
        }

        let blue_filter = color_filter("blue");
        let query = [1.5f64, 2.5].to_vec();
        let results = hnsw.search(&query, Some(10), Some(&blue_filter)).unwrap();

        let mut seen_id_0 = false;
        let mut seen_id_1 = false;
        let mut seen_id_2 = false;
        for candidate in results {
            if candidate.id == 0 {
                seen_id_0 = true;
            }
            if candidate.id == 1 {
                seen_id_1 = true;
            }
            if candidate.id == 2 {
                seen_id_2 = true;
            }
        }
        assert!(seen_id_0);
        assert!(!seen_id_1);
        assert!(seen_id_2);
    }

    #[test]
    fn test_metadata_no_schema_should_fail_with_filter() {
        let mut hnsw = HNSW::new(2, None, None, None, None, None, None);
        let seed: u64 = 5678;
        let mut rng = StdRng::seed_from_u64(seed);
        let vec = [0.0f64, 0.0].to_vec();
        hnsw.insert(&vec, None, None, &mut rng).unwrap();
        let blue_filter = color_filter("blue");
        let result = hnsw.search(&vec, Some(1), Some(&blue_filter));
        assert!(result.is_err());
    }

    #[test]
    fn test_metadata_attribute_not_found() {
        let schema = vec!["color".to_string()];
        let mut hnsw = HNSW::new(2, None, None, None, None, None, Some(schema));
        let seed: u64 = 555;
        let mut rng = StdRng::seed_from_u64(seed);
        let meta = vec![Metadata::Str("blue".to_string())];
        let vec = [5.0f64, 5.0].to_vec();
        hnsw.insert(&vec, Some(&meta), None, &mut rng).unwrap();

        // Define a filter for a nonexistent attribute "category".
        let filter = Filter {
            attribute: "category".to_string(),
            condition: Arc::new(|_value: &Metadata| true),
        };
        let result = hnsw.search(&vec, Some(1), Some(&filter));
        assert!(result.is_err());
    }
}

/// Our Python-facing HNSW type. It wraps HNSW specialized to f64.
#[pyclass]
pub struct PyHNSW {
    inner: HNSW,
}

#[pymethods]
impl PyHNSW {
    /// Create a new HNSW index.
    ///
    /// - `dim`: dimension of the vectors.
    /// - `layers`, `m`, `ef_construction`, `ef_search` are optional.
    /// - `metric`: an optional string ("l2", "cosine", "inner_product") defaults to "l2".
    #[new]
    pub fn new(
        dim: usize,
        layers: Option<usize>,
        m: Option<usize>,
        ef_construction: Option<usize>,
        ef_search: Option<usize>,
        metric: Option<String>,
    ) -> PyResult<Self> {
        let metric_enum = match metric.as_deref() {
            Some("cosine") => Some(Metric::Cosine),
            Some("inner_product") => Some(Metric::InnerProduct),
            _ => Some(Metric::L2),
        };
        Ok(PyHNSW {
            inner: HNSW::new(
                dim,
                layers,
                m,
                ef_construction,
                ef_search,
                metric_enum,
                None,
            ),
        })
    }

    /// Insert a vector into the index.
    ///
    /// The vector is expected as a list of floats.
    pub fn insert(&mut self, vector: Vec<f64>) -> PyResult<()> {
        self.inner
            .insert(&vector, None, None, &mut rand::rng())
            .map_err(|e| PyValueError::new_err(format!("Insert error: {:?}", e)))
    }

    /// Search for the k nearest neighbors.
    ///
    /// Returns a list of tuples `(id, distance)`.
    pub fn search(&self, query: Vec<f64>, k: Option<usize>) -> PyResult<Vec<(f64, Vec<f64>)>> {
        let k = k.unwrap_or(1);
        self.inner
            .search(&query, Some(k), None)
            .map(|candidates| {
                candidates
                    .into_iter()
                    .map(|c| (c.distance.into_inner(), self.inner.vector_from_id(c.id)))
                    .collect()
            })
            .map_err(|e| PyValueError::new_err(format!("Search error: {:?}", e)))
    }

    /// Create an index from a list of vectors.
    pub fn create(&mut self, vectors: Vec<Vec<f64>>) -> PyResult<()> {
        let vecs: Vec<&[f64]> = vectors.iter().map(|v| v.as_slice()).collect();
        self.inner
            .create(vecs, None, None, &mut rand::rng())
            .map_err(|e| PyValueError::new_err(format!("Create error: {:?}", e)))
    }
}
