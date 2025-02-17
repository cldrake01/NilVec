use crate::candidate::Candidate;
use crate::filter::Filter;
use crate::metadata::Metadata;
use crate::metric::Metric;
use ordered_float::OrderedFloat;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Errors for Flat index operations.
#[derive(Debug)]
pub enum FlatError {
    EmptyIndex,
    EmptyVectors,
    MetadataMismatch,
    NoSchema,
    EmptySchema,
    AttributeNotFound,
}

impl std::fmt::Display for FlatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for FlatError {}

/// The Flat index.
/// - Vectors are stored flat in a Vec<f64> (each vector occupies `dim` consecutive values).
/// - A parallel Vec<bool> tracks tombstones.
/// - A metric function is stored to compute distances.
/// - Optionally, a schema (a Vec<String>) and metadata are provided.
pub struct Flat {
    pub dim: usize,
    pub vectors: Vec<f64>,
    pub tombstones: Vec<bool>,
    pub metric: fn(&[f64], &[f64]) -> f64,
    pub schema: Option<Vec<String>>,
    pub metadata: Vec<Metadata>,
    pub version: AtomicUsize,
}

impl Flat {
    /// Creates a new Flat index.
    /// The optional `metric` selects a distance function (default: Euclidean).
    /// The optional `schema` can be provided if you wish to attach metadata.
    pub fn new(dim: usize, metric: Option<Metric>, schema: Option<Vec<String>>) -> Self {
        let metric_fn = match metric {
            Some(Metric::Cosine) => Flat::cosine_similarity,
            Some(Metric::InnerProduct) => Flat::dot_product,
            _ => Flat::euclidean_distance,
        };
        Self {
            dim,
            vectors: Vec::new(),
            tombstones: Vec::new(),
            metric: metric_fn,
            schema,
            metadata: Vec::new(),
            version: AtomicUsize::new(0),
        }
    }

    fn vector_from_id(&self, id: usize) -> Vec<f64> {
        self.vectors[id * self.dim..(id + 1) * self.dim].to_vec()
    }

    /// Searches for the k nearest neighbors to `query` (a slice of f64 of length `dim`).
    /// If a filter is provided, only candidates whose metadata passes the condition are returned.
    pub fn search(
        &self,
        query: &[f64],
        k: Option<usize>,
        filter: Option<&Filter>,
    ) -> Result<Vec<Candidate>, FlatError> {
        let k_ = k.unwrap_or(1);
        let num_vectors = self.vectors.len() / self.dim;
        if num_vectors == 0 {
            return Err(FlatError::EmptyIndex);
        }

        // Compute all candidates in parallel.
        let mut candidates: Vec<Candidate> = (0..num_vectors)
            .into_par_iter()
            .map(|i| {
                let vector = &self.vectors[i * self.dim..(i + 1) * self.dim];
                let d = (self.metric)(query, vector);
                Candidate {
                    distance: OrderedFloat(d),
                    id: i,
                }
            })
            .collect();

        // Sort candidates by distance (lowest first).
        candidates.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        // Sequentially filter candidates with error handling.
        let mut count = 0;
        let mut results = Vec::new();
        let meta_count = if let Some(ref schema) = self.schema {
            schema.len()
        } else {
            0
        };

        let mut iter = candidates.into_iter();
        while let Some(candidate) = iter.next() {
            if let Some(filter) = filter {
                let attr_index = self.attribute_index(&filter.attribute)?;
                if meta_count == 0 {
                    return Err(FlatError::EmptySchema);
                }
                let meta_idx = candidate.id * meta_count + attr_index;
                if meta_idx >= self.metadata.len() {
                    return Err(FlatError::AttributeNotFound);
                }
                let meta = &self.metadata[meta_idx];
                if !(filter.condition)(meta) {
                    continue;
                }
            }
            if !self.tombstones[candidate.id] {
                results.push(candidate);
                count += 1;
                if count >= k_ {
                    break;
                }
            }
        }
        Ok(results)
    }

    /// Inserts a new vector into the index.
    /// The provided `vector` must have length equal to `dim`.
    /// Optionally, a slice of Metadata (one per attribute in the schema) may be provided.
    pub fn insert(
        &mut self,
        vector: &[f64],
        metadata: Option<&[Metadata]>,
    ) -> Result<(), FlatError> {
        if vector.len() != self.dim {
            panic!(
                "Vector length {} does not match index dimension {}",
                vector.len(),
                self.dim
            );
        }
        self.vectors.extend_from_slice(vector);
        self.tombstones.push(false);
        // self.version.fetch_add(1, Ordering::Release);
        if let Some(meta) = metadata {
            self.metadata.extend_from_slice(meta);
        }
        Ok(())
    }

    /// Builds the index from a set of vectors.
    /// If metadata is provided, there must be one slice of Metadata per vector.
    pub fn create(
        &mut self,
        vectors: &[&[f64]],
        metadata: Option<&[&[Metadata]]>,
    ) -> Result<(), FlatError> {
        if vectors.is_empty() {
            return Err(FlatError::EmptyVectors);
        }
        if let (Some(meta_list), Some(_)) = (metadata, &self.schema) {
            if meta_list.len() != vectors.len() {
                return Err(FlatError::MetadataMismatch);
            }
            for (&vec, &meta) in vectors.iter().zip(meta_list.iter()) {
                self.insert(vec, Some(meta))?;
            }
            return Ok(());
        }
        for &vec in vectors {
            self.insert(vec, None)?;
        }
        Ok(())
    }

    /// Finds the nearest neighbor to `query` and marks it as deleted.
    pub fn delete_nearest(&mut self, query: &[f64]) -> Result<(), FlatError> {
        let nearest = self.search(query, Some(1), None)?;
        if !nearest.is_empty() {
            self.delete(nearest[0].id);
        }
        Ok(())
    }

    /// Marks a node (vector) as deleted.
    pub fn delete(&mut self, id: usize) {
        if id < self.tombstones.len() {
            self.tombstones[id] = true;
        }
    }

    /// Cleans the index by removing all tombstoned nodes.
    /// This moves surviving vectors (and their associated metadata) into a contiguous block.
    pub fn clean(&mut self) -> Result<(), FlatError> {
        let meta_count = if let Some(ref schema) = self.schema {
            schema.len()
        } else {
            0
        };
        let num_vectors = self.vectors.len() / self.dim;
        let mut write_index = 0;
        for i in 0..num_vectors {
            if !self.tombstones[i] {
                if write_index != i {
                    // Move vector i to the write_index position.
                    let src_range = i * self.dim..(i + 1) * self.dim;
                    let dst_start = write_index * self.dim;
                    self.vectors.copy_within(src_range, dst_start);
                    self.tombstones[write_index] = false;
                    if self.schema.is_some() && meta_count > 0 {
                        let src_start = i * meta_count;
                        let dst_start = write_index * meta_count;
                        for j in 0..meta_count {
                            self.metadata[dst_start + j] = self.metadata[src_start + j].clone();
                        }
                    }
                }
                write_index += 1;
            }
        }
        self.vectors.truncate(write_index * self.dim);
        self.tombstones.truncate(write_index);
        if self.schema.is_some() && meta_count > 0 {
            self.metadata.truncate(write_index * meta_count);
        }
        Ok(())
    }

    /// Returns the index of the given attribute (provided as a string) within the schema.
    pub fn attribute_index(&self, name: &str) -> Result<usize, FlatError> {
        let schema = self.schema.as_ref().ok_or(FlatError::NoSchema)?;
        if schema.is_empty() {
            return Err(FlatError::EmptySchema);
        }
        for (i, attr) in schema.iter().enumerate() {
            if attr == name {
                return Ok(i);
            }
        }
        Err(FlatError::AttributeNotFound)
    }

    // --- Distance Functions ---

    /// Euclidean distance.
    pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        let mut sum = 0.0;
        for i in 0..len {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }

    /// Dot product.
    pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        let mut sum = 0.0;
        for i in 0..len {
            sum += a[i] * b[i];
        }
        sum
    }

    /// Cosine similarity.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot = Flat::dot_product(a, b);
        let norm_a = Flat::dot_product(a, a).sqrt();
        let norm_b = Flat::dot_product(b, b).sqrt();
        dot / (norm_a * norm_b)
    }
}

//
// Tests
//
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_basic_initialization() {
        let index = Flat::new(2, None, None);
        assert_eq!(index.vectors.len(), 0);
        assert_eq!(index.tombstones.len(), 0);
    }

    #[test]
    fn test_flat_insert_and_search() {
        let mut index = Flat::new(2, None, None);
        let points = vec![
            vec![-1.0, -1.0],
            vec![-1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, -1.0],
            vec![0.0, 0.0],
        ];
        for pt in &points {
            index.insert(pt, None).unwrap();
        }
        // There should be 5 vectors × 2 dimensions = 10 numbers stored.
        assert_eq!(index.vectors.len(), 10);

        let query = vec![0.1, 0.1];
        let results = index.search(&query, Some(2), None).unwrap();
        assert!(!results.is_empty());
        // Expect that the center (vector [0.0, 0.0]) is the nearest (ID=4).
        assert_eq!(results[0].id, 4);
    }

    #[test]
    fn test_flat_delete_nearest_and_tombstone() {
        let mut index = Flat::new(2, None, None);
        let points = vec![
            vec![0.0, 0.0],   // ID=0
            vec![1.0, 1.0],   // ID=1
            vec![-1.0, -1.0], // ID=2
        ];
        for pt in &points {
            index.insert(pt, None).unwrap();
        }
        assert_eq!(index.vectors.len(), 6);

        let query = vec![1.0, 1.0];
        let results = index.search(&query, Some(1), None).unwrap();
        assert_eq!(results[0].id, 1);

        index.delete_nearest(&query).unwrap();
        assert!(index.tombstones[1]);

        let results_after = index.search(&query, Some(1), None).unwrap();
        assert!(!results_after.is_empty());
        assert_ne!(results_after[0].id, 1);
    }

    #[test]
    fn test_flat_clean_tombstoned_nodes() {
        let mut index = Flat::new(2, None, None);
        let points = vec![
            vec![2.0, 2.0],   // ID=0
            vec![-2.0, 2.0],  // ID=1
            vec![-2.0, -2.0], // ID=2
            vec![2.0, -2.0],  // ID=3
        ];
        for pt in &points {
            index.insert(pt, None).unwrap();
        }
        // Tombstone vectors with IDs 1 and 2.
        index.tombstones[1] = true;
        index.tombstones[2] = true;
        index.clean().unwrap();
        // Now only 2 vectors should remain (2 × 2 = 4 numbers).
        assert_eq!(index.vectors.len(), 4);
        for flag in &index.tombstones {
            assert!(!*flag);
        }
    }

    #[test]
    fn test_flat_distance_functions() {
        let a = vec![1.0, 0.0];
        let b = vec![-0.5, 0.866];
        let c = vec![-0.5, -0.866];

        let d_ab = Flat::euclidean_distance(&a, &b);
        let d_ac = Flat::euclidean_distance(&a, &c);
        let d_bc = Flat::euclidean_distance(&b, &c);
        // For an equilateral triangle, side ≈ 1.732
        assert!((d_ab - 1.732).abs() < 1e-3);
        assert!((d_ac - 1.732).abs() < 1e-3);
        assert!((d_bc - 1.732).abs() < 1e-3);
        assert!((Flat::euclidean_distance(&a, &a) - 0.0).abs() < 1e-12);

        let dot_ab = Flat::dot_product(&a, &b);
        assert!((dot_ab + 0.5).abs() < 1e-3);

        let sim_ab = Flat::cosine_similarity(&a, &b);
        assert!((sim_ab + 0.5).abs() < 1e-3);

        let sim_aa = Flat::cosine_similarity(&a, &a);
        assert!((sim_aa - 1.0).abs() < 1e-12);
    }

    /// A helper that builds a filter to match a metadata “color” against a target.
    fn color_filter(color: &str) -> Filter {
        let color_owned = color.to_owned();
        Filter {
            attribute: "color".to_string(),
            condition: Arc::new(move |value: &Metadata| {
                if let Metadata::Str(ref s) = value {
                    s == &color_owned
                } else {
                    false
                }
            }),
        }
    }
    #[test]
    fn test_flat_metadata_and_filter() {
        // Create a schema with one attribute: "color".
        let schema = vec!["color".to_string()];
        let mut index = Flat::new(2, Some(Metric::Cosine), Some(schema));
        {
            let meta_blue = vec![Metadata::Str("blue".to_string())];
            index
                .insert(&[0.0, 0.0], Some(meta_blue.as_slice()))
                .unwrap();
        }
        {
            let meta_red = vec![Metadata::Str("red".to_string())];
            index
                .insert(&[10.0, 10.0], Some(meta_red.as_slice()))
                .unwrap();
        }
        let blue_filter = color_filter("blue");
        let results = index
            .search(&[0.1, 0.0], Some(2), Some(&blue_filter))
            .unwrap();
        let mut found_blue = false;
        let mut found_red = false;
        for candidate in results {
            if candidate.id == 0 {
                found_blue = true;
            }
            if candidate.id == 1 {
                found_red = true;
            }
        }
        assert!(found_blue);
        assert!(!found_red);
    }

    #[test]
    fn test_flat_no_schema() {
        let mut index = Flat::new(2, Some(Metric::L2), None);
        index.insert(&[1.0, 2.0], None).unwrap();
        let blue_filter = color_filter("blue");
        let result = index.search(&[1.0, 2.0], Some(1), Some(&blue_filter));
        assert!(result.is_err());
    }

    #[test]
    fn test_flat_attribute_not_found() {
        // Suppose our schema only has "color".
        let schema = vec!["color".to_string()];
        let mut index = Flat::new(2, Some(Metric::L2), Some(schema));
        {
            let meta = vec![Metadata::Str("blue".to_string())];
            index.insert(&[5.0, 5.0], Some(meta.as_slice())).unwrap();
        }
        // Define a filter for a non-existent attribute "category".
        let missing_attr_filter = Filter {
            attribute: "category".to_string(),
            condition: Arc::new(|_value: &Metadata| true),
        };
        let result = index.search(&[5.0, 5.0], Some(1), Some(&missing_attr_filter));
        assert!(result.is_err());
    }
}

/// Our Python-facing Flat index.
#[pyclass]
pub struct PyFlat {
    inner: Flat,
}

#[pymethods]
impl PyFlat {
    /// Create a new Flat index.
    ///
    /// Parameters:
    /// - `dim`: dimension of each vector.
    /// - `metric`: an optional string ("l2", "cosine", "inner_product") defaulting to "l2".
    /// - `schema`: an optional list of strings representing the schema of the vectors.
    #[new]
    pub fn new(dim: usize, metric: Option<String>, schema: Option<Vec<String>>) -> PyResult<Self> {
        let metric_enum = match metric.as_deref() {
            Some("cosine") => Some(Metric::Cosine),
            Some("inner_product") => Some(Metric::InnerProduct),
            Some("l2") => Some(Metric::L2),
            None => Some(Metric::L2),
            _ => return Err(PyValueError::new_err("Unsupported metric.")),
        };
        Ok(PyFlat {
            inner: Flat::new(dim, metric_enum, schema),
        })
    }

    /// Insert a vector into the index.
    ///
    /// The vector is expected as a list of floats whose length equals the index dimension.
    pub fn insert(
        &mut self,
        vector: Vec<f64>,
        metadata: Option<Vec<(String, PyObject)>>,
        py: Python,
    ) -> PyResult<()> {
        // If metadata is provided, convert each tuple into a Metadata value.
        let meta_converted = if let Some(meta_tuples) = metadata {
            // Ensure a schema was set when the index was created.
            let schema = self.inner.schema.as_ref().ok_or_else(|| {
                PyValueError::new_err("No schema provided in HNSW instance; cannot use metadata")
            })?;
            // Build a map from attribute name to PyObject.
            let meta_map: HashMap<String, PyObject> = meta_tuples.into_iter().collect();
            // Create a vector of Metadata in the order defined by the schema.
            let mut meta_vec = Vec::with_capacity(schema.len());
            for attr in schema {
                if let Some(py_val) = meta_map.get(attr) {
                    // Directly extract the value from the PyObject.
                    let md = if let Ok(s) = py_val.extract::<String>(py) {
                        Metadata::Str(s)
                    } else if let Ok(i) = py_val.extract::<i64>(py) {
                        Metadata::Int(i)
                    } else if let Ok(f) = py_val.extract::<f64>(py) {
                        Metadata::Float(f)
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "Unsupported metadata type for attribute '{}'",
                            attr
                        )));
                    };
                    meta_vec.push(md);
                } else {
                    return Err(PyValueError::new_err(format!(
                        "Missing metadata for attribute '{}'",
                        attr
                    )));
                }
            }
            Some(meta_vec)
        } else {
            None
        };

        self.inner
            .insert(&vector, meta_converted.as_deref())
            .map_err(|e| PyValueError::new_err(format!("Insert error: {:?}", e)))
    }

    /// Search for the k nearest neighbors.
    ///
    /// Returns a list of tuples `(id, distance)`.
    #[pyo3(signature = (query, k=None, filter=None))]
    pub fn search(
        &self,
        query: Vec<f64>,
        k: Option<usize>,
        filter: Option<(String, PyObject)>,
        py: Python,
    ) -> PyResult<Vec<(f64, Vec<f64>)>> {
        let k = k.unwrap_or(1);
        let candidates = if let Some((attribute, expected_value)) = filter {
            // Ensure that a schema was set when the index was created.
            if self.inner.schema.is_none() {
                return Err(PyValueError::new_err(
                    "No schema provided in Flat instance; cannot use metadata filtering",
                ));
            }
            // Convert the expected value from Python into a Metadata value.
            let expected_md = if let Ok(s) = expected_value.extract::<String>(py) {
                Metadata::Str(s)
            } else if let Ok(i) = expected_value.extract::<i64>(py) {
                Metadata::Int(i)
            } else if let Ok(f) = expected_value.extract::<f64>(py) {
                Metadata::Float(f)
            } else {
                return Err(PyValueError::new_err(
                    "Unsupported type for metadata filter",
                ));
            };

            // Create a Filter that checks for equality.
            let filter_obj = Filter {
                attribute,
                condition: Arc::new(move |meta: &Metadata| meta == &expected_md),
            };
            self.inner
                .search(&query, Some(k), Some(&filter_obj))
                .map_err(|e| PyValueError::new_err(format!("Search error: {:?}", e)))?
        } else {
            self.inner
                .search(&query, Some(k), None)
                .map_err(|e| PyValueError::new_err(format!("Search error: {:?}", e)))?
        };

        Ok(candidates
            .into_iter()
            .map(|c| (c.distance.into_inner(), self.inner.vector_from_id(c.id)))
            .collect())
    }

    /// Create an index from a list of vectors.
    pub fn create(
        &mut self,
        vectors: Vec<Vec<f64>>,
        metadata: Option<Vec<Vec<(String, PyObject)>>>,
        py: Python,
    ) -> PyResult<()> {
        // Convert vectors into slices.
        let vecs: Vec<&[f64]> = vectors.iter().map(|v| v.as_slice()).collect();

        if let Some(meta_lists) = metadata {
            if meta_lists.len() != vectors.len() {
                return Err(PyValueError::new_err(
                    "Length of metadata list must match number of vectors",
                ));
            }
            // Get the schema from the inner HNSW.
            let schema = self.inner.schema.as_ref().ok_or_else(|| {
                PyValueError::new_err("No schema provided in HNSW instance; cannot use metadata")
            })?;
            // In this block, build the owned metadata vectors.
            let mut meta_vecs: Vec<Vec<Metadata>> = Vec::with_capacity(meta_lists.len());
            for meta_tuples in meta_lists {
                let meta_map: HashMap<String, PyObject> = meta_tuples.into_iter().collect();
                let mut meta_vec: Vec<Metadata> = Vec::with_capacity(schema.len());
                for attr in schema {
                    if let Some(py_val) = meta_map.get(attr) {
                        let md = if let Ok(s) = py_val.extract::<String>(py) {
                            Metadata::Str(s)
                        } else if let Ok(i) = py_val.extract::<i64>(py) {
                            Metadata::Int(i)
                        } else if let Ok(f) = py_val.extract::<f64>(py) {
                            Metadata::Float(f)
                        } else {
                            return Err(PyValueError::new_err(format!(
                                "Unsupported metadata type for attribute '{}'",
                                attr
                            )));
                        };
                        meta_vec.push(md);
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "Missing metadata for attribute '{}'",
                            attr
                        )));
                    }
                }
                meta_vecs.push(meta_vec);
            }
            // Create a vector of references that borrow from meta_vecs.
            let meta_refs: Vec<&[Metadata]> = meta_vecs.iter().map(|v| v.as_slice()).collect();
            // Call inner.create while meta_vecs (and thus meta_refs) are still alive.
            self.inner
                .create(vecs.as_slice(), Some(meta_refs.as_slice()))
                .map_err(|e| PyValueError::new_err(format!("Create error: {:?}", e)))
        } else {
            self.inner
                .create(vecs.as_slice(), None)
                .map_err(|e| PyValueError::new_err(format!("Create error: {:?}", e)))
        }
    }
}
