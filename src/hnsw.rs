use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::PyObject;
use rand::Rng;

/// Type alias for storage indices.
pub type StorageIdx = i32;

/// A candidate result from search.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub distance: f32,
    pub id: StorageIdx,
}

impl Candidate {
    pub fn new(distance: f32, id: StorageIdx) -> Self {
        Self { distance, id }
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // For candidate ordering we compare distances
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// A simple metadata type.
#[derive(Clone, Debug, PartialEq)]
pub enum Metadata {
    Str(String),
    Int(i64),
    Float(f64),
}

/// The HNSW index implemented using flat arrays.
pub struct HNSW {
    // Link structure:
    pub assign_probas: Vec<f64>,
    pub cum_nneighbor_per_level: Vec<i32>,
    pub levels: Vec<i32>,           // stored as level+1
    pub offsets: Vec<usize>,        // start index in neighbors[] for each vector
    pub neighbors: Vec<StorageIdx>, // flat array of neighbor IDs (-1 means empty)

    /// Global entry point (index into nodes); -1 if none.
    pub entry_point: StorageIdx,
    /// Maximum level in the index.
    pub max_level: i32,
    pub ef_construction: i32,
    pub ef_search: i32,
    pub check_relative_distance: bool,
    pub search_bounded_queue: bool,

    // Data vectors:
    pub vectors: Vec<f32>,
    pub dim: usize,

    // Now each vector’s metadata is stored as a vector of (attribute, Metadata) pairs.
    pub metadata: Option<Vec<Vec<(String, Metadata)>>>,
    /// Distance function.
    pub metric: fn(&[f32], &[f32]) -> f32,
}

/// The HNSW index implemented using flat arrays.
/// Note: metadata is now stored as an Option<Vec<Vec<(String, Metadata)>>>,
/// i.e. each vector may have multiple (attribute, value) pairs.

impl HNSW {
    /// Create a new empty index.
    pub fn new(
        dim: usize,
        m: i32,
        ef_construction: i32,
        ef_search: i32,
        metric: Option<fn(&[f32], &[f32]) -> f32>,
    ) -> Self {
        let mut assign_probas = Vec::new();
        let mut cum_nneighbor_per_level = Vec::new();
        cum_nneighbor_per_level.push(0);
        let mut nn = 0;
        let level_mult = 1.0_f32 / (m as f32).ln();
        let mut level = 0;
        // Adjusted threshold so that for m=16 we get ~5 levels.
        loop {
            let proba = ((-(level as f32) / level_mult).exp()) * (1.0 - (-1.0 / level_mult).exp());
            if proba < 1e-6 {
                break;
            }
            assign_probas.push(proba as f64);
            if level == 0 {
                nn += m * 2;
            } else {
                nn += m;
            }
            cum_nneighbor_per_level.push(nn);
            level += 1;
        }
        Self {
            assign_probas,
            cum_nneighbor_per_level,
            levels: Vec::new(),
            offsets: vec![0],
            neighbors: Vec::new(),
            entry_point: -1,
            max_level: -1,
            ef_construction,
            ef_search,
            check_relative_distance: true,
            search_bounded_queue: true,
            vectors: Vec::new(),
            dim,
            metadata: None, // initially no metadata
            metric: metric.unwrap_or(Self::euclidean_distance),
        }
    }

    /// Number of neighbors at a given level.
    pub fn nb_neighbors(&self, layer_no: usize) -> i32 {
        self.cum_nneighbor_per_level[layer_no + 1] - self.cum_nneighbor_per_level[layer_no]
    }

    /// Cumulative number of neighbors up to (and excluding) level_no.
    pub fn cum_nb_neighbors(&self, layer_no: usize) -> i32 {
        self.cum_nneighbor_per_level[layer_no]
    }

    /// For a given vector `no` at level `layer_no`, return (begin, end) indices into neighbors[].
    pub fn neighbor_range(&self, no: usize, layer_no: usize) -> (usize, usize) {
        let o = self.offsets[no];
        let begin = o + self.cum_nneighbor_per_level(layer_no) as usize;
        let end = o + self.cum_nb_neighbors(layer_no + 1) as usize;
        (begin, end)
    }

    fn cum_nneighbor_per_level(&self, layer_no: usize) -> i32 {
        self.cum_nneighbor_per_level[layer_no]
    }

    /// Greedy descent at level `level`.
    pub fn greedy_update_nearest(
        &self,
        query: &[f32],
        mut nearest: StorageIdx,
        mut d_nearest: f32,
        level: usize,
    ) -> (StorageIdx, f32) {
        // If the current node does not have neighbors for this level, return immediately.
        if (self.levels[nearest as usize] as usize) <= level {
            return (nearest, d_nearest);
        }
        loop {
            let (begin, end) = self.neighbor_range(nearest as usize, level);
            let mut improved = false;
            for j in begin..end {
                let candidate = self.neighbors[j];
                if candidate < 0 {
                    break;
                }
                let d = (self.metric)(query, self.get_vector(candidate as usize));
                if d < d_nearest {
                    d_nearest = d;
                    nearest = candidate;
                    improved = true;
                }
            }
            // If no improvement was found or the new candidate doesn’t have this level, stop.
            if !improved || (self.levels[nearest as usize] as usize) <= level {
                break;
            }
        }
        (nearest, d_nearest)
    }

    /// Search at base level (level 0).
    /// The optional filter closure now takes a reference to a Vec<(String, Metadata)>.
    pub fn search_base_layer(
        &self,
        query: &[f32],
        entry: StorageIdx,
        ef: usize,
        filter: Option<&dyn Fn(&Vec<(String, Metadata)>) -> bool>,
    ) -> Vec<Candidate> {
        let d_entry = (self.metric)(query, self.get_vector(entry as usize));
        let mut candidates = vec![Candidate::new(d_entry, entry)];
        let mut visited = vec![false; self.levels.len()];
        visited[entry as usize] = true;
        let mut i = 0;
        while i < candidates.len() {
            let current = &candidates[i];
            if self.check_relative_distance
                && current.distance > candidates.last().unwrap().distance
            {
                break;
            }
            let (begin, end) = self.neighbor_range(current.id as usize, 0);
            for j in begin..end {
                let neighbor = self.neighbors[j];
                if neighbor < 0 || visited[neighbor as usize] {
                    continue;
                }
                visited[neighbor as usize] = true;
                if let Some(filter_fn) = filter {
                    if let Some(meta_vecs) = &self.metadata {
                        let meta = &meta_vecs[neighbor as usize];
                        if !filter_fn(meta) {
                            continue;
                        }
                    }
                }
                let d = (self.metric)(query, self.get_vector(neighbor as usize));
                if candidates.len() < ef || d < candidates.last().unwrap().distance {
                    candidates.push(Candidate::new(d, neighbor));
                    candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
                    if candidates.len() > ef {
                        candidates.pop();
                    }
                }
            }
            i += 1;
        }
        candidates
    }

    /// Get the vector corresponding to a given id.
    pub fn get_vector(&self, id: usize) -> &[f32] {
        let start = id * self.dim;
        &self.vectors[start..start + self.dim]
    }

    /// Helper: add a link from node `src` to node `dst` at level `level`.
    pub fn add_link(&mut self, src: usize, dst: StorageIdx, level: usize) {
        let (begin, end) = self.neighbor_range(src, level);
        for i in begin..end {
            if self.neighbors[i] < 0 {
                self.neighbors[i] = dst;
                return;
            }
        }
        self.neighbors[end - 1] = dst;
    }

    /// Helper: shrink candidate list to at most max_links.
    pub fn shrink_candidate_list(
        &self,
        mut candidates: Vec<Candidate>,
        max_links: i32,
    ) -> Vec<StorageIdx> {
        candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        candidates
            .into_iter()
            .take(max_links as usize)
            .map(|c| c.id)
            .collect()
    }

    /// Full insertion procedure that builds links.
    /// The metadata parameter is now Option<Vec<(String, Metadata)>>.
    pub fn insert<R: Rng>(
        &mut self,
        vector: &[f32],
        metadata: Option<Vec<(String, Metadata)>>,
        rng: &mut R,
    ) -> usize {
        // (Optionally normalize vector if desired.)
        self.vectors.extend_from_slice(vector);
        let level = self.random_level(rng);
        self.levels.push(level + 1); // stored as level+1
        let new_offset =
            *self.offsets.last().unwrap() + self.cum_nb_neighbors((level + 1) as usize) as usize;
        self.offsets.push(new_offset);
        self.neighbors.resize(*self.offsets.last().unwrap(), -1);
        if let Some(ref mut meta_storage) = self.metadata {
            // Push the metadata vector for this new vector.
            meta_storage.push(metadata.unwrap_or_else(|| Vec::new()));
        }
        let new_index = self.levels.len() - 1;

        if self.entry_point == -1 {
            self.entry_point = new_index as StorageIdx;
            self.max_level = level;
            return new_index;
        }
        let mut current_entry = self.entry_point;
        for l in ((level + 1) as usize..=self.max_level as usize).rev() {
            let (ne, _) = self.greedy_update_nearest(
                vector,
                current_entry,
                (self.metric)(vector, self.get_vector(current_entry as usize)),
                l,
            );
            current_entry = ne;
        }
        for l in (0..=level as usize).rev() {
            let candidates =
                self.search_base_layer(vector, current_entry, self.ef_construction as usize, None);
            let max_links = self.nb_neighbors(l);
            let final_neighbors = self.shrink_candidate_list(candidates, max_links);
            let (begin, end) = self.neighbor_range(new_index, l);
            let mut pos = begin;
            for &cand in final_neighbors.iter() {
                self.neighbors[pos] = cand;
                pos += 1;
            }
            while pos < end {
                self.neighbors[pos] = -1;
                pos += 1;
            }
            for &cand in final_neighbors.iter() {
                if self.levels[cand as usize] - 1 >= l as i32 {
                    self.add_link(cand as usize, new_index as StorageIdx, l);
                }
            }
            if !final_neighbors.is_empty() {
                current_entry = final_neighbors[0];
            }
        }
        if level > self.max_level {
            self.max_level = level;
            self.entry_point = new_index as StorageIdx;
        }
        new_index
    }

    /// Choose a random level for a new node.
    pub fn random_level<R: Rng>(&self, rng: &mut R) -> i32 {
        let mut f = rng.random::<f64>();
        for (level, &p) in self.assign_probas.iter().enumerate() {
            if f < p {
                return level as i32;
            }
            f -= p;
        }
        (self.assign_probas.len() - 1) as i32
    }

    /// Full search procedure.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&dyn Fn(&Vec<(String, Metadata)>) -> bool>,
    ) -> Vec<Candidate> {
        if self.entry_point == -1 {
            return Vec::new();
        }
        let mut current = self.entry_point;
        let mut d = (self.metric)(query, self.get_vector(current as usize));
        for l in (1..=self.max_level as usize).rev() {
            let (new_current, new_d) = self.greedy_update_nearest(query, current, d, l);
            current = new_current;
            d = new_d;
        }
        let mut candidates =
            self.search_base_layer(query, current, self.ef_search as usize, filter);
        candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        candidates.truncate(k);
        candidates
    }

    /// Euclidean distance.
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()
    }

    /// Dot product.
    fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0;
        for i in 0..a.len().min(b.len()) {
            sum += a[i] * b[i];
        }
        sum
    }

    /// Cosine similarity.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot = Self::dot_product(a, b);
        let norm_a = Self::dot_product(a, a).sqrt();
        let norm_b = Self::dot_product(b, b).sqrt();
        dot / (norm_a * norm_b)
    }
}

#[pyclass]
pub struct PyHNSW {
    inner: HNSW,
}

#[pymethods]
impl PyHNSW {
    /// Create a new HNSW index.
    ///
    /// * `dim`: dimension of the vectors.
    /// * `m`: maximum number of neighbors per node (default 16).
    /// * `ef_construction`: expansion factor during insertion (default 200).
    /// * `ef_search`: expansion factor during query (default 50).
    /// * `metric`: an optional string ("l2", "angular", "inner_product") defaults to "l2".
    /// * `schema`: an optional list of attribute names (not used in the new version).
    #[new]
    #[pyo3(signature = (dim, m=None, ef_construction=None, ef_search=None, metric=None, _schema=None))]
    pub fn new(
        dim: usize,
        m: Option<usize>,
        ef_construction: Option<usize>,
        ef_search: Option<usize>,
        metric: Option<String>,
        _schema: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let m = m.unwrap_or(16) as i32;
        let ef_construction = ef_construction.unwrap_or(200) as i32;
        let ef_search = ef_search.unwrap_or(50) as i32;
        let metric_fn: fn(&[f32], &[f32]) -> f32 = match metric.as_deref() {
            // Some("cosine") => |a: &[f32], b: &[f32]| {
            //     let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            //     let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            //     let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            //     1.0 - (dot / (norm_a * norm_b))
            // },
            // Some("dot_product") => {
            //     |a: &[f32], b: &[f32]| -(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>())
            // }
            Some("cosine") => HNSW::cosine_similarity,
            Some("dot_product") => HNSW::dot_product,
            Some("euclidean") | None => HNSW::euclidean_distance,
            _ => {
                return Err(PyValueError::new_err(
                    "Unsupported metric. Expected 'cosine', 'dot_product', or 'euclidean'.",
                ))
            }
        };
        Ok(PyHNSW {
            inner: HNSW::new(dim, m, ef_construction, ef_search, Some(metric_fn)),
        })
    }

    /// Insert a vector into the index.
    ///
    /// The vector is expected as a list of floats.
    /// Optionally, metadata can be provided as a dictionary (e.g. {"category": "blue", ...})
    /// which will be converted into a Vec<(String, PyObject)>.
    #[pyo3(signature = (vector, metadata=None))]
    pub fn insert(
        &mut self,
        vector: Vec<f64>,
        metadata: Option<Vec<(String, PyObject)>>,
        py: Python,
    ) -> PyResult<()> {
        // Convert input vector (f64) to f32.
        let vector_f32: Vec<f32> = vector.into_iter().map(|x| x as f32).collect();

        // Convert metadata: since we support multiple metadata fields per vector,
        // we expect metadata: Option<Vec<(String, PyObject)>>.
        let meta_converted: Option<Vec<(String, Metadata)>> = if let Some(meta_tuples) = metadata {
            let mut out = Vec::with_capacity(meta_tuples.len());
            for (attr, py_obj) in meta_tuples {
                let md = if let Ok(s) = py_obj.extract::<String>(py) {
                    Metadata::Str(s)
                } else if let Ok(i) = py_obj.extract::<i64>(py) {
                    Metadata::Int(i)
                } else if let Ok(f) = py_obj.extract::<f64>(py) {
                    Metadata::Float(f)
                } else {
                    return Err(PyValueError::new_err(format!(
                        "Unsupported metadata type for attribute '{}'",
                        attr
                    )));
                };
                out.push((attr, md));
            }
            Some(out)
        } else {
            None
        };

        let mut rng = rand::rng();
        self.inner.insert(&vector_f32, meta_converted, &mut rng);
        Ok(())
    }

    /// Search for the k nearest neighbors.
    ///
    /// Returns a list of tuples `(distance, vector)`.
    /// If `filter` is provided, it is converted to a Metadata value and only candidates
    /// whose metadata equal that value are returned.
    #[pyo3(signature = (query, k=None, filter=None))]
    pub fn search(
        &self,
        query: Vec<f64>,
        k: Option<usize>,
        filter: Option<PyObject>,
        py: Python,
    ) -> PyResult<Vec<(f64, Vec<f64>)>> {
        let k = k.unwrap_or(1);
        let query_f32: Vec<f32> = query.into_iter().map(|x| x as f32).collect();
        let filter_fn: Option<Box<dyn Fn(&Vec<(String, Metadata)>) -> bool>> = if let Some(f_obj) =
            filter
        {
            // Expect filter to be a tuple: (attribute, value)
            let (attribute, expected_value) = f_obj
                .extract::<(String, PyObject)>(py)
                .map_err(|_| PyValueError::new_err("Filter must be a tuple (attribute, value)"))?;
            let expected_md = if let Ok(s) = expected_value.extract::<String>(py) {
                Metadata::Str(s)
            } else if let Ok(i) = expected_value.extract::<i64>(py) {
                Metadata::Int(i)
            } else if let Ok(f) = expected_value.extract::<f64>(py) {
                Metadata::Float(f)
            } else {
                return Err(PyValueError::new_err(
                    "Unsupported type for metadata filter.",
                ));
            };
            Some(Box::new(move |meta_vec: &Vec<(String, Metadata)>| {
                meta_vec
                    .iter()
                    .any(|(attr, md)| attr == &attribute && md == &expected_md)
            }))
        } else {
            None
        };
        let candidates = self.inner.search(&query_f32, k, filter_fn.as_deref());
        let results = candidates
            .into_iter()
            .map(|c| {
                let vec_f64: Vec<f64> = self
                    .inner
                    .get_vector(c.id as usize)
                    .iter()
                    .map(|&x| x as f64)
                    .collect();
                (c.distance as f64, vec_f64)
            })
            .collect();
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    // use std::cmp::Ordering;
    // use std::f32::EPSILON;
    // use std::sync::Arc;

    #[test]
    fn test_hnsw_basic_initialization() {
        // Our new HNSW::new takes (dim, m, ef_construction, ef_search, metric)
        let hnsw1 = HNSW::new(2, 16, 200, 50, None);
        // In our new version, the number of layers is the length of assign_probas.
        assert_eq!(hnsw1.assign_probas.len(), 5); // for m=16 we expect 5 levels
        assert_eq!(hnsw1.ef_construction, 200);
        assert_eq!(hnsw1.ef_search, 50);

        let hnsw2 = HNSW::new(2, 8, 100, 10, None);
        // For m=8 we expect fewer layers (e.g. 3)
        assert_eq!(hnsw2.assign_probas.len(), 7);
        assert_eq!(hnsw2.ef_construction, 100);
        assert_eq!(hnsw2.ef_search, 10);
    }

    #[test]
    fn test_hnsw_insert_and_search() {
        let mut hnsw = HNSW::new(2, 16, 200, 50, None);
        let seed: u64 = 42;
        let mut rng = StdRng::seed_from_u64(seed);

        // Points as arrays of [2]f32.
        let points: &[[f32; 2]] = &[
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [0.0, 0.0],
        ];

        for pt in points {
            // For simplicity, we pass None for metadata.
            hnsw.insert(pt, None, &mut rng);
        }
        assert_eq!(hnsw.vectors.len() / hnsw.dim, 5);

        // Since linking isn’t fully implemented, force the entry point to the center.
        hnsw.entry_point = 4;
        let query = vec![0.1, 0.0];
        let results = hnsw.search(&query, 2, None);
        assert!(!results.is_empty());
        let best_candidate = &results[0];
        // Expect that the returned candidate’s vector is very near [0.0, 0.0]
        let center = hnsw.get_vector(best_candidate.id as usize);
        for (a, b) in center.iter().zip([0.0, 0.0].iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_hnsw_distance_metrics() {
        let mut hnsw = HNSW::new(2, 16, 200, 50, None);
        let seed: u64 = 1234;
        let mut rng = StdRng::seed_from_u64(seed);

        let points: &[[f32; 2]] = &[
            [1.0, 0.0],     // ID 0
            [-0.5, 0.866],  // ID 1
            [-0.5, -0.866], // ID 2
        ];

        for pt in points {
            hnsw.insert(pt, None, &mut rng);
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

    // Helper to build a simple color filter.
    // Now each candidate’s metadata is a Vec<(String, Metadata)>.
    fn color_filter(color: &str) -> impl Fn(&Vec<(String, Metadata)>) -> bool {
        let color_owned = color.to_owned();
        move |meta_vec: &Vec<(String, Metadata)>| {
            meta_vec.iter().any(|(attr, md)| {
                attr == "color"
                    && match md {
                        Metadata::Str(s) => s == &color_owned,
                        _ => false,
                    }
            })
        }
    }

    #[test]
    fn test_metadata_basic_insert_and_filter() {
        // In the new version we simulate metadata by storing a single value per vector.
        let mut hnsw = HNSW::new(2, 16, 200, 50, None);
        // Initialize metadata storage.
        hnsw.metadata = Some(Vec::new());
        let seed: u64 = 1234;
        let mut rng = StdRng::seed_from_u64(seed);
        // Insert vector [1.0, 2.0] with color "blue".
        {
            let vec_a = [1.0f32, 2.0].to_vec();
            hnsw.insert(
                &vec_a,
                Some(vec![(
                    "color".to_string(),
                    Metadata::Str("blue".to_string()),
                )]),
                &mut rng,
            );
        }
        // Insert vector [2.0, 3.0] with color "red".
        {
            let vec_b = [2.0f32, 3.0].to_vec();
            hnsw.insert(
                &vec_b,
                Some(vec![(
                    "color".to_string(),
                    Metadata::Str("red".to_string()),
                )]),
                &mut rng,
            );
        }
        // Insert vector [10.0, 10.0] with color "blue".
        {
            let vec_c = [10.0f32, 10.0].to_vec();
            hnsw.insert(
                &vec_c,
                Some(vec![(
                    "color".to_string(),
                    Metadata::Str("blue".to_string()),
                )]),
                &mut rng,
            );
        }
        let blue_filter = color_filter("blue");
        let query = vec![1.5f32, 2.5f32];
        let results = hnsw.search(&query, 10, Some(&blue_filter));
        // Verify that every returned candidate has metadata matching "blue".
        for candidate in results {
            if let Some(ref meta_vec) = hnsw.metadata {
                let meta = &meta_vec[candidate.id as usize];
                assert!(blue_filter(meta));
            }
        }
    }
}
