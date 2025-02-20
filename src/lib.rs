mod candidate;
mod filter;
pub mod flat;
pub mod hnsw;
pub mod metadata;
pub mod metric;

use crate::flat::PyFlat;
use crate::hnsw::PyHNSW;
use pyo3::prelude::*;

#[pymodule]
fn nilvec(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFlat>()?;
    m.add_class::<PyHNSW>()?;
    Ok(())
}
