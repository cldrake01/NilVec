mod hnsw;

use pyo3::prelude::*;
use crate::hnsw::PyHNSW;

#[pymodule]
fn nilvec(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHNSW>()?;
    Ok(())
}
