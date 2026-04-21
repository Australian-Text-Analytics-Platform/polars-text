use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_polars::PolarsAllocator;

mod concordance;
pub mod expressions;
mod plan_paths;
mod pos_tagging;
mod token_frequencies;
mod tokenizer;
mod topic_modeling;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

#[pymodule]
fn _internal(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    _m.add_function(wrap_pyfunction!(topic_modeling_py, _m)?)?;
    _m.add_function(wrap_pyfunction!(token_frequencies_py, _m)?)?;
    _m.add_function(wrap_pyfunction!(list_source_paths_py, _m)?)?;
    _m.add_function(wrap_pyfunction!(replace_source_paths_py, _m)?)?;
    Ok(())
}

#[pyfunction(name = "list_source_paths")]
fn list_source_paths_py(path: &str) -> PyResult<Vec<String>> {
    plan_paths::list_source_paths(std::path::Path::new(path))
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction(name = "replace_source_paths")]
fn replace_source_paths_py(path: &str, mapper: &Bound<'_, PyDict>) -> PyResult<usize> {
    let mut map = std::collections::HashMap::with_capacity(mapper.len());
    for (k, v) in mapper.iter() {
        let key: String = k.extract()?;
        let value: String = v.extract()?;
        map.insert(key, value);
    }
    plan_paths::replace_source_paths(std::path::Path::new(path), &map)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction(name = "topic_modeling")]
#[pyo3(signature = (texts, min_points=5, eps=None, max_terms=5, seed=42))]
fn topic_modeling_py(
    py: Python<'_>,
    texts: Vec<String>,
    min_points: usize,
    eps: Option<f32>,
    max_terms: usize,
    seed: u64,
) -> PyResult<(Py<PyAny>, Vec<Vec<(i64, f32)>>)> {
    topic_modeling::topic_modeling_py(py, texts, min_points, eps, max_terms, seed)
}

#[pyfunction(name = "token_frequencies")]
#[pyo3(signature = (texts))]
fn token_frequencies_py(py: Python<'_>, texts: Vec<String>) -> PyResult<Py<PyAny>> {
    token_frequencies::token_frequencies_py(py, texts)
}
