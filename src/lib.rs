use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

mod cache;
mod concordance;
pub mod expressions;
mod lindera_dict;
mod offsets;
mod token_frequencies;
mod tokenizer;
pub mod topic_modeling;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

#[pymodule]
fn _internal(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    _m.add_function(wrap_pyfunction!(token_frequencies_py, _m)?)?;
    _m.add_function(wrap_pyfunction!(prefetch_tokenizer_py, _m)?)?;
    _m.add_function(wrap_pyfunction!(loaded_tokenizers_py, _m)?)?;
    _m.add_function(wrap_pyfunction!(prefetch_embedder_py, _m)?)?;
    _m.add_function(wrap_pyfunction!(loaded_embedders_py, _m)?)?;
    Ok(())
}

#[pyfunction(name = "token_frequencies")]
#[pyo3(signature = (texts, model))]
fn token_frequencies_py(py: Python<'_>, texts: Vec<String>, model: String) -> PyResult<Py<PyAny>> {
    token_frequencies::token_frequencies_py(py, texts, Some(model.as_str()))
}

#[pyfunction(name = "prefetch_tokenizer")]
#[pyo3(signature = (model_id))]
fn prefetch_tokenizer_py(model_id: &str) -> PyResult<()> {
    tokenizer::ensure_tokenizer_for_model(Some(model_id))
        .map(|_| ())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

#[pyfunction(name = "loaded_tokenizers")]
fn loaded_tokenizers_py() -> Vec<String> {
    tokenizer::loaded_model_ids()
}

/// Download/load the ONNX Runtime embedder for `repo_id` (default model if `None`) so
/// a later `run_topic_modeling` call doesn't pay the load cost. Mirrors
/// `prefetch_tokenizer`.
#[pyfunction(name = "prefetch_embedder")]
#[pyo3(signature = (repo_id=None))]
fn prefetch_embedder_py(repo_id: Option<String>) -> PyResult<()> {
    topic_modeling::embedding::ensure_embedder(repo_id.as_deref())
        .map(|_| ())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
}

/// Repo ids of embedders currently resident in the in-process registry.
#[pyfunction(name = "loaded_embedders")]
fn loaded_embedders_py() -> Vec<String> {
    topic_modeling::embedding::loaded_embedder_ids()
}
