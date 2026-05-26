use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

mod concordance;
pub mod expressions;
mod lindera_dict;
mod offsets;
mod token_frequencies;
mod tokenizer;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

#[pymodule]
fn _internal(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    _m.add_function(wrap_pyfunction!(token_frequencies_py, _m)?)?;
    _m.add_function(wrap_pyfunction!(prefetch_tokenizer_py, _m)?)?;
    _m.add_function(wrap_pyfunction!(loaded_tokenizers_py, _m)?)?;
    Ok(())
}

#[pyfunction(name = "token_frequencies")]
#[pyo3(signature = (texts))]
fn token_frequencies_py(py: Python<'_>, texts: Vec<String>) -> PyResult<Py<PyAny>> {
    token_frequencies::token_frequencies_py(py, texts)
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
